import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

torch.manual_seed(42)

@dataclass
class KimiK3config:
    d_model: int = 512
    n_layer:int = 16
    n_heads: int = 8
    d_head:int = 64

    kda_ratio: int = 3

    n_experts: int = 16
    n_active:int = 4
    n_shared: int = 2
    d_latent_moe: int = 256
    d_expert: int = 512

    n_attn_blocks: int = 4

    g_min: float = -5.0
    chuck_size: int = 64

    beta1_situ: float = 4.0
    beta2_situ: float = 25.0

    vocab_size: int = 32000
    max_seq_len: int = 4096
    dropout: float = 0.0
    eps: float = 1e-6

config = KimiK3config()


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.gamma

class ShortConv(nn.Module):
    def __init__(self,d,kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(d,d,kernel_size,padding=kernel_size-1,groups=d,bias=True)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x=x.transpose(1,2)
        x=self.conv(x)
        x=x[:,:,:x.shape[2]]
        return x.transpose(1,2)

class SiTUGLU(nn.Module):
    def __init__(self,d_in,d_out,beta1=4.0,beta2=25.0):
        super().__init__()
        self.beta1=beta1
        self.beta2=beta2
        self.W_gate= nn.Linear(d_in,d_out, bias=False)
        self.W_up = nn.Linear(d_in,d_out,bias=False)

        def forward(self,x:torch.Tensor) -> torch.Tensor:
            gate_input = self.W_gate(x)
            up_input = self.W_up(x)

            gate = self.beta1 * torch.tanh(gate_input/self.beta1) * torch.sigmoid(gate_input)

            up = self.beta2 * torch.tanh(up_input/self.beta2)

            return gate * up

class QuantileBalancing:
    def __init__(self,n_experts,n_active,n_bins = 200):
        self.n_experts = n_experts
        self.n_active = n_active
        self.n_bins = n_bins

        self.biases = torch.zeros(n_experts)

    def compute_cutoffs(self,scores:torch.Tensor) -> torch.Tensor:
        topk1_scores, _ = scores.topk(self.n_active + 1, dim=-1)
        return topk1_scores[:, -1]

    def update_biases(self, scores:torch.Tensor)-> torch.Tensor:
        batch_size = scores.shape[0]
        target_load = batch_size * self.n_active / self.n_experts
        target_quantile = 1.0 - self.n_active / self.n_experts

        with torch.no_grad():
            biased_scores = scores + self.biases.to(scores.device)
            cutoffs = self.compute_cutoffs(biased_scores)

            margins = scores - cutoffs.unsqueeze(1)

            new_biases = torch.zeros(self.n_experts)
            for j in range(self.n_experts):
                expert_margins = margins[:, j]
                quantile_val = torch.quantile(expert_margins.float(), 1.0 - target_quantile)
                new_biases[j] = -quantile_val

            new_biases = new_biases - new_biases.mean()
            self.biases = new_biases
        return self.biases

    def route(self,scores:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        biased_scores = scores + self.biases.to(scores.device)
        _, topk_indices = biased_scores.topk(self.n_active, dim=-1)

        selected_scores = scores.gather(1, topk_indices)
        weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)

        return topk_indices, weights


class ExpertFFN(nn.Module):
    def __init__(self,d_latent,d_inner,beta1,beta2):
        super().__init__()
        self.act = SiTUGLU(d_latent,d_inner,beta1,beta2)
        self.down = nn.Linear(d_inner,d_latent,bias=False)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.down(self.act(x))

class StableLatentMoe(nn.Module):
    def __init__(self,config:KimiK3config):
        super().__init__()
        d = config.d_model
        l = config.d_latent_moe

        self.shared_experts = nn.ModuleList([nn.Sequential(
            SitUGLU(d,config.d_expert,config.beta1_situ,config.beta2_situ),
            nn.Linear(config.d_expert,d,bias=False))
            for _ in range(config.n_shared)
        ])

        self.W_down = nn.Linear(d, l, bias=False)
        self.router = nn.Linear(d, config.n_experts, bias=False)

        self.experts = nn.ModuleList([ExpertFFN(l, config.d_expert, config.beta1_situ, config.beta2_situ) for _ in range(config.n_experts)])

        self.W_up = nn.Linear(l,d,bias=False)

        self.norm_before_up = RMSNorm(l)

        self.qb = QuantileBalancing(config.n_experts, config.n_active)
        self.n_active = config.n_active
        self.n_experts = config.n_experts

    def forward(self,x:torch.Tensor,update_biases:bool) -> torch.Tensor:
        B,S,d = x.shape

        shared_out = sum(expert(x) for expert in self.shared_experts)


        z = self.W_down(x)
        z_flat = z.view(B*S,-1)

        x_flat = x.view(B*S,d)
        router_scores = torch.sigmoid(self.router(x_flat))

        if update_biases and self.training:
            self.qb.update_biases(router_scores.detach())
        indices, weights = self.qb.route(router_scores)

        u = torch.zeros(B*S,z_flat.shape[1],device=x.device)

        for token_idx in range(B*S):
            token_z = z_flat[token_idx]
            token_indices = indices[token_idx]
            token_weights = weights[token_idx]

            for i, (expert_idx, w) in enumerate(zip(token_indices.to_list(), token_weights.to_list())):
                expert_output = self.experts[expert_idx](token_z.unsqueeze(0))
                u[token_idx] += w * expert_output.squeeze(0)

        u = self.norm_before_up(u)

        routed_out = self.W_up(u).view(B,S,d)

        return routed_out + shared_out


class KimiDeltaAttention(nn.Module):
    def __init__(self,config:KimiK3config):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.chuck_size = config.chuck_size
        self.g_min = config.g_min

        self.W_q = nn.Linear(d , config.n_heads * config.d_head, bias=False)
        self.W_k = nn.Linear(d, config.n_heads * config.d_head, bias=False)
        self.W_v = nn.Linear(d, config.n_heads * config.d_head, bias=False)

        self.conv_q = ShortConv(self.n_heads * self.d_head)
        self.conv_k = ShortConv(self.n_heads * self.d_head)

        self.W_beta = nn.Linear(d, self.n_heads, bias=True)

        d_rank = max(16,d//16)
        self.W_decay_down = nn.Linear(d,d_rank,bias=False)
        self.W_decay_up = nn.Linear(d_rank,self.n_heads *self.d_head,bias=False)

        self.b_decay = nn.Parameter(torch.zeros(self.n_heads ,1))

        self.W_gate = nn.Linear(d,self.n_heads * self.d_head, bias=False)

        self.W_o = nn.Linear(self.n_heads * self.d_head,d,bias=False)

        self.head_norm = RMSNorm(self.d_head)


    def _compute_decay(self,x:torch.Tensor) -> torch.Tensor:
        B,S,_ = x.shape
        z = self.W_decay_up(self.W_decay_down(x))
        z=z.view(B,S,self.n_heads,self.d_head)

        z = z + self.b_decay.unsqueeze(0).unsqueeze(0)

        A = self.log_A.exp().unsqueeze(0).unsqueeze(0)
        log_decay = self.g_min * torch.sigmoid(A*z)

        alpha = torch.exp(log_decay)

        return alpha

    def _recurrent_forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,alpha:torch.Tensor,beta:torch.Tensor) -> torch.Tensor:
        B,S,H,dk = q.shape
        device = q.device
        dv = v.shape[-1]
        outputs = []

        S = torch.zeros(B,H,dk,dv,device=device,dtype=q.dtype)

        for t in range(q.shape[1]):
            q_t = q[:,t]
            k_t = k[:,t]
            v_t = v[:,t]
            alpha_t = alpha[:,t]
            beta_t = beta[:,t]

            state = state * alpha_t.unsqueeze(-1)
            k_read = torch.einsum('bhk,bhkv->bhv', k_t, state)
            erase = torch.einsum('bhk,bhkv->bhv', beta_t,k_t, k_read)
            state = state - erase

            write = torch.einsum('bhk,bhkv->bhv', beta_t,k_t, v_t)
            state = state + write

            o_t = torch.einsum('bhkv,bhkv->bhv', q_t, state)
            outputs.append(o_t)

        return torch.stack(outputs, dim=1)

    def _chunkwise_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
       
        B, S, H, dk = q.shape
        C = self.chunk_size
        n_chunks = (S + C - 1) // C

        outputs = []
        state = torch.zeros(B, H, dk, dk, device=q.device, dtype=q.dtype)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * C
            end   = min(start + C, S)
            c_len = end - start


            q_c     = q[:, start:end]      
            k_c     = k[:, start:end]
            v_c     = v[:, start:end]
            alpha_c = alpha[:, start:end]  
            beta_c  = beta[:, start:end]  

            
            log_alpha_c = torch.log(alpha_c.clamp(min=1e-10))
            log_gamma = log_alpha_c.cumsum(dim=1)  
            gamma = torch.exp(log_gamma)           

            
            inter_out = torch.einsum(
                'bchd,bhdv->bchv',
                gamma * q_c,  
                state          
            )   

           
            k_scaled = k_c / gamma.clamp(min=1e-10)   

            
            q_gamma = (gamma * q_c).permute(0, 2, 1, 3)   
            k_scaled_p = k_scaled.permute(0, 2, 3, 1)     
            A = torch.matmul(q_gamma, k_scaled_p)      

          
            causal = torch.tril(torch.ones(c_len, c_len, device=q.device))
            A = A * causal.unsqueeze(0).unsqueeze(0)

           
            kS = torch.einsum('bchd,bhdv->bchv', k_c, state)  
            Ve = v_c - kS                                        

           
            Ve_weighted = Ve * beta_c.unsqueeze(-1)           

            
            Ve_p = Ve_weighted.permute(0, 2, 1, 3)            
            intra_out = torch.matmul(A, Ve_p)                  
            intra_out = intra_out.permute(0, 2, 1, 3)          

            chunk_out = inter_out + intra_out                 
            outputs.append(chunk_out)

           
            for t in range(c_len):
                k_t     = k_c[:, t]
                v_t     = v_c[:, t]
                alpha_t = alpha_c[:, t]
                beta_t  = beta_c[:, t]

                state = state * alpha_t.unsqueeze(-1)
                k_read = torch.einsum('bhd,bhdv->bhv', k_t, state)
                erase  = torch.einsum('bh,bhd,bhv->bhdv', beta_t, k_t, k_read)
                state  = state - erase
                write  = torch.einsum('bh,bhd,bhv->bhdv', beta_t, k_t, v_t)
                state  = state + write

        return torch.cat(outputs, dim=1)  

    def forward(
        self,
        x: torch.Tensor,                          
        state: Optional[torch.Tensor] = None,    
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, d = x.shape

      
        q = self.W_q(x)   
        k = self.W_k(x)
        v = self.W_v(x)  

        
        q = self.conv_q(q)
        k = self.conv_k(k)

       
        q = F.silu(q)
        k = F.silu(k)

        
        q = q.view(B, S, self.n_heads, self.d_head)
        k = k.view(B, S, self.n_heads, self.d_head)
        v = v.view(B, S, self.n_heads, self.d_head)
        v = F.silu(v)   

       
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        
        alpha = self._compute_decay(x)             
        beta  = torch.sigmoid(self.W_beta(x))      

       
        if S <= self.chunk_size or not self.training:
           
            out = self._recurrent_forward(q, k, v, alpha, beta)
        else:
            
            out = self._chunkwise_forward(q, k, v, alpha, beta)
       

        out = self.head_norm(out)  
        gate = torch.sigmoid(self.W_gate(x))  
        gate = gate.view(B, S, self.n_heads, self.d_head)
        out = gate * out
        out = out.contiguous().view(B, S, self.n_heads * self.d_head)
        out = self.W_o(out)   
        final_state = None  
        return out, final_state

class GatedMLA(nn.Module):
    def __init__(self, config: KimiK3Config):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.d_head  = config.d_head

        self.d_latent_kv = d // 4
        self.W_kv_compress = nn.Linear(d, self.d_latent_kv, bias=False)
        self.W_k_uncompress = nn.Linear(self.d_latent_kv, self.n_heads * self.d_head, bias=False)
        self.W_v_uncompress = nn.Linear(self.d_latent_kv, self.n_heads * self.d_head, bias=False)

        self.W_q = nn.Linear(d, self.n_heads * self.d_head, bias=False)

        self.W_gate = nn.Linear(d, self.n_heads * self.d_head, bias=False)
        self.W_o    = nn.Linear(self.n_heads * self.d_head, d, bias=False)

        self.scale = self.d_head ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, d = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_head)
        Q = Q.transpose(1, 2)

        c = self.W_kv_compress(x)

        if kv_cache is not None:
            c = torch.cat([kv_cache, c], dim=1)

        K = self.W_k_uncompress(c).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v_uncompress(c).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if kv_cache is None:
            causal_mask = torch.tril(torch.ones(S, S, device=x.device)).bool()
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2)

        gate = torch.sigmoid(self.W_gate(x)).view(B, S, self.n_heads, self.d_head)
        out = gate * out

        out = out.contiguous().view(B, S, -1)
        out = self.W_o(out)

        return out, c


class BlockAttnRes(nn.Module):
    def __init__(self, config: KimiK3Config):
        super().__init__()
        d = config.d_model

        self.pseudo_query = nn.Parameter(torch.randn(d) * 0.02)
        self.key_norm = RMSNorm(d)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def compute_attn_weights(
        self,
        sources: List[torch.Tensor],
    ) -> torch.Tensor:
        B = sources[0].shape[0]
        n = len(sources)

        keys = torch.stack(sources, dim=1)
        keys_normed = self.key_norm(keys)

        q = self.pseudo_query.unsqueeze(0).unsqueeze(0)
        scores = (q * keys_normed).sum(dim=-1)

        temperature = self.log_temperature.exp()
        weights = F.softmax(scores / temperature, dim=-1)

        return weights

    def forward(
        self,
        sources: List[torch.Tensor],
        current_partial: torch.Tensor,
    ) -> torch.Tensor:
        all_sources = sources + [current_partial]
        weights = self.compute_attn_weights(all_sources)

        values = torch.stack(all_sources, dim=1)
        out = (weights.unsqueeze(-1) * values).sum(dim=1)

        return out


class BlockAttnResManager:
    def __init__(self, n_layers: int, n_blocks: int, d_model: int):
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.block_size = n_layers // n_blocks
        self.d_model = d_model

        self.reset()

    def reset(self, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        self.block_summaries = []
        self.current_partial = None
        self.current_block = 0
        self.layer_in_block = 0

    def update(self, layer_output: torch.Tensor) -> List[torch.Tensor]:
        sources = list(self.block_summaries)

        if self.current_partial is not None:
            sources.append(self.current_partial)
        
        if self.current_partial is None:
            self.current_partial = layer_output.clone()
        else:
            self.current_partial = self.current_partial + layer_output

        self.layer_in_block += 1
        if self.layer_in_block >= self.block_size:
            self.block_summaries.append(self.current_partial.clone())
            self.current_partial = None
            self.layer_in_block = 0
            self.current_block += 1

        return sources

def newton_schulz_5(G: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
    assert G.dim() == 2, "Newton-Schulz expects 2D matrices"

    X = G / (G.norm() + 1e-8)

    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(n_steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X

    return X


class PerHeadMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        n_heads: int = 8,
        d_head: int = 64,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            n_heads=n_heads,
            d_head=d_head,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr          = group['lr']
            momentum    = group['momentum']
            nesterov    = group['nesterov']
            ns_steps    = group['ns_steps']
            n_heads     = group['n_heads']
            d_head      = group['d_head']
            is_attn     = group.get('is_attention', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                buf.mul_(momentum).add_(grad)

                if nesterov:
                    g = grad + momentum * buf
                else:
                    g = buf

                if is_attn and g.dim() == 2:
                    d_out, d_in = g.shape

                    if d_out == n_heads * d_head:
                        g_heads = g.view(n_heads, d_head, d_in)
                        g_ortho = torch.zeros_like(g_heads)

                        for h in range(n_heads):
                            head_grad = g_heads[h]
                            g_ortho[h] = newton_schulz_5(head_grad, n_steps=ns_steps)

                        update = g_ortho.view(n_heads * d_head, d_in)
                    else:
                        update = newton_schulz_5(g, n_steps=ns_steps)

                    update = update * (g.norm() / (update.norm() + 1e-8))

                else:
                    update = g

                p.add_(update, alpha=-lr)

class KimiK3Block(nn.Module):
    def __init__(self, config: KimiK3Config, is_mla: bool = False):
        super().__init__()
        self.is_mla = is_mla

        if is_mla:
            self.attention = GatedMLA(config)
        else:
            self.attention = KimiDeltaAttention(config)

        self.ffn = StableLatentMoE(config)

        self.norm_attn = RMSNorm(config.d_model)
        self.norm_ffn  = RMSNorm(config.d_model)

        self.attn_res = BlockAttnRes(config)

    def forward(
        self,
        x: torch.Tensor,
        attn_res_sources: List[torch.Tensor],
        current_partial: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, d = x.shape

        if attn_res_sources or current_partial is not None:
            attn_res_out_list = []
            for s_idx in range(S):
                sources_s = [src[:, s_idx, :] if src.dim() == 3 else src
                             for src in attn_res_sources]
                partial_s = current_partial[:, s_idx, :] if current_partial is not None else \
                            torch.zeros(B, d, device=x.device)
                res = self.attn_res(sources_s, partial_s)
                attn_res_out_list.append(res)
            h = torch.stack(attn_res_out_list, dim=1)
        else:
            h = x

        new_kv_cache = None
        if self.is_mla:
            attn_out, new_kv_cache = self.attention(self.norm_attn(h), kv_cache)
        else:
            attn_out, _ = self.attention(self.norm_attn(h))

        h = h + attn_out

        h = h + self.ffn(self.norm_ffn(h))

        return h, new_kv_cache


class KimiK3(nn.Module):
    def __init__(self, config: KimiK3Config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            is_mla = ((i + 1) % (config.kda_ratio + 1) == 0)
            self.blocks.append(KimiK3Block(config, is_mla=is_mla))

        self.final_mla = GatedMLA(config)
        self.final_norm = RMSNorm(config.d_model)

        self.attn_res_manager = BlockAttnResManager(
            n_layers=config.n_layers,
            n_blocks=config.n_attn_blocks,
            d_model=config.d_model
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List] = None,
    ) -> torch.Tensor:
        B, S = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)

        x = x * math.sqrt(self.config.d_model)

        self.attn_res_manager.reset(batch_size=B, device=device)

        embedding_summary = x.mean(dim=1)

        block_summaries = [embedding_summary]
        current_partial = None

        all_new_kv_caches = []
        mla_cache_idx = 0

        for layer_idx, block in enumerate(self.blocks):
            layer_cache = None
            if block.is_mla and kv_caches is not None:
                layer_cache = kv_caches[mla_cache_idx]

            sources = block_summaries[:]

            sources_pt = [s.unsqueeze(1).expand(B, S, -1) for s in sources]
            partial_pt = current_partial.unsqueeze(1).expand(B, S, -1) if \
                         current_partial is not None else None

            x, new_kv = block(x, sources_pt, partial_pt, layer_cache)

            if block.is_mla and new_kv is not None:
                all_new_kv_caches.append(new_kv)
                mla_cache_idx += 1

            layer_summary = x.mean(dim=1)

            if current_partial is None:
                current_partial = layer_summary
            else:
                current_partial = current_partial + layer_summary

            block_size = self.config.n_layers // self.config.n_attn_blocks
            if (layer_idx + 1) % block_size == 0:
                block_summaries.append(current_partial)
                current_partial = None

        x_normed = self.final_norm(x)
        x_final, _ = self.final_mla(x_normed)
        x = x + x_final

        logits = self.lm_head(x)

        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Building Kimi K3 (small config)...")
model = KimiK3(config)
total = count_parameters(model)
print(f"Total parameters: {total:,}")

batch_size, seq_len = 2, 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

with torch.no_grad():
    logits = model(input_ids)

print(f"\nInput shape:  {input_ids.shape}")
print(f"Output shape: {logits.shape}")
print(f"All shapes correct: {logits.shape == (batch_size, seq_len, config.vocab_size)}")


