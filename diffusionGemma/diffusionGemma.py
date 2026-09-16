import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

torch.manual_seed(42)


@dataclass
class DiffusionGemmaConfig:
    vocab_size:    int   = 32000
    d_model:       int   = 512
    n_heads:       int   = 8
    n_kv_heads:    int   = 4
    n_layers:      int   = 6
    d_ff:          int   = 2048
    canvas_size:   int   = 64
    dropout:       float = 0.0
    max_seq_len:   int   = 2048
    T_max:         int   = 48
    tau_max:       float = 0.8
    tau_min:       float = 0.4
    eb_gamma:      float = 0.1


config = DiffusionGemmaConfig()


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t    = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:, :, :seq_len]
        sin = self.sin_cached[:, :, :seq_len]
        return q*cos + self.rotate_half(q)*sin, k*cos + self.rotate_half(k)*sin


class GemmaAttention(nn.Module):
    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups   = config.n_heads // config.n_kv_heads
        self.head_dim   = config.d_model // config.n_heads
        self.scale      = self.head_dim ** -0.5

        d = config.d_model
        self.q_proj = nn.Linear(d, config.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, d,    bias=False)

        self.rotary  = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x:              torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, _ = x.shape

        Q = self.q_proj(x).view(B, S, self.n_heads,    self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        Q, K = self.rotary(Q, K, seq_len=S)

        if self.n_groups > 1:
            K = K.repeat_interleave(self.n_groups, dim=1)
            V = V.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn = torch.nan_to_num(F.softmax(scores, dim=-1))
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class GemmaMLP(nn.Module):
    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class DiffusionGemmaLayer(nn.Module):
    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__()
        self.norm1     = RMSNorm(config.d_model)
        self.norm2     = RMSNorm(config.d_model)
        self.attention = GemmaAttention(config)
        self.mlp       = GemmaMLP(config)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class SelfConditioningMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm      = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.up_proj   = nn.Linear(d_model, d_model * 2, bias=False)
        self.down_proj = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        S = self.norm(S)
        gate = F.gelu(self.gate_proj(S))
        up   = self.up_proj(S)
        return self.down_proj(gate * up)


class DiffusionGemma(nn.Module):
    def __init__(self, config: DiffusionGemmaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.self_cond_mlp = SelfConditioningMLP(config.d_model)

        self.layers = nn.ModuleList([
            DiffusionGemmaLayer(config)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

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
        prompt_ids:    torch.Tensor,
        canvas_ids:    torch.Tensor,
        self_cond:     Optional[torch.Tensor] = None,
        prompt_mask:   Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, P = prompt_ids.shape
        C = canvas_ids.shape[1]
        scale = math.sqrt(self.config.d_model)

        prompt_embs = self.embed_tokens(prompt_ids) * scale

        canvas_embs = self.embed_tokens(canvas_ids) * scale

        if self_cond is None:
            self_cond = torch.zeros(B, C, self.config.d_model,
                                    device=canvas_ids.device,
                                    dtype=canvas_embs.dtype)
        canvas_embs = canvas_embs + self.self_cond_mlp(self_cond)

        x = torch.cat([prompt_embs, canvas_embs], dim=1)

        if prompt_mask is not None:
            canvas_mask = torch.ones(B, C, device=canvas_ids.device, dtype=torch.long)
            full_mask   = torch.cat([prompt_mask, canvas_mask], dim=1)
            ext_mask    = full_mask.unsqueeze(1).unsqueeze(2)
        else:
            ext_mask = None

        for layer in self.layers:
            x = layer(x, ext_mask)
        x = self.norm(x)

        canvas_out = x[:, P:, :]

        logits = self.lm_head(canvas_out)

        return logits

    def compute_self_cond(
        self,
        logits:      torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        probs = F.softmax(logits / temperature, dim=-1)

        S = torch.matmul(probs, self.embed_tokens.weight)

        return S


def entropy_bounded_sampling(
    logits:      torch.Tensor,
    temperature: float,
    gamma:       float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, V = logits.shape
    device   = logits.device

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    probs_flat = probs.view(B * C, V)
    candidates = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
    candidates = candidates.view(B, C)

    log_probs = torch.log(probs.clamp(min=1e-10))
    entropies = -(probs * log_probs).sum(dim=-1)

    committed = torch.zeros(B, C, dtype=torch.bool, device=device)

    for b in range(B):
        sorted_idx     = entropies[b].argsort()
        sorted_entropy = entropies[b][sorted_idx]

        cumsum = 0.0
        for i, idx in enumerate(sorted_idx.tolist()):
            H_i = sorted_entropy[i].item()

            new_sum = cumsum + H_i
            if new_sum - H_i > gamma:
                break

            committed[b, idx] = True
            cumsum = new_sum

    random_tokens = torch.randint(0, V, (B, C), device=device)
    new_canvas    = torch.where(committed, candidates, random_tokens)

    return new_canvas, committed


def compute_temperature(
    step:  int,
    T_max: int,
    tau_max: float = 0.8,
    tau_min: float = 0.4
) -> float:
    if T_max <= 1:
        return tau_min
    t = step / (T_max - 1)
    return tau_max + t * (tau_min - tau_max)


def should_stop(
    prev_canvas: torch.Tensor,
    curr_canvas: torch.Tensor,
    min_steps:   int = 4
) -> bool:
    return (prev_canvas == curr_canvas).all().item()


@torch.no_grad()
def generate(
    model:        DiffusionGemma,
    prompt_ids:   torch.Tensor,
    config:       DiffusionGemmaConfig,
    max_canvases: int = 1,
    prompt_mask:  Optional[torch.Tensor] = None,
    verbose:      bool = False
) -> torch.Tensor:
    model.eval()
    B, P = prompt_ids.shape
    C    = config.canvas_size
    V    = config.vocab_size
    device = prompt_ids.device

    all_generated = []

    for canvas_idx in range(max_canvases):
        if verbose:
            print(f"\n--- Canvas {canvas_idx + 1}/{max_canvases} ---")

        canvas = torch.randint(0, V, (B, C), device=device)

        self_cond = torch.zeros(B, C, config.d_model, device=device)

        prev_canvas = None

        for step in range(config.T_max):

            tau = compute_temperature(step, config.T_max, config.tau_max, config.tau_min)

            logits = model(
                prompt_ids   = prompt_ids,
                canvas_ids   = canvas,
                self_cond    = self_cond,
                prompt_mask  = prompt_mask
            )

            self_cond = model.compute_self_cond(logits, tau)

            new_canvas, committed = entropy_bounded_sampling(
                logits, temperature=tau, gamma=config.eb_gamma
            )

            n_committed = committed.float().mean().item()
            if verbose:
                print(f"  Step {step+1:3d} | tau={tau:.3f} | "
                      f"committed={n_committed*100:.1f}% | "
                      f"canvas_changed={not (canvas == new_canvas).all().item()}")

            if prev_canvas is not None and step >= 4:
                if should_stop(prev_canvas, new_canvas):
                    if verbose:
                        print(f"  Converged at step {step+1}")
                    canvas = new_canvas
                    break

            prev_canvas = canvas
            canvas      = new_canvas

        all_generated.append(canvas)

        if canvas_idx < max_canvases - 1:
            prompt_ids  = torch.cat([prompt_ids, canvas], dim=1)
            if prompt_mask is not None:
                canvas_mask = torch.ones(B, C, device=device, dtype=torch.long)
                prompt_mask = torch.cat([prompt_mask, canvas_mask], dim=1)

    return torch.cat(all_generated, dim=1)


def compute_diffusion_loss(
    model:      DiffusionGemma,
    input_ids:  torch.Tensor,
    config:     DiffusionGemmaConfig,
    prompt_len: int = 0
) -> torch.Tensor:
    B = input_ids.shape[0]
    C = config.canvas_size
    V = config.vocab_size
    device = input_ids.device

    prompt_ids = input_ids[:, :prompt_len]
    canvas_ids = input_ids[:, prompt_len:prompt_len + C]

    assert canvas_ids.shape[1] == C, \
        f"Need at least {prompt_len + C} tokens, got {input_ids.shape[1]}"

    t = torch.rand(B, 1, device=device)

    noise_mask = torch.rand(B, C, device=device) < t

    random_tokens = torch.randint(0, V, (B, C), device=device)
    noisy_canvas  = torch.where(noise_mask, random_tokens, canvas_ids)

    tau_train = 1.0

    logits = model(
        prompt_ids = prompt_ids,
        canvas_ids = noisy_canvas,
        self_cond  = None
    )

    if noise_mask.any():
        logits_flat  = logits.view(B * C, V)
        targets_flat = canvas_ids.view(B * C)
        mask_flat    = noise_mask.view(B * C)

        loss_all  = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss_masked = loss_all[mask_flat]

        if len(loss_masked) > 0:
            t_expanded = t.expand(B, C).contiguous().view(B * C)[mask_flat]
            weighted_loss = (loss_masked / (t_expanded + 1e-8)).mean()
            return weighted_loss

    return torch.tensor(0.0, device=device, requires_grad=True)


def train_step(
    model:      DiffusionGemma,
    optimizer:  torch.optim.Optimizer,
    input_ids:  torch.Tensor,
    config:     DiffusionGemmaConfig,
    prompt_len: int = 8
) -> float:
    model.train()

    loss = compute_diffusion_loss(model, input_ids, config, prompt_len)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def decode_canvas(
    canvas: torch.Tensor,
    tokenizer,
    skip_special_tokens: bool = True
) -> List[str]:
    return [
        tokenizer.decode(canvas[b].tolist(), skip_special_tokens=skip_special_tokens)
        for b in range(canvas.shape[0])
    ]


def measure_generation_speed(
    model:      DiffusionGemma,
    prompt_ids: torch.Tensor,
    config:     DiffusionGemmaConfig,
    n_runs:     int = 5
) -> dict:
    import time

    model.eval()
    times = []

    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            output = generate(model, prompt_ids, config)
        elapsed = time.time() - start
        tokens  = output.shape[0] * output.shape[1]
        times.append(tokens / elapsed)

    return {
        'mean_tokens_per_sec':   sum(times) / len(times),
        'tokens_per_generation': config.canvas_size,
        'batch_size':            prompt_ids.shape[0]
    }


def run_demo():
    from collections import defaultdict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    demo_config = DiffusionGemmaConfig(
        vocab_size  = 1000,
        d_model     = 128,
        n_heads     = 4,
        n_kv_heads  = 2,
        n_layers    = 2,
        d_ff        = 512,
        canvas_size = 32,
        T_max       = 20,
        tau_max     = 0.8,
        tau_min     = 0.4,
        eb_gamma    = 0.1
    )

    model = DiffusionGemma(demo_config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    n_train = 200
    seq_len = demo_config.canvas_size + 8
    train_data = torch.randint(4, demo_config.vocab_size, (n_train, seq_len))

    print("\n--- Training (masked diffusion objective) ---")
    batch_size = 16
    n_epochs   = 10

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, n_train, batch_size):
            batch = train_data[i:i+batch_size].to(device)
            loss  = train_step(model, optimizer, batch, demo_config, prompt_len=8)
            total_loss += loss
            n_batches  += 1
        print(f"Epoch {epoch+1}/{n_epochs} | Avg Loss: {total_loss/n_batches:.4f}")

    print("\n--- Generation (denoising loop) ---")
    model.eval()

    prompt = torch.randint(4, demo_config.vocab_size, (1, 8)).to(device)

    print("Starting with random canvas tokens...")
    print("Running entropy-bounded denoising...")

    output = generate(
        model       = model,
        prompt_ids  = prompt,
        config      = demo_config,
        max_canvases = 1,
        verbose     = True
    )

    print(f"\nGenerated canvas shape: {output.shape}")
    print(f"Sample token IDs: {output[0, :10].tolist()}")

    print("\n--- Multi-canvas generation (2 canvases = 64 tokens) ---")
    output_multi = generate(
        model        = model,
        prompt_ids   = prompt,
        config       = demo_config,
        max_canvases = 2,
        verbose      = False
    )
    print(f"Multi-canvas output shape: {output_multi.shape}")

    print("\n--- Self-conditioning visualization ---")
    canvas = torch.randint(0, demo_config.vocab_size, (1, demo_config.canvas_size)).to(device)
    S_zero = torch.zeros(1, demo_config.canvas_size, demo_config.d_model).to(device)

    with torch.no_grad():
        logits_step1 = model(prompt, canvas, S_zero)
        S_step1      = model.compute_self_cond(logits_step1, tau=0.8)

        logits_step2 = model(prompt, canvas, S_step1)

    print(f"Logits step 1 (S=zeros): mean={logits_step1.mean().item():.4f}")
    print(f"Logits step 2 (S=S^1):   mean={logits_step2.mean().item():.4f}")
    print("Self-conditioning changes the logits — previous step's uncertainty informs the next.")

    print("\nDemo complete.")


if __name__ == '__main__':
    run_demo()