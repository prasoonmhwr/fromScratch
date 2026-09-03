import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)

D_MODEL        = 768
D_FF           = 3072
D_INTERMEDIATE = 3072
D_EMBED        = 768
N_HEADS        = 12
N_KV_HEADS     = 4
N_LAYERS       = 24
VOCAB_SIZE     = 256000
MAX_SEQ_LEN    = 512
DROPOUT        = 0.0
MRL_DIMS       = [128, 256, 512, 768]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 5000, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :, :seq_len]
        sin = self.sin_cached[:, :, :seq_len]
        return (q*cos + self.rotate_half(q)*sin,
                k*cos + self.rotate_half(k)*sin)


class GemmaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        assert n_heads % n_kv_heads == 0

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups   = n_heads // n_kv_heads
        self.head_dim   = d_model // n_heads
        self.scale      = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model,    bias=False)

        self.rotary  = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class GemmaEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        self.norm1     = RMSNorm(d_model)
        self.norm2     = RMSNorm(d_model)
        self.attention = GemmaAttention(d_model, n_heads, n_kv_heads, dropout, max_seq_len)
        self.mlp       = GemmaMLP(d_model, d_ff)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class GemmaEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_kv_heads: int, n_layers: int, d_ff: int,
                 dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            GemmaEncoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(input_ids) * math.sqrt(D_MODEL)
        ext_mask = attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, ext_mask)
        return self.norm(x)


class MeanPooling(nn.Module):
    def forward(self, token_embs: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        m       = mask.unsqueeze(-1).expand_as(token_embs).float()
        sum_emb = (token_embs * m).sum(dim=1)
        n_real  = m.sum(dim=1).clamp(min=1e-9)
        return sum_emb / n_real


class EmbeddingProjectionHead(nn.Module):
    def __init__(self, d_model: int, d_intermediate: int,
                 d_embed: int, mrl_dims: List[int] = None):
        super().__init__()
        self.proj1    = nn.Linear(d_model, d_intermediate, bias=False)
        self.proj2    = nn.Linear(d_intermediate, d_embed, bias=False)
        self.mrl_dims = mrl_dims or [d_embed]

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def get_mrl_embeddings(self, x: torch.Tensor) -> List[torch.Tensor]:
        full = self.forward(x, normalize=False)
        return [F.normalize(full[:, :dim], p=2, dim=-1) for dim in self.mrl_dims]


class EmbeddingGemma(nn.Module):
    def __init__(
        self,
        vocab_size:     int   = VOCAB_SIZE,
        d_model:        int   = D_MODEL,
        n_heads:        int   = N_HEADS,
        n_kv_heads:     int   = N_KV_HEADS,
        n_layers:       int   = N_LAYERS,
        d_ff:           int   = D_FF,
        d_intermediate: int   = D_INTERMEDIATE,
        d_embed:        int   = D_EMBED,
        dropout:        float = DROPOUT,
        max_seq_len:    int   = MAX_SEQ_LEN,
        mrl_dims:       List[int] = None
    ):
        super().__init__()
        self.encoder = GemmaEncoder(
            vocab_size, d_model, n_heads, n_kv_heads,
            n_layers, d_ff, dropout, max_seq_len
        )
        self.pooling = MeanPooling()
        self.head    = EmbeddingProjectionHead(
            d_model, d_intermediate, d_embed, mrl_dims or MRL_DIMS
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                normalize: bool = True) -> torch.Tensor:
        token_embs = self.encoder(input_ids, attention_mask)
        pooled     = self.pooling(token_embs, attention_mask)
        return self.head(pooled, normalize)

    def get_mrl_embeddings(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> List[torch.Tensor]:
        token_embs = self.encoder(input_ids, attention_mask)
        pooled     = self.pooling(token_embs, attention_mask)
        return self.head.get_mrl_embeddings(pooled)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.02, hardness_alpha: float = 5.0):
        super().__init__()
        self.tau   = temperature
        self.alpha = hardness_alpha

    def forward(self, q: torch.Tensor, p: torch.Tensor,
                n: Optional[torch.Tensor] = None) -> torch.Tensor:
        B   = q.size(0)
        sim = torch.matmul(q, p.T) / self.tau

        if n is None:
            return F.cross_entropy(sim, torch.arange(B, device=q.device))

        with torch.no_grad():
            neg_sim_diag = (q * n).sum(dim=-1)
            h_weights    = torch.exp(self.alpha * neg_sim_diag)

        sim_neg    = torch.matmul(q, n.T) / self.tau
        eye        = torch.eye(B, device=q.device).bool()
        denom_ib   = sim.masked_fill(eye, 0).exp().sum(dim=1)
        denom_hard = h_weights * sim_neg.diagonal().exp()
        numer      = sim.diagonal().exp()

        return (-torch.log(numer / (numer + denom_ib + denom_hard + 1e-9))).mean()


class SpreadOutRegularizer(nn.Module):
    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        B    = q.size(0)
        n    = B * (B - 1)
        diag = torch.eye(B, device=q.device).bool()
        sq   = torch.matmul(q, q.T).masked_fill(diag, 0)
        sp   = torch.matmul(p, p.T).masked_fill(diag, 0)
        return (sq**2).sum() / n + (sp**2).sum() / n


class EmbeddingDistillationLoss(nn.Module):
    def forward(self, sq: torch.Tensor, tq: torch.Tensor,
                sp: torch.Tensor, tp: torch.Tensor,
                sn: Optional[torch.Tensor] = None,
                tn: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.mse_loss(sq @ sp.T, tq @ tp.T)
        if sn is not None and tn is not None:
            loss = loss + F.mse_loss(sq @ sn.T, tq @ tn.T)
            loss = loss + F.mse_loss(sp @ sn.T, tp @ tn.T)
        return loss


class EmbeddingGemmaLoss(nn.Module):
    def __init__(self, temperature: float = 0.02, hardness_alpha: float = 5.0,
                 mrl_dims: List[int] = None):
        super().__init__()
        self.contrastive  = ContrastiveLoss(temperature, hardness_alpha)
        self.spread_out   = SpreadOutRegularizer()
        self.distillation = EmbeddingDistillationLoss()
        self.mrl_dims     = mrl_dims or MRL_DIMS

    def forward(
        self,
        student_q: List[torch.Tensor],
        student_p: List[torch.Tensor],
        student_n: Optional[List[torch.Tensor]] = None,
        teacher_q: Optional[torch.Tensor] = None,
        teacher_p: Optional[torch.Tensor] = None,
        teacher_n: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        dev   = student_q[0].device
        total = torch.tensor(0.0, device=dev)
        log   = {}

        for i, dim in enumerate(self.mrl_dims):
            q  = student_q[i]
            p  = student_p[i]
            n  = student_n[i] if student_n else None
            lc = self.contrastive(q, p, n)
            ls = self.spread_out(q, p)
            ld = torch.tensor(0.0, device=dev)

            if teacher_q is not None and dim == max(self.mrl_dims):
                ld = self.distillation(q, teacher_q, p, teacher_p, n, teacher_n)

            log.update({f'LC_{dim}d': lc.item(), f'LS_{dim}d': ls.item()})
            if teacher_q is not None and dim == max(self.mrl_dims):
                log[f'LD_{dim}d'] = ld.item()

            total = total + lc + ls + ld

        return total, log


TASK_PROMPTS = {
    'retrieval':      {'query': 'task: search query | query: ',
                       'passage': 'task: search result | text: '},
    'similarity':     {'query': 'task: sentence similarity | sentence: ',
                       'passage': 'task: sentence similarity | sentence: '},
    'classification': {'query': 'task: classify | text: ',
                       'passage': 'task: classify | text: '},
}


def apply_task_prompt(text: str, task: str, is_query: bool) -> str:
    key = 'query' if is_query else 'passage'
    return TASK_PROMPTS.get(task, TASK_PROMPTS['retrieval'])[key] + text


class EmbeddingDataset(Dataset):
    def __init__(self, examples: List[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, max_q: int = 128, max_p: int = 512):
    queries   = [apply_task_prompt(ex['query'],    ex.get('task', 'retrieval'), True)  for ex in batch]
    positives = [apply_task_prompt(ex['positive'], ex.get('task', 'retrieval'), False) for ex in batch]
    q_enc = tokenizer(queries,   max_length=max_q, padding=True, truncation=True, return_tensors='pt')
    p_enc = tokenizer(positives, max_length=max_p, padding=True, truncation=True, return_tensors='pt')
    out = {
        'q_ids':  q_enc['input_ids'],  'q_mask': q_enc['attention_mask'],
        'p_ids':  p_enc['input_ids'],  'p_mask': p_enc['attention_mask'],
    }
    if all(ex.get('negative') for ex in batch):
        negatives = [apply_task_prompt(ex['negative'], ex.get('task', 'retrieval'), False) for ex in batch]
        n_enc = tokenizer(negatives, max_length=max_p, padding=True, truncation=True, return_tensors='pt')
        out['n_ids']  = n_enc['input_ids']
        out['n_mask'] = n_enc['attention_mask']
    return out


def train_epoch(
    model: EmbeddingGemma,
    criterion: EmbeddingGemmaLoss,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
    log_every: int = 50
) -> float:
    model.train()
    if teacher_model:
        teacher_model.eval()
    total, n = 0.0, 0

    for step, batch in enumerate(dataloader):
        q_ids  = batch['q_ids'].to(device);  q_mask = batch['q_mask'].to(device)
        p_ids  = batch['p_ids'].to(device);  p_mask = batch['p_mask'].to(device)
        has_n  = 'n_ids' in batch
        n_ids  = batch['n_ids'].to(device)  if has_n else None
        n_mask = batch['n_mask'].to(device) if has_n else None

        sq = model.get_mrl_embeddings(q_ids, q_mask)
        sp = model.get_mrl_embeddings(p_ids, p_mask)
        sn = model.get_mrl_embeddings(n_ids, n_mask) if has_n else None

        tq = tp = tn = None
        if teacher_model is not None:
            with torch.no_grad():
                tq = teacher_model(q_ids, q_mask)
                tp = teacher_model(p_ids, p_mask)
                if has_n:
                    tn = teacher_model(n_ids, n_mask)

        loss, log = criterion(sq, sp, sn, tq, tp, tn)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        n += 1
        if step % log_every == 0:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f} | " +
                  " | ".join(f"{k}:{v:.4f}" for k, v in log.items()))

    return total / n


def model_soup(checkpoint_paths: List[str], model_class,
               model_kwargs: dict, device=torch.device('cpu')) -> nn.Module:
    print(f"Souping {len(checkpoint_paths)} checkpoints...")
    souped = model_class(**model_kwargs).to(device)
    state  = {k: torch.zeros_like(v) for k, v in souped.state_dict().items()}

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device)
        ckpt = ckpt.get('model_state_dict', ckpt)
        for k in state:
            if k in ckpt:
                state[k] = state[k] + ckpt[k]

    n = len(checkpoint_paths)
    for k in state:
        if state[k].is_floating_point():
            state[k] = state[k] / n

    souped.load_state_dict(state)
    print(f"Done — {n} checkpoints averaged.")
    return souped


@torch.no_grad()
def encode_texts(model: EmbeddingGemma, texts: List[str], tokenizer,
                 device: torch.device, task: str = 'retrieval',
                 is_query: bool = True, embed_dim: int = 768,
                 batch_size: int = 32) -> torch.Tensor:
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        prompted = [apply_task_prompt(t, task, is_query) for t in texts[i:i+batch_size]]
        enc  = tokenizer(prompted, max_length=512, padding=True,
                         truncation=True, return_tensors='pt')
        ids  = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        emb  = F.normalize(model(ids, mask)[:, :embed_dim], p=2, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def evaluate_retrieval(model: EmbeddingGemma, queries: List[str],
                       passages: List[str], relevant_indices: List[int],
                       tokenizer, device: torch.device,
                       embed_dim: int = 768) -> dict:
    q_embs = encode_texts(model, queries,  tokenizer, device, is_query=True,  embed_dim=embed_dim)
    p_embs = encode_texts(model, passages, tokenizer, device, is_query=False, embed_dim=embed_dim)
    sims   = q_embs @ p_embs.T

    mrr, recall = 0.0, {1: 0, 5: 0, 10: 0}
    for row, rel in zip(sims, relevant_indices):
        rank = (row.argsort(descending=True) == rel).nonzero(as_tuple=True)[0].item() + 1
        if rank <= 10:
            mrr += 1.0 / rank
        for k in [1, 5, 10]:
            if rank <= k:
                recall[k] += 1

    n       = len(queries)
    metrics = {
        'MRR@10': mrr/n, 'R@1': recall[1]/n,
        'R@5': recall[5]/n, 'R@10': recall[10]/n
    }
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def run_demo():
    from collections import defaultdict
    from functools import partial

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    class ToyTokenizer:
        def __init__(self, vocab_size=2000):
            self.v = defaultdict(lambda: 1)
            self.v.update({'[PAD]': 0, '[CLS]': 2, '[SEP]': 3})
            self._n = 4
            self.vocab_size = vocab_size

        def _id(self, w):
            if w not in self.v:
                if self._n < self.vocab_size:
                    self.v[w] = self._n
                    self._n += 1
                else:
                    return 1
            return self.v[w]

        def __call__(self, texts, max_length=128, padding=True,
                     truncation=True, return_tensors='pt'):
            seqs  = [[2] + [self._id(w.lower()) for w in t.split()] + [3] for t in texts]
            if truncation:
                seqs = [s[:max_length] for s in seqs]
            ml    = max(len(s) for s in seqs) if padding else 0
            masks = [[1]*len(s) + [0]*(ml-len(s)) for s in seqs]
            seqs  = [s + [0]*(ml-len(s)) for s in seqs] if padding else seqs
            return {
                'input_ids':      torch.tensor(seqs,  dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long)
            }

    tokenizer = ToyTokenizer(2000)

    raw = [
        ('what is machine learning',
         'Machine learning is AI that learns patterns from data',
         'The stock market rose three percent yesterday'),
        ('how does photosynthesis work',
         'Plants convert sunlight water and CO2 into glucose and oxygen',
         'Photosynthesis is a very long word with many syllables'),
        ('best way to learn programming',
         'Practice daily by building small projects and solving problems',
         'Programming languages were invented in the nineteen fifties'),
        ('what causes inflation',
         'Inflation rises when money supply grows faster than the economy',
         'Inflation is mentioned often in newspapers and magazines'),
        ('explain neural networks',
         'Neural networks are computing systems inspired by biological brains',
         'Networks can refer to television channels or computers'),
    ]
    toy_data = [
        {'query': q, 'positive': p, 'negative': n, 'task': 'retrieval'}
        for q, p, n in raw
    ] * 40

    for ex in toy_data:
        for field in ['query', 'positive', 'negative']:
            for word in ex[field].lower().split():
                tokenizer._id(word)

    dataset    = EmbeddingDataset(toy_data)
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_q=32, max_p=64)
    )

    model = EmbeddingGemma(
        vocab_size=2000, d_model=128, n_heads=4, n_kv_heads=2,
        n_layers=2, d_ff=512, d_intermediate=512, d_embed=128,
        dropout=0.1, max_seq_len=64, mrl_dims=[32, 64, 128]
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = EmbeddingGemmaLoss(temperature=0.05, mrl_dims=[32, 64, 128])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\n--- Training ---")
    for epoch in range(5):
        avg = train_epoch(model, criterion, optimizer, dataloader, device, log_every=25)
        print(f"Epoch {epoch+1}/5 | Avg Loss: {avg:.4f}")

    print("\n--- Retrieval Evaluation ---")
    eval_queries  = [q for q, p, n in raw]
    eval_passages = [p for q, p, n in raw] + [n for q, p, n in raw][:2]
    relevant      = list(range(len(eval_queries)))
    evaluate_retrieval(model, eval_queries, eval_passages, relevant, tokenizer, device, embed_dim=128)

    print("\n--- Cosine Similarity Demo ---")
    model.eval()
    pairs = [
        ("what is machine learning", "Machine learning is AI that learns from data"),
        ("what is machine learning", "The Eiffel Tower is located in Paris France"),
        ("how to bake bread",        "Mix flour yeast water and salt then bake at 375"),
        ("how to bake bread",        "Swimming is excellent cardiovascular exercise"),
    ]
    for q_text, p_text in pairs:
        q_enc = tokenizer([q_text], max_length=32, padding=True, truncation=True, return_tensors='pt')
        p_enc = tokenizer([p_text], max_length=64, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            qe = model(q_enc['input_ids'].to(device), q_enc['attention_mask'].to(device))
            pe = model(p_enc['input_ids'].to(device), p_enc['attention_mask'].to(device))
        sim   = (qe * pe).sum().item()
        label = "HIGH ✓" if sim > 0.5 else "low  ✗"
        print(f"  [{label}] ({sim:.3f})  '{q_text[:38]}' ↔ '{p_text[:38]}'")

    print("\nDemo complete.")


if __name__ == '__main__':
    run_demo()