import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================
# UTILITY FUNCTIONS
# =====================

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len: int, device=torch.device('cpu')) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

def create_decoder_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    seq_len = tgt.size(1)
    pad_mask = create_padding_mask(tgt, pad_idx)
    causal_mask = create_causal_mask(seq_len, tgt.device)
    return pad_mask & causal_mask.bool()


# =====================
# CORE COMPONENTS
# =====================

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights)
    return torch.matmul(weights, V), weights


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
    def forward(self, x):
        return self.emb(x) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, s, d = x.size()
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        b = Q.size(0)
        Q, K, V = self.split_heads(self.W_q(Q)), self.split_heads(self.W_k(K)), self.split_heads(self.W_v(V))
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        x, w = scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.n_heads * self.d_k)
        return self.W_o(x), w


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        a, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop(a))
        a, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop(a))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x = self.pe(self.emb(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.pe(self.emb(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_heads, n_layers, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, n_layers, d_ff, dropout, max_len)
        self.proj = nn.Linear(d_model, tgt_vocab)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.proj(self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask))


# Inference

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    sos_idx: int,
    eos_idx: int,
    max_len: int = 100,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
   
    model.eval()
    batch_size = src.size(0)
    encoder_output = model.encode(src, src_mask)  
    decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(max_len):
        
        tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
        decoder_output = model.decode(
            decoder_input, encoder_output, src_mask, tgt_mask
        )  
        logits = model.output_projection(decoder_output[:, -1, :]) 
        next_token = logits.argmax(dim=-1, keepdim=True)  
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        finished |= (next_token.squeeze(-1) == eos_idx)

        if finished.all():
            break

    return decoder_input  