import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        Q = self.WQ(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.WK(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.WV(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        output, attn = self.attention(Q, K, V, mask)
        output = output.transpose(1,2).reshape(batch_size, -1, self.num_heads * self.d_k)

        return self.fc(output)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x2 = self.mha(x, x, x, mask)
        x = self.norm1(x + x2)

        x2 = self.ff(x)
        x = self.norm2(x + x2)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
        for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

def future_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)

class DecoderLayer(nn.Module):
    def __init__(self,d_model,heads,d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,heads)
        self.cross_attn = MultiHeadAttention(d_model,heads)
        self.ff = FeedForward(d_model,d_ff)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self,x,enc,src_mask,tgt_mask):
        x = self.n1(x + self.self_attn(x,x,x,tgt_mask))
        x = self.n2(x + self.cross_attn(x,enc,enc,src_mask))
        x = self.n3(x + self.ff(x))
        return x

class Decoder(nn.Module):
    def __init__(self,vocab,d_model,N,heads,d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab,d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model,heads,d_ff) for _ in range(N)])
        self.fc = nn.Linear(d_model, vocab)

    def forward(self,x,enc,src_mask,tgt_mask):
        x = self.pos(self.embed(x))
        for layer in self.layers:
            x = layer(x,enc,src_mask,tgt_mask)
        return self.fc(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=2, heads=8, d_ff=2048):
        super().__init__()
        self.encoder = Encoder(src_vocab,d_model,N,heads,d_ff)
        self.decoder = Decoder(tgt_vocab,d_model,N,heads,d_ff)

    def forward(self,src,tgt,src_mask,tgt_mask):
        enc = self.encoder(src,src_mask)
        return self.decoder(tgt,enc,src_mask,tgt_mask)



if __name__ == "__main__":
    model = Transformer(src_vocab=5000, tgt_vocab=5000)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(300):
    src, tgt_in, tgt_out = batch()  
    src_mask = None
    tgt_mask = future_mask(tgt_in.size(1))

    logits = model(src,tgt_in,src_mask,tgt_mask)
    loss = loss_fn(logits.view(-1,logits.size(-1)), tgt_out.view(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss {loss.item():.4f}")
