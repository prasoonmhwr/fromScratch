import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import math

torch.manual_seed(42)

# =====================
# HYPERPARAMETERS
# =====================
D_MODEL    = 64
N_CLASSES  = 10
IMG_SIZE   = (32, 32)
PATCH_SIZE = (16, 16)
N_CHANNELS = 1
N_HEADS    = 8
N_LAYERS   = 6
BATCH_SIZE = 128
EPOCHS     = 10
LR         = 0.005


# =====================
# MODEL COMPONENTS
# =====================

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.linear_project = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = math.sin(pos / (10000 ** (i / d_model)))
                else:
                    pe[pos][i] = math.cos(pos / (10000 ** ((i-1) / d_model)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pe
        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.W_q = nn.Linear(d_model, head_size, bias=False)
        self.W_k = nn.Linear(d_model, head_size, bias=False)
        self.W_v = nn.Linear(d_model, head_size, bias=False)

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_size)
        return F.softmax(scores, dim=-1) @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, d_model // n_heads) for _ in range(n_heads)])
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.W_o(torch.cat([h(x) for h in self.heads], dim=-1))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        assert d_model % n_heads == 0
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embedding = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(d_model, n_patches + 1)
        self.encoder_layers = nn.Sequential(*[TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.classifier = nn.Sequential(nn.Linear(d_model, n_classes), nn.Softmax(dim=-1))

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.encoder_layers(x)
        return self.classifier(x[:, 0])


# =====================
# TRAINING
# =====================

transform = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])
train_set = MNIST(root='./data', train=True,  download=True, transform=transform)
test_set  = MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, shuffle=True,  batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_set,  shuffle=False, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(D_MODEL, N_CLASSES, IMG_SIZE, PATCH_SIZE, N_CHANNELS, N_HEADS, N_LAYERS).to(device)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")