import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional


# =====================
# THEORETICAL BOUNDS
# =====================

def lower_bound_mse(b):  return 1.0 / 4.0**b
def lower_bound_prod(b, d, y_norm_sq=1.0): return y_norm_sq / d / 4.0**b
def turboquant_mse_ub(b): return math.sqrt(3) * math.pi / 2 / 4.0**b


# =====================
# LLOYD-MAX CODEBOOKS
# =====================

def lloyd_max(n_levels, n_iter=1000, n_pts=100000):
    samples, _ = torch.randn(n_pts).sort()
    x_min, x_max = samples[n_pts//100].item(), samples[-n_pts//100].item()
    C = torch.linspace(x_min, x_max, n_levels)
    for _ in range(n_iter):
        old = C.clone()
        A = (samples.unsqueeze(1) - C.unsqueeze(0)).abs().argmin(1)
        C = torch.stack([samples[A==k].mean() if (A==k).sum()>0 else C[k] for k in range(n_levels)])
        if (C-old).abs().max() < 1e-8: break
    B = (C[:-1]+C[1:])/2
    return B, C

CODEBOOKS = {b: dict(zip(['boundaries','centroids'], lloyd_max(2**b))) for b in [1,2,3,4]}


# =====================
# RANDOM ROTATION
# =====================

class RandomRotation(nn.Module):
    def __init__(self, d, seed=42):
        super().__init__()
        self.d = d
        self.d_pad = 2**math.ceil(math.log2(d))
        g = torch.Generator(); g.manual_seed(seed)
        D = torch.randint(0,2,(self.d_pad,),generator=g).float()*2-1
        self.register_buffer('D', D)

    def _hadamard(self, x):
        n, h = x.shape[-1], 1
        r = x.clone()
        while h < n:
            r = r.view(*r.shape[:-1],-1,2*h)
            l, rr = r[...,:h].clone(), r[...,h:].clone()
            r[...,:h], r[...,h:] = l+rr, l-rr
            r = r.view(*r.shape[:-2],-1); h*=2
        return r/math.sqrt(n)

    def forward(self, x):
        if self.d < self.d_pad: x = F.pad(x,(0,self.d_pad-self.d))
        return self._hadamard(x*self.D)

    def inverse(self, x):
        x = self._hadamard(x)*self.d_pad/math.sqrt(self.d_pad)
        x = self._hadamard(x)*self.D
        return x[...,:self.d]


# =====================
# MSE TURBOQUANT
# =====================

class TurboQuantMSE(nn.Module):
    def __init__(self, d, b, seed=42):
        super().__init__()
        self.d, self.b = d, b
        self.rotation = RandomRotation(d, seed)
        self.scale = math.sqrt(self.rotation.d_pad)
        cb = CODEBOOKS[b]
        self.register_buffer('boundaries', cb['boundaries'])
        self.register_buffer('centroids', cb['centroids'])

    def quantize(self, x):
        norms = x.norm(dim=-1, keepdim=True)
        idx = torch.bucketize(self.rotation(x/(norms+1e-9))*self.scale, self.boundaries)
        return norms.squeeze(-1), idx

    def dequantize(self, norms, idx):
        return self.rotation.inverse(self.centroids[idx]/self.scale)[...,:self.d]*norms.unsqueeze(-1)

    def forward(self, x):
        return self.dequantize(*self.quantize(x))


# =====================
# QJL
# =====================

class QJL(nn.Module):
    def __init__(self, d, seed=123):
        super().__init__()
        self.scale = math.sqrt(math.pi/2)/d
        g = torch.Generator(); g.manual_seed(seed)
        self.register_buffer('S', torch.randn(d,d,generator=g))

    def quantize(self, x):
        z = torch.sign(x@self.S.T)
        return torch.where(z==0, torch.ones_like(z), z)

    def dequantize(self, z): return self.scale*(z@self.S)

    def inner_product_estimate(self, z, y):
        Sy = (y@self.S.T)
        if Sy.dim()==1: Sy=Sy.unsqueeze(0)
        return self.scale*(z*Sy).sum(-1)

    def forward(self, x): return self.dequantize(self.quantize(x))


# =====================
# IP TURBOQUANT
# =====================

class TurboQuantIP(nn.Module):
    def __init__(self, d, b, mse_seed=42, qjl_seed=123):
        super().__init__()
        self.d, self.b = d, b
        self.mse = TurboQuantMSE(d, b-1, mse_seed)
        self.qjl = QJL(d, qjl_seed)

    def quantize(self, x):
        norms, idx = self.mse.quantize(x)
        residual = x - self.mse.dequantize(norms, idx)
        bits = self.qjl.quantize(residual)
        return norms, idx, bits

    def estimate_ip(self, norms, idx, bits, y):
        x_mse = self.mse.dequantize(norms, idx)
        return (x_mse*y.unsqueeze(0)).sum(-1) + self.qjl.inner_product_estimate(bits, y)

    def dequantize(self, norms, idx, bits):
        return self.mse.dequantize(norms, idx) + self.qjl.dequantize(bits)

    def forward(self, x):
        return self.dequantize(*self.quantize(x))