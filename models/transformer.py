import torch
from torch import nn

from einops import rearrange

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class DualPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.normx = nn.LayerNorm(dim)
        self.normy = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, **kwargs):
        return self.fn(self.normx(x), self.normy(y), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        # qk = self.to_qk(x).chunk(2, dim = -1) #
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.heads) # q,k from the zero feature
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h = self.heads) # v from the reference features
        v = rearrange(self.to_v(y), 'b n (h d) -> b h n d', h = self.heads) 

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                DualPreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))


    def forward(self, x, y): # x is the cropped, y is the foreign reference
        bs,c,h,w = x.size()

        # img to embedding
        x = x.view(bs,c,-1).permute(0,2,1)
        y = y.view(bs,c,-1).permute(0,2,1)

        for attn, ff in self.layers:
            x = attn(x, y) + x
            x = ff(x) + x

        x = x.view(bs,h,w,c).permute(0,3,1,2)
        return x

class RETURNX(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, y): # x is the cropped, y is the foreign reference 
        return x