
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Cross_Attention(nn.Module):
    def __init__(self, dim_q,dim_k):
        super().__init__()


        self.query = nn.Linear(dim_q, dim_q, bias=False)
        self.key = nn.Linear(dim_k, dim_q, bias=False)
        self.value = nn.Linear(dim_k, dim_q, bias=False)
        self.out = nn.Linear(dim_q, dim_q, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)

    def forward(self, q,k):

        query = self.query(q)
        key = self.key(k).transpose(-1, -2)
        value = self.value(k)
        sp_similarity_matrix = torch.matmul(query, key)
        sp_similarity_matrix = self.softmax(self.psi(sp_similarity_matrix.unsqueeze(0)).squeeze(0))
        out = torch.matmul(sp_similarity_matrix, value)
        out = self.out(out)
        return out

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=[1,1]):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class DFF_Transformer(nn.Module):
    def __init__(self, dim_q,dim_k, depth=1, mlp_dim=256, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Cross_Attention(dim_q,dim_k),
                PreNorm(dim_q, FeedForward(dim_q, mlp_dim, dropout = dropout))
            ]))
        self.reverse  = Reconstruct(dim_q,dim_q,1)

        self.norm = nn.LayerNorm(dim_q)
        self.norm_ = nn.LayerNorm(dim_k)

        self.layers_= nn.ModuleList([])
        for _ in range(depth):
            self.layers_.append(nn.ModuleList([
                 Cross_Attention(dim_k, dim_q),
                PreNorm(dim_k, FeedForward(dim_k, mlp_dim, dropout=dropout))
            ]))
        self.reverse_= Reconstruct(dim_k, dim_k, 1)
    def forward(self, q,k):
        b_r, c_r, h_r, w_r = q.shape
        q = q.flatten(2).transpose(-1, -2)  # B N C
        k = k.flatten(2).transpose(-1, -2)  # B N C
        q=self.norm(q)
        k=self.norm_(k)
        x=q
        for cross_attn, ff in self.layers:
            x = cross_attn(q,k) + x
            x = ff(x) + x
        x=self.reverse(x)

        x_=k
        for cross_attn, ff in self.layers_:
            x_ = cross_attn(k, q) + x_
            x_= ff(x_) + x_
        x_= self.reverse_(x_)
        return x,x_

class DFF(nn.Module):
    def __init__(self, dim_l, dim_s):
        super().__init__()
        self.cross_atten1=Cross_Attention(dim_l, dim_s)
        self.cross_atten2=Cross_Attention(dim_s, dim_l)

        self.reverse1 = Reconstruct(dim_l, dim_l, 1)
        self.reverse2 = Reconstruct(dim_s, dim_s, 1)

    def forward(self, q, k):
        #q=f1 k=f0
        # b_r, c_r, h_r, w_r = q.shape
        q = q.flatten(2).transpose(-1, -2)  # B N C
        k = k.flatten(2).transpose(-1, -2)  # B N C
        x1=self.cross_atten1(q,k)
        x1= self.reverse1(x1)

        x2 = self.cross_atten2(k, q)
        x2 = self.reverse2(x2)
        return x1,x2


################################################

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

#TIF module
class Cross_Att(nn.Module):
    def __init__(self, dim_s, dim_l):
        super().__init__()
        #dim_s ！！>x_h
        #dim_l ！！>x_l
        # self.transformer_s = Transformer(dim=dim_s, depth=1, heads=3, dim_head=32, mlp_dim=128)
        self.transformer_s = Transformer(dim=dim_s, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.transformer_l = Transformer(dim=dim_l, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.norm_s = nn.LayerNorm(dim_s)
        self.norm_l = nn.LayerNorm(dim_l)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_s = nn.Linear(dim_s, dim_l)
        self.linear_l = nn.Linear(dim_l, dim_s)

        ###
        self.norm_s_ = nn.LayerNorm(dim_s)
        self.norm_l_ = nn.LayerNorm(dim_l)
        self.linear_l_=nn.Linear(dim_l, dim_s)
    def forward(self, r, e):
       b_e, c_e, h_e, w_e = e.shape
       e = e.reshape(b_e, c_e, -1).permute(0, 2, 1)
       b_r, c_r, h_r, w_r = r.shape
       r = r.reshape(b_r, c_r, -1).permute(0, 2, 1)#B N C
       e_t = torch.flatten(self.avgpool(self.norm_l(e).transpose(1,2)), 1)
       r_t = torch.flatten(self.avgpool(self.norm_s(r).transpose(1,2)), 1) #B C
       e_t = self.linear_l(e_t).unsqueeze(1)
       r_t = self.linear_s(r_t).unsqueeze(1)#B 1 C

       r_pre= self.transformer_s(torch.cat([e_t, r],dim=1))
       r = r_pre[:, 1:, :]
       r_semantic=r_pre[:,0,:] #c=64
       e_pre= self.transformer_l(torch.cat([r_t, e],dim=1))
       e = e_pre[:, 1:, :]
       e_semantic=e_pre[:,0,:] #c=128
       e = e.permute(0, 2, 1).reshape(b_e, c_e, h_e, w_e)
       r = r.permute(0, 2, 1).reshape(b_r, c_r, h_r, w_r)

       ###
       r_semantic=self.norm_s_(r_semantic).squeeze(1)
       e_semantic=self.norm_l_(e_semantic).squeeze(1)
       e_semantic=self.linear_l_(e_semantic)
       semantic_embed=r_semantic+e_semantic

       return r, e,semantic_embed

