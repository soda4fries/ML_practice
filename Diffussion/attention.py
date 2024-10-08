import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):

    def __init__(
        self, n_heads: int, 
        d_model: int, 
        in_proj_bias=True, 
        out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (b,s,d)

        b, s, d = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (b, n_head, s, d_head)
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        qk = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(qk, dtype=torch.bool).triu(1)
            qk.masked_fill_(mask, -torch.inf)

        qk /= torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))

        qk = F.softmax(qk, dim=-1)

        # (b, h, s, d_head)
        output = qk @ v

        output = output.transpose(1, 2).reshape((b, s, d))
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):

    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        d_cross: int,
        in_proj_bias=True,
        out_proj_bias=True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x : self q (b,s,d_embed)
        # y : cross kv (b,s,d_cross) (b, 77, 768)
        x_b, x_s, x_d = x.shape

        view_shape = (x_b, -1, self.n_heads, self.d_head)

        # (b,s_x,d)
        q = self.q_proj(x)

        # (b,s_y,d)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(view_shape).transpose(1, 2)
        k = k.view(view_shape).transpose(1, 2)
        v = v.view(view_shape).transpose(1, 2)

        # (b,s_x, s_y)
        attn = q @ k.transpose(-1, -2)
        attn /= math.sqrt(self.d_head)

        attn = F.softmax(attn, dim=-1)

        # (b, s_x, s_y) @ (b, s_y, d) -> (b, s_x, d)
        output = attn @ v

        output = output.transpose(1, 2).reshape((x_b, x_s, x_d))
        return self.out_proj(output)
