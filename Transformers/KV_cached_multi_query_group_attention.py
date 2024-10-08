import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(
        self, n_kv_heads: int, n_heads: int, dim: int, max_batch_size: int, seq_len: int
    ):
        super().__init__()

        # total kv heads
        self.n_kv_heads = n_kv_heads
        # total heads
        self.n_heads_q = n_heads

        # heads grouped by queries. ratio
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # dims visible
        self.head_dim = dim / n_heads

        self.wq = nn.Linear(
            in_features=dim, out_features=n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            in_features=dim, out_features=self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            in_features=dim, out_features=self.n_kv_heads * self.head_dim, bias=False
        )

        self.wo = nn.Linear(in_features=dim, out_features=dim, bias=False)

        self.cache_k = torch.zeros((max_batch_size, seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((max_batch_size, seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.Tensor, start: int, freqs_complex: torch.Tensor):
        # x: (b, 1, dim)
        # freq_complex: rotational embedding
        b, s, _ = x.shape

        # (B, 1, H_Q * h_dim) (B, 1, dim)
        xq = self.wq(x)
        
        # (B, 1, h_kv * h_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        
        