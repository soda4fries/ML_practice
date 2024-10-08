import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab_size: int, n_dim: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab_size, n_dim)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_dim))

    def forward(self, tokens):
        # token: (b,s)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_dim: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_dim)
        self.attention = SelfAttention(n_head, n_dim)
        self.layernorm_2 = nn.LayerNorm(n_dim)
        self.linear_1 = nn.Linear(n_dim, 4 * n_dim)
        self.linear_2 = nn.Linear(4 * n_dim, n_dim)

    def forward(self, x: torch.Tensor):
        # x: (b, s, dim)

        residule = x

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residule

        residule = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)

        # quick gelu
        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residule

        return x


class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([CLIPLayer(12, 768) for i in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor):
        tokens = tokens.type(torch.long) # type: ignore

        x = self.embedding(tokens)

        for layer in self.layers: # type: ignore
            x = layer(x)

        output = self.layernorm(x)

        # (b, s, d)
        return output
