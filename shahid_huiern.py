import math

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.utils.data


class MyDataset(Dataset):
    def __init__(self, n_sample: int):
        super().__init__()
        self.x0 = torch.arange(0, n_sample, .1, dtype=torch.float32)
        self.x1 = torch.arange(0, n_sample, .1, dtype=torch.float32)
        self.y = torch.arange(0, n_sample, .1, dtype=torch.float32)

    def __getitem__(self, item):
        return {
            "x": torch.cat([self.x0[item].unsqueeze(dim=0), self.x1[item].unsqueeze(dim=0)], dim=0),
            "y": self.y[item]
        }

    def __len__(self):
        return len(self.y)


class MyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        x = x @ self.weight
        if self.bias is not None:
            x + self.bias
        return x


class MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:

            batch_mean = x.mean(dim=0) # compresses all dimension
            batch_var = x.var(dim=0, unbiased=False)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:

            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * x_norm + self.beta


class MyLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        # Learnable parameters for scaling and shifting (gamma and beta)
        self.gamma = nn.Parameter(torch.ones(num_features))  # Scale
        self.beta = nn.Parameter(torch.zeros(num_features))  # Shift
        self.eps = eps  # Small epsilon to prevent division by zero

    def forward(self, x):
        # Compute mean and variance across feature dimension (dim=-1) or last dimension. 
        feature_mean = x.mean(dim=-1, keepdim=True) 
        feature_var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input across the feature dimension
        x_norm = (x - feature_mean) / torch.sqrt(feature_var + self.eps)

        # Scale and shift using learnable parameters gamma and beta
        return self.gamma * x_norm + self.beta


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "dimension must be divisible by number of heads"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        d_k = q.shape[-1]

        # (b, h, s, d_k) @ (b, h, d_k, s) -> (b, h, s, s)
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill(mask == 0, -1e9)
        # (b, h , sl, sl)
        attention_score = attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        # (b, h, s, s) @ (b, h, s, d_k) -> (b, h, s, d_k)
        return attention_score @ v

    def forward(self, q, k, v, mask):
        # (b, s, d)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # (b, h, s, d_k)
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

        # x (b, h, s, d_k), attention_score (b, h, s, s)
        x = MultiHeadAttention.attention(q, k, v, mask, self.dropout)

        # (b, s, d)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        # (b, s, d)
        return self.w_o(x)


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()

        # x is passed one after another, must match dimensions
        self.seq_layers = nn.Sequential(
            MyLinear(input_dim, hidden_dim),
            MyBatchNorm(hidden_dim),
            nn.ReLU()
        )

        # access using index, can use if else if required
        self.module_list_layers = nn.ModuleList([
            MyLinear(hidden_dim, hidden_dim),
            MyLayerNorm(hidden_dim),
            nn.ReLU(),

        ])
        self.adjust_dim = nn.Linear(1, 15 * 8)
        self.attn = MultiHeadAttention(d_model=15 * 8, h=8, dropout=.15)
        self.readjust_dim = nn.Linear(15 * 8, 1)

        # Final linear layer
        self.output_layer = MyLinear(hidden_dim, output_dim)

    def forward(self, x):
        # our feature is simple just 1 number. but it can be a complex feature. it will attend among features along s
        # (b, 2)
        # (b, s)
        shape_before = x.shape
        # (b, s, f)
        # (b, 2, 1)
        # (b, s, 1) (b, s, 1)
        x = x.unsqueeze(dim=-1)
        # (b, 2, 15*8) # adjust feature dimension, just make it a bit bigger for no reason kek
        x = self.adjust_dim(x)
        #so it will attendend between two numbers (among features), its attending among s, so in sentence 1 word is 5 dimension, and sentence is (hello hi) and (hi hi hi) the data will be (2,3, 5) first sentence must be padded. so its (hello, hi, pad) (hi,hi,hi)

        x = self.attn(x, x, x, None)  # self attention, q, k, v same. cross attention k,v from another model.
        x = self.readjust_dim(x)
        x = x.view(shape_before)

        # Pass through the nn.Sequential block
        x = self.seq_layers(x)

        # Pass through the nn.ModuleList block
        for layer in self.module_list_layers:
            x = layer(x)
            # sequenctial unless MyLinear. can be other condition also, lets say u want to bring KV from another model.
            if isinstance(layer, MyLinear):
                x = nn.ReLU()(x)

        x = self.output_layer(x)
        return x


if __name__ == '__main__':

    x = torch.randn(10, 6, 4)
    a, b = x.chunk(2, dim=0)
    print("a: ", a.shape, " b: ", b.shape)
    print(x.shape)
    x = x.view(2, -1, 6, 4)
    print("chunking whole", x.shape)

    x = x.mT
    print("Transpose", x.shape)
    x = x @ x.mT
    print("matmul:", x.shape)

    model = MyLinear(2, 1)
    dataset = MyDataset(100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        for batch in dataloader:
            optim.zero_grad()
            result = model(batch['x'])
            loss = (batch['y'] - result)
            loss = pow(loss, 2).sum()
            loss.backward()
            optim.step()

    print(model(torch.tensor([[1, 1]], dtype=torch.float32)))

    model = MyModel(2, 100, 1)
    dataset = MyDataset(10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in dataloader:
            optim.zero_grad()
            result = model(batch['x'])
            loss = (batch['y'] - result)
            loss = pow(loss, 2).sum()
            loss.backward()
            optim.step()

    print(model(torch.tensor([[1, 1]], dtype=torch.float32)))
