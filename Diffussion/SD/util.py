import torch

def rescale(x, input_range, output_range, clamp=False):
    i_max, i_min = input_range
    o_max, o_min = output_range
    
    x -= i_min
    x *= (o_max - o_min) / (i_max - i_min)
    x += o_min
    if clamp:
        x = x.clamp(o_min, o_max)
    return x

def get_time_embedding(timestamp):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # indexing None same as unsqeezing that dim
    x = torch.tensor([timestamp], dtype=torch.float32)[:,None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)])