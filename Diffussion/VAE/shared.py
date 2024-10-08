import torch
from torch import nn
from torch.nn import functional as F
from Diffussion.attention import SelfAttention


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def foward(self, x: torch.tensor):
        # x: (b, f, h, w)

        residule = x

        n, c, h, w = x.shape

        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x += residule


class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channel)
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding_mode=1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channel, out_channel, kernel_size=1, padding=0
            )

    def forward(self, x: torch.tensor):
        # x : (b,c,h,w)

        residule = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residule)
