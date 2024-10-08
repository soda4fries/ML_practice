import torch

from torch import nn
from torch.nn import functional as F

from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):

    def __init__(self, n_embed: int):
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.Tensor):
        # x: (1, 320)
        # (1, 1280)
        return self.linear_2(F.silu(self.linear_1(x)))


class UpSample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (b,c,2h,2w)
        return self.conv(F.interpolate(x, scale_factor=2))


class UNET_OutputLayer(nn.Module):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.silu(self.groupnorm(x)))


class UNet_residualBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, n_time=1280):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channel)

        self.groupnorm_merged = nn.GroupNorm(32, out_channel)
        self.conv_merged = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channel, out_channel, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        # time : (1, 1280) -> (1, out_channel)
        time = self.linear_time(F.silu(time))

        residual = x
        x = self.conv(F.silu(self.groupnorm(x)))

        merged = self.groupnorm_merged(x + time.unsqueeze(-1).unsqueeze(-1))
        merged = self.conv_merged(F.silu(merged))

        return merged + self.residual_layer(residual)


class UNet_attentionBlock(nn.Module):

    def __init__(self, n_head: int, n_dim: int, d_context: int = 768):
        super().__init__()
        channels = n_head * n_dim

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_head, channels, d_context, in_proj_bias=False
        )
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, (2 * 4) * channels)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x : (b,c,h,w) context: (b, s, d)
        b, c, h, w = x.shape
        residual_long = x

        x = self.conv_input(self.groupnorm(x))
        x = x.view((b, c, h * w))
        x = x.transpose(-1, -2)

        residule_short = x
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residule_short

        residule_short = x
        x = self.layernorm_2(x)
        self.attention_2(x, context)
        x += residule_short

        residule_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residule_short

        x = x.transpose(-1, -2).view((b, c, h, w))

        return self.conv_output(x) + residual_long


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor: # type: ignore
        for layer in self:
            if isinstance(layer, UNet_attentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_residualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
            
        return x


class UNet(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = nn.Module(
            [
                # (b,4,h/8,w/8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNet_residualBlock(320, 320), UNet_attentionBlock(8, 40)
                ),
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNet_residualBlock(320, 640), UNet_attentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNet_residualBlock(640, 640), UNet_attentionBlock(8, 80)
                ),
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNet_residualBlock(640, 1280), UNet_attentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNet_residualBlock(1280, 1280), UNet_attentionBlock(8, 160)
                ),
                # (b,1280,h/64,w/64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(UNet_residualBlock(1280, 1280)),
                SwitchSequential(UNet_residualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNet_residualBlock(1280, 1280),
            UNet_attentionBlock(8, 160),
            UNet_residualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList(
            [
                SwitchSequential(UNet_residualBlock(2560, 1280)),
                SwitchSequential(UNet_residualBlock(2560, 1280)),
                SwitchSequential(UNet_residualBlock(2560, 1280), UpSample(1280)),
                SwitchSequential(
                    UNet_residualBlock(2560, 1280), UNet_attentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNet_residualBlock(2560, 1280), UNet_attentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNet_residualBlock(1920, 1280),
                    UNet_attentionBlock(8, 160),
                    UpSample(1280),
                ),
                SwitchSequential(
                    UNet_residualBlock(1920, 640), UNet_attentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNet_residualBlock(960, 640),
                    UNet_attentionBlock(8, 80),
                    UpSample(640),
                ),
                SwitchSequential(
                    UNet_residualBlock(960, 320), UNet_attentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNet_residualBlock(640, 320), UNet_attentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNet_residualBlock(640, 320), UNet_attentionBlock(8, 40)
                ),
            ]
        )


class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (b, 4, h/8, w/8)
        # context (b, seqlen, d_embed)

        # time (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (b, 320, h/8, w/8)
        output = self.unet(latent, context, time)

        # (b 4, h/8, w/8)
        output = self.final(output)

        return output
