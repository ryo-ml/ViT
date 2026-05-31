import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int) -> None:
        super().__init__()

        pe = torch.zeros(num_tokens, d_model)
        pos = torch.arange(0, num_tokens).unsqueeze(1) # (n, 1)
        i = torch.arange(0, d_model, 2).unsqueeze(0) # (1, d_model / 2)

        div_term = torch.exp(
            i * (-math.log(10000.0) / d_model) # (1, d_model / 2)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0) # (1, num_tokens, d_model)

        self.register_buffer('pe', pe)

    def forward(self) -> torch.Tensor:
        return self.pe

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int) -> None:
        super().__init__()

        self.pe = nn.Parameter(
            torch.randn(1, num_tokens, d_model)
        )

    def forward(self) -> torch.Tensor:
        return self.pe

class PatchEmbedding(nn.Module):
    """
    Patchifies and embeds.
    """
    def __init__(self, in_channels: int, d_model: int, patch_size: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x (B, C, H, W): 2d images
        """
        # (B, C, H, W) -> (B, d_model, H/P, H/W) -> (B, d_model, N) -> (B, N, d_model)
        return self.flatten(self.conv(x)).permute(0, 2, 1)
