import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    self-attention
    """
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f'{d_model} must be divisible by {num_heads}'

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.scale = (self.head_dim) ** (-0.5)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x (B, N, D): patch embeddings
        """
        B, N, D = x.size()
        assert D == self.d_model, f'Expected input dim {self.d_model}, but got {D}'

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] #(B, num_heads, N, head_dim)

        raw_scores = q @ k.transpose(-2, -1) * self.scale # (B, num_heads, N, N)
        scores = self.attn_dropout(self.softmax(raw_scores))
        attn = scores @ v # (B, num_heads, N, head_dim)
        x = self.proj_dropout(self.proj(attn.permute(0, 2, 1, 3).reshape(B, N, self.d_model))) # (B, N, d_model)
        return x

class Block(nn.Module):
    """
    transformer encoder
    """
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float,
        qkv_bias: bool,
    ) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = Attention(
            num_heads,
            d_model,
            dropout=dropout,
            qkv_bias=qkv_bias
        )

        self.proj_norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.proj_dropout1 = nn.Dropout(dropout)
        self.proj2 = nn.Linear(4 * d_model, d_model)
        self.proj_dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x (B, N, D): patch embeddings
        """
        x = x + self.attn(self.attn_norm(x))
        x_mlp = self.proj_dropout1(self.gelu(self.proj1(self.proj_norm(x))))
        x_mlp = self.proj_dropout2(self.proj2(x_mlp))
        x = x + x_mlp
        return x
