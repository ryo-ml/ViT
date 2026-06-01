import torch
import torch.nn as nn

from src.models.modules import Block
from src.models.embeddings import LearnablePositionalEmbedding, PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: tuple,
        patch_size: int,
        in_channels: int,
        d_model: int,
        dropout: float,
        num_layers: int,
        num_heads: int,
        qkv_bias: bool,
    ):
        super().__init__()

        height, width = img_size
        num_patches =  (height // patch_size) * (width // patch_size)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size
        )
        self.pos_embed = LearnablePositionalEmbedding(
            num_tokens=num_patches + 1,
            d_model=d_model,
        )
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(
                num_heads=num_heads,
                d_model=d_model,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        x (B, C, H, W): 2d images
        """
        B, _, _, _ = x.size()

        cls_tokens = self.cls_token.expand(B, -1, -1) # (1, 1, d_model) -> (B, 1, d_model)
        x = self.patch_embed(x) # (B, N, d_model)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N + 1, d_model)
        x = self.emb_dropout(x + self.pos_embed())

        for block in self.blocks:
            x = block(x)
        return x
