import torch

from src.models.vision_transformer import VisionTransformer

def test_vit() -> None:
    model = VisionTransformer(
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,
        d_model=768,
        dropout=0.1,
        num_layers=12,
        num_heads=8,
        qkv_bias=True,
    )

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    # num_tokens = num_patches + cls_token = 196 + 1 = 197
    assert y.size() == (2, 197, 768)
