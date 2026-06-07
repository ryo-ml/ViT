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
        num_classes=100,
    )

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    # y: (B, num_classes)
    assert y.size() == (2, 100)
