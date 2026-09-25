"""Forward-pass shape contracts of the shared encoders and the pretraining heads (CPU, tiny)."""

import pytest
import torch

from tutorial_rs import IMG_SIZE, build_vit_s8, build_vit_t8
from tutorial_ts import SERIES_LEN, build_ts_encoder


@pytest.mark.parametrize(("builder", "width"), [(build_vit_s8, 384), (build_vit_t8, 192)])
def test_vit_shapes(builder, width):
    encoder = builder().eval()
    assert encoder.embed_dim == width
    images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    grid = encoder.grid_size
    with torch.no_grad():
        tokens, attn = encoder(images, return_attn=True)
        assert tokens.shape == (2, 1 + grid * grid, width)
        assert attn.shape[:2] == (2, encoder.blocks[-1].attn.num_heads)
        for pool in ("cls", "mean"):
            assert encoder.forward_features(images, pool=pool).shape == (2, width)


def test_ts_encoder_shapes():
    encoder = build_ts_encoder().eval()
    series = torch.randn(3, SERIES_LEN, 1)
    with torch.no_grad():
        for pool in ("cls", "mean"):
            assert encoder.forward_features(series, pool=pool).shape == (3, encoder.embed_dim)


def test_heads_follow_encoder_width():
    """SimCLR / MAE / DINO heads read the width off the encoder, so vit_t8 works everywhere."""
    import train_contrastive
    import train_dino
    import train_mae

    images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert train_contrastive.SimCLRModel(build_vit_t8())(images).shape == (2, 128)
        assert train_dino.DINOModel(build_vit_t8(), out_dim=64)(images).shape == (2, 64)
        pred, mask = train_mae.MAEModel(build_vit_t8(), decoder_dim=64, decoder_heads=2)(images)
    num_patches = (IMG_SIZE // 8) ** 2
    assert pred.shape == (2, num_patches, 8 * 8 * 3)
    assert mask.shape == (2, num_patches)
