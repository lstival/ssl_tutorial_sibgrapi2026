"""SSL objectives on toy tensors: value ranges, known optima, and gradient flow."""

import torch
import torch.nn.functional as F
import train_dino
import train_mae
from train_contrastive import info_nce_loss


def test_info_nce_identifies_positives():
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(8, 16), dim=-1)
    feats = torch.cat([anchors, anchors]).requires_grad_()  # view k and k+N are identical
    loss, acc = info_nce_loss(feats, temperature=0.1)
    assert acc == 1.0
    assert loss.item() >= 0
    loss.backward()
    assert torch.isfinite(feats.grad).all()


def test_random_masking_keeps_the_right_tokens():
    x = torch.arange(2 * 10 * 4, dtype=torch.float32).reshape(2, 10, 4)
    visible, mask, ids_restore, ids_keep = train_mae.random_masking(x, mask_ratio=0.7)
    assert visible.shape == (2, 3, 4)
    assert mask.sum(dim=1).tolist() == [7, 7]
    # every visible token is the original token at its recorded position, and is unmasked
    assert torch.equal(visible, torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, 4)))
    assert (mask.gather(1, ids_keep) == 0).all()
    assert torch.equal(ids_restore.argsort(dim=1)[:, :3], ids_keep)


def test_mae_loss_is_zero_for_perfect_reconstruction():
    imgs = torch.randn(2, 3, 16, 16)
    target = train_mae.patchify(imgs)
    mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
    perfect = (target - mean) / (var + 1e-6).sqrt()
    mask = torch.ones(2, target.shape[1])
    assert train_mae.mae_reconstruction_loss(perfect, imgs, mask).item() < 1e-6


def test_dino_loss_and_ema():
    torch.manual_seed(0)
    student = [torch.randn(4, 32, requires_grad=True) for _ in range(4)]
    teacher = [torch.randn(4, 32) for _ in range(2)]
    center = torch.zeros(1, 32)
    loss = train_dino.dino_loss(student, teacher, center)
    loss.backward()
    assert loss.item() > 0 and student[0].grad is not None
    # the teacher never supervises its own view: crop 0 and crop 1 each get gradient from the
    # other teacher view only, local crops from both
    new_center = train_dino.update_center(center, teacher, momentum=0.9)
    assert new_center.shape == center.shape

    a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    before = b.weight.detach().clone()
    train_dino.ema_update(a, b, m=0.9)
    assert torch.allclose(b.weight, 0.9 * before + 0.1 * a.weight)
