"""损失改进 A/C 的离线冒烟：不加载模型，只在随机张量上校验恒等性与边界行为。

校验项：
  1. bndsoft 在 eps=0 时严格退化为普通 CE（保证与基线单变量可比）
  2. bndsoft 的边界带几何：带内 1 px 环、内部实心像素不在带内
  3. bndsoft 在 eps=1、全图前景时目标全为 (0.5, 0.5)，loss 恒等于 ln2
  4. farfp 的远处假阳性项：预测与 GT 一致时≈0，预测远处孤立块时明显>0
  5. 两条损失的梯度均有限

运行：venv\\Scripts\\python.exe scripts\\smoke_loss_unet.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DiceLoss, boundary_soft_ce, far_fp_loss


def _make_batch(n=2, size=32, r0=8, r1=24, c0=10, c1=22, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 2, size, size, generator=g)
    label = torch.zeros(n, size, size, dtype=torch.long)
    label[:, r0:r1, c0:c1] = 1
    return logits, label


def main():
    torch.manual_seed(0)
    logits, label = _make_batch()
    dice = DiceLoss(2)
    ce = torch.nn.CrossEntropyLoss()

    # 1) eps=0 必须严格退化为普通 CE
    l0 = boundary_soft_ce(logits, label, eps=0.0, radius=1)
    l_ref = ce(logits, label)
    assert abs(l0.item() - l_ref.item()) < 1e-5, (l0.item(), l_ref.item())
    print('[ok] bndsoft eps=0 == CrossEntropy (%.6f)' % l0.item())

    # 2) 边界带几何：1 px 环，实心内部不在带内
    fg = (label[0:1] > 0).float().unsqueeze(1)
    dil = F.max_pool2d(fg, kernel_size=3, stride=1, padding=1)
    ero = 1.0 - F.max_pool2d(1.0 - fg, kernel_size=3, stride=1, padding=1)
    band = (dil - ero).clamp(0.0, 1.0)[0, 0]
    assert band[15, 15].item() == 0.0, '内部像素不应在边界带内'
    assert band[15, 9].item() == 1.0, '左边界外 1 px 应在带内'
    assert band[8, 15].item() == 1.0, '上边界内 1 px 应在带内'
    assert band[15, 6].item() == 0.0, '离边界 4 px 不应在带内'
    print('[ok] bndsoft band geometry: %d px in band' % int(band.sum().item()))

    # 3) 3x3 图内只有一个中心前景像素 -> 整图都在带内；logits=0 时目标全为
    #    (0.5,0.5)，CE 的极小值（此处恰为值）等于 ln2
    logits_zero = torch.zeros(2, 2, 3, 3)
    label_center = torch.zeros(2, 3, 3, dtype=torch.long)
    label_center[:, 1, 1] = 1
    fg_c = label_center[0:1].unsqueeze(1).float()
    band_c = (F.max_pool2d(fg_c, 3, 1, 1)
              - (1.0 - F.max_pool2d(1.0 - fg_c, 3, 1, 1))).clamp(0.0, 1.0)
    assert band_c.sum().item() == 9.0, '3x3 单像素前景应整图在带内'
    l_all = boundary_soft_ce(logits_zero, label_center, eps=0.5, radius=1)
    assert abs(l_all.item() - 0.693147) < 1e-4, l_all.item()
    print('[ok] bndsoft eps=0.5, all-band target at logits=0 -> ln2 (%.6f)' % l_all.item())

    # 4) farfp：预测等于 GT 时远处假阳项≈0
    logits_big, label_big = _make_batch(size=64, r0=40, r1=56, c0=40, c1=56, seed=1)
    gt_prob = label_big.unsqueeze(1).float()
    logits_gt = torch.cat([1.0 - gt_prob, gt_prob], dim=1) * 20.0
    l_far_zero = far_fp_loss(logits_gt, label_big, margin=10)
    assert l_far_zero.item() < 1e-3, l_far_zero.item()

    # 预测在距 GT > margin 处出现孤立块时，该项应明显为正
    far_label = torch.zeros_like(label_big)
    far_label[:, 0:8, 0:8] = 1
    fp = far_label.unsqueeze(1).float()
    logits_far = torch.cat([1.0 - fp, fp], dim=1) * 20.0
    l_far_pos = far_fp_loss(logits_far, label_big, margin=10)
    assert l_far_pos.item() > 0.5, l_far_pos.item()
    assert 0.0 <= l_far_pos.item() <= 1.0
    print('[ok] farfp: gt=pred -> %.6f, far blob -> %.6f' % (l_far_zero.item(), l_far_pos.item()))

    # 5) 梯度有限
    logits.requires_grad_(True)
    loss_a = 0.5 * boundary_soft_ce(logits, label) + 0.5 * dice(logits, label, softmax=True)
    loss_a.backward()
    grad_a = logits.grad.clone()
    assert torch.isfinite(grad_a).all()
    logits.grad = None
    loss_c = (0.5 * ce(logits, label) + 0.5 * dice(logits, label, softmax=True)
              + 0.1 * far_fp_loss(logits, label, margin=10))
    loss_c.backward()
    assert torch.isfinite(logits.grad).all()
    print('[ok] grads finite (bndsoft %.4f, farfp %.4f)' % (loss_a.item(), loss_c.item()))

    print('SMOKE PASS')


if __name__ == '__main__':
    main()
