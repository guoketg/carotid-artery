"""区域 Hausdorff 距离损失（#82）冒烟：不加载模型，只校验距离剖面与损失行为。

校验项（与 docs/experiments.md §2.1 一一对应）：
  1. 精确性：二值输入下 d+ 与 scipy 棋盘距离在 I 层内逐像素相等，且随 I 单调不减
  2. 可微性：float64 下 autograd 梯度与中心差分相对误差 < 1e-3
  3. 恒等性：rhd_w=0 时总损失与 0.5CE+0.5Dice 逐位相同（diff 0.0）
  4. 尺度探针：打印 ce / dice / rhd，报告与 0.5CE+0.5Dice 的倍率
  5. 空前景守卫：GT 全空时返回 0 且无 NaN（混合 batch 只算有效样本）

运行：venv\\Scripts\\python.exe scripts\\smoke_rhd_loss.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DiceLoss, positive_distance_map_2d, regional_hd_loss


def _blob_label(size=32, r0=8, r1=24, c0=10, c1=22, batch=1):
    label = torch.zeros(batch, size, size, dtype=torch.long)
    label[:, r0:r1, c0:c1] = 1
    return label


def main():
    torch.manual_seed(0)
    ce = torch.nn.CrossEntropyLoss()
    dice = DiceLoss(2)
    label = _blob_label()

    # 1) 精确性：软腐蚀在二值输入上 = 棋盘距离（截断在 I）
    gt = (label > 0).float().unsqueeze(1)
    ref = ndi.distance_transform_cdt(~(label[0].numpy() > 0), metric='chessboard')
    for iters in (2, 4, 8):
        d = positive_distance_map_2d(gt, iters)[0, 0].numpy()
        expect = np.minimum(ref, iters).astype(np.float32)
        assert np.array_equal(d, expect), (iters, np.abs(d - expect).max())
    d4 = positive_distance_map_2d(gt, 4)[0, 0].numpy()
    d8 = positive_distance_map_2d(gt, 8)[0, 0].numpy()
    assert np.all(d8 >= d4 - 1e-6), 'd+ 应随 I 单调不减'
    print('[1] d+ == chessboard distance (I=2/4/8) ok, max d+ =', float(d8.max()))

    # 2) 可微性：float64 + 中心差分
    g = torch.Generator().manual_seed(1)
    logits64 = torch.randn(1, 2, 24, 24, generator=g, dtype=torch.float64)
    lab = _blob_label(size=24, r0=6, r1=18, c0=8, c1=16)
    logits64.requires_grad_(True)
    loss = regional_hd_loss(logits64, lab, term='sym', iters=6)
    loss.backward()
    grad = logits64.grad.detach().clone()
    rng = np.random.RandomState(0)
    max_rel = 0.0
    for _ in range(6):
        r, c = rng.randint(0, 24, size=2)
        ch = rng.randint(0, 2)
        h = 1e-2
        with torch.no_grad():
            lp = logits64.clone()
            lm = logits64.clone()
            lp[0, ch, r, c] += h
            lm[0, ch, r, c] -= h
            fd = (regional_hd_loss(lp, lab, term='sym', iters=6)
                  - regional_hd_loss(lm, lab, term='sym', iters=6)) / (2 * h)
        a = grad[0, ch, r, c].item()
        rel = abs(a - fd.item()) / max(1e-8, abs(fd.item()) + abs(a))
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-3, max_rel
    print('[2] autograd vs finite difference ok, max rel err = %.2e' % max_rel)

    # 3) 恒等性：rhd_w=0 与 0.5CE+0.5Dice 逐位相同
    logits = torch.randn(2, 2, 32, 32, generator=g)
    lab2 = _blob_label(batch=2)
    base = 0.5 * ce(logits, lab2) + 0.5 * dice(logits, lab2, softmax=True)
    with_w0 = (0.5 * ce(logits, lab2) + 0.5 * dice(logits, lab2, softmax=True)
               + 0.0 * regional_hd_loss(logits, lab2))
    assert (with_w0 - base).abs().item() == 0.0, (with_w0 - base).abs().item()
    print('[3] rhd_w=0 identical to 0.5CE+0.5Dice, diff = 0.0')

    # 4) 尺度探针：偏软的 logits 更接近训练早期
    soft = torch.randn(2, 2, 32, 32, generator=g) * 0.5
    soft.requires_grad_(True)
    l_ce = ce(soft, lab2)
    l_dice = dice(soft, lab2, softmax=True)
    l_rhd = regional_hd_loss(soft, lab2, term='sym', iters=6)
    ref_ce_dice = 0.5 * l_ce + 0.5 * l_dice
    assert torch.isfinite(l_rhd) and 0.0 <= l_rhd.item() <= 6.0, l_rhd.item()
    ratio = l_rhd.item() / (ref_ce_dice.item() + 1e-8)
    print('[4] ce %.4f | dice %.4f | rhd %.4f | 0.5CE+0.5Dice %.4f | ratio %.2fx'
          % (l_ce.item(), l_dice.item(), l_rhd.item(), ref_ce_dice.item(), ratio))

    # 5) 空前景守卫
    empty = torch.zeros(1, 24, 24, dtype=torch.long)
    l_empty = regional_hd_loss(torch.randn(1, 2, 24, 24, generator=g), empty, iters=6)
    assert torch.isfinite(l_empty).item() and l_empty.item() == 0.0, l_empty.item()
    mix = torch.stack([empty[0], _blob_label(size=24)[0]])
    l_mix = regional_hd_loss(torch.randn(2, 2, 24, 24, generator=g), mix, iters=6)
    assert torch.isfinite(l_mix) and l_mix.item() > 0
    print('[5] empty foreground -> 0.0, mixed batch ok (%.4f)' % l_mix.item())

    print('ALL PASS')


if __name__ == '__main__':
    main()
