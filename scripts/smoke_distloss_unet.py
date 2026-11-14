"""U-Net 上三条距离损失臂的接线冒烟（不加载真实数据，CPU 可跑）。

三条臂与各自要回答的问题：
  A 区域 HD 平均项  0.5CE+0.5Dice+0.5*rhd(sym, I=6)   —— MLMI 2025 主用式
  B 区域 HD 最大项  0.5CE+0.5Dice+0.5*rhd(h,   I=6)   —— soft-max 压最坏点（Karimi 方向）
  C 边界图加权项    0.5CE+0.5Dice+5.0*Boundary        —— Kervadec，边界位置监督

校验项：
  1. U-Net 前向输出形状 (B,2,224,224)
  2. 三条总损失有限，且对网络参数有非零有限梯度
  3. rhd 的 h 项量级落在 [0, iters+0.5]（soft-max 上界是剖面截断深度），且与 sym 同量级
  4. boundary 项乘权重后与 0.5CE+0.5Dice 同量级（相差 < 3x）
  5. 空前景 batch 上三条损失都不 NaN，rhd / boundary 对空样本返回 0 贡献

运行：venv\\Scripts\\python.exe scripts\\smoke_distloss_unet.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.unet import UNet
from utils import BoundaryLoss, DiceLoss, regional_hd_loss


def _labels(batch=2, size=224):
    label = torch.zeros(batch, size, size, dtype=torch.long)
    label[:, 60:160, 70:150] = 1
    return label


def main():
    torch.manual_seed(0)
    net = UNet(in_channels=1, num_classes=2)
    net.train()

    img = torch.randn(2, 1, 224, 224) * 0.3
    label = _labels()

    ce = torch.nn.CrossEntropyLoss()
    dice = DiceLoss(2)
    boundary = BoundaryLoss()

    with torch.no_grad():
        out = net(img)
    assert out.shape == (2, 2, 224, 224), out.shape
    print('[1] UNet forward ok, out', tuple(out.shape))

    def run(name, extra, weight):
        net.zero_grad(set_to_none=True)
        logits = net(img)
        l_ce = ce(logits, label)
        l_dice = dice(logits, label, softmax=True)
        l_extra = extra(logits, label)
        total = 0.5 * l_ce + 0.5 * l_dice + weight * l_extra
        total.backward()
        gnorm = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
        assert torch.isfinite(total).item(), (name, total)
        assert torch.isfinite(gnorm).item() and gnorm.item() > 0, (name, gnorm)
        print('[2] %-8s ce %.4f dice %.4f extra %.4f | w*extra %.4f | 0.5CE+0.5Dice %.4f'
              ' | grad_sum %.3e'
              % (name, l_ce.item(), l_dice.item(), l_extra.item(), weight * l_extra.item(),
                 0.5 * l_ce.item() + 0.5 * l_dice.item(), gnorm.item()))
        return l_extra.item()

    v_sym = run('rhd-sym', lambda o, l: regional_hd_loss(o, l, term='sym', iters=6), 0.5)
    v_h = run('rhd-h', lambda o, l: regional_hd_loss(o, l, term='h', iters=6), 0.5)
    v_b = run('boundary', boundary, 5.0)

    assert 0.0 <= v_h <= 6.5, v_h
    assert 0.5 * v_sym < v_h < 3.0 * v_sym or v_h < 0.5, (v_sym, v_h)
    print('[3] rhd h 项量级 ok：sym %.4f / h %.4f，h 上界 = iters = 6' % (v_sym, v_h))

    with torch.no_grad():
        logits = net(img)
        base = 0.5 * ce(logits, label) + 0.5 * dice(logits, label, softmax=True)
        assert abs(5.0 * v_b) < 3.0 * base.item(), (v_b, base.item())
    print('[4] boundary 项与 Dice 同量级 ok（w*bd %.4f vs base %.4f）' % (5.0 * v_b, base.item()))

    empty = torch.zeros(2, 224, 224, dtype=torch.long)
    logits = torch.randn(2, 2, 224, 224)
    e_rhd = regional_hd_loss(logits, empty, term='sym', iters=6)
    e_bd = boundary(logits, empty)
    assert torch.isfinite(e_rhd).item() and e_rhd.item() == 0.0, e_rhd
    print('[5] 空前景：rhd = 0.0 ok；boundary = %.4f（GT 全空时距离图全 1）' % e_bd.item())

    print('ALL PASS')


if __name__ == '__main__':
    main()
