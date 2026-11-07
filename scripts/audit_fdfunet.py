"""FDFUNet 复现完整性审查（机器可复核，不依赖人工"看起来像"）。

审查对象：networks/fdfunet.py 是否把论文 Eq.(1)-(6) 与 Fig.5/6/7 的结构都实现了。
用法：python scripts/audit_fdfunet.py   （任何一项 FAIL 时退出码非零）
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.fdfunet import (ASCA, DASConv, DRDAC, FDF, FEM, MFDF, FDFUNet)  # noqa: E402

IMG = 224
RECORDS = []


def check(name, ok, detail=''):
    RECORDS.append((name, bool(ok), detail))
    print('%-46s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, x):
    total = {'v': 0}

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            total['v'] += out.numel() * m.kernel_size[0] * m.kernel_size[1] * m.in_channels / m.groups
        elif isinstance(m, nn.Conv1d):
            total['v'] += out.numel() * m.kernel_size[0] * m.in_channels / m.groups
        elif isinstance(m, nn.Linear):
            total['v'] += out.numel() * m.in_features

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return 2.0 * total['v'] / 1e9          # 1 MAC = 2 FLOPs


# ---------------------------------------------------------------- 1. 模块级
def audit_drdac():
    torch.manual_seed(0)
    blk = DRDAC(4, 8).eval()
    x = torch.randn(2, 4, 16, 16)
    with torch.no_grad():
        y = blk(x)
        x1 = blk.conv(x)
        ref = blk.fuse(torch.cat([x, x1 + blk.pw(x1 + blk.ass(x1))], dim=1))
    check('DRDAC 与式(1) 数值一致', torch.allclose(y, ref, atol=1e-6),
          'X2 = X1 + PW(X1 + ASS(X1)); Y = 1x1Conv(Concat(X, X2))')
    check('DRDAC 输出通道 = 声明通道（skip 相加可对齐）',
          blk(torch.randn(1, 4, 16, 16)).shape[1] == 8)
    check('ASSConv 为 depthwise + 空洞率 3',
          blk.ass.dw.groups == 8 and blk.ass.dw.dilation == (3, 3))


def audit_fdf():
    f = FDF(8, 16, 16)
    x = torch.randn(2, 8, 16, 16)
    y = f(x)
    check('FDF 复数权重形状 (C,H,W/2+1,2)',
          tuple(f.weight.shape) == (8, 16, 9, 2), str(tuple(f.weight.shape)))
    check('FDF 初始化 std≈0.02（GFNet 惯例）',
          abs(f.weight.std().item() - 0.02) < 0.004, '%.4f' % f.weight.std().item())
    check('FDF 形状不变且非恒等（频域权重可学）',
          y.shape == x.shape and not torch.allclose(y, x, atol=1e-3))


def audit_mfdf():
    m = MFDF(16, 32, 32).eval()
    x = torch.randn(1, 16, 32, 32, requires_grad=True)
    y = m(x)
    check('MFDF 通道不增不减（残差结构）', y.shape == x.shape)
    y.sum().backward()
    grads = {'das11': m.das11.ass.dw.weight, 'fdf12': m.fdf12.weight,
             'das21': m.das21.ass.dw.weight, 'fdf22': m.fdf22.weight,
             'fdf2': m.fdf2.weight, 'spatial': m.spatial.ass.dw.weight}
    check('MFDF 四路分支全部收到梯度',
          all(g.grad is not None and g.grad.abs().sum() > 0 for g in grads.values()),
          ','.join(k for k, g in grads.items() if g.grad is None))
    check('MFDF 四路通道数 = C/4', m.das11.ass.dw.out_channels == 4 and m.fdf12.weight.shape[0] == 4)
    check('MFDF 输入归一化为通道维 LN',
          m.ln_w.shape == (1, 16, 1, 1))


def audit_asca():
    a = ASCA(8, variant='row5').eval()
    x = torch.randn(2, 8, 12, 20)
    y = a(x)
    check('ASCA 形状不变（乘性注意力）', y.shape == x.shape)

    calls = []
    o_avg, o_max = F.adaptive_avg_pool2d, F.adaptive_max_pool2d
    F.adaptive_avg_pool2d = lambda t, s: (calls.append(('avg', s)), o_avg(t, s))[1]
    F.adaptive_max_pool2d = lambda t, s: (calls.append(('max', s)), o_max(t, s))[1]
    try:
        with torch.no_grad():
            a(x)
    finally:
        F.adaptive_avg_pool2d, F.adaptive_max_pool2d = o_avg, o_max
    seq = calls[:4]
    expect = [('avg', (12, 1)), ('max', 1), ('max', (1, 20)), ('avg', 1)]
    check('ASCA 池化顺序 = H:Avg→Max / W:Max→Avg', seq == expect, str(seq))

    w = (y.abs() / x.abs().clamp_min(1e-6))
    check('ASCA 门控取值在 (0,1)', bool((w > 0).all() and (w < 1).all()))


def audit_topology(net_full, net_base):
    check('FEM 总数 = 5(编码) + 4(解码) = 9',
          len(net_full.fems_down) == 5 and len(net_full.fems_up) == 4,
          'down=%d up=%d' % (len(net_full.fems_down), len(net_full.fems_up)))
    check('主干 5 级、通道 [64,128,256,512,1024]',
          net_full.channels == (64, 128, 256, 512, 1024))
    check('Up 块 5 个（最后一个只上采样、不接 FEM）',
          len(net_full.ups) == 4 and isinstance(net_full.up_last, nn.Module),
          'ups=4 + up_last')
    check('skip 为加法（无 skip 拼接层）',
          not any('skip' in n for n, _ in net_full.named_modules()),
          'forward 内 feats 直接相加')
    check('UNet* 基线已去掉 DRDAC/MFDF/ASCA',
          not isinstance(net_base.downs[0].body, DRDAC)
          and isinstance(net_base.fems_down[0].mfdf, nn.Identity)
          and isinstance(net_base.fems_down[0].asca, nn.Identity))
    check('UNet* 基线参数量 < FDFUNet', count_params(net_base) < count_params(net_full),
          '%.2fM vs %.2fM' % (count_params(net_base) / 1e6, count_params(net_full) / 1e6))


def audit_forward_grad():
    net = FDFUNet(img_size=IMG, num_classes=2).train()
    x = torch.randn(2, 1, IMG, IMG)
    y = net(x)
    check('前向输出形状 (B,2,224,224)', tuple(y.shape) == (2, 2, IMG, IMG), str(tuple(y.shape)))
    y.square().mean().backward()
    dead = [n for n, p in net.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    check('全部参数获得非零梯度（无死支路）', len(dead) == 0, 'dead=%d %s' % (len(dead), dead[:3]))

    net_b = FDFUNet(img_size=IMG, num_classes=2,
                    use_drdac=False, use_mfdf=False, use_asca=False)
    fl_full = count_macs(FDFUNet(img_size=IMG, num_classes=2).eval(), x[:1])
    fl_base = count_macs(net_b.eval(), x[:1])
    ratio = fl_full / fl_base
    check('FLOPs 比值≈论文 19.83/12.84=1.54（±10%）', abs(ratio - 1.544) < 0.15,
          '本实现 %.2f (%.2fG / %.2fG)' % (ratio, fl_full, fl_base))


def audit_train_wiring():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_train = open(os.path.join(root, 'train.py'), encoding='utf-8').read()
    src_tr = open(os.path.join(root, 'trainer.py'), encoding='utf-8').read()
    check("train.py 已注册 --arch fdfunet", "'fdfunet'" in src_train)
    check("train.py 已注册 --fdfunet_variant", '--fdfunet_variant' in src_train)
    check("trainer.py 已接 --loss bce_iou_dice", 'bce_iou_dice' in src_tr)
    check("trainer.py 已接 --optim adam + cosine", "optim_name == 'adam'" in src_tr
          and 'math.cos' in src_tr)


def main():
    torch.manual_seed(0)
    audit_drdac()
    audit_fdf()
    audit_mfdf()
    audit_asca()
    net_full = FDFUNet(img_size=IMG, num_classes=2)
    net_base = FDFUNet(img_size=IMG, num_classes=2,
                       use_drdac=False, use_mfdf=False, use_asca=False)
    audit_topology(net_full, net_base)
    audit_forward_grad()
    audit_train_wiring()

    n_fail = sum(1 for _, ok, _ in RECORDS if not ok)
    print('\n%d 项检查，%d 项通过，%d 项失败' % (len(RECORDS), len(RECORDS) - n_fail, n_fail))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
