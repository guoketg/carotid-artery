# coding=utf-8
"""冒烟：FGR 位置研究（--fgr_pos dec/enc/skip）+ 编码器融合 EFF / EFF-v2（2026-11-13）。

CPU-only（GPU 让给训练队列）。断言四件：
  1. 恒等起步：同 seed 下 dec/enc/skip/eff 四臂与基线 logits 逐位相等；
     module 级 FGR(enc 形状) 与 EFF（whole/band/lf-diff 三档）均满足 y == x。
  2. 参数量：每种位置的增量都恰好是 Σ(8C²+3C)（UNet 四级 64/128/256/512
     → 2,788,160），EFF 因 proj 与通道翻倍抵消，数值相同；EFF-v2 band 档
     多出的两枚 C→C 核比单核多 C 个 bias/级（+960）。
  3. 梯度链路：EFF 第 0 步 fuse 有梯度（gate/proj 数学上必为 0），1 步 SGD 后
     gate 非零，2 步后 proj 非零——「一条链上只有一个零」不被冻死；band 档
     两枚核都要在第 0 步有梯度。
  4. train.py 旗标接线：--fgr_pos enc 不带 --fgr 1、--eff 配 --arch baseline、
     --eff_mode/--eff_lfdiff 不带 --eff 1 都必须 parser.error（exit 2）。
"""
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch

from networks.fgr import EFF, FGR
from networks.unet import UNet

SEED = 1234
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')
CHS = (64, 128, 256, 512)
EXPECT = sum(8 * c * c + 3 * c for c in CHS)
EXPECT_BAND = EXPECT + sum(CHS)          # band 档两枚 C→C 核多 +C bias/级


def build_unet(**kw):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    return UNet(in_channels=1, num_classes=2, **kw)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def check(x):
    arms = [
        ('baseline', {}, 0),
        ('fgr_dec', dict(fgr=True, fgr_pos='dec'), EXPECT),
        ('fgr_enc', dict(fgr=True, fgr_pos='enc'), EXPECT),
        ('fgr_skip', dict(fgr=True, fgr_pos='skip'), EXPECT),
        ('eff', dict(eff=True), EXPECT),
        ('eff_band', dict(eff=True, eff_mode='band'), EXPECT_BAND),
        ('eff_diff', dict(eff=True, eff_lfdiff=True), EXPECT),
        ('eff_band_diff', dict(eff=True, eff_mode='band', eff_lfdiff=True), EXPECT_BAND),
    ]
    base = build_unet(**arms[0][1]).eval()
    p0 = n_params(base)
    with torch.no_grad():
        y0 = base(x)
    print('[UNet] baseline params=%d' % p0)
    ok = True
    for name, kw, expect in arms[1:]:
        net = build_unet(**kw).eval()
        p = n_params(net)
        with torch.no_grad():
            y = net(x)
        dmax = (y - y0).abs().max().item()
        delta = p - p0
        good = dmax == 0.0 and delta == expect
        ok = ok and good
        print('  %-13s params=+%d (expect %d)  identity=%s  %s'
              % (name, delta, expect, dmax, 'OK' if good else 'FAIL'))
        del net, y
    del base, y0
    return ok


def module_check():
    torch.manual_seed(SEED)
    x8 = torch.randn(2, 8, 8, 8)
    deep = torch.randn(2, 16, 4, 4)
    f = FGR(8, 8, 8)
    d_fgr = (f(x8) - x8).abs().max().item()
    deltas = []
    for mode, lfd in (('whole', False), ('band', False), ('whole', True), ('band', True)):
        e = EFF(8, 8, 8, 16, mode=mode, lf_diff=lfd)
        deltas.append((e(x8, deep) - x8).abs().max().item())
    print('[module] FGR identity=%.1e  EFF whole/band/lfdiff/band+diff=%s'
          % (d_fgr, ['%.1e' % v for v in deltas]))
    return d_fgr == 0.0 and all(v == 0.0 for v in deltas)


def eff_grad_check():
    ok = True
    for mode in ('whole', 'band'):
        torch.manual_seed(SEED)
        m = EFF(8, 8, 8, 16, mode=mode)
        x = torch.randn(2, 8, 8, 8)
        deep = torch.randn(2, 16, 4, 4)
        fuse_names = (['fuse_hf', 'fuse_lf'] if mode == 'band' else ['fuse'])
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        fuse_g, gate_g, proj_g = [], [], []
        for _ in range(3):
            opt.zero_grad()
            m(x, deep).pow(2).mean().backward()
            fuse_g.append(sum(getattr(m, n).weight.grad.abs().sum().item()
                              for n in fuse_names))
            gate_g.append(m.gate.weight.grad.abs().sum().item())
            proj_g.append(m.proj.weight.grad.abs().sum().item())
            opt.step()
        print('[grad] EFF-%-5s fuse=%s gate=%s proj=%s'
              % (mode, ['%.2e' % v for v in fuse_g], ['%.2e' % v for v in gate_g],
                 ['%.2e' % v for v in proj_g]))
        good = (fuse_g[0] > 0 and gate_g[0] == 0.0 and proj_g[0] == 0.0
                and gate_g[1] > 0 and proj_g[2] > 0)
        ok = ok and good
    return ok


def arg_check():
    cases = [
        (['--dataset', 'Carotid', '--fgr_pos', 'enc'], 'fgr_pos'),
        (['--dataset', 'Carotid', '--arch', 'baseline', '--eff', '1'], 'eff'),
        (['--dataset', 'Carotid', '--arch', 'unet', '--eff_mode', 'band'], 'eff_mode'),
    ]
    ok = True
    for argv, key in cases:
        r = subprocess.run([PYTHON, 'train.py'] + argv, cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        tail = (r.stderr or r.stdout or '').strip().splitlines()
        tail = tail[-1] if tail else ''
        good = r.returncode == 2 and key in tail
        ok = ok and good
        print('[args] %s -> exit=%d %s  %s'
              % (' '.join(argv[-3:]), r.returncode, tail, 'OK' if good else 'FAIL'))
    return ok


def check_r50():
    """R50-TransUNet（--arch baseline --vit_layers 0）编码器位置接线。

    编码器四级实际形状 112²/64、55²/256（padding hack 前）、28²/512、14²/1024，
    增量 = Σ(8C²+3C) = 11,046,848（比解码器位置的 691,568 大 16 倍，用户明确
    「不管容量」）。恒等比对必须 eval：ViT 嵌入层 Dropout(0.1) 在 train 下随机。
    """
    import copy
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer

    def build(**kw):
        cfg = copy.deepcopy(CONFIGS['R50-ViT-B_16'])
        cfg.n_classes = 2
        cfg.n_skip = 3
        cfg.transformer.num_layers = 0
        cfg.patches.grid = (int(224 / 16), int(224 / 16))
        torch.manual_seed(SEED)
        return VisionTransformer(cfg, img_size=224, num_classes=2, **kw).eval()

    expect = sum(8 * c * c + 3 * c for c in (64, 256, 512, 1024))
    x = torch.randn(1, 1, 224, 224)
    base = build()
    p0 = n_params(base)
    with torch.no_grad():
        y0 = base(x)
    net = build(fgr=True, fgr_pos='enc')
    p = n_params(net)
    with torch.no_grad():
        y = net(x)
    dmax = (y - y0).abs().max().item()
    delta = p - p0
    good = dmax == 0.0 and delta == expect
    print('[R50] baseline params=%d  enc delta=+%d (expect %d)  identity=%s  %s'
          % (p0, delta, expect, dmax, 'OK' if good else 'FAIL'))
    return good


ok = True
x = torch.randn(1, 1, 224, 224)
ok &= check(x)
ok &= module_check()
ok &= eff_grad_check()
ok &= arg_check()
ok &= check_r50()
print('SMOKE OK' if ok else 'SMOKE FAILED')
sys.exit(0 if ok else 1)
