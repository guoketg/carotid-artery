# coding=utf-8
"""冒烟：FBGR 分带档（--fgr_mode band）+ full 档（HFE+FBGR）（2026-11-11 队列前置检查）。

CPU-only（GPU 让给在跑的训练队列）。断言四件（docs/module-suite-design.md §3.2/§7.2）：
  1. 恒等起步：同 seed 下 fgr_whole / fgr_band 与各自基线 logits 逐位相等；
     hfe+fgr_band 与 hfe 基线逐位相等；R50 fgr_band 与 R50 基线逐位相等；
     module 级 FGR(mode='band') y == x。
  2. 参数增量：band − whole == ΣC（UNet 四级 512/256/128/64 → +960；R50 四级
     256/128/64/16 → +464；full 臂 − hfe 基线同为 +960）。
  3. 门不被冻死：第 0 步梯度直达 fuse_hf/fuse_lf（gate 数学上必为 0，因 fuse 零核
     → dL/dw = 0），一步 SGD 后 gate 梯度非零——fgr.py docstring「经 fuse 移动后」。
  4. train.py 旗标接线：--fgr_mode band 不带 --fgr 1 必须 parser.error（exit 2）。
"""
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch

from networks.fgr import FGR
from networks.unet import UNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

SEED = 1234
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')


def build_r50(**kw):
    """复刻 train.py：同 seed、R50-ViT-B_16、vitL0、skip3、n_classes=2、load_from 预训练。"""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = 2
    cfg.n_skip = 3
    cfg.transformer.num_layers = 0
    cfg.patches.grid = (14, 14)   # train.py：vit_name 含 R50 时 grid = img_size/vit_patches_size
    net = ViT_seg(cfg, img_size=224, num_classes=2, **kw)   # CPU：GPU 让给训练队列
    net.load_from(weights=np.load(cfg.pretrained_path))
    return net


def build_unet(**kw):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    return UNet(in_channels=1, num_classes=2, **kw)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def check(family, builder, arms, x):
    """arms[0] 为基线，其余各臂比对 logits 与参数量。返回 (ok, {name: params})。

    恒等比对统一在 eval() 下做：ViT 嵌入层带 Dropout(0.1)（vit_seg_modeling.py L172），
    train 模式两次 forward 的随机 mask 不同会造成假失配。
    """
    base_net = builder(**arms[0][1]).eval()
    params = {arms[0][0]: n_params(base_net)}
    with torch.no_grad():
        y0 = base_net(x)
    print('[%s] baseline %s: params=%d' % (family, arms[0][0], params[arms[0][0]]))
    ok = True
    for name, kw in arms[1:]:
        net = builder(**kw).eval()
        params[name] = n_params(net)
        with torch.no_grad():
            y = net(x)
        dmax = (y - y0).abs().max().item()
        tag = 'IDENTITY' if dmax == 0.0 else 'MISMATCH delta=%.3e' % dmax
        ok = ok and dmax == 0.0
        print('  %-14s params=%d (+%d)  %s' % (name, params[name],
                                               params[name] - params[arms[0][0]], tag))
        del net, y
    del base_net, y0
    return ok, params


def grad_check():
    """第 0 步梯度直达 fuse（gate 数学上为 0），fuse 移动后 gate 有梯度、不被冻死。"""
    torch.manual_seed(SEED)
    m = FGR(16, 28, 28, mode='band')
    x = torch.randn(2, 16, 28, 28)
    ident = (m(x) - x).abs().max().item()   # 恒等起步（module 级）
    m(x).pow(2).mean().backward()
    g_fuse = (m.fuse_hf.weight.grad.abs().sum()
              + m.fuse_lf.weight.grad.abs().sum()).item()
    g_gate0 = m.gate.weight.grad.abs().sum().item()   # fuse 零核 → dL/dw = 0，必为 0
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    opt.step()
    opt.zero_grad()
    m(x).pow(2).mean().backward()
    g_gate1 = m.gate.weight.grad.abs().sum().item()
    print('[grad] module identity=%.1e  step0: fuse=%.3e gate=%.3e(预期0)  '
          'step1: gate=%.3e' % (ident, g_fuse, g_gate0, g_gate1))
    return ident == 0.0 and g_fuse > 0 and g_gate0 == 0.0 and g_gate1 > 0


def arg_check():
    """--fgr_mode band 不带 --fgr 1 必须 parser.error（exit 2，任何目录副作用之前）。"""
    r = subprocess.run([PYTHON, 'train.py', '--dataset', 'Carotid',
                        '--fgr_mode', 'band'],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    msg = (r.stderr or r.stdout or '').strip().splitlines()
    tail = msg[-1] if msg else ''
    print('[args] exit=%d  %s' % (r.returncode, tail))
    return r.returncode == 2 and 'fgr_mode' in tail


ok = True
x = torch.randn(1, 1, 224, 224)

# ---- 1. UNet：整带（重构回归）+ 分带，参数差必须 == 960 ----
ok1, up = check('UNet', build_unet, [
    ('baseline', {}),
    ('fgr_whole', dict(fgr=True)),
    ('fgr_band', dict(fgr=True, fgr_mode='band')),
], x)
ok1 &= (up['fgr_band'] - up['fgr_whole']) == 960
print('[UNet] band - whole = %d (expect 960)' % (up['fgr_band'] - up['fgr_whole']))

# ---- 2. full 档：hfe+fgr_band vs hfe 基线；FGR 增量必须与它在 baseline 上的增量一致 ----
ok2, uph = check('UNet hfe', build_unet, [
    ('hfe', dict(edge_in=True)),
    ('hfe+fgr_band', dict(edge_in=True, fgr=True, fgr_mode='band')),
], x)
ok2 &= (uph['hfe+fgr_band'] - uph['hfe']) == (up['fgr_band'] - up['baseline'])
print('[UNet hfe] full - hfe = %d (expect %d, same FGR stack on both bases)'
      % (uph['hfe+fgr_band'] - uph['hfe'], up['fgr_band'] - up['baseline']))

# ---- 3. R50：分带档透传（vit_seg_modeling 侧挂载）----
ok3, rp = check('R50 vitL0', build_r50, [
    ('baseline', {}),
    ('fgr_band', dict(fgr=True, fgr_mode='band')),
], x)
delta_r50 = rp['fgr_band'] - rp['baseline']
# 网络参数差必须恰好等于四级 FGR band 模块的参数和（256/128/64/16 @ 28/56/112/224）
mods = [FGR(c, s, s, mode='band')
        for c, s in zip((256, 128, 64, 16), (28, 56, 112, 224))]
expect = sum(n_params(m) for m in mods)
ok3 &= delta_r50 == expect
print('[R50] band-baseline = %d (expect %d = sum of 4 band modules)'
      % (delta_r50, expect))

# ---- 4. 梯度与旗标接线 ----
ok &= grad_check()
ok &= arg_check()

ok = ok and ok1 and ok2 and ok3
print('SMOKE OK' if ok else 'SMOKE FAILED')
sys.exit(0 if ok else 1)
