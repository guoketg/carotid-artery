# coding=utf-8
"""冒烟：FGR+DDBE 组合臂 / UNet 弱骨干 FGR 臂（2026-11-10 队列前置检查）。

同 seed=1234 构建基线与各臂，固定输入前向一次，比对 logits：
  - R50 ddbe4+fccombo：DDBE.fuse 与 FGR.fuse 都是零初始化 → logits 与基线逐位相等
  - UNet fgr / unet ddbe4 / unet ddbe4+fgr：同上（unet 挂载已移到 init_kaiming 之后，
    同 seed 基础权重与不挂载版本逐位一致）
另打印各臂参数量与新增模块参数。
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch

from networks.unet import UNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

SEED = 1234


def build_r50(**kw):
    """复刻 train.py：同 seed、R50-ViT-B_16、vitL0、skip3、n_classes=2、load_from 预训练。"""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = 2
    cfg.n_skip = 3
    cfg.transformer.num_layers = 0
    cfg.patches.grid = (14, 14)   # train.py：vit_name 含 R50 时 grid = img_size/vit_patches_size
    net = ViT_seg(cfg, img_size=224, num_classes=2, **kw).cuda()
    net.load_from(weights=np.load(cfg.pretrained_path))
    return net


def build_unet(**kw):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    return UNet(in_channels=1, num_classes=2, **kw).cuda()


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def check(family, builder, arms):
    torch.manual_seed(SEED + 999)
    x = torch.randn(1, 1, 224, 224).cuda()
    base_net = builder(**arms[0][1])
    base_p = n_params(base_net)
    with torch.no_grad():
        y0 = base_net(x)
    print('[%s] baseline: params=%d, logits[0,0,0,:3]=%s'
          % (family, base_p, y0.flatten()[:3].tolist()))
    ok = True
    for name, kw in arms[1:]:
        net = builder(**kw)
        p = n_params(net)
        with torch.no_grad():
            y = net(x)
        dmax = (y - y0).abs().max().item()
        tag = 'IDENTITY' if dmax == 0.0 else 'MISMATCH delta=%.3e' % dmax
        ok = ok and dmax == 0.0
        print('  %-14s params=%d (+%d)  vs baseline: %s' % (name, p, p - base_p, tag))
    return ok


ok = True

# ---- R50 强骨干：FGR+DDBE 组合（ddbe→fgr 同点串联） ----
ok &= check('R50 vitL0', build_r50, [
    ('baseline', {}),
    ('ddbe4+fgr', dict(ddbe_level=4, fgr=True)),          # haar 默认
    ('fgr_only', dict(fgr=True)),
    ('ddbe4_only', dict(ddbe_level=4)),
])

# ---- UNet 弱骨干：仅 FGR / FGR+DDBE ----
ok &= check('UNet', build_unet, [
    ('baseline', {}),
    ('fgr', dict(fgr=True)),
    ('ddbe4', dict(ddbe_level=4)),
    ('ddbe4+fgr', dict(ddbe_level=4, fgr=True)),
])

print('SMOKE OK' if ok else 'SMOKE FAILED')
sys.exit(0 if ok else 1)
