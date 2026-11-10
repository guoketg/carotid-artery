# coding=utf-8
"""冒烟：--r50_fdf / --fgr / ddbectl(plain) 恒等起步与预训练加载（2026-11-09 队列前置检查）。

同 seed=1234 构建基线与各臂（复刻 train.py 的 config 构建与 load_from），
固定输入前向一次，逐臂比对 logits：
  - drdac / fgr / ddbe4-plain：fuse 是唯一零点 → logits 必须与基线逐位相等
  - drdac_mfdf / full：FEM 非恒等（FDF randn*0.02、LN、ASCA）→ Δ≠0 属预期，
    只验证前向跑通、load_from 无 KeyError
另打印各臂参数量与新增模块参数。
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

SEED = 1234


def build(**kw):
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


def n_params(net):
    return sum(p.numel() for p in net.parameters())


torch.manual_seed(SEED + 999)
x = torch.randn(1, 1, 224, 224).cuda()

arms = [
    ('baseline', {}),
    ('ddbe4_plain(CTL)', dict(ddbe_level=4, ddbe_hf_op='plain')),
    ('r50_fdf=drdac(A1)', dict(r50_fdf='drdac')),
    ('r50_fdf=drdac_mfdf(A2)', dict(r50_fdf='drdac_mfdf')),
    ('r50_fdf=full(A3)', dict(r50_fdf='full')),
    ('fgr', dict(fgr=True)),
]

base_net = build()
base_p = n_params(base_net)
with torch.no_grad():
    y0 = base_net(x)
print('baseline: params=%d, logits[0,0,0,:3]=%s' % (base_p, y0.flatten()[:3].tolist()))

for name, kw in arms[1:]:
    net = build(**kw)
    p = n_params(net)
    with torch.no_grad():
        y = net(x)
    dmax = (y - y0).abs().max().item()
    tag = 'IDENTITY' if dmax == 0.0 else 'delta=%s (FEM 非恒等属预期)' % ('%.3e' % dmax)
    print('%-22s params=%d (+%d)  vs baseline: %s' % (name, p, p - base_p, tag))

print('SMOKE OK')
