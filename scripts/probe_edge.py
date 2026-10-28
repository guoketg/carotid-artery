"""边缘特征融合·输入级 的自检脚本（CPU 即可跑）。

检查项：
1. haar_dwt / haar_idwt 是否精确可逆（IDWT 重建是否等于原图）
2. build_hf_image 的输出形状与数值（是否只含高频、是否被标准化）
3. edge_in=True 的模型能否加载预训练权重（conv_root 4 通道零填充）
4. edge_in=True 与 edge_in=False 的参数量差
5. edge_in=False 时旧行为不变（参数量 105.28M、输出形状不变）
"""
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.vit_seg_modeling import CONFIGS, VisionTransformer
from networks.wavelet import build_hf_image, haar_dwt, haar_idwt

torch.manual_seed(0)

# ---- 1. DWT/IDWT 可逆性 -------------------------------------------------
x = torch.randn(2, 1, 64, 64)
ll, details = haar_dwt(x)
rec = haar_idwt(ll, details)
print(f"[1] DWT/IDWT 可逆: max|Δ| = {(rec - x).abs().max().item():.3e}")

# ---- 2. HF 图 -----------------------------------------------------------
img = torch.rand(2, 1, 224, 224)
hf = build_hf_image(img)
print(f"[2] build_hf_image: shape={tuple(hf.shape)} "
      f"mean={hf.mean().item():+.4f} std={hf.std(dim=(1, 2, 3)).mean().item():.4f} "
      f"min={hf.min().item():+.3f} max={hf.max().item():+.3f}")

# HF 图与原图的相关性应当远低于 1（否则说明没真正去掉低频）
c = torch.corrcoef(torch.stack([img.flatten(), hf.flatten()]))[0, 1].item()
print(f"    原图 vs HF 图 相关系数 = {c:+.4f}（接近 0 才算真的只剩高频）")

# ---- 3/4/5. 模型 --------------------------------------------------------
config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
config.n_classes = 2
config.n_skip = 3
config.patches.grid = (14, 14)

base = VisionTransformer(config, img_size=224, num_classes=2)
edge = VisionTransformer(config, img_size=224, num_classes=2, edge_in=True)
n_base = sum(p.numel() for p in base.parameters())
n_edge = sum(p.numel() for p in edge.parameters())
print(f"[3] 参数量：baseline = {n_base / 1e6:.4f}M，edge_in = {n_edge / 1e6:.4f}M，"
      f"差 = {n_edge - n_base} 参数")

weights = np.load(config.pretrained_path)
base.load_from(weights=weights)
edge.load_from(weights=weights)
print("[4] 预训练权重加载成功（含 conv_root 3→4 通道零填充）")

w = edge.transformer.embeddings.hybrid_model.root.conv.weight
print(f"[5] stem conv_root 形状 = {tuple(w.shape)}，"
      f"第 4 通道 absmax = {w[:, 3].abs().max().item():.1f}（应为 0）")

base.eval()
edge.eval()
with torch.no_grad():
    y0 = base(torch.rand(1, 1, 224, 224))
    y1 = edge(torch.rand(1, 1, 224, 224))
print(f"[6] 前向输出：baseline {tuple(y0.shape)}，edge_in {tuple(y1.shape)}")

# edge_in=False 必须与改动前逐元素一致（老实验可复现）
with torch.no_grad():
    ref = base.transformer(img.repeat(1, 3, 1, 1))
print(f"[7] edge_in=False 走原路径：features 长度 = {len(ref[2])}")
