"""融合「轴」实验：频率信息该走通道，还是该走 QKV 注意力？

背景（docs/technical-design.md 频率注入位置 一节）：
  - 历史双骨干方案 = 主路原图 / 侧路 LH+HL+HH，4 个接点用**跨支交叉注意力（QKV）**融合；
    实测四项指标全差（status.md 4.5.6）。
  - 当前输入级方案 = stem 直接吃 4 通道（原图 3 + 小波 HF 1），**没有任何注意力**。

用户的判断：低频/高频本质是「通道」的事，QKV 那种跨空间位置的注意力是错的轴。

本脚本在**同一个 TinyCNN 尺度**上把两条路摆在一起比：
  raw          单分支，只吃原图（基线）
  raw_hf       输入级通道拼接（原图 1ch + HF 1ch → 同一个卷积）  ← 当前已实现的边缘融合
  feat_concat  双分支 + **通道拼接**融合（同一位置的低频特征与高频特征逐像素合并）
  feat_attn    双分支 + **QKV 交叉注意力**融合（模拟双骨干的融合轴）

  feat_concat vs feat_attn 是本脚本的核心对比：分支、参数量级、融合位置全同，
  只换融合算子（通道轴 vs 空间注意力轴）。

评估：Dice + ASSD(mm) + HD95(mm)。ASSD/HD95 在 224 网格上算，
间距按各样本 NIfTI 头**逐轴换算**（zoom * 原图边长/224），保证单位是 mm。

用法：python scripts/probe_fusion_axis.py [--subset SPARC] [--epochs 20] [--seeds 1234,2024]
"""
import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset, LIST_DIR, DATA_ROOT, SUBSETS
from networks.wavelet import build_hf_image
from metrics_carotid import compute_case_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANTS = ["raw", "raw_hf", "feat_concat", "feat_attn",
            "feat_concat_raw", "feat_attn_raw", "feat_attn_zero",
            "raw_bias"]


# ---------------------------------------------------------------- 网络
def encoder(in_ch=1, width=16):
    """与历史 TinyCNN 完全同构的 4 级 stride-2 编码器 → 14×14×64"""
    return nn.Sequential(
        nn.Conv2d(in_ch, width, 3, stride=2, padding=1), nn.GroupNorm(4, width), nn.ReLU(),
        nn.Conv2d(width, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
    )


class TinyAttnFusion(nn.Module):
    """跨支交叉注意力（模拟双骨干的 QKV 融合轴）。
    γ 零初始化 → 初始等价于「只用主路」，与项目里 ReZero 风格一致，避免死锁。"""

    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, a, b):
        B, C, H, W = a.shape
        dh = C // self.heads
        q = self.q(a).reshape(B, H * W, self.heads, dh).transpose(1, 2)
        k = self.k(b).reshape(B, H * W, self.heads, dh).transpose(1, 2)
        v = self.v(b).reshape(B, H * W, self.heads, dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)          # (B,heads,HW,dh)
        o = o.transpose(1, 2).reshape(B, C, H, W)
        return a + self.gamma * self.proj(o)


class TinyConcatFusion(nn.Module):
    """通道拼接融合：同位置的低频特征与高频特征逐像素合并，再 1×1 回投影。
    γ 零初始化 → 初始等价于「只用主路」。"""

    def __init__(self, dim=64):
        super().__init__()
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, a, b):
        return a + self.gamma * self.fuse(torch.cat([a, b], dim=1))


class TinySeg(nn.Module):
    """四种变体共用同一个头，保证对比公平"""

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        # *_raw  后缀 = 对照组：侧路喂「第二份原图」而不是 HF 图。
        # *_zero 后缀 = 对照组：侧路喂全零。用来分离
        #   「融合架构本身的容量/正则化效应」与「HF 内容本身的贡献」：
        #   若 side=零也能涨到同样水平，则涨幅来自架构而非频率信息。
        self.side_is_hf = not variant.endswith("_raw") and not variant.endswith("_zero")
        base = variant
        for suf in ("_raw", "_zero"):
            if base.endswith(suf):
                base = base[:-len(suf)]
        if base == "raw":
            self.enc = encoder(1)
            if variant == "raw_bias":
                # 对照：单分支 + 一个可学习的逐通道常数偏置。
                # feat_attn_zero（侧路喂全零）在数学上退化成「主路 + 常数偏置」，
                # 用这个变体验证那 +0.27 是不是纯粹来自常数偏置（标定/阈值漂移），
                # 而不是来自频率信息或注意力的建模能力。
                self.bias = nn.Parameter(torch.zeros(1, 64, 1, 1))
        elif base == "raw_hf":
            self.enc = encoder(2)
        else:
            self.enc_a = encoder(1)      # 主路：原图
            self.enc_b = encoder(1)      # 侧路：HF 图 / 第二份原图 / 全零
            self.fuse = TinyConcatFusion(64) if base == "feat_concat" else TinyAttnFusion(64)
        self.head = nn.Conv2d(64, 2, 1)

    def forward(self, x, hf=None):
        if self.variant == "raw":
            f = self.enc(x)
        elif self.variant == "raw_bias":
            f = self.enc(x) + self.bias
        elif self.variant == "raw_hf":
            f = self.enc(torch.cat([x, hf], dim=1))
        else:
            if self.side_is_hf:
                side = hf
            elif self.variant.endswith("_raw"):
                side = x
            else:
                side = torch.zeros_like(x)
            f = self.fuse(self.enc_a(x), self.enc_b(side))
        f = F.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False)
        return self.head(f)


# ---------------------------------------------------------------- 评估
def sample_spacing(root, rel):
    """224 网格上每像素的物理间距(mm)，逐轴换算：zoom * 原图边长 / 224"""
    hdr = nib.load(os.path.join(root, rel)).header
    oh, ow = hdr.get_data_shape()[:2]
    zy, zx = hdr.get_zooms()[:2]
    return (float(zy) * oh / 224.0, float(zx) * ow / 224.0)


@torch.no_grad()
def evaluate(model, images, labels, variant, hf_cache, spacings, bs=16):
    model.eval()
    rows = []
    for i in range(0, images.shape[0], bs):
        x = images[i:i + bs].to(DEVICE)
        hf = hf_cache[i:i + bs].to(DEVICE)
        pred = torch.softmax(model(x, hf), dim=1)[:, 1].cpu().numpy() > 0.5
        gt = labels[i:i + bs].numpy() > 0.5
        for j in range(pred.shape[0]):
            rows.append(compute_case_metrics(pred[j], gt[j], spacing=spacings[i + j]))
    out = {}
    for k in ["dice", "assd", "hd95"]:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["abs_delta_area"] = float(np.mean([abs(r["delta_area"]) for r in rows]))
    return out


def train_one(variant, train_img, train_lab, test_img, test_lab, spacings, epochs, seed):
    torch.manual_seed(seed)
    model = TinySeg(variant).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_img.shape[0]
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 16):
            idx = perm[i:i + 16]
            x = train_img[idx].to(DEVICE)
            y = train_lab[idx].to(DEVICE)
            with torch.no_grad():
                hf = build_hf_image(x)
            pred = model(x, hf)
            loss = F.cross_entropy(pred, y.long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        hf_test = torch.cat([build_hf_image(test_img[i:i + 16].to(DEVICE))
                             for i in range(0, test_img.shape[0], 16)])
    m = evaluate(model, test_img, test_lab, variant, hf_test, spacings)
    m["nparam"] = sum(p.numel() for p in model.parameters())
    m["sec"] = time.time() - t0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="SPARC")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="1234,2024")
    args = ap.parse_args()

    folder = SUBSETS[args.subset]
    tr = CarotidDataset(root=DATA_ROOT, list_file=os.path.join(LIST_DIR, args.subset, "train.txt"),
                        split="train", img_size=224, augment=False)
    te = CarotidDataset(root=DATA_ROOT, list_file=os.path.join(LIST_DIR, args.subset, "test.txt"),
                        split="test", img_size=224, augment=False)
    print(f"[data] {args.subset}: train {len(tr)} / test {len(te)}")

    tr_i = torch.stack([tr[i]["image"] for i in range(len(tr))])
    tr_l = torch.stack([tr[i]["label"] for i in range(len(tr))])
    te_i = torch.stack([te[i]["image"] for i in range(len(te))])
    te_l = torch.stack([te[i]["label"] for i in range(len(te))])
    spacings = [sample_spacing(DATA_ROOT, te[i]["case"]) for i in range(len(te))]
    print(f"[data] 224 网格间距(mm) 例：{spacings[0][0]:.4f} × {spacings[0][1]:.4f}")

    # 输出网格天花板：真值自己压到 14×14 再放大
    small = F.adaptive_avg_pool2d(te_l.float().unsqueeze(1), (14, 14))
    up = F.interpolate(small, size=(224, 224), mode="nearest")
    p, g = (up > 0.25).float(), te_l.float().unsqueeze(1)
    inter = (p * g).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))
    print(f"[ceiling] 真值压到 14×14 再放大 → Dice 上限 = {(2*inter/den.clamp_min(1)).mean() * 100:.2f}%"
          f"（小网络的输出网格就这么粗）\n")

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"{'变体':<14} {'参数':>8} {'Dice':>8} {'ASSD(mm)':>9} {'HD95(mm)':>9} {'|ΔArea|':>9}")
    allm = {}
    for variant in VARIANTS:
        runs = []
        for sd in seeds:
            m = train_one(variant, tr_i, tr_l, te_i, te_l, spacings, args.epochs, sd)
            runs.append(m)
            print(f"  {variant:<14} seed {sd}: dice {m['dice'] * 100:.2f}%  assd {m['assd']:.4f}  "
                  f"hd95 {m['hd95']:.4f}  |dA| {m['abs_delta_area']:.3f}")
        agg = {k: float(np.mean([r[k] for r in runs])) for k in ["dice", "assd", "hd95", "abs_delta_area"]}
        agg["nparam"] = runs[0]["nparam"]
        allm[variant] = agg
        print(f"{variant:<14} {agg['nparam']/1e6:>7.3f}M {agg['dice'] * 100:>7.2f}% {agg['assd']:>9.4f} "
              f"{agg['hd95']:>9.4f} {agg['abs_delta_area']:>9.3f}\n")

    b = allm["raw"]
    print("=== 相对 raw 的差值 ===")
    for v in VARIANTS[1:]:
        m = allm[v]
        print(f"  {v:<14} Dice {(m['dice']-b['dice']) * 100:+.2f}%   ASSD {m['assd']-b['assd']:+.4f} mm   "
              f"HD95 {m['hd95']-b['hd95']:+.4f} mm   |ΔArea| {m['abs_delta_area']-b['abs_delta_area']:+.3f} mm²")


if __name__ == "__main__":
    main()
