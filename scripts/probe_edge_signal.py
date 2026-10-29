"""边缘特征融合的**先行诊断**（不等 200 epoch 队列）。

两个问题、两个实验：

A. 边缘算子选得对不对？
   把 teacher 定的「小波高频 IDWT 重建（LL 置零）」与 technical-design.md 里点名的
   后续对照（Sobel / Laplacian）以及旧的高频能量图放在同一把尺子上量：
   算出来的边缘图与真值斑块边界的吻合度。边界吻合 = 算子抓对了位置。

B. 第 4 个通道到底有没有**信息增量**？
   门槛实验（项目既有套路，见 experiments.md 的 E1）：一个 ~0.1M 的小 CNN，
   分别吃 (1) 原图 / (2) 原图+HF / (3) 只吃 HF，比 test Dice。
   小网络跑得快，结论只认「相对差值」，不认绝对值。

用法：python scripts/probe_edge_signal.py [--subset SPARC] [--epochs 20]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset, LIST_DIR, DATA_ROOT, SUBSETS
from networks.wavelet import build_hf_image, haar_dwt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- A. 算子质量
def sobel_mag(x):
    """x: (B,1,H,W) -> Sobel 梯度幅值"""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device)
    ky = kx.t().contiguous()
    kx = kx.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def laplacian_mag(x):
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device)
    k = k.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, k, padding=1).abs()


def hf_energy_map(x):
    """旧方案：细节能量图 sqrt(LH²+HL²+HH²)，只留能量、丢方向"""
    x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
    ll, details = haar_dwt(torch.log1p(x_pos))
    e = torch.sqrt(sum(d * d for d in details) + 1e-8)
    # DWT 降了半分辨率，插值回原尺寸再参与比较
    return F.interpolate(e, size=x.shape[-2:], mode="bilinear", align_corners=False)


def _auroc(score, pos):
    """用秩和算 AUROC：score 越大越像正样本(pos=1)"""
    order = score.argsort()
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, score.numel() + 1, device=score.device, dtype=score.dtype)
    n_pos = pos.sum()
    n_neg = pos.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return ((ranks[pos > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def edge_localization_score(edge, label, topk=0.05):
    """边缘图与真值边界带的吻合度。

    label: (B,1,H,W) 0/1；边界带 = 形态学梯度（外扩一圈的差集）。
    返回 (边界命中率, 分离比)：命中率 = 边缘图 top-k 像素落在边界带里的比例；
    分离比 = 边界带内平均响应 / 带外平均响应（>1 说明边缘图在边界处更亮）。
    """
    dil = F.max_pool2d(label, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-label, kernel_size=3, stride=1, padding=1)
    band = (dil - ero)                       # 2px 宽的边界带
    e = edge.abs()

    b, _, h, w = e.shape
    k = max(1, int(h * w * topk))
    flat = e.reshape(b, -1)
    thr = flat.topk(k, dim=1).values[:, -1:].reshape(b, 1, 1, 1)
    pred = (e >= thr).float()
    hit = (pred * band).sum(dim=(1, 2, 3)) / pred.sum(dim=(1, 2, 3)).clamp_min(1)
    rand_h = k / (h * w)
    precision = hit / rand_h                    # 相对随机猜测的提升倍数

    inside = (e * band).sum(dim=(1, 2, 3)) / band.sum(dim=(1, 2, 3)).clamp_min(1)
    outside = (e * (1 - band)).sum(dim=(1, 2, 3)) / (1 - band).sum(dim=(1, 2, 3)).clamp_min(1)
    sep = (inside / outside.clamp_min(1e-8))
    auroc = float(np.mean([_auroc(e[i].flatten(), band[i].flatten()) for i in range(b)]))
    return precision.mean().item(), sep.mean().item(), auroc


def part_a(images, labels, n=60):
    print("\n=== A. 边缘算子与真值边界的吻合度（真值来自斑块标注）===")
    x = images[:n].to(DEVICE)
    y = labels[:n].to(DEVICE).float().unsqueeze(1)   # (B,1,H,W)

    ops = {
        "小波高频 IDWT（本次采用）": lambda t: build_hf_image(t),
        "Sobel 梯度幅值（对照）": sobel_mag,
        "Laplacian 幅值（对照）": laplacian_mag,
        "小波细节能量图（旧方案）": hf_energy_map,
    }
    print(f"{'算子':<26} {'边界命中率(×随机)':>16} {'分离比':>10} {'AUROC':>8}")
    for name, fn in ops.items():
        e = fn(x)
        prec, sep, auc = edge_localization_score(e, y)
        print(f"{name:<26} {prec:>15.2f}x {sep:>10.2f} {auc:>8.3f}")


# ---------------------------------------------------------------- B. 门槛实验
class TinyCNN(nn.Module):
    """~0.1M 参数的小 CNN，只用来测「多给一个通道有没有用」"""

    def __init__(self, in_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        f = self.body(x)
        f = F.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False)
        return self.head(f)


def make_inputs(batch, variant, hf_cache):
    """variant: 'raw' | 'raw_hf' | 'hf'"""
    img = batch["image"]
    if variant == "raw":
        return img
    hf = hf_cache
    if variant == "hf":
        return hf
    return torch.cat([img, hf], dim=1)


@torch.no_grad()
def evaluate(model, images, labels, variant, hf_cache, bs=16):
    model.eval()
    dices = []
    for i in range(0, images.shape[0], bs):
        x = images[i:i + bs].to(DEVICE)
        y = labels[i:i + bs].to(DEVICE)
        hf = hf_cache[i:i + bs].to(DEVICE)
        inp = make_inputs({"image": x}, variant, hf)
        p = torch.softmax(model(inp), dim=1)[:, 1] > 0.5
        g = y > 0.5
        inter = (p & g).sum(dim=(1, 2)).float()
        den = p.sum(dim=(1, 2)).float() + g.sum(dim=(1, 2)).float()
        dices.extend((2 * inter / den.clamp_min(1)).cpu().numpy())
    return float(np.mean(dices))


def part_b(train_img, train_lab, test_img, test_lab, epochs, seed=1234):
    print("\n=== B. 门槛实验：多一个 HF 通道有没有信息增量（0.1M 小 CNN）===")
    results = {}
    for variant in ["raw", "raw_hf", "hf"]:
        torch.manual_seed(seed)
        model = TinyCNN(1 if variant in ("raw", "hf") else 2).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = train_img.shape[0]
        g = torch.Generator().manual_seed(seed)
        t0 = time.time()

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 16):
                idx = perm[i:i + 16]
                x = train_img[idx].to(DEVICE)
                y = train_lab[idx].to(DEVICE)
                with torch.no_grad():
                    hf = build_hf_image(x)
                inp = make_inputs({"image": x}, variant, hf)
                loss = F.cross_entropy(model(inp), y.long())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()

        # 测试集的 HF 预先算好，避免每轮重复算
        with torch.no_grad():
            hf_test = torch.cat([build_hf_image(test_img[i:i + 16].to(DEVICE))
                                 for i in range(0, test_img.shape[0], 16)])
        d = evaluate(model, test_img, test_lab, variant, hf_test)
        nparam = sum(p.numel() for p in model.parameters())
        results[variant] = d
        print(f"  变体 {variant:<8} in_ch={1 if variant in ('raw','hf') else 2}  "
              f"参数 {nparam/1e6:.3f}M  test Dice = {d * 100:.2f}%   ({time.time()-t0:.0f}s)")

    print(f"\n  差值：raw+HF − raw = {(results['raw_hf'] - results['raw']) * 100:+.2f}%")
    print(f"        HF 单独       = {results['hf'] * 100:.2f}%（边缘图自己能做到多少）")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="SPARC")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    folder = SUBSETS[args.subset]
    tr = CarotidDataset(root=DATA_ROOT, list_file=os.path.join(LIST_DIR, args.subset, "train.txt"),
                        split="train", img_size=224, augment=False)
    te = CarotidDataset(root=DATA_ROOT, list_file=os.path.join(LIST_DIR, args.subset, "test.txt"),
                        split="test", img_size=224, augment=False)
    print(f"[data] {args.subset}: train {len(tr)} / test {len(te)}")

    t0 = time.time()
    tr_i = torch.stack([tr[i]["image"] for i in range(len(tr))])
    tr_l = torch.stack([tr[i]["label"] for i in range(len(tr))])
    te_i = torch.stack([te[i]["image"] for i in range(len(te))])
    te_l = torch.stack([te[i]["label"] for i in range(len(te))])
    print(f"[data] 全部读进内存 {time.time() - t0:.0f}s，"
          f"train {tuple(tr_i.shape)} / test {tuple(te_i.shape)}")

    part_a(te_i, te_l)
    part_b(tr_i, tr_l, te_i, te_l, args.epochs)


if __name__ == "__main__":
    main()
