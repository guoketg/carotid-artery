"""边缘算子对比可视化：把四个候选边缘图放在同一张图里看效果。

动机（status.md §3.3 / 对话澄清）：边缘图用的是「小波高频 IDWT 重建（LL 置零）」，
被排除的能量图、未跑的 Sobel / Laplacian 只有 probe_edge_signal.py 里的定量分数，
缺一张肉眼可比的图。本脚本从 SPARC test 取 3 个不同病例（斑块大/中/小），
每个病例并排画：原图+真值轮廓、小波高频重建、细节能量图、Sobel、Laplacian，
全部叠同一条真值斑块轮廓，直接看「谁把边界抓得准」。

用法：python scripts/vis_edge_operators.py [--subset SPARC]
输出：docs/assets/edge-fusion/edge-operators-compare.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset, LIST_DIR, DATA_ROOT, SUBSETS, PROJECT_ROOT
from networks.wavelet import build_hf_image
from scripts.probe_edge_signal import sobel_mag, laplacian_mag, hf_energy_map

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "assets", "edge-fusion")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def pick_cases(ds, n=3):
    """按斑块面积取大/中/小三档，且保证来自不同病例（同一病例切片几乎一样）。"""
    areas, pids = [], []
    for i in range(len(ds)):
        lab = ds[i]["label"]
        areas.append(int((lab > 0).sum()))
        pids.append(ds[i]["case"].split("/")[1].split("_")[0])
    areas = np.array(areas)
    order = np.argsort(areas)
    chosen, used_pid = [], set()
    for rank in [order[-1], order[len(order) // 2], order[len(order) // 10]]:
        if pids[rank] not in used_pid:
            chosen.append(int(rank))
            used_pid.add(pids[rank])
        if len(chosen) == n:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="SPARC")
    args = ap.parse_args()

    ds = CarotidDataset(root=DATA_ROOT,
                        list_file=os.path.join(LIST_DIR, args.subset, "test.txt"),
                        split="test", img_size=224, augment=False)
    idx = pick_cases(ds)
    print(f"[cases] {[(i, ds[i]['case'], int((ds[i]['label'] > 0).sum())) for i in idx]}")

    batch = torch.stack([ds[i]["image"] for i in idx])          # (3,1,224,224)
    labels = torch.stack([ds[i]["label"] for i in idx]).float()  # (3,224,224)

    with torch.no_grad():
        hf = build_hf_image(batch)
        energy = hf_energy_map(batch)
        sobel = sobel_mag(batch)
        lap = laplacian_mag(batch)

    # 每个量级图按自身 99.5 分位截断（四者的动态范围差几个数量级，不能共用色标）
    def norm_mag(t):
        out = np.empty_like(t)
        for j in range(t.shape[0]):
            hi = np.percentile(t[j], 99.5)
            out[j] = np.clip(t[j], 0, max(hi, 1e-8))
        return out

    cols = [
        ("原图 + 真值轮廓", lambda j: batch[j, 0].numpy(), "gray", None),
        ("小波高频 IDWT 重建（本项目采用）", lambda j: hf[j, 0].numpy(), "RdBu_r", (-3, 3)),
        ("细节能量图 √(LH²+HL²+HH²)（旧方案）", lambda j: norm_mag(energy[:, 0].numpy())[j], "inferno", None),
        ("Sobel 梯度幅值（对照，未跑训练）", lambda j: norm_mag(sobel[:, 0].numpy())[j], "inferno", None),
        ("Laplacian 幅值（对照，未跑训练）", lambda j: norm_mag(lap[:, 0].numpy())[j], "inferno", None),
    ]

    fig, axes = plt.subplots(len(idx), len(cols), figsize=(17, 3.6 * len(idx)))
    for r in range(len(idx)):
        case = ds[idx[r]]["case"]
        tag = "/".join(case.split("/")[-2:])
        area = int((labels[r] > 0).sum())
        for c, (title, get, cmap, lim) in enumerate(cols):
            ax = axes[r, c]
            img = get(r)
            ax.imshow(img, cmap=cmap,
                      vmin=None if lim is None else lim[0],
                      vmax=None if lim is None else lim[1])
            ax.contour(labels[r], levels=[0.5], colors="#ff2d2d", linewidths=0.9)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{tag}\n斑块 {area}px", fontsize=9)
    fig.suptitle("边缘算子对比（红线 = 真值斑块轮廓，各行大/中/小斑块各一例；量级图按各自 99.5 分位截断）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "edge-operators-compare.png")
    fig.savefig(out, dpi=180)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
