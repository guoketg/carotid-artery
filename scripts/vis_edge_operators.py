"""边缘算子对比可视化：把四个候选边缘图放在同一张图里看效果。

动机（status.md §3.3 / 对话澄清）：边缘图用的是「小波高频 IDWT 重建（LL 置零）」，
被排除的能量图、未跑的 Sobel / Laplacian 只有 probe_edge_signal.py 里的定量分数，
缺一张肉眼可比的图。本脚本从 SPARC test 按斑块**回声类型**选例（低回声软斑块 /
混合斑块 / 实性斑块，各 2 例、不同病例），每个病例并排画：原图+真值轮廓、
小波低频 LL 重建、小波高频 IDWT 重建、细节能量图、Sobel、Laplacian，
全部叠同一条真值斑块轮廓，直接看「谁把边界抓得准」——低回声斑块灰度不明显，
是最考验算子的一档。

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

from scipy import ndimage

from datasets.dataset_carotid import CarotidDataset, LIST_DIR, DATA_ROOT, SUBSETS, PROJECT_ROOT
from networks.wavelet import build_hf_image, haar_dwt, haar_idwt
from scripts.probe_edge_signal import sobel_mag, laplacian_mag, hf_energy_map

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "assets", "edge-fusion")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def build_lf_image(x):
    """与 build_hf_image 对偶：只保留 LL 子带的 IDWT 重建（低频/大轮廓图）。

    同样走对数域，输出前 expm1 拉回原动态范围，不做标准化（低频图近似原图的
    平滑版，量级与原图一致，用 gray 显示即可）。
    """
    x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
    xl = torch.log1p(x_pos)
    ll, details = haar_dwt(xl)
    lf = haar_idwt(ll, [torch.zeros_like(d) for d in details])
    return torch.expm1(lf)


def echo_contrast(img, lab):
    """斑块回声对比度：(斑块内均值 - 周围环带均值) / 全图 std。

    img: (H,W) 归一化灰度；lab: (H,W) 0/1。值越负 → 斑块内越暗（低回声/软斑块），
    越接近 0 → 与周围等回声、灰度上越难分辨，两者都是「灰度不明显」的情形。
    """
    inside = lab > 0
    if inside.sum() == 0:
        return float("nan")
    ring = ndimage.binary_dilation(inside, iterations=6) & \
        ~ndimage.binary_dilation(inside, iterations=2)
    denom = max(float(img.std()), 1e-6)
    return float((img[inside].mean() - img[ring].mean()) / denom)


def pick_type_cases(ds, n_per_kind=2):
    """按回声类型选例：低回声（最暗）/ 混合（内部最参差）/ 实性（最亮），各 n_per_kind 例。

    数据集没有斑块回声类型的标注，用两个可计算的代理量近似临床分型：
    - 回声对比度 con（见 echo_contrast）：越负 → 斑块内越暗 → 低回声/软斑块；
    - 斑块内 std（异质性）：内部灰度越参差 → 强弱混杂 → 混合回声斑块；
      既不暗也不参差、对比度最高的一端 → 均匀强回声（实性/纤维/钙化）。
    全部保证来自不同病例。
    """
    stats = []
    for i in range(len(ds)):
        img = ds[i]["image"][0].numpy()
        lab = ds[i]["label"].numpy()
        area = int((lab > 0).sum())
        if area < 30:                    # 斑块太小的切片不参与，环带统计不稳
            continue
        stats.append({
            "i": i,
            "pid": ds[i]["case"].split("/")[1].split("_")[0],
            "area": area,
            "con": echo_contrast(img, lab),
            "het": float(img[lab > 0].std()),
        })

    by_con = sorted(stats, key=lambda s: s["con"])
    m = 4 * n_per_kind                    # 两端各取一小批候选，中间批里挑混合
    low_pool = by_con[:m]
    high_pool = by_con[-m:]
    mixed_pool = sorted(by_con[m:-m], key=lambda s: -s["het"])

    def take(pool, kind):
        out, used = [], set()
        for s in pool:
            if s["pid"] in used:
                continue
            out.append((s["i"], kind, s["con"]))
            used.add(s["pid"])
            if len(out) == n_per_kind:
                break
        return out

    return take(low_pool, "低回声/软斑块") + \
        take(mixed_pool, "混合斑块") + \
        take(high_pool, "实性(强回声)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="SPARC")
    args = ap.parse_args()

    ds = CarotidDataset(root=DATA_ROOT,
                        list_file=os.path.join(LIST_DIR, args.subset, "test.txt"),
                        split="test", img_size=224, augment=False)
    picked = pick_type_cases(ds)
    idx = [i for i, _, _ in picked]
    kinds = [k for _, k, _ in picked]
    cons = [c for _, _, c in picked]
    print("[cases] " + ", ".join(
        f"{k} {ds[i]['case']} con={c:+.2f}" for i, k, c in zip(idx, kinds, cons)))

    batch = torch.stack([ds[i]["image"] for i in idx])          # (N,1,224,224)
    labels = torch.stack([ds[i]["label"] for i in idx]).float()  # (N,224,224)

    with torch.no_grad():
        lf = build_lf_image(batch)      # 小波低频：LL 单独 IDWT 重建
        hf = build_hf_image(batch)      # 小波高频：LL 置零 IDWT 重建（本项目采用）
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
        ("小波低频 LL 重建（大轮廓/回声水平）", lambda j: lf[j, 0].numpy(), "gray", None),
        ("小波高频 IDWT 重建（本项目采用）", lambda j: hf[j, 0].numpy(), "RdBu_r", (-3, 3)),
        ("细节能量图 √(LH²+HL²+HH²)（旧方案）", lambda j: norm_mag(energy[:, 0].numpy())[j], "inferno", None),
        ("Sobel 梯度幅值（对照，未跑训练）", lambda j: norm_mag(sobel[:, 0].numpy())[j], "inferno", None),
        ("Laplacian 幅值（对照，未跑训练）", lambda j: norm_mag(lap[:, 0].numpy())[j], "inferno", None),
    ]

    fig, axes = plt.subplots(len(idx), len(cols), figsize=(21, 3.4 * len(idx)))
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
                ax.set_ylabel(f"{tag}\n{kinds[r]} 斑块{area}px 回声对比{cons[r]:+.2f}", fontsize=8)
    fig.suptitle("边缘算子对比（红线 = 真值斑块轮廓；行 = 低回声/混合/实性各 2 例；量级图按各自 99.5 分位截断）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "edge-operators-compare.png")
    fig.savefig(out, dpi=180)
    print(f"[saved] {out}")


def vis_lf_hf(ds):
    """第二张图：低频/高频分解对照，重点看软斑块（低回声、灰度不明显）。"""
    picked = pick_echo_cases(ds)
    n_soft = 5
    idx = [i for i, _ in picked]
    cons = [c for _, c in picked]
    print("[echo-cases] " + ", ".join(
        f"{ds[i]['case']} con={c:+.2f}" for i, c in zip(idx, cons)))

    batch = torch.stack([ds[i]["image"] for i in idx])
    labels = torch.stack([ds[i]["label"] for i in idx]).float()

    with torch.no_grad():
        lf = build_lf_image(batch)                      # 低频：LL 单独重建
        hf = build_hf_image(batch)                      # 高频：LL 置零重建（本项目采用）
        hf_abs = hf.abs()                               # 高频幅值：只看边缘强度

    # HF 已标准化，用发散色标 ±3；|HF| 按各自 99.5 分位截断
    def norm_mag(t):
        out = np.empty_like(t)
        for j in range(t.shape[0]):
            hi = np.percentile(t[j], 99.5)
            out[j] = np.clip(t[j], 0, max(hi, 1e-8))
        return out

    n = len(idx)
    cols = [
        ("原图 + 真值轮廓", lambda j: batch[j, 0].numpy(), "gray", None, None),
        ("低频 LL 重建（大轮廓/回声水平）", lambda j: lf[j, 0].numpy(), "gray", None, None),
        ("高频 HF 重建（本项目边缘图）", lambda j: hf[j, 0].numpy(), "RdBu_r", (-3, 3), None),
        ("|HF| 幅值（边缘强度）", lambda j: norm_mag(hf_abs[:, 0].numpy())[j], "inferno", None, None),
    ]

    fig, axes = plt.subplots(n, len(cols), figsize=(14, 3.4 * n))
    for r in range(n):
        case = ds[idx[r]]["case"]
        tag = "/".join(case.split("/")[-2:])
        area = int((labels[r] > 0).sum())
        kind = "软斑块(低回声)" if r < n_soft else "强回声(对照)"
        for c, (title, get, cmap, lim, _) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(get(r), cmap=cmap,
                      vmin=None if lim is None else lim[0],
                      vmax=None if lim is None else lim[1])
            ax.contour(labels[r], levels=[0.5], colors="#ff2d2d", linewidths=0.9)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{tag}\n{kind} 斑块{area}px 回声对比{cons[r]:+.2f}", fontsize=8)
    fig.suptitle("低频/高频分解 × 斑块回声对比（前 5 行=低回声软斑块，后 3 行=强回声对照；红线=真值轮廓）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = os.path.join(OUT_DIR, "lf-hf-plaques.png")
    fig.savefig(out, dpi=180)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
