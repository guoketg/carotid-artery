# -*- coding: utf-8 -*-
"""斑块标签形态学可视化：膨胀 / 腐蚀 / 两种相减（外边界、内边界）/ 形态学梯度。

口径说明：运算放在 224×224 网格上，与训练、边缘监督分支共用同一分辨率（label 由
原图最近邻放大到 224 后再算），结构元取 3×3 全连通（8 邻域，等价 cv2 的
np.ones((3,3))），迭代 1 次。腐蚀按 0 边界处理，避免画布外侧被当成前景。

三种相减都在同一张掩膜上算：
  外边界 = 膨胀 − 原标签        （多出来的那一圈，向外扩 1 px）
  内边界 = 原标签 − 腐蚀        （被削掉的那一圈，向内收 1 px）
  形态学梯度 = 膨胀 − 腐蚀      （上面两圈的并集，宽 2 px）

输出三张图：
  label-morph-overview.png  整体效果（每例：原标签 / 膨胀 / 腐蚀 / 轮廓对比）
  label-morph-pixels.png    像素级相减（每例：外边界 / 内边界 / 梯度 / 三色叠加，带像素网格）
  label-morph-stats.png     全测试集的面积变化统计

用法：
  python scripts/vis_label_morph.py                    # 默认 SPARC test
  python scripts/vis_label_morph.py --subset Zhongnan
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset, DATA_ROOT, LIST_DIR, PROJECT_ROOT

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "assets", "label-morph")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SE = np.ones((3, 3), dtype=bool)   # 3×3 全连通结构元（8 邻域）

C_ORIG = "#ff3b30"   # 原标签 红
C_OUT = "#ffcc00"    # 外边界（膨胀新增）黄
C_IN = "#00d5ff"     # 内边界（腐蚀削掉）青
C_LABEL_GRAY = 0.62  # 像素放大图里原标签的灰


def morph_ops(label, iterations=1, se=SE):
    """二值掩膜的一次完整形态学分解，返回 6 个 0/1 uint8 掩膜。"""
    B = (label > 0.5)
    D = ndimage.binary_dilation(B, structure=se, iterations=iterations, border_value=0)
    E = ndimage.binary_erosion(B, structure=se, iterations=iterations, border_value=0)
    return {
        "orig": B.astype(np.uint8),
        "dilate": D.astype(np.uint8),
        "erode": E.astype(np.uint8),
        "outer": (D & ~B).astype(np.uint8),      # 膨胀 − 原标签
        "inner": (B & ~E).astype(np.uint8),      # 原标签 − 腐蚀
        "grad": (D & ~E).astype(np.uint8),       # 膨胀 − 腐蚀
    }


def overlay(ax, gray, mask, color, alpha=0.6):
    """灰度底图 + 单色半透明掩膜覆盖。"""
    ax.imshow(gray, cmap="gray")
    rgba = np.zeros(gray.shape + (4,), dtype=np.float32)
    rgba[mask > 0, :3] = to_rgb(color)
    rgba[mask > 0, 3] = alpha
    ax.imshow(rgba, interpolation="nearest")


def square_crop(mask, pad=10):
    """取掩膜外接框并向两侧补成正方形，避免拉伸变形。返回 (y0,y1,x0,x1)。"""
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + pad + 1
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + pad + 1
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    half = max(y1 - y0, x1 - x0) / 2.0
    H, W = mask.shape
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    # 越界时整体平移回画布内，尺寸保持不变
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > H:
        y0 -= (y1 - H)
        y1 = H
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if x1 > W:
        x0 -= (x1 - W)
        x1 = W
    return max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)


def tip_point(mask):
    """轮廓上离质心最远的点，用作局部放大窗口中心（尖端处形态学改动最明显）。"""
    cy, cx = ndimage.center_of_mass(mask)
    edge = mask.astype(bool) & ~ndimage.binary_erosion(mask.astype(bool), SE)
    ys, xs = np.where(edge)
    d = (ys - cy) ** 2 + (xs - cx) ** 2
    k = int(np.argmax(d))
    return ys[k], xs[k]


def collect_stats(ds):
    """逐切片统计面积变化，用于挑代表病例和画统计图。"""
    rows = []
    for i in range(len(ds)):
        ops = morph_ops(ds[i]["label"].numpy().astype(np.float32))
        area = int(ops["orig"].sum())
        if area == 0:                        # 无斑块切片不参与统计
            continue
        rows.append({
            "i": i,
            "case": ds[i]["case"],
            "pid": ds[i]["case"].split("/")[1].split("_")[0],
            "area": area,
            "d_area": int(ops["dilate"].sum()),
            "e_area": int(ops["erode"].sum()),
            "outer": int(ops["outer"].sum()),
            "inner": int(ops["inner"].sum()),
            "grad": int(ops["grad"].sum()),
        })
    return rows


def pick_cases(rows, n=6):
    """按斑块面积分位挑代表样本（小→大），保证来自不同病例。"""
    targets = np.percentile(np.array([r["area"] for r in rows]), np.linspace(5, 95, n))
    picked, used = [], set()
    for t in targets:
        for r in sorted(rows, key=lambda r: abs(r["area"] - t)):
            if r["pid"] in used:
                continue
            picked.append(r)
            used.add(r["pid"])
            break
    return picked


def tag_text(ax, s, fontsize=7.5, loc="bottom"):
    """在图内左下角标注病例信息，避免 ylabel 在窄轴上被挤压旋转。"""
    ax.text(0.02, 0.02 if loc == "bottom" else 0.98, s, transform=ax.transAxes,
            fontsize=fontsize, color="white", va="bottom" if loc == "bottom" else "top",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.2))


# ---------------------------------------------------------------- 图 1：整体
def mask_rgb(ops, sl, base_key, ring=None):
    """纯掩膜图：黑底，base 掩膜画白，ring 指定时把差集环染成它的颜色。

    ring 可以是 (键, 颜色) 或 [(键, 颜色), ...]，用于把梯度拆成外环+内环。
    """
    h, w = ops["orig"][sl].shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    if base_key is not None:
        img[ops[base_key][sl] > 0] = 0.92
    if ring is not None:
        for key, color in (ring if isinstance(ring, list) else [ring]):
            img[ops[key][sl] > 0] = to_rgb(color)
    return img


def plot_overview(ds, picked, subset, split, iters):
    """图 1：只看标签本身（不带超声底图），黑底白掩膜 + 差集上色。"""
    n = len(picked)
    titles = ["原标签", f"膨胀 {iters} 次", f"腐蚀 {iters} 次",
              "膨胀 − 原标签（外环）", "原标签 − 腐蚀（内环）", "膨胀 − 腐蚀（梯度）"]
    fig, axes = plt.subplots(n, 6, figsize=(15.0, 2.7 * n))
    for r, rec in enumerate(picked):
        ops = morph_ops(ds[rec["i"]]["label"].numpy().astype(np.float32), iters)
        y0, y1, x0, x1 = square_crop(ops["dilate"], pad=12)
        sl = np.s_[y0:y1, x0:x1]

        panels = [
            (mask_rgb(ops, sl, "orig", None), f"{rec['area']} px"),
            (mask_rgb(ops, sl, "dilate", ("outer", C_OUT)), f"{rec['d_area']} px"),
            (mask_rgb(ops, sl, "erode", ("inner", C_IN)), f"{rec['e_area']} px"),
            (mask_rgb(ops, sl, None, ("outer", C_OUT)), f"{rec['outer']} px"),
            (mask_rgb(ops, sl, None, ("inner", C_IN)), f"{rec['inner']} px"),
            (mask_rgb(ops, sl, None, [("outer", C_OUT), ("inner", C_IN)]),
             f"{rec['grad']} px"),
        ]
        for c, (img, note) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal")
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
            ax.set_xlabel(note, fontsize=8.5, color="#333333")

        tag_text(axes[r, 0], f"{rec['pid']}", loc="top", fontsize=7)

    fig.suptitle(f"斑块标签形态学（只看掩膜，{subset}/{split}，224×224 网格，"
                 f"3×3 全连通结构元，迭代 {iters}）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT_DIR, "label-morph-overview.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------ 图 2：像素级
def plot_pixels(ds, picked, subset, split, iters, half=22):
    """局部放大到「一个像素 = 一个格子」，1 px 宽的环才看得见。"""
    recs = picked[::max(1, len(picked) // 3)][:3] or picked[:1]
    n = len(recs)
    fig, axes = plt.subplots(n, 4, figsize=(11.0, 3.0 * n))
    for r, rec in enumerate(recs):
        ops = morph_ops(ds[rec["i"]]["label"].numpy().astype(np.float32), iters)
        cy, cx = tip_point(ops["orig"])
        H, W = ops["orig"].shape
        y0, y1 = max(cy - half, 0), min(cy + half, H)
        x0, x1 = max(cx - half, 0), min(cx + half, W)
        sl = np.s_[y0:y1, x0:x1]

        def rgb_panel(mask_key=None, show_orig=True, show_outer=False, show_inner=False):
            """按状态组装 RGB：背景深灰、原标签浅灰、外环黄、内环青。"""
            h, w = ops["orig"][sl].shape
            img = np.zeros((h, w, 3), dtype=np.float32) + 0.10
            if show_orig:
                img[ops["orig"][sl] > 0] = C_LABEL_GRAY
            if mask_key == "outer":
                img[ops["dilate"][sl] > 0] = C_LABEL_GRAY        # 膨胀后的掩膜作底
                img[ops["outer"][sl] > 0] = to_rgb(C_OUT)
            if mask_key == "inner":
                img[ops["orig"][sl] > 0] = C_LABEL_GRAY          # 原掩膜作底
                img[ops["inner"][sl] > 0] = to_rgb(C_IN)
            if mask_key == "grad":
                img[ops["grad"][sl] > 0] = 1.0
                img[ops["outer"][sl] > 0] = to_rgb(C_OUT)
                img[ops["inner"][sl] > 0] = to_rgb(C_IN)
            if show_outer:
                img[ops["outer"][sl] > 0] = to_rgb(C_OUT)
            if show_inner:
                img[ops["inner"][sl] > 0] = to_rgb(C_IN)
            return img

        panels = [
            ("外边界 = 膨胀 − 原标签", "outer", f"{rec['outer']} px（+{rec['outer'] / rec['area'] * 100:.1f}%）"),
            ("内边界 = 原标签 − 腐蚀", "inner", f"{rec['inner']} px（−{rec['inner'] / rec['area'] * 100:.1f}%）"),
            ("形态学梯度 = 膨胀 − 腐蚀", "grad", f"{rec['grad']} px（宽 2 px）"),
            ("三色叠加：原(灰)/外环(黄)/内环(青)", None,
             f"外 {rec['outer']} · 内 {rec['inner']} px"),
        ]
        for c, (title, key, note) in enumerate(panels):
            ax = axes[r, c]
            img = rgb_panel(key, show_orig=(key is None), show_outer=(key is None),
                            show_inner=(key is None))
            h, w = img.shape[:2]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which="minor", color="w", linewidth=0.35, alpha=0.45)
            ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
            ax.set_xlim(-0.5, w - 0.5), ax.set_ylim(h - 0.5, -0.5)
            ax.set_aspect("equal")
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=9.5)
            ax.set_xlabel(note, fontsize=8.5, color="#333333")

        axes[r, 0].set_ylabel("")
        tag_text(axes[r, 0], f"{rec['pid']} · 面积 {rec['area']} px", loc="top")

    fig.suptitle(f"形态学相减的像素级细节（{subset}/{split}，3×3 结构元，迭代 {iters}；"
                 f"窗口取斑块尖端 {2 * half}×{2 * half} px）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_DIR, "label-morph-pixels.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------- 图 3：统计
def plot_stats(rows, subset, split):
    areas = np.array([r["area"] for r in rows])
    outer = np.array([r["outer"] for r in rows])
    inner = np.array([r["inner"] for r in rows])
    grad = np.array([r["grad"] for r in rows])
    collapse = np.array([r["e_area"] == 0 for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.scatter(areas, outer / areas * 100, s=14, c=C_OUT, alpha=0.8, label="外边界 / 面积")
    ax.scatter(areas, inner / areas * 100, s=14, c=C_IN, alpha=0.8, label="内边界 / 面积")
    ax.scatter(areas, grad / areas * 100, s=14, c="#666666", alpha=0.5, label="梯度 / 面积")
    ax.set_xscale("log")
    ax.set_xlabel("斑块面积（px，对数轴）")
    ax.set_ylabel("相对面积变化（%）")
    ax.set_title("一枚像素宽的环占多大比例：斑块越小，形态学改动越剧烈", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    bins = np.linspace(0, areas.max(), 30)
    ax.hist(areas[~collapse], bins=bins, color="#8a8a8a", alpha=0.8,
            label=f"腐蚀后仍存活（{int((~collapse).sum())} 张）")
    if collapse.any():
        ax.hist(areas[collapse], bins=bins, color=C_IN, alpha=0.9,
                label=f"腐蚀后消失（{int(collapse.sum())} 张）")
    ax.set_xlabel("斑块面积（px）")
    ax.set_ylabel("切片数")
    ax.set_title("腐蚀对斑块存亡的影响", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(f"形态学算子对斑块面积的影响（{subset}/{split}，有斑块切片 {len(rows)} 张）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUT_DIR, "label-morph-stats.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="SPARC")
    ap.add_argument("--split", default="test")
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()

    ds = CarotidDataset(root=DATA_ROOT,
                        list_file=os.path.join(LIST_DIR, args.subset, args.split + ".txt"),
                        split=args.split, img_size=224, augment=False)

    rows = collect_stats(ds)
    areas = np.array([r["area"] for r in rows])
    outer = np.array([r["outer"] for r in rows])
    inner = np.array([r["inner"] for r in rows])
    collapse = np.array([r["e_area"] == 0 for r in rows])

    print(f"[{args.subset}/{args.split}] 有斑块切片 {len(rows)} / {len(ds)}，"
          f"斑块面积 {areas.min()}~{areas.max()} px（中位 {np.median(areas):.0f}）")
    print(f"  膨胀：平均 +{outer.mean():.1f} px，占面积 {(outer / areas).mean() * 100:.2f}%")
    print(f"  腐蚀：平均 −{inner.mean():.1f} px，占面积 {(inner / areas).mean() * 100:.2f}%")
    print(f"  形态学梯度：平均 {np.array([r['grad'] for r in rows]).mean():.1f} px")
    print(f"  腐蚀后完全消失：{int(collapse.sum())} 张（{collapse.mean() * 100:.1f}%）")

    picked = pick_cases(rows, n=6)
    print("[cases] " + " | ".join(
        f"{r['case'].split('/')[-1]} {r['area']}px" for r in picked))

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_overview(ds, picked, args.subset, args.split, args.iters)
    plot_pixels(ds, picked, args.subset, args.split, args.iters)
    plot_stats(rows, args.subset, args.split)


if __name__ == "__main__":
    main()

