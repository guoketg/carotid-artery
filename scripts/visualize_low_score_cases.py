"""低分样本可视化 + 指标表（只读：读 error_slices_*.csv / pred_cache_*.npz / 原图）。

产出：
  docs/assets/low-score/{SUBSET}/{序号}_{case}.png   逐样本图（每张切片一个文件，3 格：
                                                     全图轮廓 / 局部放大轮廓 / 局部放大 FP-FN）
  docs/assets/low-score/shape_scatter.png            Dice 与低分特征的散点
  logs/analysis/low_score_tables.md                  文档用的 markdown 表格

口径与评测一致：224 概率 → 最近邻回原网格（order=0），GT 用原图；距离指标统一用 HD。
选样：库内 Dice ≤ p25 且 HD ≥ p75，按 Dice 升序取前 N 张。
图里只画轮廓和 FP/FN，不重算任何指标。
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

ROOT = r"E:\mkl\code\datasets\Carotid2D"
NPZ = {"SPARC": "logs/analysis/pred_cache_SPARC.npz",
       "Zhongnan": "logs/analysis/pred_cache_Zhongnan.npz"}
CSV = {"SPARC": "logs/analysis/error_slices_SPARC.csv",
       "Zhongnan": "logs/analysis/error_slices_Zhongnan.csv"}
FIG_DIR = os.path.join(BASE_DIR, "docs", "assets", "low-score")
TAB_PATH = os.path.join(BASE_DIR, "logs", "analysis", "low_score_tables.md")

NUM_COLS = ["dice", "assd", "hd", "gt_area_mm2", "elongation", "minor_mm",
            "cnr", "rel_echo", "pred_ratio"]


def load_subset(subset):
    with open(os.path.join(BASE_DIR, CSV[subset]), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUM_COLS:
            r[k] = float(r[k])
        r["low_dice"] = r["low_dice"].strip().lower() in ("1", "true")
    hd_p75 = float(np.percentile([r["hd"] for r in rows], 75))
    for r in rows:
        r["high_hd"] = r["hd"] >= hd_p75
    return rows, hd_p75


def boundary(mask):
    mask = mask.astype(bool)
    ero = ndi.binary_erosion(mask, np.ones((3, 3), bool), border_value=0)
    return mask & ~ero


def read_case(rel):
    img = nib.load(os.path.join(ROOT, rel)).get_fdata(dtype=np.float32)
    gt = nib.load(os.path.join(ROOT, rel.replace(".nii.gz", "_label.nii.gz"))).get_fdata(dtype=np.float32)
    if img.ndim == 3:
        img = img[..., img.shape[-1] // 2]
    if gt.ndim == 3:
        gt = gt[..., gt.shape[-1] // 2]
    return img, gt > 0.5


def pred_at_original(prob224, shape):
    """与 metrics_carotid.original_grid_pair 同口径：最近邻回原网格。"""
    p = ndi.zoom(prob224.astype(np.float32),
                 (shape[0] / prob224.shape[0], shape[1] / prob224.shape[1]),
                 order=0, mode="nearest")
    return p > 0.5


def crop_box(gt_b, pr_b, margin_frac=0.35, min_margin=12):
    """以 GT∪Pred 的包围盒为中心裁一块局部放大区域（越界自动收拢）。"""
    m = gt_b | pr_b
    ys, xs = np.where(m)
    if ys.size == 0:
        return slice(None), slice(None)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = gt_b.shape
    my = max(min_margin, int((y1 - y0) * margin_frac))
    mx = max(min_margin, int((x1 - x0) * margin_frac))
    return (slice(max(0, y0 - my), min(h, y1 + my)),
            slice(max(0, x0 - mx), min(w, x1 + mx)))


def panel(ax, img, gt_b, pr_b, mode, title):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    if mode == "contour":
        ov = np.zeros(img.shape + (4,), np.float32)
        ov[boundary(gt_b)] = (0.1, 1.0, 0.2, 1.0)
        ov[boundary(pr_b)] = (1.0, 0.15, 0.1, 1.0)
        ax.imshow(ov)
        ax.set_xlabel("GT=green  Pred=red", fontsize=8)
    else:
        fp = pr_b & ~gt_b
        fn = gt_b & ~pr_b
        ov = np.zeros(img.shape + (4,), np.float32)
        ov[fp] = (1.0, 0.1, 0.1, 0.55)
        ov[fn] = (0.2, 0.4, 1.0, 0.55)
        ax.imshow(ov)
        ax.set_xlabel("FP=red  FN=blue", fontsize=8)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])


def worst_rows(rows, n):
    both = [r for r in rows if r["low_dice"] and r["high_hd"]]
    both.sort(key=lambda r: r["dice"])
    return both[:n]


def fig_one(subset, r, prob):
    img, gt_b = read_case(r["rel"])
    pr_b = pred_at_original(prob.astype(np.float32), gt_b.shape)
    name = r["rel"].split("/")[-1].replace(".nii.gz", "")
    sl = crop_box(gt_b, pr_b)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=170)
    panel(axes[0], img, gt_b, pr_b, "contour", "full slice")
    panel(axes[1], img[sl], gt_b[sl], pr_b[sl], "contour", "plaque zoom")
    panel(axes[2], img[sl], gt_b[sl], pr_b[sl], "error",
          "error (GT %d px)" % int(gt_b.sum()))
    fig.suptitle(
        "%s | %s\nDice %.3f   ASSD %.3f mm   HD %.2f mm   area %.1f mm2   "
        "elongation %.1f   minor %.2f mm   CNR %.2f   pred/GT %.2f"
        % (subset, name, r["dice"], r["assd"], r["hd"], r["gt_area_mm2"],
           r["elongation"], r["minor_mm"], r["cnr"], r["pred_ratio"]),
        fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def fig_scatter(data):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=150)
    specs = [("elongation", "elongation (major/minor)"),
             ("minor_mm", "plaque minor axis (mm)"),
             ("cnr", "foreground/background CNR")]
    for ax, (key, label) in zip(axes, specs):
        for name, rows, color in (("SPARC", data["SPARC"][0], "#1f77b4"),
                                  ("Zhongnan", data["Zhongnan"][0], "#ff7f0e")):
            x = np.array([r[key] for r in rows])
            y = np.array([r["dice"] for r in rows])
            bad = np.array([r["low_dice"] and r["high_hd"] for r in rows])
            ax.scatter(x[~bad], y[~bad], s=12, c=color, alpha=0.35, label="%s other" % name)
            ax.scatter(x[bad], y[bad], s=26, c=color, edgecolors="k", linewidths=0.5,
                       label="%s low Dice & high HD" % name)
            rho, p = spearmanr(x, y)
            ax.annotate("%s rho=%.2f p=%.3f" % (name, rho, p),
                        xy=(0.02, 0.04 if name == "SPARC" else 0.11),
                        xycoords="axes fraction", fontsize=7, color=color,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Dice", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle("Per-slice Dice vs low-score characteristics (both subsets, original-grid metrics)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, "shape_scatter.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def med(rows, key):
    return float(np.median([r[key] for r in rows]))


def build_tables(data):
    lines = ["# 低分样本明细（自动生成，请勿手改）", ""]
    for subset, (rows, hd_p75) in data.items():
        n = len(rows)
        nb = sum(1 for r in rows if r["low_dice"] and r["high_hd"])
        lines += ["## %s（n=%d，低 Dice ∩ 高 HD = %d）" % (subset, n, nb), "",
                  "| case | Dice | ASSD | HD | 面积 mm² | elong | minor mm | CNR | rel_echo | pred_ratio |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in worst_rows(rows, 12):
            lines.append("| %s | %.3f | %.3f | %.2f | %.1f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
                r["rel"].split("/")[-1], r["dice"], r["assd"], r["hd"], r["gt_area_mm2"],
                r["elongation"], r["minor_mm"], r["cnr"], r["rel_echo"], r["pred_ratio"]))
        lo = [r for r in rows if r["low_dice"]]
        rest = [r for r in rows if not r["low_dice"]]
        hh = [r for r in rows if r["high_hd"]]
        hn = [r for r in rows if not r["high_hd"]]
        lines += ["", "| 分组对比（中位数） | 低 Dice | 其余 | 高 HD | 其余 |", "|---|---|---|---|---|"]
        for key, label in (("elongation", "elongation"), ("minor_mm", "minor mm"),
                           ("cnr", "CNR"), ("rel_echo", "rel_echo"),
                           ("gt_area_mm2", "面积 mm²"), ("dice", "Dice"), ("hd", "HD")):
            lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (label, med(lo, key), med(rest, key),
                                                                med(hh, key), med(hn, key)))
        lines += ["", "| Spearman（本库内） | rho | p |", "|---|---|---|"]
        for key in ("elongation", "minor_mm", "cnr", "rel_echo"):
            rho, p = spearmanr([r[key] for r in rows], [r["dice"] for r in rows])
            lines.append("| %s ~ Dice | %.3f | %.3f |" % (key, rho, p))
        for key in ("gt_area_mm2", "minor_mm", "elongation"):
            rho, p = spearmanr([r[key] for r in rows], [r["hd"] for r in rows])
            lines.append("| %s ~ HD | %.3f | %.3f |" % (key, rho, p))
        lines.append("")
        q = np.percentile([r["dice"] for r in rows], 25)
        lines.append("- Dice p25 = %.4f；HD p75 = %.4f mm" % (q, hd_p75))
        lines.append("")
    with open(TAB_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return TAB_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    data = {s: load_subset(s) for s in ("SPARC", "Zhongnan")}
    for subset, (rows, _) in data.items():
        probs = np.load(os.path.join(BASE_DIR, NPZ[subset]), allow_pickle=True)
        idx = {str(c): i for i, c in enumerate(probs["cases"])}
        out_dir = os.path.join(FIG_DIR, subset)
        os.makedirs(out_dir, exist_ok=True)
        for k, r in enumerate(worst_rows(rows, args.n), 1):
            name = r["rel"].split("/")[-1].replace(".nii.gz", "")
            fig = fig_one(subset, r, probs["probs"][idx[r["rel"]]])
            out = os.path.join(out_dir, "%02d_%s.png" % (k, name))
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print("[fig]", out)
    print("[fig]", fig_scatter(data))
    print("[tab]", build_tables(data))


if __name__ == "__main__":
    main()
