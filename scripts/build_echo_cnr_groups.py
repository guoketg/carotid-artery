"""按「回声」与「对比噪声率」分别分簇的样本分组表（替换旧的二维 k-means 版本）。

旧版（scripts/build_cluster_weights.py）有三个问题，本脚本逐条改掉：
  1. 把「回声」算成了 斑块均值 − 环带均值（那是灰度差/对比度，不是回声强度本身）；
     本脚本的回声 = 斑块内部（GT 内）像素归一化灰度的均值，不减环带。
  2. 把回声与 CNR 拼成二维做 k-means，等于在二维平面上画一条直线，
     而两维高度共线，实际退化成一次阈值切分；本脚本把两者**分开**，各自一维分档。
  3. 权重 1.5 / 0.5 是硬编码的；本脚本的比值由 --ratio 给，且分档点取 train 中位数，
     因此两档各占约 50%，权重均值天然为 1。

两个指标（都只用 GT + 图像本身算，不涉及预测）：
  echo = mean(I[GT])                                    斑块内平均灰度
  cnr  = |mean(I[GT]) − mean(I[ring])| / sqrt(0.5·(var_fg + var_ring))
                                                        对比噪声率（沿用环带作为背景参考）
  其中 I 为整图 1%~99% 截断 + z-score 后的灰度，ring = GT 外扩 --ring_px 圈且不含 GT。

分档（只用 train 拟合阈值）：
  以 train 上该指标的中位数为界，<= 阈值 为"低"档，> 阈值 为"高"档。
  1 维上按中位数二分与 1 维 k-means(k=2) 的切点等价（仅边界取整点不同）。

产物（每个子集两组文件）：
  logs/analysis/groups_<Subset>_echo.csv     列：rel,echo,group
  logs/analysis/groups_<Subset>_cnr.csv      列：rel,cnr,group
  logs/analysis/sample_weights_<Subset>_echo_group.csv   列：rel,w
  logs/analysis/sample_weights_<Subset>_cnr_group.csv    列：rel,w
训练侧用 --loss sw --sw_mode cluster_slice --sw_file <上面的权重表> 即可，其它口径不变。

用法：
  .\\venv\\Scripts\\python.exe scripts\\build_echo_cnr_groups.py                 # 全量
  .\\venv\\Scripts\\python.exe scripts\\build_echo_cnr_groups.py --limit 20      # 冒烟
  .\\venv\\Scripts\\python.exe scripts\\build_echo_cnr_groups.py --ratio 2 --hard high
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_carotid import DATA_ROOT, LIST_DIR, normalize_image, read_list  # noqa: E402
from scripts.build_sample_weights import OUT_DIR, load_pair  # noqa: E402


def slice_echo_cnr(rel, ring_px):
    """返回 (echo, cnr)。两者都基于整图 z-score 后的灰度。"""
    img, gt = load_pair(rel)
    if gt.sum() == 0:
        return np.nan, np.nan
    img = normalize_image(img)
    fg = img[gt]
    ring = ndi.binary_dilation(gt, iterations=ring_px) & ~gt
    if ring.sum() == 0:
        return float(fg.mean()), np.nan
    bg = img[ring]
    echo = float(fg.mean())
    pooled = float(np.sqrt(0.5 * (fg.var() + bg.var())))
    cnr = float(abs(fg.mean() - bg.mean()) / max(pooled, 1e-6))
    return echo, cnr


def split_at_train_median(values_tr, values_all):
    """以 train 中位数为界二分：<= 阈值 = low，> 阈值 = high。"""
    thr = float(np.median(values_tr[np.isfinite(values_tr)]))
    group = np.where(values_all <= thr, "low", "high")
    group[~np.isfinite(values_all)] = "na"
    return thr, group


def group_difficulty(subset, rels, groups, metrics):
    """用现有 A0 逐切片指标看两档的难度（仅诊断，不参与分组）。"""
    out = {}
    for g in ("low", "high"):
        hits = [metrics[r] for r, gg in zip(rels, groups) if gg == g and r in metrics]
        out[g] = (len(hits),
                  np.mean([h[0] for h in hits]) if hits else float("nan"),
                  np.mean([h[1] for h in hits]) if hits else float("nan"),
                  np.mean([h[2] for h in hits]) if hits else float("nan"))
    return out


def read_metrics(subset):
    p = os.path.join(OUT_DIR, "error_slices_%s.csv" % subset)
    if not os.path.isfile(p):
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {r["rel"].replace("\\", "/"): (float(r["dice"]) * 100, float(r["assd"]), float(r["hd"]))
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="SPARC,Zhongnan")
    ap.add_argument("--ring_px", type=int, default=6, help="CNR 的背底环带宽度（原始像素）")
    ap.add_argument("--ratio", type=float, default=3.0,
                    help="难档 : 易档 的权重比（默认 3 => 1.5 / 0.5，两档中位数均分故均值=1）")
    ap.add_argument("--hard", default="low", choices=["low", "high"],
                    help="哪一档给高权重；默认 low（低回声/低对比度一侧）")
    ap.add_argument("--limit", type=int, default=0, help="冒烟：每个子集只取前 N 张（<=50）")
    args = ap.parse_args()

    w_hi = 2.0 * args.ratio / (1.0 + args.ratio)     # ratio=3 -> 1.5
    w_lo = 2.0 / (1.0 + args.ratio)                  # ratio=3 -> 0.5
    suffix = "_smoke" if args.limit else ""          # 冒烟产物绝不覆盖全量表
    os.makedirs(OUT_DIR, exist_ok=True)

    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        train = read_list(os.path.join(LIST_DIR, subset, "train.txt"))
        test = read_list(os.path.join(LIST_DIR, subset, "test.txt"))
        if args.limit:
            train, test = train[:args.limit], test[:max(1, args.limit // 4)]
        rels = train + test
        n_tr = len(train)
        vals = np.array([slice_echo_cnr(r, args.ring_px) for r in rels], dtype=float)
        echo, cnr = vals[:, 0], vals[:, 1]
        metrics = read_metrics(subset)

        print("\n===== %s =====" % subset)
        print("  slices: train %d / test %d   ring_px=%d   ratio=%.1f (w %s=%.2f, %s=%.2f)"
              % (n_tr, len(test), args.ring_px, args.ratio,
                 args.hard, w_hi, "high" if args.hard == "low" else "low", w_lo))

        for name, v in (("echo", echo), ("cnr", cnr)):
            thr, grp = split_at_train_median(v[:n_tr], v)
            n_na = int((grp == "na").sum())
            print("  [%s] 阈值(train 中位数) %.4f   train 低 %d / 高 %d   全量缺失 %d"
                  % (name, thr, int((grp[:n_tr] == "low").sum()),
                     int((grp[:n_tr] == "high").sum()), n_na))

            path = os.path.join(OUT_DIR, "groups_%s_%s%s.csv" % (subset, name, suffix))
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["rel", name, "group"])
                wr.writerows([(r, "%.6f" % x if np.isfinite(x) else "", g)
                              for r, x, g in zip(rels, v, grp)])

            # 权重表：难档 w_hi，易档 w_lo；train 均值应为 1（两档中位数均分）
            w = np.where(grp == "high", w_hi if args.hard == "high" else w_lo,
                         w_hi if args.hard == "low" else w_lo)
            w[grp == "na"] = 1.0
            wpath = os.path.join(OUT_DIR, "sample_weights_%s_%s_group%s.csv" % (subset, name, suffix))
            with open(wpath, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["rel", "w"])
                wr.writerows([(r, float(x)) for r, x in zip(rels, w)])
            print("    -> %s（train 权重均值 %.3f）"
                  % (os.path.basename(wpath), w[:n_tr].mean()))

            if metrics:
                diff = group_difficulty(subset, test, grp[n_tr:], metrics)
                for g in ("low", "high"):
                    n, d, a, h = diff[g]
                    print("    test 分组 %-4s n=%3d  Dice %.2f / ASSD %.3f / HD %.3f"
                          % (g, n, d, a, h))
                gap = abs(diff["high"][1] - diff["low"][1])
                if np.isfinite(gap):
                    print("    两档 Dice 差 %.2f（诊断用，未参与分组与权重）" % gap)
                else:
                    print("    两档 Dice 差：某一档在 test 上无样本，无法计算")


if __name__ == "__main__":
    main()
