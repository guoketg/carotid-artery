"""生成逐切片损失权重表（回声 / 边界对比度 → 连续权重）。

权重不手设分档，而是由指标连续换算：
  1. 指标在 train 分布中的分位数 u ∈ [0,1]，u 越大表示该切片越「难」（越该抬权）
     echo：rel_echo 越小越难（低回声，与周围腔/管壁难分）
     cnr ：cnr 越小越难（斑块边界与周围环境对比度低）
  2. w = exp(γ·(u − 0.5))，γ = ln(RATIO)
     这样最难的样本权重是最易样本的 RATIO 倍，且相邻分位之间是平滑过渡而非跳变
  3. 归一化到 train 均值 1 并 clip

注意：训练时权重会在 batch 内再归一化，因此**权重的绝对尺度对训练没有影响**，
唯一有意义的自由度就是 RATIO（最难 : 最易）。默认 RATIO=3。

阈值（分位参考分布）只在 train 上拟合，test 只查表，避免信息泄漏。

产物（每个子集 3 份，列 rel,w）：
  logs/analysis/sample_weights_<Subset>_echo.csv
  logs/analysis/sample_weights_<Subset>_cnr.csv
  logs/analysis/sample_weights_<Subset>_both.csv   两者相乘后重新归一化 + clip

用法：
  .\\venv\\Scripts\\python.exe scripts\\build_sample_weights.py
"""
import csv
import math
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_carotid import DATA_ROOT, LIST_DIR, read_list, normalize_image  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "logs", "analysis")

RATIO = 3.0                      # 最难 : 最易 的权重比（唯一自由度）
GAMMA = math.log(RATIO)
CLIP = (0.4, 2.5)
RING_PX = 6


def load_pair(rel):
    path = os.path.join(DATA_ROOT, rel)
    img = nib.load(path).get_fdata(dtype=np.float32)
    gt = nib.load(path.replace(".nii.gz", "_label.nii.gz")).get_fdata(dtype=np.float32)
    if img.ndim == 3:
        img = img[..., img.shape[-1] // 2]
    if gt.ndim == 3:
        gt = gt[..., gt.shape[-1] // 2]
    return img, gt > 0.5


def slice_attributes(rel):
    """rel_echo = 前景均值 − 外环均值（>0 高回声）；cnr = 两者差异 / 合并标准差。"""
    img, gt = load_pair(rel)
    if gt.sum() == 0:
        return np.nan, np.nan
    img = normalize_image(img)
    fg = img[gt]
    ring = ndi.binary_dilation(gt, iterations=RING_PX) & ~gt
    if ring.sum() == 0:
        return np.nan, np.nan
    ref = img[ring]
    diff = float(fg.mean() - ref.mean())
    pooled = float(np.sqrt(0.5 * (fg.var() + ref.var())))
    return diff, abs(diff) / max(pooled, 1e-6)


def hard_rank(values, ref, lower_is_harder):
    """指标 → 难度分位 u ∈ [0,1]；参考分布 ref 来自 train。"""
    ref = np.sort(np.asarray(ref, dtype=float))
    u = np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref)
    return 1.0 - u if lower_is_harder else u


def to_weight(u):
    """难度分位 → 权重：w = exp(γ(u−0.5))，再归一化到均值 1 并 clip。"""
    w = np.exp(GAMMA * (np.asarray(u, dtype=float) - 0.5))
    w = w / w.mean()
    return np.clip(w, CLIP[0], CLIP[1])


def report(name, values, w):
    q = np.quantile(w, [0.0, 0.1, 0.5, 0.9, 1.0])
    print("  %-4s 权重 p0/p10/p50/p90/p100 = %.2f / %.2f / %.2f / %.2f / %.2f"
          "  最难:最易 = %.2f:1"
          % (name, q[0], q[1], q[2], q[3], q[4], q[4] / max(q[0], 1e-6)))
    # 权重对自身指标应严格单调（Spearman = ±1），作为接线自检
    from scipy import stats
    rho = stats.spearmanr(values, w).statistic
    print("        spearman(指标, 权重) = %+.3f（单调性自检）" % rho)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("RATIO = %.1f:1（最难:最易），γ = %.4f" % (RATIO, GAMMA))
    for subset in ("SPARC", "Zhongnan"):
        split_dir = os.path.join(LIST_DIR, subset)
        train = read_list(os.path.join(split_dir, "train.txt"))
        test = read_list(os.path.join(split_dir, "test.txt"))
        print("\n[%s] train %d / test %d" % (subset, len(train), len(test)))

        feats = {rel: slice_attributes(rel) for rel in train + test}

        tr_echo = [feats[r][0] for r in train if np.isfinite(feats[r][0])]
        tr_cnr = [feats[r][1] for r in train if np.isfinite(feats[r][1])]
        print("  train 参考分布：rel_echo 中位 %+.3f（IQR %.3f），cnr 中位 %.3f（IQR %.3f）"
              % (np.median(tr_echo), np.subtract(*np.quantile(tr_echo, [0.75, 0.25])),
                 np.median(tr_cnr), np.subtract(*np.quantile(tr_cnr, [0.75, 0.25]))))

        rows = {"echo": [], "cnr": [], "both": []}
        miss = 0
        echo_vals, cnr_vals = [], []
        for rel in train + test:
            echo, cnr = feats[rel]
            if not (np.isfinite(echo) and np.isfinite(cnr)):
                miss += 1
                echo, cnr = float(np.median(tr_echo)), float(np.median(tr_cnr))
            echo_vals.append(echo)
            cnr_vals.append(cnr)

        n_tr = len(train)
        echo_vals = np.array(echo_vals)
        cnr_vals = np.array(cnr_vals)
        # 阈值分布只用 train 段拟合
        u_echo = hard_rank(echo_vals, echo_vals[:n_tr], lower_is_harder=True)
        u_cnr = hard_rank(cnr_vals, cnr_vals[:n_tr], lower_is_harder=True)
        w_echo = to_weight(u_echo)
        w_cnr = to_weight(u_cnr)
        w_both = to_weight(0.5 * (u_echo + u_cnr))

        # both 单独归一化（以 train 段均值为 1）
        w_both = w_both / w_both[:n_tr].mean()
        w_both = np.clip(w_both, CLIP[0], CLIP[1])
        w_echo = w_echo / w_echo[:n_tr].mean()
        w_cnr = w_cnr / w_cnr[:n_tr].mean()

        if miss:
            print("  警告：%d 张切片属性缺失，按 train 中位数（中性）处理" % miss)

        for mode, w in (("echo", w_echo), ("cnr", w_cnr), ("both", w_both)):
            for rel, v in zip(train + test, w):
                rows[mode].append((rel, float(v)))
            report(mode, (echo_vals if mode == "echo" else
                          cnr_vals if mode == "cnr" else 0.5 * (u_echo + u_cnr)), w)
            path = os.path.join(OUT_DIR, "sample_weights_%s_%s.csv" % (subset, mode))
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["rel", "w"])
                writer.writerows(rows[mode])
            print("        train 均值 %.3f（min %.2f / max %.2f） test 均值 %.3f → %s"
                  % (w[:n_tr].mean(), w[:n_tr].min(), w[:n_tr].max(),
                     w[n_tr:].mean(), os.path.basename(path)))


if __name__ == "__main__":
    main()
