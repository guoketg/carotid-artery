"""聚类版样本权重表：全训练集 k-means（k=2），按簇给权。

与 build_sample_weights.py 的连续分位版并列，二者共用同一套训练代码，
只是权重表的生成方式不同：
  连续版：单指标 → 分位 → exp 映射（每个切片权重都不同）
  聚类版：二维特征 [rel_echo, cnr] → k-means 二分 → 簇级权重（簇内同权）

分组口径（避免信息泄漏）：
  - 特征标准化参数、k-means 中心都只在 train 上拟合，test 只做 assign
  - 低回声簇 = 簇中心 rel_echo 更小的那一簇 → 权重 1.5
    高回声簇 = 另一簇                      → 权重 0.5
    两者严格 3:1，均值 1

两种粒度：
  slice   ：以切片为单位聚类（每张切片独立归簇）
  patient ：同一患者的多张切片取特征中位数后归簇，整例共用同一权重
            （临床上回声分级是患者/斑块级；切片级聚类会让 71% 的 SPARC 患者
             内部出现两种标签）

产物：
  logs/analysis/sample_weights_<Subset>_cluster_slice.csv
  logs/analysis/sample_weights_<Subset>_cluster_patient.csv

用法：
  .\\venv\\Scripts\\python.exe scripts\\build_cluster_weights.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_carotid import LIST_DIR, read_list  # noqa: E402
from scripts.build_sample_weights import OUT_DIR, slice_attributes  # noqa: E402

W_LOW, W_HIGH = 1.5, 0.5          # 低回声簇 / 高回声簇，3:1，均值 1
METRIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "model", "TU_Carotid224")


def kmeans2d(x, k=2, inits=30, iters=100, seed=1234):
    """numpy 版 k-means（环境无 sklearn）；多次初始化取惯性最小的一次。"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(inits):
        c = x[rng.choice(len(x), k, replace=False)].copy()
        for _ in range(iters):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, c.copy(), lab.copy())
    return best[1], best[2]


def patient_of(rel):
    return rel.replace("\\", "/").split("/")[-2]


def aggregate_patient(rows):
    """rows: [(rel, echo, cnr)] → 患者级 (pid, 中位 echo, 中位 cnr, [rel...])"""
    by = {}
    for rel, e, c in rows:
        by.setdefault(patient_of(rel), []).append((rel, e, c))
    out = []
    for pid, vs in by.items():
        out.append((pid, float(np.median([v[1] for v in vs])),
                    float(np.median([v[2] for v in vs])), [v[0] for v in vs]))
    return out


def group_metrics(subset, rels):
    """用 A0 基线逐切片指标看分组难度（仅诊断用，不参与权重计算）。"""
    names = {"SPARC": "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d",
             "Zhongnan": "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_c2d"}
    path = os.path.join(METRIC_DIR, names[subset], "metrics.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        mm = {r["case"].replace("\\", "/"): r for r in csv.DictReader(f)}
    hit = [mm[r] for r in rels if r in mm]
    if not hit:
        return None
    return tuple(float(np.mean([float(r[k]) for r in hit])) for k in ("dice", "assd", "hd"))


def nearest(points, centers):
    """points: (n,2) 原始特征；返回最近簇中心下标。"""
    return np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for subset in ("SPARC", "Zhongnan"):
        split_dir = os.path.join(LIST_DIR, subset)
        train = read_list(os.path.join(split_dir, "train.txt"))
        test = read_list(os.path.join(split_dir, "test.txt"))
        feats = {rel: slice_attributes(rel) for rel in train + test}

        rows = [(r, feats[r][0], feats[r][1]) for r in train + test
                if np.isfinite(feats[r][0]) and np.isfinite(feats[r][1])]
        tr_rows = [r for r in rows if r[0] in set(train)]
        te_rows = [r for r in rows if r[0] in set(test)]

        echo = np.array([r[1] for r in tr_rows])
        cnr = np.array([r[2] for r in tr_rows])
        mu = np.array([echo.mean(), cnr.mean()])
        sd = np.array([echo.std(), cnr.std()]) + 1e-8

        print("\n===== %s =====" % subset)
        print("  train 切片 %d / test 切片 %d" % (len(tr_rows), len(te_rows)))

        for gran in ("slice", "patient"):
            if gran == "slice":
                x_tr = (np.stack([[r[1], r[2]] for r in tr_rows]) - mu) / sd
                centers, _ = kmeans2d(x_tr)
                tr_lab = nearest((np.stack([[r[1], r[2]] for r in tr_rows]) - mu) / sd, centers)
                te_lab = nearest((np.stack([[r[1], r[2]] for r in te_rows]) - mu) / sd, centers) \
                    if te_rows else np.array([], dtype=int)
            else:
                p_tr = aggregate_patient(tr_rows)
                p_te = aggregate_patient(te_rows)
                x_tr = (np.stack([[p[1], p[2]] for p in p_tr]) - mu) / sd
                centers, _ = kmeans2d(x_tr)
                p_lab = nearest((np.stack([[p[1], p[2]] for p in p_tr]) - mu) / sd, centers)
                pid_lab_tr = dict(zip([p[0] for p in p_tr], p_lab))
                te_p_lab = nearest((np.stack([[p[1], p[2]] for p in p_te]) - mu) / sd, centers) \
                    if p_te else np.array([], dtype=int)
                pid_lab_te = dict(zip([p[0] for p in p_te], te_p_lab))
                tr_lab = np.array([pid_lab_tr[patient_of(r[0])] for r in tr_rows])
                te_lab = np.array([pid_lab_te[patient_of(r[0])] for r in te_rows]) \
                    if te_rows else np.array([], dtype=int)

            # 低回声簇 = 中心 rel_echo 更小的簇
            tr_echo_of = np.array([np.mean([r[1] for r, l in zip(tr_rows, tr_lab) if l == i])
                                   for i in range(2)])
            hypo = int(np.argmin(tr_echo_of))
            w = np.where(tr_lab == hypo, W_LOW, W_HIGH)
            w_te = np.where(te_lab == hypo, W_LOW, W_HIGH)

            print("  [%s] 簇中心(标准化) %s / %s；低回声簇=簇%d，权重 %.1f : %.1f"
                  % (gran, np.round(centers[0], 2), np.round(centers[1], 2),
                     hypo, W_LOW, W_HIGH))
            for i in range(2):
                tr_i = [r for r, l in zip(tr_rows, tr_lab) if l == i]
                te_i = [r for r, l in zip(te_rows, te_lab) if l == i]
                gm = group_metrics(subset, [r[0] for r in te_i])
                tag = "低回声簇" if i == hypo else "高回声簇"
                print("    %s train %3d 切片 (%.0f%%) / test %3d 切片  rel_echo %+.2f  cnr %.2f"
                      % (tag, len(tr_i), 100.0 * len(tr_i) / len(tr_rows), len(te_i),
                         np.mean([r[1] for r in tr_i]), np.mean([r[2] for r in tr_i])),
                      end="")
                if gm:
                    print("  | A0 test Dice %.2f / ASSD %.3f / HD %.3f" % (gm[0] * 100, gm[1], gm[2]))
                else:
                    print()

            path = os.path.join(OUT_DIR, "sample_weights_%s_cluster_%s.csv" % (subset, gran))
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["rel", "w"])
                wr.writerows([(r[0], float(x)) for r, x in zip(rows, np.concatenate([w, w_te]))])
            print("    → %s（train 低回声 %d / 高回声 %d）"
                  % (os.path.basename(path), int((w == W_LOW).sum()), int((w == W_HIGH).sum())))


if __name__ == "__main__":
    main()
