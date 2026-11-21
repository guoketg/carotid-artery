"""表示空间距离加权表（repr_dist）：全训练集 k-means，权重取「到本簇中心的距离」。

命名与口径说明（重要）：
  本方法**不使用回声与 CNR**。回声 / 对比度这两个灰度标量既不进入聚类特征，
  也不参与权重构造，簇与临床声学表型（低回声 / 高回声）没有任何对应关系。
  实测核对：权重与 rel_echo 的 Spearman 相关 SPARC −0.18、Zhongnan +0.09，
  与 cnr 为 −0.04 / +0.22，符号还跨库翻转（Temp/check_repr_vs_echo.py）。
  真要按回声 / CNR 加权，用 scripts/build_echo_cnr_groups.py 产出的
  sample_weights_<Subset>_echo_group.csv / _cnr_group.csv，两者不可混称。

与 build_cluster_weights.py 的区别（表结构一致，列 rel,w）：
  - 聚类特征：ImageNet 预训练 ResNet152 的 avgpool 特征（2048 维，L2 归一化）
              -> 只在 train 上拟合 PCA(16) -> 标准化 -> k-means(k=4)
    手工版用的是二维标量 [rel_echo, cnr]。文献（Thiam et al., Computers 2025, 14(11):457，
    "Dealing with Class Overlap Through Cluster-Based Sample Weighting"）的做法是在
    表示空间聚类、并以样本到簇中心的距离构造权重；本脚本按这条口径实现。
  - 权重形式：连续、逐样本不同。d = 到本簇中心的欧氏距离，簇内 z-score 后取 train 内分位 u，
            w = exp(γ(u − 0.5))，γ = ln3（最难 : 最易 = 3:1），clip 到 [0.5, 2.0]，
            最后归一化到 train 均值 1。
            距离越大 = 越靠近簇间交界（越"重叠"）= 权重越高，与文献一致。
  - 防泄漏：PCA / 标准化参数 / 簇中心 / 距离分位参考分布全部只在 train 上拟合，test 只 assign。

产物：logs/analysis/sample_weights_<Subset>_repr_dist_k4.csv
（未入队、无训练读数；训练时需显式 --sw_file 指这张表，--sw_mode 里没有对应取值）

用法：
  .\\venv\\Scripts\\python.exe scripts\\build_repr_dist_weights.py
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_carotid import LIST_DIR, read_list  # noqa: E402
from scripts.build_cluster_weights import kmeans2d, nearest  # noqa: E402
from scripts.build_sample_weights import OUT_DIR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(OUT_DIR, "repr_dist_feats_resnet152.npz")
PROBE_CACHE = os.path.join(ROOT, "Temp", "probe_feats.npz")
K = 4                      # 簇数（离线探针里 k=4 的簇间难度差大于 k=2，见 docs/results.md）
PCA_DIM = 16
RATIO = 3.0                # 最难 : 最易
GAMMA = math.log(RATIO)
CLIP = (0.5, 2.0)
SIZE = 224


def _load_pair(rel):
    import nibabel as nib
    from scipy import ndimage as ndi

    def _z(vol):
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        vol = np.clip(vol, lo, hi)
        return (vol - vol.mean()) / max(vol.std(), 1e-6)

    from datasets.dataset_carotid import DATA_ROOT
    p = os.path.join(DATA_ROOT, rel)
    img = nib.load(p).get_fdata(dtype=np.float32)
    if img.ndim == 3:
        img = img[..., img.shape[-1] // 2]
    img = _z(img)
    return ndi.zoom(img, (SIZE / img.shape[0], SIZE / img.shape[1]), order=3, mode="nearest")


def extract_features(subset):
    """ResNet152 avgpool 特征；CPU 前向，不占 GPU。"""
    import torch
    import torchvision

    train = read_list(os.path.join(LIST_DIR, subset, "train.txt"))
    test = read_list(os.path.join(LIST_DIR, subset, "test.txt"))
    rels = train + test
    torch.set_num_threads(os.cpu_count() or 4)
    model = torchvision.models.resnet152(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(rels), 16):
            x = torch.cat([torch.from_numpy(_load_pair(r))[None, None]
                           for r in rels[i:i + 16]], 0)
            x = ((x.repeat(1, 3, 1, 1) - mean) / std)
            feats.append(model(x).numpy())
            if (i + 16) % 160 < 16:
                print("    features %d/%d" % (min(i + 16, len(rels)), len(rels)), flush=True)
    return train, test, np.concatenate(feats)


def all_features():
    if os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=True)
    elif os.path.isfile(PROBE_CACHE):          # 复用离线探针的同一份特征（同口径）
        z = np.load(PROBE_CACHE, allow_pickle=True)
        print("  reuse probe feature cache")
    else:
        out = {}
        for s in ("SPARC", "Zhongnan"):
            tr, te, f = extract_features(s)
            out[s + "_tr"], out[s + "_te"], out[s + "_f"] = np.array(tr, dtype=object), \
                np.array(te, dtype=object), f
        np.savez(CACHE, **out)
        z = np.load(CACHE, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    Z = all_features()
    for subset in ("SPARC", "Zhongnan"):
        tr = [r.replace("\\", "/") for r in Z[subset + "_tr"]]
        te = [r.replace("\\", "/") for r in Z[subset + "_te"]]
        F = Z[subset + "_f"].astype(np.float64)
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        n_tr = len(tr)

        # PCA 只在 train 上拟合
        mu = F[:n_tr].mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(F[:n_tr] - mu, full_matrices=False)
        P = (F - mu) @ vt[:PCA_DIM].T
        psd = P[:n_tr].std(0) + 1e-8
        Ptr, Pte = P[:n_tr] / psd, P[n_tr:] / psd

        centers, _ = kmeans2d(Ptr, k=K)
        lab_tr, lab_te = nearest(Ptr, centers), nearest(Pte, centers)

        # 到本簇中心的距离 -> 簇内 z-score -> train 分位 -> exp 映射
        d_tr = np.linalg.norm(Ptr - centers[lab_tr], axis=1)
        d_te = np.linalg.norm(Pte - centers[lab_te], axis=1)
        z_tr, z_te = np.zeros(n_tr), np.zeros(len(te))
        for c in range(K):
            m_tr, m_te = lab_tr == c, lab_te == c
            if not m_tr.any():
                continue
            mc, sc = d_tr[m_tr].mean(), max(d_tr[m_tr].std(), 1e-8)
            z_tr[m_tr] = (d_tr[m_tr] - mc) / sc
            z_te[m_te] = (d_te[m_te] - mc) / sc
        ref = np.sort(z_tr)
        u_tr = np.searchsorted(ref, z_tr, side="right") / len(ref)
        u_te = np.searchsorted(ref, z_te, side="right") / len(ref)
        w_tr = np.exp(GAMMA * (u_tr - 0.5))
        w_te = np.exp(GAMMA * (u_te - 0.5))
        w_tr = np.clip(w_tr / w_tr.mean(), *CLIP)
        w_te = np.clip(w_te / w_tr.mean(), *CLIP)

        print("\n===== %s（train %d / test %d，k=%d，PCA %d 维）====="
              % (subset, n_tr, len(te), K, PCA_DIM))
        for c in range(K):
            m = lab_tr == c
            mt = lab_te == c
            print("  cluster %d: train %3d (%4.0f%%) / test %3d  w 均值 %.2f（%.2f~%.2f）"
                  % (c, int(m.sum()), 100.0 * m.sum() / n_tr, int(mt.sum()),
                     w_tr[m].mean(), w_tr[m].min(), w_tr[m].max()))
        print("  train 权重：均值 %.3f  min %.2f  max %.2f  3:1 覆盖 u∈[0,1]"
              % (w_tr.mean(), w_tr.min(), w_tr.max()))

        path = os.path.join(OUT_DIR, "sample_weights_%s_repr_dist_k4.csv" % subset)
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["rel", "w"])
            wr.writerows([(r, float(x)) for r, x in zip(tr + te, np.concatenate([w_tr, w_te]))])
        print("  -> %s" % os.path.basename(path))


if __name__ == "__main__":
    main()
