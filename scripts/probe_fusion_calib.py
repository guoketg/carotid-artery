"""融合轴实验的「反常」追查：那 +0.24 Dice 到底来自什么？

上一轮（scripts/probe_fusion_axis.py）发现：
    raw 0.4923 | raw_hf 0.5069 | feat_concat 0.4969
    feat_attn 0.7323 | feat_attn_raw 0.7135 | feat_attn_zero 0.7579
侧路喂**全零**（完全没有 HF 内容）反而最高 → 涨幅与频率信息无关。

两个待分辨的候选解释：
  (H1) 校准漂移：feat_attn_zero 里 k/v 全是常数 → 注意力输出与空间无关 →
       融合项退化成「给主干特征加一个逐通道常数偏置」。偏置会整体平移 logit，
       改变预测面积，从而在固定阈值 0.5 下大幅改变 Dice。
       若成立，|ΔArea| 应显著小于 raw，且「oracle 最优阈值 Dice」应接近 raw。
  (H2) 真实建模增益：注意力/双分支带来 raw 没有的全局上下文或集成效应。
       若成立，oracle 最优阈值 Dice 也会显著高于 raw。

做法：同一套训练，只换评估口径。
  dice@0.5      固定阈值 Dice（现在用的口径）
  dice@oracle   每个样本扫阈值取最大 Dice（剔掉标定影响后的「真实分割质量」）
  dice@area     用「预测面积 = 真值面积」的最优阈值（另一种去标定口径）
  signed ΔArea  预测面积 - 真值面积（mm²），正=过分割
  dsc@0.5-@orc  两者的差 = 标定贡献了多少 Dice

用法：python scripts/probe_fusion_calib.py [--subset SPARC] [--epochs 20] [--seeds 1234,2024]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset, LIST_DIR, DATA_ROOT, SUBSETS
from networks.wavelet import build_hf_image
from probe_fusion_axis import TinySeg, VARIANTS, sample_spacing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = np.arange(0.05, 0.96, 0.01)


def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float(2.0 * inter / den) if den > 0 else 1.0


def best_threshold_dice(prob, gt):
    """在阈值网格上取最优 Dice；同时返回使预测面积等于真值面积的阈值下 Dice。"""
    gt_area = int(gt.sum())
    best_d, area_d, best_t = 0.0, 0.0, 0.5
    if gt_area == 0:
        return 1.0, 1.0, 0.5
    for t in THRESHOLDS:
        pred = prob > t
        d = dice(pred, gt)
        if d > best_d:
            best_d, best_t = d, t
        pa = int(pred.sum())
        if abs(pa - gt_area) < abs(int((prob > best_t).sum()) - gt_area):
            pass
        area_d = max(area_d, 0.0)  # 占位，下面统一算
    # 「面积匹配」口径：取预测面积最接近真值面积的阈值
    diffs = []
    for t in THRESHOLDS:
        pred = prob > t
        diffs.append((abs(int(pred.sum()) - gt_area), dice(pred, gt), t))
    diffs.sort()
    area_d = diffs[0][1]
    return best_d, area_d, best_t


@torch.no_grad()
def evaluate_calib(model, images, labels, hf_cache, spacings, bs=16):
    model.eval()
    rows = []
    for i in range(0, images.shape[0], bs):
        x = images[i:i + bs].to(DEVICE)
        hf = hf_cache[i:i + bs].to(DEVICE)
        prob = torch.softmax(model(x, hf), dim=1)[:, 1].cpu().numpy()
        gt = labels[i:i + bs].numpy() > 0.5
        for j in range(prob.shape[0]):
            g = gt[j]
            pred = prob[j] > 0.5
            sy, sx = spacings[i + j]
            signed_area = (int(pred.sum()) - int(g.sum())) * sy * sx
            d05 = dice(pred, g)
            d_orc, d_area, t = best_threshold_dice(prob[j], g)
            rows.append({"dice05": d05, "dice_orc": d_orc, "dice_area": d_area,
                         "signed_area": signed_area, "t_orc": t})
    out = {}
    for k in ["dice05", "dice_orc", "dice_area", "signed_area", "t_orc"]:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["calib_gain"] = out["dice05"] - out["dice_orc"]
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
            loss = F.cross_entropy(model(x, hf), y.long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        hf_test = torch.cat([build_hf_image(test_img[i:i + 16].to(DEVICE))
                             for i in range(0, test_img.shape[0], 16)])
    m = evaluate_calib(model, test_img, test_lab, hf_test, spacings)
    m["nparam"] = sum(p.numel() for p in model.parameters())
    m["sec"] = time.time() - t0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="SPARC")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="1234,2024")
    ap.add_argument("--variants", default="raw,raw_hf,feat_concat,feat_attn,feat_attn_zero")
    args = ap.parse_args()

    tr = CarotidDataset(root=DATA_ROOT, list_file=os.path.join(LIST_DIR, args.subset, "train.txt"),
                        split="train", img_size=224, augment=False)
    te = CarotidDataset(root=DATA_ROOT, list_file=os.path.join(LIST_DIR, args.subset, "test.txt"),
                        split="test", img_size=224, augment=False)
    tr_i = torch.stack([tr[i]["image"] for i in range(len(tr))])
    tr_l = torch.stack([tr[i]["label"] for i in range(len(tr))])
    te_i = torch.stack([te[i]["image"] for i in range(len(te))])
    te_l = torch.stack([te[i]["label"] for i in range(len(te))])
    spacings = [sample_spacing(DATA_ROOT, te[i]["case"]) for i in range(len(te))]
    print(f"[data] {args.subset}: train {len(tr)} / test {len(te)}\n")

    variants = args.variants.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    agg = {}
    for v in variants:
        runs = []
        for sd in seeds:
            m = train_one(v, tr_i, tr_l, te_i, te_l, spacings, args.epochs, sd)
            runs.append(m)
            print(f"  {v:<14} seed {sd}: dice@0.5 {m['dice05'] * 100:.2f}%  dice@orc {m['dice_orc'] * 100:.2f}%  "
                  f"dice@area {m['dice_area'] * 100:.2f}%  signedΔA {m['signed_area']:+.3f}  "
                  f"calib_gain {(m['dice05'] - m['dice_orc']) * 100:+.2f}%  ({m['sec']:.0f}s)")
        a = {k: float(np.mean([r[k] for r in runs])) for k in
             ["dice05", "dice_orc", "dice_area", "signed_area", "calib_gain"]}
        a["nparam"] = runs[0]["nparam"]
        agg[v] = a
        print(f"{v:<14} {a['nparam']/1e6:>6.3f}M | dice@0.5 {a['dice05'] * 100:.2f}% | dice@orc {a['dice_orc'] * 100:.2f}% "
              f"| dice@area {a['dice_area'] * 100:.2f}% | signedΔA {a['signed_area']:+.3f} "
              f"| calib_gain {(a['dice05'] - a['dice_orc']) * 100:+.2f}%\n")

    if "raw" in agg:
        b = agg["raw"]
        print("=== 相对 raw ===")
        for v in variants:
            if v == "raw":
                continue
            a = agg[v]
            print(f"  {v:<14} Δdice@0.5 {(a['dice05']-b['dice05']) * 100:+.2f}%   "
                  f"Δdice@orc {(a['dice_orc']-b['dice_orc']) * 100:+.2f}%   "
                  f"Δdice@area {(a['dice_area']-b['dice_area']) * 100:+.2f}%   "
                  f"ΔsignedΔA {a['signed_area']-b['signed_area']:+.3f}")


if __name__ == "__main__":
    main()
