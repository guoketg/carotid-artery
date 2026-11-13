"""阶段 0：离线验证「距离尾部代理」与真实 HD 的相关性。

目的：在写任何损失代码之前，确认"最差 q% 边界距离误差"这一代理量
能不能预测逐切片 HD。若不能，说明 HD 的尾部由标注/成像极限主导，
尾部聚合损失不会有效，直接止损。

做法（不训练、不占 GPU，复用 logs/analysis/pred_cache_*.npz）：
  1. 取 224 概率图 -> 二值预测 -> 重采样回原始网格（mm spacing）；
  2. 构造边界误差样本集：
       FP 侧 = 预测边界像素到 GT 的距离（外正内负，取绝对值）
       FN 侧 = GT 边界上未被预测覆盖的像素，到最近预测前景的距离
  3. 代理量：mean（均值型，≈ASSD 口径）与 top-q%（q=1/5/10/20/30，尾部型，≈HD 口径）；
  4. 与 pred_geometry_*.csv 里的真实 hd / assd 求 Spearman，并统计
     高 HD 组（≥p75）的检出重合度。

用法：
  .\\venv\\Scripts\\python.exe scripts\\analyze_hd_tail_surrogate.py --subset both
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from metrics_carotid import original_grid_pair

ROOT_PATH = r'E:\mkl\code\datasets\Carotid2D'
ANA_DIR = os.path.join(BASE_DIR, 'logs', 'analysis')
QS = [1, 5, 10, 20, 30]
STRUCT = np.ones((3, 3), bool)


def boundary_errors(pred, gt, spacing):
    """返回边界误差样本向量（mm）。FP 侧按 GT 带符号距离，FN 侧按到预测的距离。"""
    din = ndi.distance_transform_edt(gt, sampling=spacing)
    dout = ndi.distance_transform_edt(~gt, sampling=spacing)

    pb = pred & ~ndi.binary_erosion(pred, structure=STRUCT)
    e_fp = np.abs(np.where(gt, -din, dout)[pb]) if pb.any() else np.zeros(0)

    gb = gt & ~ndi.binary_erosion(gt, structure=STRUCT)
    unc = gb & ~pred
    if unc.any():
        d_pred = ndi.distance_transform_edt(~pred, sampling=spacing)
        e_fn = d_pred[unc]
    else:
        e_fn = np.zeros(0)

    return np.concatenate([e_fp, e_fn])


def top_q_mean(e, q):
    if e.size == 0:
        return np.nan
    k = max(1, int(round(e.size * q / 100.0)))
    return float(np.sort(e)[-k:].mean())


def run_subset(subset):
    cache = os.path.join(ANA_DIR, 'pred_cache_%s.npz' % subset)
    geom = os.path.join(ANA_DIR, 'pred_geometry_%s.csv' % subset)
    if not (os.path.exists(cache) and os.path.exists(geom)):
        print('[%s] 缺少缓存，跳过' % subset)
        return None

    z = np.load(cache, allow_pickle=True)
    probs, cases = z['probs'], [str(c) for c in z['cases']]
    ref = {}
    with open(geom, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = r['rel'].replace('\\', '/')
            ref[key] = (float(r['hd']), float(r['assd']), float(r['dice']))

    rows = []
    for prob, case in zip(probs, cases):
        key = case.replace('\\', '/')
        if key not in ref:
            continue
        pred224 = prob.astype(np.float32) >= 0.5
        pred_o, gt_o, spacing = original_grid_pair(pred224, ROOT_PATH, case)
        if pred_o.sum() == 0 or gt_o.sum() == 0:
            continue
        e = boundary_errors(pred_o, gt_o, spacing)
        if e.size == 0:
            continue
        rec = {'case': key, 'hd': ref[key][0], 'assd': ref[key][1], 'dice': ref[key][2],
               'n_bnd': int(e.size), 'mean': float(e.mean()), 'max': float(e.max())}
        for q in QS:
            rec['top%d' % q] = top_q_mean(e, q)
        rows.append(rec)

    if not rows:
        print('[%s] 无有效样本' % subset)
        return None

    print('\n===== %s  n=%d =====' % (subset, len(rows)))
    hd = np.array([r['hd'] for r in rows])
    assd = np.array([r['assd'] for r in rows])
    dice = np.array([r['dice'] for r in rows])
    keys = ['mean', 'max'] + ['top%d' % q for q in QS]

    print('%-8s %10s %10s %10s' % ('proxy', 'rho(HD)', 'rho(ASSD)', 'rho(Dice)'))
    for k in keys:
        v = np.array([r[k] for r in rows])
        r1 = stats.spearmanr(v, hd).correlation
        r2 = stats.spearmanr(v, assd).correlation
        r3 = stats.spearmanr(v, dice).correlation
        print('%-8s %10.3f %10.3f %10.3f' % (k, r1, r2, r3))

    # 高 HD 组检出：以代理量排前 25% 与真实 HD 前 25% 的重合率
    k = max(1, len(rows) // 4)
    gt_top = set(np.argsort(-hd)[:k])
    print('\n高 HD 组（前 %d 张）检出率：' % k)
    for kk in ['mean', 'max'] + ['top%d' % q for q in QS]:
        v = np.array([r[kk] for r in rows])
        pred_top = set(np.argsort(-v)[:k])
        print('  %-8s %5.1f%%' % (kk, 100.0 * len(gt_top & pred_top) / k))

    out = os.path.join(ANA_DIR, 'hd_tail_surrogate_%s.csv' % subset)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('csv -> %s' % out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='SPARC', choices=['SPARC', 'Zhongnan', 'both'])
    args = ap.parse_args()
    for s in (['SPARC', 'Zhongnan'] if args.subset == 'both' else [args.subset]):
        run_subset(s)


if __name__ == '__main__':
    main()
