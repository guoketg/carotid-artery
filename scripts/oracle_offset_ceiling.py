"""把已有预测做「等宽内缩/外扩」扫描，量化共性错误里可被修正的比例（上界）。

背景：前面实测发现低 Dice 切片的误差几乎全部是与 GT 边界平行的窄带（多检像素 100%
落在 GT 边界 2 px 内、无孤立连通域），因此先回答一个问题——如果能把轮廓整体沿法向
移动 1~2 px，Dice / HD95 能回来多少？这决定「边界带监督」值不值得做。

不训练、不需要模型权重，只用 logs/analysis/pred_cache_*.npz 与 GT。

用法：
  .\\venv\\Scripts\\python.exe scripts\\oracle_offset_ceiling.py
"""
import csv
import os
import sys

import numpy as np
from scipy import ndimage as ndi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from metrics_carotid import compute_case_metrics, original_grid_pair

ROOT_PATH = r'E:\mkl\code\datasets\Carotid2D'
OUT_DIR = os.path.join(BASE_DIR, 'logs', 'analysis')
KS = [-3, -2, -1, 0, 1, 2, 3]   # 负=内缩（腐蚀），正=外扩（膨胀），单位：原网格像素


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r + 0.5


def shifted(mask, k):
    if k == 0:
        return mask
    out = ndi.binary_erosion(mask, disk(abs(k))) if k < 0 else ndi.binary_dilation(mask, disk(k))
    return out if out.any() else mask


def run(subset):
    cache = np.load(os.path.join(OUT_DIR, 'pred_cache_%s.npz' % subset), allow_pickle=True)
    probs, cases = cache['probs'], cache['cases']
    # 低 Dice 判定沿用 error_slices csv 的分位数
    low = {}
    csv_path = os.path.join(OUT_DIR, 'error_slices_%s.csv' % subset)
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            low[r['rel']] = r['low_dice'] == 'True'

    recs = []
    for prob, case in zip(probs, cases):
        pred224 = prob.astype(np.float32) >= 0.5
        pred_o, gt, spacing = original_grid_pair(pred224, ROOT_PATH, case)
        row = {'rel': case, 'low_dice': low.get(case, False)}
        for k in KS:
            m = compute_case_metrics(shifted(pred_o, k), gt, spacing=spacing)
            row['dice_k%+d' % k] = m['dice']
            row['hd95_k%+d' % k] = m['hd95']
        row['best_k'] = max(KS, key=lambda k: row['dice_k%+d' % k])
        row['gain'] = row['dice_k%+d' % row['best_k']] - row['dice_k+0']
        row['hd_gain'] = row['hd95_k+0'] - row['hd95_k%+d' % row['best_k']]
        recs.append(row)

    n = len(recs)
    d0 = np.array([r['dice_k+0'] for r in recs])
    gain = np.array([r['gain'] for r in recs])
    lo = np.array([r['low_dice'] for r in recs], bool)
    print('\n=== %s  n=%d  基线dice=%.4f ===' % (subset, n, d0.mean()))
    print('-- 每张切片独立选最优内缩/外扩（oracle 上界）')
    print('最优 k 分布：' + '  '.join('%+d:%d' % (k, sum(r['best_k'] == k for r in recs)) for k in KS))
    print('dice 平均提升 +%.4f（全体）  +%.4f（低Dice组 %d 张）'
          % (gain.mean(), gain[lo].mean(), lo.sum()))
    print('低Dice组：%.0f%% 的切片最优 k!=0，中位提升 +%.4f'
          % (100 * np.mean([r['best_k'] != 0 for r in recs if r['low_dice']]),
             np.median([r['gain'] for r in recs if r['low_dice']])))
    print('HD95 平均下降 %.4f mm（全体）  %.4f mm（低Dice组）'
          % (np.mean([r['hd_gain'] for r in recs]),
             np.mean([r['hd_gain'] for r in recs if r['low_dice']])))

    print('\n-- 全局统一 k（部署时不能按切片用 GT 选 k，只能固定一个偏移量）')
    print('%-4s%10s%10s%10s%10s' % ('k', 'dice_all', 'Δall', 'dice_low', 'Δlow'))
    for k in KS:
        d = np.array([r['dice_k%+d' % k] for r in recs])
        print('%-4d%10.4f%10.4f%10.4f%10.4f'
              % (k, d.mean(), d.mean() - d0.mean(), d[lo].mean(), d[lo].mean() - d0[lo].mean()))

    out_csv = os.path.join(OUT_DIR, 'oracle_offset_%s.csv' % subset)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print('csv -> %s' % out_csv)


if __name__ == '__main__':
    for s in ('SPARC', 'Zhongnan'):
        run(s)
