"""误差是否偏向斑块的某一侧（管腔侧=暗 / 外膜侧=亮）。

前一步已确认低 Dice 切片的误差是与 GT 边界平行的窄带，本脚本再看它是否**单侧**：
以 GT 主轴为参照，把预测边界分成「暗侧 / 亮侧」两类，比较两侧的有符号偏移，
以及多检/漏检像素在两侧的分布。不需要模型权重。

用法：
  .\\venv\\Scripts\\python.exe scripts\\analyze_error_side.py
"""
import csv
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from metrics_carotid import original_grid_pair

ROOT_PATH = r'E:\mkl\code\datasets\Carotid2D'
OUT_DIR = os.path.join(BASE_DIR, 'logs', 'analysis')


def normalize(vol):
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    v = np.clip(vol, lo, hi)
    return (v - v.mean()) / max(v.std(), 1e-6)


def load_img(rel):
    img = nib.load(os.path.join(ROOT_PATH, rel)).get_fdata(dtype=np.float32)
    if img.ndim == 3:
        img = img[..., img.shape[-1] // 2]
    return normalize(img)


def signed_dist(gt, spacing):
    din = ndi.distance_transform_edt(gt, sampling=spacing)
    dout = ndi.distance_transform_edt(~gt, sampling=spacing)
    return np.where(gt, -din, dout)


def side_stats(pred, gt, img, spacing):
    ys, xs = np.nonzero(gt)
    c = np.array([ys.mean(), xs.mean()])
    pts = np.stack([ys, xs], 1) - c
    w, v = np.linalg.eigh(np.cov(pts.T))
    u = v[:, 0]                      # 次轴方向（近似斑块法向）
    half_major = 2.0 * np.sqrt(max(w[-1], 1e-6))

    yy, xx = np.mgrid[0:gt.shape[0], 0:gt.shape[1]]
    minor = (yy - c[0]) * u[0] + (xx - c[1]) * u[1]
    major = (yy - c[0]) * -u[1] + (xx - c[1]) * u[0]
    core = np.abs(major) <= 0.8 * half_major   # 去掉两端，只看侧壁

    band = ndi.binary_dilation(gt, iterations=8) & ~ndi.binary_dilation(gt, iterations=3)
    band &= core
    if band.sum() < 20:
        return None
    i_pos = float(img[band & (minor > 0)].mean())
    i_neg = float(img[band & (minor < 0)].mean())
    if abs(i_pos - i_neg) < 0.05:
        return None                       # 两侧亮度不可区分，跳过
    dark_sign = 1.0 if i_pos < i_neg else -1.0

    sd = signed_dist(gt, spacing)
    pb = pred & ~ndi.binary_erosion(pred) & core
    if pb.sum() < 10:
        return None
    off = sd[pb]
    is_dark = (minor[pb] * dark_sign) > 0
    if is_dark.sum() < 5 or (~is_dark).sum() < 5:
        return None

    missed = gt & ~pred & core
    extra = pred & ~gt & core
    frac_extra_dark = float((minor[extra] * dark_sign > 0).mean()) if extra.any() else np.nan
    frac_miss_dark = float((minor[missed] * dark_sign > 0).mean()) if missed.any() else np.nan
    return {
        'off_dark_mm': float(np.median(off[is_dark])),
        'off_bright_mm': float(np.median(off[~is_dark])),
        'extra_frac_dark': frac_extra_dark,
        'miss_frac_dark': frac_miss_dark,
        'extra_px': int(extra.sum()), 'miss_px': int(missed.sum()),
        'i_dark': min(i_pos, i_neg), 'i_bright': max(i_pos, i_neg),
    }


def run(subset):
    cache = np.load(os.path.join(OUT_DIR, 'pred_cache_%s.npz' % subset), allow_pickle=True)
    probs, cases = cache['probs'], cache['cases']
    low = {}
    with open(os.path.join(OUT_DIR, 'error_slices_%s.csv' % subset), newline='',
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            low[r['rel']] = r['low_dice'] == 'True'

    rows = []
    for prob, case in zip(probs, cases):
        pred224 = prob.astype(np.float32) >= 0.5
        pred_o, gt, spacing = original_grid_pair(pred224, ROOT_PATH, case)
        st = side_stats(pred_o, gt, load_img(case), spacing)
        if st is None:
            continue
        st.update({'rel': case, 'low_dice': low.get(case, False)})
        rows.append(st)

    lo = [r for r in rows if r['low_dice']]
    hi = [r for r in rows if not r['low_dice']]
    print('\n=== %s  可判定切片 %d / %d（两侧亮度可区分） ===' % (subset, len(rows), len(cases)))
    for name, rs in (('低Dice组', lo), ('其余', hi)):
        if not rs:
            continue
        print('%s n=%d：暗侧偏移中位 %+.3f mm，亮侧 %+.3f mm；暗侧更外扩的切片占 %.0f%%'
              % (name, len(rs),
                 np.median([r['off_dark_mm'] for r in rs]),
                 np.median([r['off_bright_mm'] for r in rs]),
                 100 * np.mean([r['off_dark_mm'] > r['off_bright_mm'] for r in rs])))
        ex = [r for r in rs if r['extra_px'] > 20]
        if ex:
            print('   多检像素在暗侧的比例：中位 %.2f（n=%d）'
                  % (np.median([r['extra_frac_dark'] for r in ex]), len(ex)))
        ms = [r for r in rs if r['miss_px'] > 20]
        if ms:
            print('   漏检像素在暗侧的比例：中位 %.2f（n=%d）'
                  % (np.median([r['miss_frac_dark'] for r in ms]), len(ms)))

    out_csv = os.path.join(OUT_DIR, 'error_side_%s.csv' % subset)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('csv -> %s' % out_csv)


if __name__ == '__main__':
    for s in ('SPARC', 'Zhongnan'):
        run(s)
