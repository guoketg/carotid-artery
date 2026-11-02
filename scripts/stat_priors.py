"""斑块先验的零成本统计验证（不训练、不占卡）。

两个候选先验，只读 GT 与原始图像就能判生死：
  1) 双边界平行性：斑块沿主轴切开后，上下两条边界的间距是否近似恒定。
     —— 决定「平行曲线/管壁连续性」正则是否值得写代码。
  2) 后方声影/衰减：斑块深侧（图像下方）的亮度是否显著低于同深度侧方参考区。
     —— 决定「声影感知」是否值得写代码；上方同带宽作为对照。

输出：logs/analysis/prior_check.csv（逐切片）+ stdout 汇总分位数。

Usage:
  .\\venv\\Scripts\\python.exe scripts\\stat_priors.py --subset SPARC --split train --limit 350
"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = r'E:\mkl\code\datasets\Carotid2D'
LIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'lists', 'lists_Carotid2D')


def load_pair(rel):
    img_path = os.path.join(ROOT, rel)
    img = nib.load(img_path).get_fdata(dtype=np.float32)
    gt = nib.load(img_path.replace('.nii.gz', '_label.nii.gz')).get_fdata(dtype=np.float32)
    if img.ndim == 3:
        img = img[..., img.shape[-1] // 2]
    if gt.ndim == 3:
        gt = gt[..., gt.shape[-1] // 2]
    return img, gt > 0.5


def boundary_parallelism(mask, n_bins=64, min_pix=4):
    """沿主轴切分，量上下边界的间距稳定性与两侧边界法向夹角。

    返回 (cv_thickness, mean_abs_d_thickness_norm, mean_normal_angle_deg)；
    无法计算时返回 None。
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, np.argmax(evals)]          # 主轴（沿斑块长边）
    u = evecs[:, np.argmin(evals)]          # 次轴（垂直长边）
    s = (pts - c) @ v
    t = (pts - c) @ u
    lo, hi = s.min(), s.max()
    if hi - lo < n_bins * 1.5:              # 主轴方向太短，分箱不可靠
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    t_top, t_bot, s_mid = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < min_pix:
            continue
        t_top.append(t[sel].min())
        t_bot.append(t[sel].max())
        s_mid.append(0.5 * (edges[b] + edges[b + 1]))
    if len(t_top) < n_bins // 2:
        return None
    t_top = ndi.median_filter(np.asarray(t_top), size=3, mode='nearest')
    t_bot = ndi.median_filter(np.asarray(t_bot), size=3, mode='nearest')
    thick = t_bot - t_top
    if thick.mean() <= 0:
        return None
    cv = float(thick.std() / thick.mean())
    dth = float(np.abs(np.diff(thick)).mean() / thick.mean())
    # 两侧边界的法向角：n = atan(dt/ds)，比较上下两条曲线
    ds = np.diff(np.asarray(s_mid))
    ds[ds <= 0] = 1.0
    ang_top = np.degrees(np.arctan(np.diff(t_top) / ds))
    ang_bot = np.degrees(np.arctan(np.diff(t_bot) / ds))
    ang_diff = float(np.abs(ang_top - ang_bot).mean())
    return cv, dth, ang_diff, float(thick.mean())


def band_mean(img, rows, cols):
    rows = np.clip(rows, 0, img.shape[0] - 1)
    cols = np.clip(cols, 0, img.shape[1] - 1)
    if len(rows) == 0 or len(cols) == 0:
        return None
    patch = img[np.ix_(rows, cols)]
    if patch.size == 0 or not np.isfinite(patch).all():
        return None
    return float(patch.mean())


def shadow_ratio(img, mask, band_h=24, gap=3, side_pad=8):
    """返回 (下方比, 上方比)。比 = 斑块带均值 / 同深度侧方参考带均值。

    下方 = 行号更大的一侧（超声探头在上，深部在下）。
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = img.shape
    cols = np.arange(x0, x1 + 1)
    out = []
    for direction in (1, -1):               # +1 下方、-1 上方
        if direction > 0:
            r_start = y1 + gap
        else:
            r_start = y0 - gap - band_h
        rows = np.arange(r_start, r_start + band_h)
        if rows.min() < 0 or rows.max() >= h:
            out.append(np.nan)
            continue
        tgt = band_mean(img, rows, cols)
        # 参考带：同深度、斑块列范围左右各外扩
        ref_cols = np.concatenate([
            np.arange(max(0, x0 - side_pad - len(cols)), max(0, x0 - side_pad)),
            np.arange(min(w, x1 + side_pad + 1), min(w, x1 + side_pad + 1 + len(cols)))])
        ref = band_mean(img, rows, ref_cols)
        if tgt is None or ref is None or abs(ref) < 1e-6:
            out.append(np.nan)
            continue
        out.append(tgt / ref)
    return out[0], out[1]


def q(vals, p):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return float('nan')
    return float(np.percentile(vals, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='SPARC')
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=0, help='0 = 全部')
    args = ap.parse_args()

    lst = os.path.join(LIST_DIR, args.subset, args.split + '.txt')
    with open(lst, 'r', encoding='utf-8') as f:
        rels = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        rels = rels[:args.limit]

    recs = []
    for i, rel in enumerate(rels):
        try:
            img, gt = load_pair(rel)
        except Exception as e:                      # 单例失败不影响整体统计
            print('skip %s (%s)' % (rel, e), file=sys.stderr)
            continue
        par = boundary_parallelism(gt)
        sh = shadow_ratio(img, gt)
        recs.append({
            'rel': rel,
            'cv_thickness': par[0] if par else np.nan,
            'd_thickness_norm': par[1] if par else np.nan,
            'normal_angle_deg': par[2] if par else np.nan,
            'thickness_px': par[3] if par else np.nan,
            'shadow_below': sh[0] if sh else np.nan,
            'shadow_above': sh[1] if sh else np.nan,
        })
        if (i + 1) % 50 == 0:
            print('  %d/%d' % (i + 1, len(rels)), file=sys.stderr)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'logs', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'prior_check_%s_%s.csv' % (args.subset, args.split))
    keys = list(recs[0].keys())
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)

    n = len(recs)
    print('\n=== %s/%s  n=%d ===' % (args.subset, args.split, n))
    for k in keys[1:]:
        vals = [r[k] for r in recs]
        print('%-18s p25=%.3f  p50=%.3f  p75=%.3f  (valid=%d)'
              % (k, q(vals, 25), q(vals, 50), q(vals, 75),
                 sum(1 for v in vals if np.isfinite(v))))
    below = [r['shadow_below'] for r in recs]
    above = [r['shadow_above'] for r in recs]
    diffs = [r['shadow_below'] - r['shadow_above'] for r in recs
             if np.isfinite(r['shadow_below']) and np.isfinite(r['shadow_above'])]
    if diffs:
        d = np.asarray(diffs)
        print('shadow below-minus-above: p25=%.3f p50=%.3f p75=%.3f  frac(neg)=%.2f'
              % (np.percentile(d, 25), np.percentile(d, 50), np.percentile(d, 75),
                 float((d < 0).mean())))
    print('csv -> %s' % out_csv)


if __name__ == '__main__':
    main()
