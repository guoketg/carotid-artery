"""逐切片误差归因：把基线的 dice / HD 与斑块自身的图像、形状特征对齐，找共性。

只读原图、GT 与已有 metrics.csv，不训练、不占 GPU。输出：
  logs/analysis/error_slices_<subset>.csv   逐切片特征 + 指标
  stdout                                   分位数、Spearman 相关、分组对比、最差样本/病例清单

用法：
  .\\venv\\Scripts\\python.exe scripts\\analyze_error_slices.py --subset SPARC
  .\\venv\\Scripts\\python.exe scripts\\analyze_error_slices.py --subset Zhongnan
  .\\venv\\Scripts\\python.exe scripts\\analyze_error_slices.py --subset both   # 两库合并，去库间差异
"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy import stats

ROOT = r'E:\mkl\code\datasets\Carotid2D'
MODEL_ROOT = r'E:\mkl\code\python_code\model\TU_Carotid224'
METRICS_CSV = {
    'SPARC': os.path.join(
        MODEL_ROOT, 'TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d',
        'metrics.csv'),
    'Zhongnan': os.path.join(
        MODEL_ROOT, 'TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_c2d',
        'metrics.csv'),
}
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'logs', 'analysis')
FEATS = ['gt_area_mm2', 'area_frac', 'major_px', 'minor_px', 'minor_mm', 'elongation',
         'depth', 'fg_mean', 'rel_echo', 'cnr', 'fg_std', 'hyper_frac',
         'bnd_grad_ratio', 'pred_ratio', 'hd95_rel', 'hd_hd95_ratio']


def norm_rel(s):
    s = s.strip().replace('\\', '/')
    return s[2:] if s.startswith('./') else s


def normalize_image(vol):
    """与 datasets/dataset_carotid.normalize_image 完全一致（1%~99% 截断 + z-score）"""
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    vol = np.clip(vol, lo, hi)
    mean, std = vol.mean(), vol.std()
    if std < 1e-6:
        std = 1.0
    return (vol - mean) / std


def load_pair(rel):
    img_path = os.path.join(ROOT, rel)
    img = nib.load(img_path).get_fdata(dtype=np.float32)
    gt = nib.load(img_path.replace('.nii.gz', '_label.nii.gz')).get_fdata(dtype=np.float32)
    if img.ndim == 3:
        img = img[..., img.shape[-1] // 2]
    if gt.ndim == 3:
        gt = gt[..., gt.shape[-1] // 2]
    return img, gt > 0.5, nib.load(img_path).header.get_zooms()[:2]


def slice_features(img_raw, gt, spacing):
    """形状 + 回声 + 边界清晰度，全部在原分辨率网格上算（与指标同网格）"""
    out = {k: np.nan for k in [
        'gt_area_px', 'gt_area_mm2', 'area_frac', 'major_px', 'minor_px',
        'elongation', 'minor_mm', 'depth', 'fg_mean', 'ring_mean', 'rel_echo',
        'cnr', 'fg_std', 'hyper_frac', 'bnd_grad_ratio']}
    n_fg = int(gt.sum())
    out['gt_area_px'] = n_fg
    if n_fg == 0:
        return out

    img = normalize_image(img_raw)
    h, w = img.shape
    sy, sx = float(spacing[0]), float(spacing[1])
    out['gt_area_mm2'] = n_fg * sy * sx
    out['area_frac'] = n_fg / float(h * w)

    ys, xs = np.nonzero(gt)
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    evals = np.linalg.eigvalsh(np.cov((pts - c).T))
    evals = np.clip(evals, 1e-6, None)
    major, minor = 2.0 * np.sqrt(evals[1]), 2.0 * np.sqrt(evals[0])
    out['major_px'], out['minor_px'] = float(major), float(minor)
    out['elongation'] = float(major / minor)
    out['minor_mm'] = float(minor * min(sy, sx))
    out['depth'] = float(ys.mean() / h)

    fg = img[gt]
    ring = ndi.binary_dilation(gt, iterations=6) & ~gt
    if ring.sum() > 0:
        ref = img[ring]
        out['fg_mean'] = float(fg.mean())
        out['fg_std'] = float(fg.std())
        out['ring_mean'] = float(ref.mean())
        out['rel_echo'] = float(fg.mean() - ref.mean())      # >0 高回声，<0 低回声
        pooled = np.sqrt(0.5 * (fg.var() + ref.var()))
        out['cnr'] = float(abs(fg.mean() - ref.mean()) / max(pooled, 1e-6))
    out['hyper_frac'] = float((fg > 1.0).mean())             # z>1 视为高回声/钙化候选

    gy, gx = np.gradient(img)
    grad = np.hypot(gy, gx)
    bnd = gt & ~ndi.binary_erosion(gt)
    if bnd.sum() > 0 and grad.mean() > 1e-8:
        out['bnd_grad_ratio'] = float(grad[bnd].mean() / grad.mean())
    return out


def read_metrics(path):
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rec = {k: float(r[k]) for k in ['dice', 'assd', 'hd95', 'hd', 'delta_area']}
            rows[norm_rel(r['case'])] = rec
    return rows


def collect(subset):
    metrics = read_metrics(METRICS_CSV[subset])
    print('[%s] metrics.csv 切片数 = %d' % (subset, len(metrics)))
    rows = []
    for i, (rel, m) in enumerate(metrics.items()):
        try:
            img, gt, spacing = load_pair(rel)
        except Exception as e:
            print('skip %s (%s)' % (rel, e), file=sys.stderr)
            continue
        rec = {'subset': subset, 'rel': rel, 'patient': rel.split('/')[1],
               'spacing_y': float(spacing[0]), 'spacing_x': float(spacing[1])}
        rec.update(m)
        rec.update(slice_features(img, gt, spacing))
        area = rec['gt_area_mm2']
        rec['pred_ratio'] = (area + rec['delta_area']) / area if area and np.isfinite(area) else np.nan
        # 相对距离：HD95 除以等效直径（mm），去掉斑块大小的量纲影响
        diam = 2.0 * np.sqrt(max(area, 1e-6) / np.pi) if np.isfinite(area) else np.nan
        rec['hd95_rel'] = rec['hd95'] / diam if diam and np.isfinite(diam) else np.nan
        # 孤立远端误检/漏检会让 HD(max) 远大于 HD95
        rec['hd_hd95_ratio'] = rec['hd'] / max(rec['hd95'], 1e-3)
        rec['low_dice'] = False
        rec['high_hd95'] = False
        rows.append(rec)
        if (i + 1) % 40 == 0:
            print('  %d/%d' % (i + 1, len(metrics)), file=sys.stderr)

    dc = np.array([r['dice'] for r in rows], float)
    hd95 = np.array([r['hd95'] for r in rows], float)
    q25, q75 = np.percentile(dc, 25), np.percentile(hd95, 75)
    for r in rows:
        r['low_dice'] = r['dice'] <= q25
        r['high_hd95'] = r['hd95'] >= q75
    return rows


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[ok], y[ok])
    return float(rho), float(p)


def partial_spearman(x, y, z):
    """去掉 z（log 面积）的秩相关：对秩做线性回归后取残差再相关"""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 12:
        return np.nan, np.nan
    rx, ry, rz = (stats.rankdata(v[ok]) for v in (x, y, z))
    A = np.stack([np.ones_like(rz), rz], axis=1)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    rho, p = stats.spearmanr(ex, ey)
    return float(rho), float(p)


def group_compare(rows, key, flag, label):
    print('\n-- %s：%s 组 vs 其余（中位数，Mann-Whitney p）' % (label, key))
    a = [r for r in rows if r[flag]]
    b = [r for r in rows if not r[flag]]
    print('%-16s%10s%10s%10s' % ('feature', key, 'rest', 'p'))
    for f in FEATS:
        va = np.array([r[f] for r in a], float)
        vb = np.array([r[f] for r in b], float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if len(va) < 5 or len(vb) < 5:
            continue
        try:
            p = stats.mannwhitneyu(va, vb, alternative='two-sided').pvalue
        except ValueError:
            p = np.nan
        print('%-16s%10.3f%10.3f%10.4f' % (f, np.median(va), np.median(vb), p))
    over = np.mean([r['pred_ratio'] > 1.05 for r in a if np.isfinite(r['pred_ratio'])])
    under = np.mean([r['pred_ratio'] < 0.95 for r in a if np.isfinite(r['pred_ratio'])])
    print('组内 pred_ratio>1.05（过分割）%.0f%%，<0.95（欠分割）%.0f%%' % (over * 100, under * 100))


def report(rows, label, top):
    dice = np.array([r['dice'] for r in rows], float)
    hd95 = np.array([r['hd95'] for r in rows], float)
    hd = np.array([r['hd'] for r in rows], float)
    print('\n=== %s  n=%d ===' % (label, len(rows)))
    print('dice mean=%.4f  p10=%.4f  p25=%.4f' % (dice.mean(), *np.percentile(dice, [10, 25])))
    print('hd95 mean=%.3f  p75=%.3f  p90=%.3f' % (hd95.mean(), *np.percentile(hd95, [75, 90])))
    print('hd   mean=%.3f  p75=%.3f  p90=%.3f' % (hd.mean(), *np.percentile(hd, [75, 90])))
    print('spearman(dice, hd95)=%.3f' % stats.spearmanr(dice, hd95)[0])
    print('HD/HD95>5 的切片 %d 张（孤立远端误检/漏检的特征）'
          % sum(r['hd_hd95_ratio'] > 5 for r in rows))

    log_area = np.log(np.array([r['gt_area_mm2'] for r in rows], float) + 1e-6)
    print('\n-- Spearman（dice / hd95 / 去面积偏相关）')
    print('%-16s%18s%18s%20s' % ('feature', 'rho(dice)', 'rho(hd95)', 'rho(dice)|area'))
    for f in FEATS:
        v = [r[f] for r in rows]
        rd, pd = spearman(v, dice)
        rh, ph = spearman(v, hd95)
        rp, pp = partial_spearman(v, dice, log_area)
        print('%-16s%8.3f(p=%.2f)%8.3f(p=%.2f)%9.3f(p=%.2f)'
              % (f, rd, pd, rh, ph, rp, pp))

    both = [r for r in rows if r['low_dice'] and r['high_hd95']]
    print('\n低Dice(<=p25) %d 张，高HD95(>=p75) %d 张，两者同时 %d 张'
          % (sum(r['low_dice'] for r in rows), sum(r['high_hd95'] for r in rows), len(both)))
    group_compare(rows, '低Dice', 'low_dice', label)
    group_compare(rows, '高HD95', 'high_hd95', label)

    # 病例级：误差是否集中在少数病人身上（同一斑块连续切片）
    groups = {}
    for r in rows:
        groups.setdefault(r['patient'], []).append(r)
    pats = sorted(groups.items(),
                  key=lambda kv: np.mean([x['dice'] for x in kv[1]]))[:8]
    print('\n-- 最差 8 例（病例级平均 Dice）')
    print('%-14s%7s%7s%8s%9s%9s' % ('patient', 'n', 'dice', 'hd95', 'relEcho', 'elong'))
    for pid, rs in pats:
        print('%-14s%7d%7.3f%8.2f%9.2f%9.2f'
              % (pid, len(rs), np.mean([x['dice'] for x in rs]),
                 np.mean([x['hd95'] for x in rs]),
                 np.nanmean([x['rel_echo'] for x in rs]),
                 np.nanmean([x['elongation'] for x in rs])))

    def dump(title, sel):
        print('\n-- %s' % title)
        print('%-6s%6s%7s%8s%10s%8s%8s%8s%8s%8s%8s' %
              ('dice', 'hd95', 'hd', 'mm2', 'elong', 'minor', 'relEcho', 'cnr', 'hyper', 'hd/hd95', 'predR'))
        for r in sel:
            print('%-6.3f%6.2f%7.2f%8.0f%10.2f%8.1f%8.2f%8.2f%8.2f%8.2f%8.2f'
                  % (r['dice'], r['hd95'], r['hd'], r['gt_area_mm2'], r['elongation'],
                     r['minor_mm'], r['rel_echo'], r['cnr'], r['hyper_frac'],
                     r['hd_hd95_ratio'], r['pred_ratio']))
            print('       %s' % r['rel'])

    dump('最低 Dice %d 张' % top, sorted(rows, key=lambda r: r['dice'])[:top])
    dump('最高 HD95 %d 张' % top, sorted(rows, key=lambda r: -r['hd95'])[:top])


def pooled(rows_all):
    """两库合并：每库内部转 z-score，消除库间难度/尺度差异后再看共性"""
    print('\n=== 合并分析（每库内 z-score） ===')
    zrows = []
    for sub in ('SPARC', 'Zhongnan'):
        rs = [r for r in rows_all if r['subset'] == sub]
        for key in ['dice', 'hd95', 'rel_echo', 'elongation', 'minor_mm', 'cnr',
                    'gt_area_mm2', 'fg_std', 'hyper_frac', 'hd95_rel', 'hd_hd95_ratio']:
            v = np.array([r[key] for r in rs], float)
            mu, sd = v.mean(), v.std() if v.std() > 1e-9 else 1.0
            for r in rs:
                r['z_' + key] = (r[key] - mu) / sd
        zrows.extend(rs)

    print('%-16s%18s%18s' % ('feature', 'rho(dice)', 'rho(hd95)'))
    for f in ['rel_echo', 'elongation', 'minor_mm', 'cnr', 'gt_area_mm2',
              'fg_std', 'hyper_frac', 'hd95_rel', 'hd_hd95_ratio']:
        v = [r['z_' + f] for r in zrows]
        rd, pd = spearman(v, [r['dice'] for r in zrows])
        rh, ph = spearman(v, [r['hd95'] for r in zrows])
        print('%-16s%8.3f(p=%.2f)%8.3f(p=%.2f)' % (f, rd, pd, rh, ph))

    low = [r for r in zrows if r['low_dice']]
    print('\n-- 合并后低Dice组 vs 其余（z 值中位数，Mann-Whitney p）')
    print('%-16s%10s%10s%10s' % ('feature', 'low', 'rest', 'p'))
    for f in ['rel_echo', 'elongation', 'minor_mm', 'cnr', 'gt_area_mm2',
              'fg_std', 'hyper_frac', 'hd95_rel', 'hd_hd95_ratio']:
        va = np.array([r['z_' + f] for r in low], float)
        vb = np.array([r['z_' + f] for r in zrows if not r['low_dice']], float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        p = stats.mannwhitneyu(va, vb, alternative='two-sided').pvalue
        print('%-16s%10.2f%10.2f%10.4f' % (f, np.median(va), np.median(vb), p))

    hi = [r for r in zrows if r['high_hd95']]
    print('\n-- 合并后高HD95组 vs 其余（z 值中位数，Mann-Whitney p）')
    print('%-16s%10s%10s%10s' % ('feature', 'high', 'rest', 'p'))
    for f in ['rel_echo', 'elongation', 'minor_mm', 'cnr', 'gt_area_mm2',
              'fg_std', 'hyper_frac', 'hd95_rel', 'hd_hd95_ratio']:
        va = np.array([r['z_' + f] for r in hi], float)
        vb = np.array([r['z_' + f] for r in zrows if not r['high_hd95']], float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        p = stats.mannwhitneyu(va, vb, alternative='two-sided').pvalue
        print('%-16s%10.2f%10.2f%10.4f' % (f, np.median(va), np.median(vb), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='SPARC', choices=['SPARC', 'Zhongnan', 'both'])
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args()

    subs = ['SPARC', 'Zhongnan'] if args.subset == 'both' else [args.subset]
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    for sub in subs:
        rows = collect(sub)
        report(rows, sub, args.top)
        out_csv = os.path.join(OUT_DIR, 'error_slices_%s.csv' % sub)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print('\ncsv -> %s' % out_csv)
        all_rows.extend(rows)
    if args.subset == 'both':
        pooled(all_rows)


if __name__ == '__main__':
    main()
