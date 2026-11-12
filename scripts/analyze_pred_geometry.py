"""预测几何诊断：在 CPU 上复现两个基线快照的 test 预测，量化「错在哪里」。

对每张切片算误差的空间形态（不是只看 dice）：
  - 边界整体偏移（mm，正=预测偏外）
  - 漏检/多检是「贴边细带」还是「深处团块」
  - 误差是否集中在斑块两端（长轴方向）
  - 预测的连通域数量（孤立碎片）
并在同一批预测上离线试几种测试时后处理，看哪种能修正该误差。

不训练、不占 GPU。输出 logs/analysis/pred_geometry_<subset>.csv 与 npz 缓存。

用法（先小样本确认速度与 dice 复现）：
  .\\venv\\Scripts\\python.exe scripts\\analyze_pred_geometry.py --subset SPARC --limit 4
  .\\venv\\Scripts\\python.exe scripts\\analyze_pred_geometry.py --subset both
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.dataset_carotid import CarotidDataset, LIST_DIR
from metrics_carotid import compute_case_metrics, original_grid_pair
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg

ROOT_PATH = r'E:\mkl\code\datasets\Carotid2D'
MODEL_ROOT = r'E:\mkl\code\python_code\model\TU_Carotid224'
SNAPSHOT = {
    'SPARC': os.path.join(MODEL_ROOT,
                          'TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d'),
    'Zhongnan': os.path.join(MODEL_ROOT,
                             'TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_c2d'),
}
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'logs', 'analysis')


def build_model(vit_layers=0, img_size=224):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = 2
    cfg.n_skip = 3
    cfg.transformer.num_layers = vit_layers
    cfg.patches.grid = (int(img_size / 16), int(img_size / 16))
    return ViT_seg(cfg, img_size=img_size, num_classes=2, edge_in=False, edge_op='wavelet',
                   wavelet_basis='haar', ddbe_level=0, acam=False, hf_branch=False,
                   hf_lambda=0.3, bdfr_v2=False, bdfr_ddbe=True, bdfr_acam=True,
                   bdfr_wave_op='haar', ffusion=False)


def load_net(subset):
    net = build_model()
    ckpt = os.path.join(SNAPSHOT[subset], 'epoch_199.pth')
    state = torch.load(ckpt, map_location='cpu')
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    return net


def signed_dist(gt, spacing):
    """到 GT 边界的带符号距离（mm）：外正内负"""
    din = ndi.distance_transform_edt(gt, sampling=spacing)
    dout = ndi.distance_transform_edt(~gt, sampling=spacing)
    return np.where(gt, -din, dout)


def geometry(pred, gt, spacing):
    px = float(np.mean(spacing))
    sd = signed_dist(gt, spacing)
    pb = pred & ~ndi.binary_erosion(pred)
    offs = sd[pb] if pb.any() else np.array([0.0])
    missed = gt & ~pred
    extra = pred & ~gt
    d_miss = ndi.distance_transform_edt(~pred, sampling=spacing)[missed] if missed.any() else np.array([0.0])
    d_extra = ndi.distance_transform_edt(gt, sampling=spacing)[extra] if extra.any() else np.array([0.0])

    lab, n = ndi.label(pred, structure=np.ones((3, 3)))
    sizes = ndi.sum(pred, lab, range(1, n + 1)) if n else np.array([0])
    n_cc = int((sizes >= 20).sum())

    # 沿长轴的归一化位置 t in [-1,1]，看误差是否集中在两端
    ys, xs = np.nonzero(gt)
    pts = np.stack([ys - ys.mean(), xs - xs.mean()], 1).astype(float)
    w, v = np.linalg.eigh(np.cov(pts.T))
    axis = v[:, -1]
    half = 2.0 * np.sqrt(max(w[-1], 1e-6))
    def end_frac(mask):
        if not mask.any():
            return 0.0
        yy, xx = np.nonzero(mask)
        t = np.abs((np.stack([yy - ys.mean(), xx - xs.mean()], 1) @ axis) / max(half, 1e-6))
        return float((t > 0.7).mean())

    area = float(gt.sum())
    return {
        'boundary_offset_mm': float(np.median(offs)),
        'abs_offset_mm': float(np.median(np.abs(offs))),
        'out_frac': float((offs > 0).mean()),
        'miss_band_frac': float((d_miss <= 2 * px).mean()) if missed.any() else np.nan,
        'miss_depth_mm': float(np.median(d_miss)) if missed.any() else 0.0,
        'extra_band_frac': float((d_extra <= 2 * px).mean()) if extra.any() else np.nan,
        'extra_depth_mm': float(np.median(d_extra)) if extra.any() else 0.0,
        'miss_frac': float(missed.sum() / area) if area else np.nan,
        'extra_frac': float(extra.sum() / area) if area else np.nan,
        'miss_end_frac': end_frac(missed),
        'extra_end_frac': end_frac(extra),
        'n_cc': n_cc,
    }


def largest_cc(mask):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def postprocs(prob224, pred224, out_path=None):
    """返回 {方法: 224 网格二值预测}，供离线比较（GT 只在重采样后使用）"""
    p = prob224 >= 0.5
    disk1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    out = {
        'raw': p,
        'largest_cc': largest_cc(p),
        'open1': ndi.binary_opening(p, structure=disk1),
        'close1': ndi.binary_closing(p, structure=disk1),
        'thr0.4': prob224 >= 0.4,
        'thr0.6': prob224 >= 0.6,
    }
    return out


def run_subset(subset, limit, top_dump=10):
    net = load_net(subset)
    ds = CarotidDataset(root=ROOT_PATH,
                        list_file=os.path.join(LIST_DIR, subset, 'test.txt'),
                        split='test', img_size=224, use_wavelet=False)
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    rows, probs, cases = [], [], []
    t0 = time.time()
    methods = ['raw', 'largest_cc', 'open1', 'close1', 'thr0.4', 'thr0.6']
    per_method = {m: [] for m in methods}
    per_method_low = {m: [] for m in methods}
    with torch.no_grad():
        for i in range(n):
            s = ds[i]
            logits = net(s['image'].unsqueeze(0))
            logits = logits['out'] if isinstance(logits, dict) else (
                logits[0] if isinstance(logits, (tuple, list)) else logits)
            prob = torch.softmax(logits, 1)[0, 1].numpy().astype(np.float16)
            pred224 = np.argmax(logits.numpy()[0], 0) > 0
            pred_o, gt_o, spacing = original_grid_pair(pred224, ROOT_PATH, s['case'])
            m = compute_case_metrics(pred_o, gt_o, spacing=spacing)
            rec = {'subset': subset, 'rel': s['case'], 'spacing_mm': float(np.mean(spacing))}
            rec.update(m)
            rec.update(geometry(pred_o, gt_o, spacing))
            rows.append(rec)
            probs.append(prob)
            cases.append(s['case'])

            for name, p224 in postprocs(prob.astype(np.float32), pred224).items():
                po, _, _ = original_grid_pair(p224, ROOT_PATH, s['case'])
                per_method[name].append(compute_case_metrics(po, gt_o, spacing=spacing)['dice'])
            if (i + 1) % 20 == 0:
                print('  %d/%d  %.1fs' % (i + 1, n, time.time() - t0), file=sys.stderr)

    # 与 metrics.csv 对齐，校验复现
    ref = {}
    mp = os.path.join(SNAPSHOT[subset], 'metrics.csv')
    with open(mp, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rel = r['case'].replace('\\', '/')
            ref[rel[2:] if rel.startswith('./') else rel] = float(r['dice'])
    diffs = [abs(r['dice'] - ref[r['rel']]) for r in rows if r['rel'] in ref]
    print('\n[%s] n=%d  dice复现最大偏差=%.5f  用时 %.1fs'
          % (subset, len(rows), max(diffs) if diffs else -1, time.time() - t0))

    dice = np.array([r['dice'] for r in rows])
    q25 = np.percentile(dice, 25)
    for r, d in zip(rows, dice):
        r['low_dice'] = d <= q25
    low_idx = [i for i, r in enumerate(rows) if r['low_dice']]

    print('dice mean=%.4f | 低Dice组 %d 张' % (dice.mean(), len(low_idx)))
    cols = ['boundary_offset_mm', 'abs_offset_mm', 'out_frac', 'miss_frac', 'extra_frac',
            'miss_band_frac', 'miss_depth_mm', 'extra_band_frac', 'extra_depth_mm',
            'miss_end_frac', 'extra_end_frac', 'n_cc']
    print('\n-- 误差几何：低Dice组 vs 其余（中位数，Mann-Whitney p）')
    print('%-20s%10s%10s%10s' % ('stat', 'low', 'rest', 'p'))
    for c in cols:
        va = np.array([rows[i][c] for i in low_idx], float)
        vb = np.array([rows[i][c] for i in range(len(rows)) if i not in set(low_idx)], float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if len(va) < 3 or len(vb) < 3:
            continue
        p = stats.mannwhitneyu(va, vb, alternative='two-sided').pvalue
        print('%-20s%10.3f%10.3f%10.4f' % (c, np.median(va), np.median(vb), p))

    print('\n-- 测试时后处理（平均 Dice，整库 / 低Dice组）')
    print('%-12s%12s%14s%10s' % ('method', 'dice_all', 'dice_low', 'n_drop'))
    base_all = np.mean(per_method['raw'])
    base_low = np.mean([per_method['raw'][i] for i in low_idx])
    for mn in methods:
        d = np.array(per_method[mn])
        dl = np.mean([d[i] for i in low_idx]) if low_idx else np.nan
        print('%-12s%12.4f%14.4f%10d' % (mn, d.mean(), dl,
                                         int((d < np.array(per_method['raw']) - 1e-9).sum())))
    print('（raw 为基线：all=%.4f low=%.4f）' % (base_all, base_low))

    def dump(title, sel):
        print('\n-- %s' % title)
        print('%-6s%7s%9s%8s%8s%8s%9s%8s%8s%6s' %
              ('dice', 'hd95', 'off_mm', 'missF', 'extraF', 'mBand', 'mDepth', 'eBand',
               'mEndF', 'nCC'))
        for r in sel:
            print('%-6.3f%7.2f%9.2f%8.3f%8.3f%8.2f%9.2f%8.2f%8.2f%6d'
                  % (r['dice'], r['hd95'], r['boundary_offset_mm'], r['miss_frac'],
                     r['extra_frac'], np.nan_to_num(r['miss_band_frac']),
                     r['miss_depth_mm'], np.nan_to_num(r['extra_band_frac']),
                     r['miss_end_frac'], r['n_cc']))
            print('       %s' % r['rel'])
    dump('最低 Dice %d 张' % top_dump, sorted(rows, key=lambda r: r['dice'])[:top_dump])
    dump('最高 HD95 %d 张' % top_dump, sorted(rows, key=lambda r: -r['hd95'])[:top_dump])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, 'pred_geometry_%s.csv' % subset)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(OUT_DIR, 'pred_cache_%s.npz' % subset),
                        probs=np.stack(probs), cases=np.array(cases))
    print('\ncsv -> %s' % out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='SPARC', choices=['SPARC', 'Zhongnan', 'both'])
    ap.add_argument('--limit', type=int, default=0, help='0=全部')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    subs = ['SPARC', 'Zhongnan'] if args.subset == 'both' else [args.subset]
    for s in subs:
        run_subset(s, args.limit)


if __name__ == '__main__':
    main()
