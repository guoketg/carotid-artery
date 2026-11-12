"""边界抖动假设的验证：翻转 TTA 平均能不能把低 Dice 切片的误差压下去。

若误差确实来自「边界定位抖动」，把 4 种翻转的预测概率平均应主要改善低 Dice 组；
若几乎不动，说明剩下的不是抖动，而是系统性形状/位置差异，后处理方向可以彻底排除。

只做前向，CPU，不训练。用法：
  .\\venv\\Scripts\\python.exe scripts\\analyze_tta_boundary.py
"""
import csv
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.dataset_carotid import CarotidDataset, LIST_DIR
from metrics_carotid import compute_case_metrics, original_grid_pair
from scripts.analyze_pred_geometry import load_net

ROOT_PATH = r'E:\mkl\code\datasets\Carotid2D'
OUT_DIR = os.path.join(BASE_DIR, 'logs', 'analysis')


def run(subset):
    net = load_net(subset)
    ds = CarotidDataset(root=ROOT_PATH,
                        list_file=os.path.join(LIST_DIR, subset, 'test.txt'),
                        split='test', img_size=224, use_wavelet=False)
    low = {}
    with open(os.path.join(OUT_DIR, 'error_slices_%s.csv' % subset), newline='',
              encoding='utf-8') as f:
        for r in csv.DictReader(f):
            low[r['rel']] = r['low_dice'] == 'True'

    base, tta, is_low = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = s['image'].unsqueeze(0)
            probs = []
            for fh in (False, True):
                for fv in (False, True):
                    z = x
                    if fh:
                        z = torch.flip(z, dims=[3])
                    if fv:
                        z = torch.flip(z, dims=[2])
                    logit = net(z)
                    logit = logit['out'] if isinstance(logit, dict) else (
                        logit[0] if isinstance(logit, (tuple, list)) else logit)
                    p = torch.softmax(logit, 1)
                    if fv:
                        p = torch.flip(p, dims=[2])
                    if fh:
                        p = torch.flip(p, dims=[3])
                    probs.append(p)
            pm = torch.stack(probs).mean(0)
            m = {}
            for name, p in (('base', probs[0]), ('tta', pm)):
                pred = p[0].argmax(0).numpy() > 0
                po, gt, sp = original_grid_pair(pred, ROOT_PATH, s['case'])
                m[name] = compute_case_metrics(po, gt, spacing=sp)
            base.append(m['base'])
            tta.append(m['tta'])
            is_low.append(low.get(s['case'], False))
            if (i + 1) % 40 == 0:
                print('  %d/%d' % (i + 1, len(ds)), file=sys.stderr)

    lo = np.array(is_low, bool)
    print('\n=== %s  n=%d ===' % (subset, len(base)))
    for key in ('dice', 'hd95'):
        b = np.array([m[key] for m in base])
        t = np.array([m[key] for m in tta])
        sgn = 1 if key == 'dice' else -1
        print('%-6s 全体 %.4f -> %.4f (Δ%+.4f)   低Dice组 %.4f -> %.4f (Δ%+.4f)'
              % (key, b.mean(), t.mean(), sgn * (t.mean() - b.mean()),
                 b[lo].mean(), t[lo].mean(), sgn * (t[lo].mean() - b[lo].mean())))
    b = np.array([m['dice'] for m in base])
    t = np.array([m['dice'] for m in tta])
    print('dice 提升的切片占比 %.0f%%（低Dice组 %.0f%%）'
          % (100 * np.mean(t > b + 1e-9), 100 * np.mean(t[lo] > b[lo] + 1e-9)))


if __name__ == '__main__':
    torch.set_num_threads(4)
    for s in ('SPARC', 'Zhongnan'):
        run(s)
