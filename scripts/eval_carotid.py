"""重算 Carotid 单个 checkpoint 的逐样本指标，可选与 baseline 做 Wilcoxon 配对检验。

用于修复 trainer.py 末尾评估漏传 use_wavelet 造成的历史 metrics.csv 错误值。

示例：
  python scripts/eval_carotid.py --subset SPARC --use_wavelet 1 --fusion cross \
      --baseline_csv F:\\code\\MyPython\\model\\TU_Carotid224\\TU_pretrain_R50-ViT-B_16_skip3_epo120_bs8_224_SPARC\\metrics.csv
"""
import argparse
import copy
import csv
import os
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon
from metrics_carotid import original_grid_pair
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset
from metrics_carotid import compute_case_metrics, original_grid_pair, summarize
from networks.vit_seg_modeling import CONFIGS, VisionTransformer


def build_model(args):
    config = copy.deepcopy(CONFIGS[args.vit_name])
    config.n_classes = 2
    config.n_skip = 3
    config.patches.grid = (args.img_size // 16, args.img_size // 16)
    if args.fusion != "none":
        from networks.wavelet_model import WaveletTransUNet
        return WaveletTransUNet(config, img_size=args.img_size, num_classes=2,
                                fusion=args.fusion, heads=args.heads)
    return VisionTransformer(config, img_size=args.img_size, num_classes=2)


def eval_ckpt(args):
    net = build_model(args).cuda()
    state = torch.load(args.ckpt, map_location="cuda")
    net.load_state_dict(state)
    net.eval()

    split_dir = os.path.join(args.list_dir, args.subset)
    ds = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "test.txt"),
                        split="test", img_size=args.img_size,
                        use_wavelet=bool(args.use_wavelet))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    records = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            logits = net(batch["image"].cuda())
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            pred_o, gt_o, spacing = original_grid_pair(pred, args.root_path, batch["case"][0])
            rec = compute_case_metrics(pred_o, gt_o, spacing=spacing)
            rec["case"] = batch["case"][0]
            records.append(rec)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "dice", "assd", "hd95", "delta_area"])
        w.writeheader()
        w.writerows(records)

    s = summarize(records)
    print(f"[eval] {args.out}")
    print(f"[eval] n={s['n_cases']} dice={s['dice']['mean']:.4f}±{s['dice']['std']:.4f} "
          f"assd={s['assd']['mean']:.3f}±{s['assd']['std']:.3f} "
          f"hd95={s['hd95']['mean']:.2f}±{s['hd95']['std']:.2f} "
          f"delta_area={s['delta_area']['mean']:.1f}±{s['delta_area']['std']:.1f} "
          f"|delta_area|={s['abs_delta_area']['mean']:.1f}")
    return records


def compare(baseline_csv, variant_csv):
    def load(p):
        with open(p, newline="") as f:
            return {r["case"]: r for r in csv.DictReader(f)}

    base, var = load(baseline_csv), load(variant_csv)
    common = sorted(set(base) & set(var))
    print(f"[compare] baseline={baseline_csv}")
    print(f"[compare] variant ={variant_csv}")
    print(f"[compare] paired cases: {len(common)}")
    print(f"{'metric':<14}{'baseline':>10}{'variant':>10}{'diff':>10}{'p(wilcoxon)':>14}")
    for key in ["dice", "assd", "hd95", "delta_area"]:
        a = np.array([float(base[c][key]) for c in common])
        b = np.array([float(var[c][key]) for c in common])
        p = wilcoxon(b, a).pvalue if np.any(a != b) else 1.0
        print(f"{key:<14}{a.mean():>10.4f}{b.mean():>10.4f}{(b - a).mean():>10.4f}{p:>14.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, choices=["SPARC", "Zhongnan"])
    p.add_argument("--use_wavelet", type=int, default=0)
    p.add_argument("--fusion", default="none", choices=["none", "se", "cross"])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--vit_name", default="R50-ViT-B_16")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--root_path", default=r"F:\code\datasets\Carotid2D")
    p.add_argument("--list_dir", default="./lists/lists_Carotid2D")
    p.add_argument("--model_root", default=r"F:\code\MyPython\model\TU_Carotid224")
    p.add_argument("--prefix", default="TU_pretrain_R50-ViT-B_16_skip3_epo120_bs8_224_")
    p.add_argument("--ckpt", default="")
    p.add_argument("--out", default="")
    p.add_argument("--baseline_csv", default="")
    args = p.parse_args()

    tag = args.subset + ("_w3" if args.use_wavelet else "")
    if args.fusion != "none":
        tag += "_" + args.fusion
    exp_dir = os.path.join(args.model_root, args.prefix + tag)
    if not args.ckpt:
        args.ckpt = os.path.join(exp_dir, "epoch_119.pth")
    if not args.out:
        args.out = os.path.join(exp_dir, "metrics.csv")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    eval_ckpt(args)
    if args.baseline_csv:
        compare(args.baseline_csv, args.out)


if __name__ == "__main__":
    main()
