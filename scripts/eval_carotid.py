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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_carotid import CarotidDataset
from metrics_carotid import compute_case_metrics, original_grid_pair, summarize
from networks.vit_seg_modeling import CONFIGS, VisionTransformer


def build_model(args):
    """与 train.py 的构造分支保持一致（baseline/dual/fusion + edge_in/ddbe），
    评估用不到预训练权重加载（直接 load_state_dict 全量覆盖）。"""
    config = copy.deepcopy(CONFIGS[args.vit_name])
    config.n_classes = 2
    config.n_skip = 3
    config.patches.grid = (args.img_size // 16, args.img_size // 16)
    if args.arch == "dual":
        from networks.dual_encoder_model import DualEncoderTransUNet
        return DualEncoderTransUNet(config, img_size=args.img_size, num_classes=2,
                                    levels=args.levels, fuse_op=args.fuse_op,
                                    window=args.window, dk_ratio=args.dk_ratio,
                                    bidir=bool(args.bidir),
                                    side_pretrained=bool(args.side_pretrained),
                                    cross_scale=bool(args.cross_scale),
                                    main_input=args.main_input,
                                    edge_in=bool(args.edge_in))
    if args.fusion != "none":
        from networks.wavelet_model import WaveletTransUNet
        return WaveletTransUNet(config, img_size=args.img_size, num_classes=2,
                                fusion=args.fusion, heads=args.heads)
    return VisionTransformer(config, img_size=args.img_size, num_classes=2,
                             edge_in=bool(args.edge_in), ddbe_level=args.ddbe,
                             acam=bool(args.acam), hf_branch=bool(args.hf_branch),
                             hf_lambda=args.hf_lambda)


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


def _case_id(slice_key):
    """切片 key 形如 PlaqueDataNii_SPARC_S/<病例>/<切片>.nii.gz → 病例 = 第二段目录"""
    parts = slice_key.replace("\\", "/").split("/")
    return parts[1] if len(parts) >= 2 else parts[0]


def _group_by_case(rows):
    """同病例多张切片先取均值再配对（experiments.md §1.6 统计口径）"""
    buckets = {}
    for r in rows:
        buckets.setdefault(_case_id(r["case"]), []).append(r)
    out = {}
    for cid, rs in buckets.items():
        agg = {"case": cid}
        for key in ["dice", "assd", "hd95", "delta_area"]:
            agg[key] = float(np.mean([float(r[key]) for r in rs]))
        out[cid] = agg
    return out


def compare(baseline_csv, variant_csv):
    def load(p):
        with open(p, newline="", encoding="utf-8") as f:
            return _group_by_case(list(csv.DictReader(f)))

    base, var = load(baseline_csv), load(variant_csv)
    common = sorted(set(base) & set(var))
    print(f"[compare] baseline={baseline_csv}")
    print(f"[compare] variant ={variant_csv}")
    print(f"[compare] paired cases: {len(common)} (case-level, per-case slice mean)")
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
    p.add_argument("--edge_in", type=int, default=0)
    p.add_argument("--fusion", default="none", choices=["none", "se", "cross"])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--arch", default="baseline", choices=["baseline", "dual"])
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--fuse_op", default="auto",
                   choices=["auto", "window", "global", "gate", "concat", "deform", "aff"])
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--dk_ratio", type=int, default=2)
    p.add_argument("--bidir", type=int, default=1)
    p.add_argument("--cross_scale", type=int, default=0)
    p.add_argument("--side_pretrained", type=int, default=1)
    p.add_argument("--main_input", default="raw", choices=["raw", "ll"])
    p.add_argument("--ddbe", type=int, default=0, help="0~4，与 train.py 同义（仅 baseline）")
    p.add_argument("--acam", type=int, default=0, help="1 = ACAM skip 对齐，与 train.py 同义（仅 baseline）")
    p.add_argument("--hf_branch", type=int, default=0, help="1 = GobletNet 式 HF 支 + FAM，与 train.py 同义（仅 baseline）")
    p.add_argument("--hf_lambda", type=float, default=0.3, help="HF 图低频回灌权重 λ，与 train.py 同义")
    p.add_argument("--vit_name", default="R50-ViT-B_16")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--root_path", default=r"E:\mkl\code\datasets\Carotid2D")
    p.add_argument("--list_dir", default="./lists/lists_Carotid2D")
    p.add_argument("--model_root", default=r"E:\mkl\code\python_code\model\TU_Carotid224")
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
