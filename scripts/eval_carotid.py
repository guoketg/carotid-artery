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
    config.n_skip = args.n_skip
    config.transformer.num_layers = args.vit_layers
    if "R50" in args.vit_name:
        config.patches.grid = (args.img_size // 16, args.img_size // 16)
    if args.arch == "unet":
        from networks.unet import UNet
        return UNet(in_channels=1, num_classes=2,
                    edge_in=bool(args.edge_in), edge_op=args.edge_op,
                    wavelet_basis=args.wavelet_basis,
                    ddbe_level=args.ddbe, acam=bool(args.acam),
                    asr=bool(args.asr),
                    fgr=bool(args.fgr), fgr_mode=args.fgr_mode,
                    fgr_pos=args.fgr_pos,
                    couple=bool(args.fgr_ddbe_couple),
                    edge_branch=bool(getattr(args, "edge_branch", 0)))
    if args.arch == "wmren":
        from networks.unet_wmren import UNetWMREN
        return UNetWMREN(in_channels=1, num_classes=2)
    if args.arch == "xnet":
        from networks.xnet import XNet
        return XNet(in_channels=1, num_classes=2)
    if args.arch == "gobletnet":
        from networks.gobletnet import GobletNet
        return GobletNet(in_channels=1, num_classes=2, hf_lambda=args.hf_lambda)
    if args.arch == "dual":
        from networks.dual_encoder_model import DualEncoderTransUNet
        return DualEncoderTransUNet(config, img_size=args.img_size, num_classes=2,
                                    levels=args.levels, fuse_op=args.fuse_op,
                                    window=args.window, dk_ratio=args.dk_ratio,
                                    bidir=bool(args.bidir),
                                    side_pretrained=bool(args.side_pretrained),
                                    cross_scale=bool(args.cross_scale),
                                    main_input=args.main_input,
                                    edge_in=bool(args.edge_in),
                                    fds=bool(args.fds))
    if args.arch == "split":
        from networks.split_encoder_model import SplitLocalGlobalTransUNet
        return SplitLocalGlobalTransUNet(config, img_size=args.img_size, num_classes=2,
                                         fuse_op=args.fuse_op, window=args.window,
                                         dk_ratio=args.dk_ratio,
                                         bidir=bool(args.bidir),
                                         fds=bool(args.fds))
    if args.fusion != "none":
        from networks.wavelet_model import WaveletTransUNet
        return WaveletTransUNet(config, img_size=args.img_size, num_classes=2,
                                fusion=args.fusion, heads=args.heads)
    return VisionTransformer(config, img_size=args.img_size, num_classes=2,
                             edge_in=bool(args.edge_in), ddbe_level=args.ddbe,
                             acam=bool(args.acam), hf_branch=bool(args.hf_branch),
                             hf_lambda=args.hf_lambda,
                             head_hr=bool(getattr(args, "head_hr", 0)),
                             edge_branch=bool(getattr(args, "edge_branch", 0)))


def load_net(args, ckpt):
    net = build_model(args).cuda()
    state = torch.load(ckpt, map_location="cuda")
    # 历史 checkpoint 的 BidirFrC 只有 gamma_a（单向）；新代码加了零初始化的 gamma_b，
    # 缺这个键时保持 0 == 旧行为逐元素等价，其余任何缺失/多余键仍报错
    missing, unexpected = net.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.endswith("gamma_b")]
    if bad or unexpected:
        raise RuntimeError(f"state_dict 不匹配: missing={bad} unexpected={list(unexpected)}")
    net.eval()
    return net


def _primary_out(logits):
    """与 trainer._primary_out 同口径：FDS 取 dict['out']，XNet 取元组首项。"""
    if isinstance(logits, dict):
        logits = logits["out"]
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits


def _dump_overlay(root_path, case, pred_o, gt_o, rec, out_dir):
    """导出原网格叠加图：灰=原图，绿=GT 轮廓，红=预测轮廓，黄=两者重合。"""
    import nibabel as nib
    from PIL import Image
    from scipy import ndimage as ndi

    img = nib.load(os.path.join(root_path, case)).get_fdata(dtype=np.float32)
    if img.ndim == 3:
        img = img[..., img.shape[-1] // 2]
    lo, hi = np.percentile(img, [1, 99])
    g = np.clip((img - lo) / max(float(hi - lo), 1e-6), 0, 1)
    rgb = np.stack([g, g, g], axis=-1)

    def edge(m):
        return m & ~ndi.binary_erosion(m, np.ones((3, 3), bool), border_value=1)

    pe, ge = edge(pred_o), edge(gt_o)
    rgb[..., 0][ge], rgb[..., 2][ge] = 0, 0   # GT 轮廓：绿
    rgb[..., 1][pe], rgb[..., 2][pe] = 0, 0   # 预测轮廓：红
    rgb[..., 0][ge & pe], rgb[..., 1][ge & pe] = 1, 1

    name = case.replace("/", "__").replace(".nii.gz", "")
    out = os.path.join(out_dir, f"{name}_hd{rec['hd']:.2f}.png")
    Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (rgb.shape[1] * 6, rgb.shape[0] * 6), Image.NEAREST).save(out)


def eval_ckpt(args):
    ckpts = [c.strip() for c in args.ens_ckpts.split(",") if c.strip()] \
        if args.ens_ckpts else [args.ckpt]
    nets = [load_net(args, c) for c in ckpts]

    dump_set = set()
    if args.dump_cases:
        os.makedirs(args.dump_dir, exist_ok=True)
        with open(args.dump_cases, "r", encoding="utf-8-sig") as f:
            dump_set = {row["case"].strip().replace("\\", "/") for row in csv.DictReader(f)}

    split_dir = os.path.join(args.list_dir, args.subset)
    ds = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "test.txt"),
                        split="test", img_size=args.img_size,
                        use_wavelet=bool(args.use_wavelet))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    records = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            probs = None
            for net in nets:
                p = torch.softmax(_primary_out(net(batch["image"].cuda())), dim=1)
                probs = p if probs is None else probs + p
            probs = probs / len(nets)
            pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
            pred_o, gt_o, spacing = original_grid_pair(pred, args.root_path, batch["case"][0])
            rec = compute_case_metrics(pred_o, gt_o, spacing=spacing)
            rec["case"] = batch["case"][0]
            records.append(rec)
            if dump_set and rec["case"] in dump_set:
                _dump_overlay(args.root_path, rec["case"], pred_o, gt_o, rec, args.dump_dir)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "dice", "assd", "hd95", "hd", "delta_area"])
        w.writeheader()
        w.writerows(records)

    s = summarize(records)
    print(f"[eval] {args.out}  (n_models={len(nets)})")
    print(f"[eval] n={s['n_cases']} dice={s['dice']['mean'] * 100:.2f}%±{s['dice']['std'] * 100:.2f}% "
          f"assd={s['assd']['mean']:.3f}±{s['assd']['std']:.3f} "
          f"hd={s['hd']['mean']:.2f}±{s['hd']['std']:.2f} "
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
        for key in ["dice", "assd", "hd95", "hd", "delta_area"]:
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
    for key in ["dice", "assd", "hd95", "hd", "delta_area"]:
        a = np.array([float(base[c][key]) for c in common])
        b = np.array([float(var[c][key]) for c in common])
        p = wilcoxon(b, a).pvalue if np.any(a != b) else 1.0
        if key == "dice":
            print(f"{key:<14}{a.mean() * 100:>9.2f}%{b.mean() * 100:>9.2f}%"
                  f"{(b - a).mean() * 100:>+9.2f}%{p:>14.4f}")
        else:
            print(f"{key:<14}{a.mean():>10.4f}{b.mean():>10.4f}{(b - a).mean():>10.4f}{p:>14.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, choices=["SPARC", "Zhongnan"])
    p.add_argument("--use_wavelet", type=int, default=0)
    p.add_argument("--edge_in", type=int, default=0)
    p.add_argument("--edge_op", default="wavelet",
                   choices=["wavelet", "wavelet_lam", "energy", "sobel", "laplacian"],
                   help="输入级边缘图算子，与 train.py 同义（仅 unet 可换）")
    p.add_argument("--wavelet_basis", default="haar",
                   choices=["haar", "db2", "db4", "sym4", "coif1", "bior1.5"],
                   help="小波基，与训练时保持一致（仅 edge_op=wavelet/wavelet_lam）")
    p.add_argument("--fusion", default="none", choices=["none", "se", "cross"])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--arch", default="baseline",
                   choices=["baseline", "dual", "split", "unet", "xnet", "gobletnet", "wmren"])
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--fuse_op", default="auto",
                   choices=["auto", "window", "global", "gate", "concat", "deform", "aff", "frc", "frc_s"])
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--dk_ratio", type=int, default=2)
    p.add_argument("--bidir", type=int, default=1)
    p.add_argument("--cross_scale", type=int, default=0)
    p.add_argument("--side_pretrained", type=int, default=1)
    p.add_argument("--main_input", default="raw", choices=["raw", "ll"])
    p.add_argument("--ddbe", type=int, default=0, help="0~4，与 train.py 同义（仅 baseline）")
    p.add_argument("--acam", type=int, default=0, help="1 = ACAM skip 对齐，与 train.py 同义（仅 baseline）")
    p.add_argument("--asr", type=int, default=0, help="1 = unet 解码器前三级挂 ASR，与 train.py 同义（仅 unet）")
    p.add_argument("--fgr", type=int, default=0, help="1 = 挂 FGR 频带改写，与 train.py 同义（unet）")
    p.add_argument("--fgr_mode", default="whole", choices=["whole", "band"],
                   help="FGR 改写核档位，与 train.py 同义")
    p.add_argument("--fgr_pos", default="dec", choices=["dec", "enc", "skip"],
                   help="FGR 挂载位置，与 train.py 同义（仅 unet 支持 enc/skip）")
    p.add_argument("--fgr_ddbe_couple", type=int, default=0,
                   help="1 = 编码器 FGR 谱门投递到解码器同级 DDBE（零参数接口）。"
                        "注意：该开关不改变 state_dict，权重文件本身看不出差异，"
                        "评估时必须与训练命令逐字一致，否则会静默用错前向")
    p.add_argument("--hf_branch", type=int, default=0, help="1 = GobletNet 式 HF 支 + FAM，与 train.py 同义（仅 baseline）")
    p.add_argument("--hf_lambda", type=float, default=0.3, help="HF 图低频回灌权重 λ，与 train.py 同义")
    p.add_argument("--head_hr", type=int, default=0,
                   help="1 = 解码器 224² 级接原图全分辨率证据支路，与 train.py 同义（仅 baseline）")
    p.add_argument("--edge_branch", type=int, default=0,
                   help="1 = 边缘监督分支（共享编码器 + 双解码器），与 train.py 同义（unet/baseline）")
    p.add_argument("--fds", type=int, default=0, help="1 = 模型带三级 FDS 辅助头（仅 dual/split）")
    p.add_argument("--vit_name", default="R50-ViT-B_16")
    p.add_argument("--vit_layers", type=int, default=12, help="ViT 层数，与 train.py 同义")
    p.add_argument("--n_skip", type=int, default=3, help="skip 数，与 train.py 同义")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--root_path", default=r"E:\mkl\code\datasets\Carotid2D")
    p.add_argument("--list_dir", default="./lists/lists_Carotid2D")
    p.add_argument("--model_root", default=r"E:\mkl\code\python_code\model\TU_Carotid224")
    p.add_argument("--prefix", default="TU_pretrain_R50-ViT-B_16_skip3_epo120_bs8_224_")
    p.add_argument("--ckpt", default="")
    p.add_argument("--ens_ckpts", default="",
                   help="逗号分隔的多个 checkpoint：逐模型 softmax 概率平均后再 argmax（多 seed 集成）")
    p.add_argument("--out", default="")
    p.add_argument("--baseline_csv", default="")
    p.add_argument("--dump_cases", default="", help="CSV（首列 case）：只导出这些病例的叠加图")
    p.add_argument("--dump_dir", default="", help="叠加图输出目录，需与 --dump_cases 同用")
    args = p.parse_args()

    tag = args.subset + ("_w3" if args.use_wavelet else "")
    if args.fusion != "none":
        tag += "_" + args.fusion
    exp_dir = os.path.join(args.model_root, args.prefix + tag)
    if not args.ckpt:
        args.ckpt = os.path.join(exp_dir, "epoch_119.pth")
    if not args.out:
        args.out = os.path.join(exp_dir, "metrics.csv")
    if args.ens_ckpts:
        missing = [c.strip() for c in args.ens_ckpts.split(",")
                   if c.strip() and not os.path.isfile(c.strip())]
        if missing:
            raise FileNotFoundError("; ".join(missing))
    elif not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    eval_ckpt(args)
    if args.baseline_csv:
        compare(args.baseline_csv, args.out)


if __name__ == "__main__":
    main()
