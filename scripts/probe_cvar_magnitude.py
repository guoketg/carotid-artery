"""CVaR 尾部项的量级探针（零训练，只前向 + 对 logits 求导）。

问题：R50 上 `tl_w=0.1` 的 cvartail 判负（ΔHD −0.05），U-Net 上同一参数 +0.82 / −0.18。
是「剂量不够」还是「选错了像素」？本脚本在不训练的前提下量三件事：

1. **项的量级**：给定 checkpoint，逐 batch 算
   - L_main = 0.5·CE + 0.5·Dice，L_cvar = cvar_distance_tail_loss(q, d0)
   - 对 logits 的梯度范数比 rho = ||∂L_cvar|| / ||∂L_main||
     → 训练时的实际梯度占比 = tl_w · rho。把「U-Net 成功臂在终点时的占比」当锚点，
       就能反解 R50 上需要多大的 tl_w 才等价。
2. **top-q 选中的是什么像素**：mean D、mean |p−gt|、D≥3px 的占比与它们的 e 贡献、
   与「真误差像素 (|p−gt|>0.5)」的精度/召回。若选中的大头是远处背景残余概率，
   那这一项其实退化成 far-FP 项（LFA 已判负），加权重也救不回来。
3. **误差尺度差异**：失配像素上的 D 分位数（R50 vs U-Net），验证「距离权重被压小」。

只读 checkpoint、只跑 CPU（GPU 上有别的队列在训），对 logits 求导而非反传全图，很快。

用法：
  .\\venv\\Scripts\\python.exe scripts\\probe_cvar_magnitude.py
  .\\venv\\Scripts\\python.exe scripts\\probe_cvar_magnitude.py --batches 8 --device cpu
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datasets.dataset_carotid import CarotidDataset          # noqa: E402
from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg     # noqa: E402
from utils import DiceLoss, cvar_distance_tail_loss, _signed_distance  # noqa: E402

MODEL_DIR = r"E:\mkl\code\python_code\model\TU_Carotid224"
LIST_DIR = os.path.join(BASE_DIR, "lists", "lists_Carotid2D")

R50 = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224"
U50 = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
CVAR = "lcvartail_W0.1Q0.1A0.05"

TARGETS = [
    # (标签, snapshot, arch, vit_layers)
    ("R50-A0", f"{R50}_SPARC_c2d", "baseline", 0),
    ("R50-cvar(w=0.1)", f"{R50}_SPARC_{CVAR}_c2d", "baseline", 0),
    ("UNet-A0", f"{U50}_SPARC_unet_c2d", "unet", 12),
    ("UNet-cvar(w=0.1)", f"{U50}_SPARC_{CVAR}_unet_c2d", "unet", 12),
]
# 探针网格：主口径 (q=0.1, d0=10) 对齐训练；另两项看「改 q / 改 d0」能否把选择拉回尾部
GRID = [(0.1, 10.0), (0.02, 10.0), (0.1, 3.0)]
CAP = 3.0


def build(arch, vit_layers, img_size=224, num_classes=2):
    if arch == "unet":
        from networks.unet import UNet
        return UNet(in_channels=1, num_classes=num_classes)
    cfg = CONFIGS_ViT_seg["R50-ViT-B_16"]
    cfg.n_classes = num_classes
    cfg.n_skip = 3
    cfg.transformer.num_layers = vit_layers
    cfg.patches.grid = (img_size // 16, img_size // 16)
    return ViT_seg(cfg, img_size=img_size, num_classes=num_classes)


def load_net(snapshot, arch, vit_layers, device):
    path = os.path.join(MODEL_DIR, snapshot, "epoch_199.pth")
    if not os.path.exists(path):
        path = os.path.join(MODEL_DIR, snapshot, "best_model.pth")
    net = build(arch, vit_layers)
    state = torch.load(path, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    return net.to(device).eval(), path


def tail_parts(logits, label, q, d0):
    """复刻 cvar_distance_tail_loss 的选点过程，额外返回选中集的构成统计。"""
    prob = torch.softmax(logits, dim=1)[:, 1]
    gt = (label > 0).float()
    gtn = gt.detach().cpu().numpy().astype(bool)
    dist = []
    for g in gtn:
        if not g.any():
            dist.append(np.zeros_like(g, dtype=np.float32))
            continue
        phi = _signed_distance(g) * float(max(g.shape))
        dist.append(np.abs(phi).astype(np.float32))
    d = torch.from_numpy(np.stack(dist)).to(prob.device).float()
    e = torch.abs(prob - gt) * (d / d0).clamp(max=CAP)
    b, h, w = e.shape
    k = max(1, int(round(h * w * float(q))))
    top = torch.topk(e.reshape(b, -1), k, dim=1)
    valid = (gt.reshape(b, -1).sum(dim=1) > 0)

    e_flat, d_flat = e.reshape(b, -1), d.reshape(b, -1)
    err_flat = torch.abs(prob - gt).reshape(b, -1)
    sel_e, sel_d, sel_err = [], [], []
    for i in range(b):
        if not valid[i]:
            continue
        idx = top.indices[i]
        sel_e.append(e_flat[i, idx].detach().cpu().numpy())
        sel_d.append(d_flat[i, idx].detach().cpu().numpy())
        sel_err.append(err_flat[i, idx].detach().cpu().numpy())
    stats = {}
    if sel_e:
        se = np.concatenate(sel_e)
        sd = np.concatenate(sel_d)
        sr = np.concatenate(sel_err)
        stats = {
            "k": k,
            "sel_D_mean": float(sd.mean()),
            "sel_D_p90": float(np.percentile(sd, 90)),
            "sel_err_mean": float(sr.mean()),
            "share_D_ge3": float((sd >= 3).mean()),
            "e_share_D_ge3": float(se[sd >= 3].sum() / max(se.sum(), 1e-12)),
            "share_true_err": float((sr > 0.5).mean()),
        }
        # 真误差像素集合（|p−gt|>0.5）被 top-k 覆盖的比例
        hit, tot = 0.0, 0.0
        for i in range(b):
            if not valid[i]:
                continue
            err_i = err_flat[i].detach().cpu().numpy()
            true_i = err_i > 0.5
            idx = top.indices[i].cpu().numpy()
            hit += true_i[idx].sum()
            tot += true_i.sum()
        stats["recall_true_err"] = float(hit / max(tot, 1.0))
    # 失配像素（|p−gt|>0.5）上的距离分位，反映"误差带离边界多远"
    mism = []
    for i in range(b):
        if not valid[i]:
            continue
        m = err_flat[i].detach().cpu().numpy() > 0.5
        if m.any():
            mism.append(d_flat[i].detach().cpu().numpy()[m])
    if mism:
        mm = np.concatenate(mism)
        stats["mism_D_median"] = float(np.median(mm))
        stats["mism_D_p90"] = float(np.percentile(mm, 90))
        stats["mism_px_per_slice"] = float(mm.size / max(len(mism), 1))
    return e, stats


def probe(label, snapshot, arch, vit_layers, args):
    device = torch.device(args.device)
    net, ckpt = load_net(snapshot, arch, vit_layers, device)
    ds = CarotidDataset(root=args.root, list_file=os.path.join(LIST_DIR, "SPARC", "train.txt"),
                        split="train", img_size=224, augment=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    ce = torch.nn.CrossEntropyLoss()
    dice = DiceLoss(2)
    torch.manual_seed(args.seed)

    acc = {g: {"main": [], "cvar": [], "rho": [], "cos": []} for g in GRID}
    stats = {g: [] for g in GRID}

    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        img, lab = batch["image"].to(device), batch["label"].to(device)
        with torch.no_grad():                 # 只前向不建图，损失侧只对 logits 求导
            logits0 = net(img)
        if isinstance(logits0, dict):
            logits0 = logits0["out"]
        logits0 = logits0[0] if isinstance(logits0, (tuple, list)) else logits0

        for q, d0 in GRID:
            logits = logits0.detach().clone().requires_grad_(True)
            l_main = 0.5 * ce(logits, lab.long()) + 0.5 * dice(logits, lab, softmax=True)
            g_main = torch.autograd.grad(l_main, logits, retain_graph=True)[0]
            l_cvar = cvar_distance_tail_loss(logits, lab, q=q, d0=d0)
            g_cvar = torch.autograd.grad(l_cvar, logits, retain_graph=True)[0]
            acc[(q, d0)]["main"].append(float(l_main))
            acc[(q, d0)]["cvar"].append(float(l_cvar))
            acc[(q, d0)]["rho"].append(float(g_cvar.norm() / (g_main.norm() + 1e-12)))
            cos = torch.nn.functional.cosine_similarity(
                g_cvar.flatten(), g_main.flatten(), dim=0)
            acc[(q, d0)]["cos"].append(float(cos))
            _, st = tail_parts(logits, lab, q, d0)
            stats[(q, d0)].append(st)

    print(f"\n## {label}  ({ckpt})")
    print(f"批次 {len(acc[GRID[0]]['main'])} × bs4，device={args.device}\n")
    print("| q | d0 | L_main | L_cvar | rho=‖g_cvar‖/‖g_main‖ | cos(g_cvar,g_main) | "
          "选中 mean D | 选中 mean\\|p−gt\\| | e 中 D≥3px 占比 | 选中为真误差占比 | 真误差召回 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for q, d0 in GRID:
        a, s = acc[(q, d0)], stats[(q, d0)]
        m = lambda k: float(np.mean([x[k] for x in s])) if s and k in s[0] else float("nan")
        rows[(q, d0)] = {
            "rho": float(np.mean(a["rho"])),
            "cvar": float(np.mean(a["cvar"])),
            "share_D_ge3": m("e_share_D_ge3"),
            "sel_D": m("sel_D_mean"),
        }
        print("| %.2f | %g | %.4f | %.5f | %.4f | %+.3f | %.2f px | %.3f | %.3f | %.3f | %.3f |"
              % (q, d0, np.mean(a["main"]), np.mean(a["cvar"]), np.mean(a["rho"]),
                 np.mean(a["cos"]), m("sel_D_mean"), m("sel_err_mean"),
                 m("e_share_D_ge3"), m("share_true_err"), m("recall_true_err")))
    print("\n失配像素（|p−gt|>0.5）上的距离分布：")
    for q, d0 in GRID:
        s = stats[(q, d0)]
        if s and "mism_D_median" in s[0]:
            print("  q=%.2f d0=%g : D 中位 %.2f px, p90 %.2f px, 每切片失配像素 %.0f"
                  % (q, d0, np.mean([x["mism_D_median"] for x in s]),
                     np.mean([x["mism_D_p90"] for x in s]),
                     np.mean([x["mism_px_per_slice"] for x in s])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"E:\mkl\code\datasets\Carotid2D")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=os.path.join(BASE_DIR, "logs", "analysis", "cvar_probe.md"))
    args = p.parse_args()
    torch.manual_seed(args.seed)

    res = {}
    for label, snap, arch, vl in TARGETS:
        try:
            res[label] = probe(label, snap, arch, vl, args)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}")

    # 反解 λ：让 cvartail 在 R50 上占的梯度份额 = U-Net 成功臂的份额
    anchor = res.get("UNet-cvar(w=0.1)", {}).get((0.1, 10.0))
    r50 = res.get("R50-A0", {}).get((0.1, 10.0)) or res.get("R50-cvar(w=0.1)", {}).get((0.1, 10.0))
    if anchor and r50:
        share_u = 0.1 * anchor["rho"]
        lam = share_u / max(r50["rho"], 1e-12)
        print("\n## 反解 tl_w")
        print(f"- U-Net 成功臂在终点的梯度份额 = 0.1 × rho({anchor['rho']:.4f}) = {share_u*100:.2f}%")
        print(f"- R50（A0 终点）rho = {r50['rho']:.4f} → 同份额需要 tl_w = {lam:.2f}")
        for s in (0.05, 0.1, 0.2, 0.3):
            print(f"  · 目标份额 {s*100:>4.0f}% → tl_w = {s/max(r50['rho'],1e-12):.2f}")
        print("\n注意：rho 在训练早期更大（模型不自信、|p−gt| 大），此处用终点 checkpoint 估计的\n"
              "是**训练末期**的份额；用它定 λ 的下界，扫 λ 时按 1×/2×/4× 取点。")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)


if __name__ == "__main__":
    main()
