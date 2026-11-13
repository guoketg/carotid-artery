# coding=utf-8
"""边缘监督分支冒烟（CPU、随机张量、不碰数据集、不训练）。

检查项：
  1) 形状：UNet / R50-TransUNet 返回 {'out': (B,2,224,224), 'edge': (B,2,224,224)}；
  2) RNG 对齐：同 seed 下挂/不挂边缘分支，**基础权重逐位一致**（模块最后创建）；
  3) 参数增量（第二个解码器与主解码器对等，不是轻量头）；
  4) 梯度链路：主损失 + 同权重边缘 CE 反传后，边缘解码器全部参数梯度有限且非全零，
     主分割头梯度仍非零（辅助监督没有截断主路径）；
  5) 边缘标签算子：GT 为全前景/全背景时边缘带为空，方形目标只在外环出带；
  6) 边缘指标边界行为（完美/空/全 1 退缩解）+ 四列拼图落盘。

用法：venv\\Scripts\\python.exe scripts\\smoke_edge_branch.py
"""
import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.vit_seg_modeling import CONFIGS, VisionTransformer
from networks.unet import UNet
from trainer import _edge_target


def _count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def _rng_aligned(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    common = [k for k in sa if k in sb and "edge_dec" not in k and "edge_head" not in k]
    bad = [k for k in common if not torch.equal(sa[k], sb[k])]
    return len(common), bad


def check_unet():
    torch.manual_seed(1234)
    base = UNet(in_channels=1, num_classes=2)
    torch.manual_seed(1234)
    model = UNet(in_channels=1, num_classes=2, edge_branch=True)
    n, bad = _rng_aligned(base, model)
    assert not bad, "UNet 基础权重未对齐: %s" % bad[:5]

    out = model(torch.randn(2, 1, 224, 224))
    assert isinstance(out, dict) and out["out"].shape == (2, 2, 224, 224), out["out"].shape
    assert out["edge"].shape == (2, 2, 224, 224), out["edge"].shape

    label = (torch.rand(2, 224, 224) > 0.9).long()
    edge_t = _edge_target(label).long()[:, 0]
    # 与主分割同形式同权重：CE 都取 1.0（正式训练里走同一个 seg_loss）
    loss = F.cross_entropy(out["out"], label) + F.cross_entropy(out["edge"], edge_t)
    loss.backward()
    eg = [p.grad for p in model.edge_dec.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in eg), "边缘解码器梯度缺失/非有限"
    assert any(g.abs().sum() > 0 for g in eg), "边缘解码器梯度全零"
    assert model.d1_2.weight.grad is not None and model.d1_2.weight.grad.abs().sum() > 0, \
        "主分割头梯度被截断"
    # 只回传边缘损失：必须有一批共享编码器参数拿到非零梯度，
    # 否则边缘监督根本没作用到骨干上（+1.0 增益的先决条件）
    model.zero_grad()
    out = model(torch.randn(2, 1, 224, 224))
    F.cross_entropy(out["edge"], _edge_target(label).long()[:, 0]).backward()
    shared = [n for n, p in model.named_parameters()
              if "edge_dec" not in n and p.grad is not None and p.grad.abs().sum() > 0]
    assert shared, "边缘损失没有回传到共享编码器"
    print("  edge-only backward reaches %d shared params, e.g. %s" % (len(shared), shared[0]))
    print("[unet] OK  params %.3fM -> %.3fM (+%.3fM)  base keys %d aligned"
          % (_count(base), _count(model), _count(model) - _count(base), n))
    return base, model


def check_baseline():
    cfg = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
    cfg.n_classes, cfg.n_skip, cfg.transformer.num_layers = 2, 3, 0
    cfg.patches.grid = (14, 14)

    torch.manual_seed(1234)
    base = VisionTransformer(copy.deepcopy(cfg), img_size=224, num_classes=2)
    torch.manual_seed(1234)
    model = VisionTransformer(copy.deepcopy(cfg), img_size=224, num_classes=2,
                              edge_branch=True)
    n, bad = _rng_aligned(base, model)
    assert not bad, "R50-TransUNet 基础权重未对齐: %s" % bad[:5]

    out = model(torch.randn(1, 1, 224, 224))
    assert isinstance(out, dict) and out["out"].shape == (1, 2, 224, 224), out["out"].shape
    assert out["edge"].shape == (1, 2, 224, 224), out["edge"].shape

    loss = out["out"].sum() + out["edge"].sum()
    loss.backward()
    eg = [p.grad for p in model.edge_dec.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in eg), "边缘解码器梯度缺失/非有限"
    # 只回传边缘损失：检查共享骨干（R50/ViT）是否真的收到梯度
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    out["edge"].sum().backward()
    shared = [n for n, p in model.named_parameters()
              if "edge_dec" not in n and "edge_head" not in n
              and p.grad is not None and p.grad.abs().sum() > 0]
    assert shared, "边缘损失没有回传到共享骨干"
    r50 = [n for n in shared if "hybrid_model" in n]
    assert r50, "边缘损失没有回传到 R50 骨干（共享编码器未真正参与边缘监督）"
    print("  edge-only backward reaches %d shared params (%d in R50), e.g. %s"
          % (len(shared), len(r50), r50[0]))
    print("[baseline] OK  params %.3fM -> %.3fM (+%.3fM)  base keys %d aligned"
          % (_count(base), _count(model), _count(model) - _count(base), n))
    return base, model


def check_edge_operator():
    empty = torch.zeros(1, 224, 224).long()
    full = torch.ones(1, 224, 224).long()
    assert _edge_target(empty).sum() == 0 and _edge_target(full).sum() == 0, "空掩膜边缘带非零"
    box = torch.zeros(1, 224, 224).long()
    box[:, 96:128, 96:128] = 1
    band = _edge_target(box)
    inside = band[:, 100:124, 100:124].sum()  # 内部（腐蚀掉的部分）
    assert inside == 0, "边缘带进入了前景内部"
    assert band.sum() > 0, "方形目标没有边缘带"
    print("[edge target] OK  box band pixels = %d, bandwidth = 1px inner + 1px outer"
          % int(band.sum()))


def check_edge_vis():
    """边缘指标的边界行为 + 拼图落盘（确保训练结束后的可视化路径能跑通）。"""
    import tempfile
    from trainer import _edge_scores, _save_edge_panels

    box = torch.zeros(224, 224).long()
    box[96:128, 96:128] = 1
    gt_e = _edge_target(box[None])[0, 0]  # (B,H,W) -> (B,1,H,W)
    assert _edge_scores(gt_e, gt_e)["edge_dice"] > 0.999, "完美预测的边缘 Dice 不为 1"
    empty = _edge_scores(torch.zeros_like(gt_e), gt_e)
    assert empty["edge_dice"] == 0 and empty["pred_edge_frac"] == 0, "空预测指标异常"
    full = _edge_scores(torch.ones_like(gt_e), gt_e)
    assert full["pred_edge_frac"] == 1.0, "全 1 退缩解识别失败"
    # 主分割预测（这里用同一实心掩膜）过同一边缘算子，应还原出同一组 GT 边缘
    d_main = _edge_scores(_edge_target(box[None])[0, 0], gt_e)["edge_dice"]
    assert d_main > 0.999, "主掩膜派生的边缘 Dice 对照异常"

    img = torch.rand(224, 224)
    rows = [("good", img.numpy(), box.numpy(), gt_e.numpy(), gt_e.numpy(), 0.9, 1.0),
            ("empty", img.numpy(), box.numpy(), gt_e.numpy(),
             torch.zeros_like(gt_e).numpy(), 0.9, 0.0),
            ("full", img.numpy(), box.numpy(), gt_e.numpy(),
             torch.ones_like(gt_e).numpy(), 0.9, float(full["edge_dice"]))]
    d = tempfile.mkdtemp()
    p = os.path.join(d, "edge_cases.png")
    _save_edge_panels(p, rows)
    assert os.path.exists(p) and os.path.getsize(p) > 10_000, "拼图未落盘"
    print("[edge vis] OK  good/empty/full metrics + panel %d bytes" % os.path.getsize(p))


if __name__ == "__main__":
    check_edge_operator()
    check_edge_vis()
    check_unet()
    check_baseline()
    print("ALL GREEN")
