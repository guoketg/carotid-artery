"""损失设计 v2 三项（几何场回归 / 斑点扰动一致性 / 多尺度深监督）的接线冒烟。

不加载真实数据，纯 CPU（不抢正在训练的 GPU），只验证"接线对不对、量级对不对"：

  1. 恒等起步：开 --ds / --sdf 只是在基础权重之后多挂头，共有参数与不挂版本逐位相同
  2. 参数量：三个头各多少参数（SDF 头、深监督两个 1×1 头、一致性零新参数）
  3. 输出契约：none → tensor；ds → dict 带 aux（56²/112² 两级）；sdf aux → dict 带 sdf；
     sdf out → 主 logits 就是 ±0.5·sdf（零水平集给掩膜，与 argmax 严格等价）
  4. 几何场目标：GT 内部为正、外部为负、按 tau 饱和；窄带 L1 有限；
     sdf_w=0 时总损失与不挂版本逐位相同（恒等起步）
  5. 深监督项：两级 aux 上采样到 224² 后算 0.5CE+0.5Dice，梯度非零有限
  6. 一致性项：同图两次前向 ≈ 0；乘性斑点扰动后落在 (0,1)，梯度有限
  7. 空前景：几何场项返回 0、不 NaN
  8. R50 侧结构检查（构造 + 前向一次，CPU 慢约 30 s）：--sdf aux --ds 同时挂通的输出键

运行：venv\\Scripts\\python.exe scripts\\smoke_lossv2.py
      venv\\Scripts\\python.exe scripts\\smoke_lossv2.py --no-r50     # 跳过 R50 结构检查
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.unet import UNet
from trainer import _sdf_target
from utils import DiceLoss


def _labels(batch=2, size=224):
    label = torch.zeros(batch, size, size, dtype=torch.long)
    label[:, 60:160, 70:150] = 1
    return label


def _make_unet(**kw):
    torch.manual_seed(1234)
    return UNet(in_channels=1, num_classes=2, **kw)


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def _ce_dice(logits, label, ce, dice):
    return 0.5 * ce(logits, label.long()) + 0.5 * dice(logits, label, softmax=True)


def _cons(p1, p2):
    """与 trainer 训练循环里同一式子：1 − 软 Dice(p1, p2)"""
    return 1.0 - (2.0 * (p1 * p2).sum() + 1.0) / (p1.sum() + p2.sum() + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-r50', action='store_true', help='跳过 R50 结构检查（更快）')
    args = ap.parse_args()
    torch.manual_seed(0)

    img = torch.randn(2, 1, 224, 224) * 0.3
    label = _labels()
    ce = torch.nn.CrossEntropyLoss()
    dice = DiceLoss(2)

    # ---- [1] 恒等起步：只多挂的头，共有权重逐位相同 ----
    base = _make_unet()
    net_ds = _make_unet(ds=True)
    net_sdf = _make_unet(sdf='aux')
    sd_base = base.state_dict()
    for name, other in (('--ds', net_ds), ('--sdf aux', net_sdf)):
        sd = other.state_dict()
        extra = sorted(set(sd) - set(sd_base))
        missing = sorted(set(sd_base) - set(sd))
        assert not missing, (name, '缺键', missing[:3])
        bad = [k for k in sd_base if not torch.equal(sd_base[k], sd[k])]
        assert not bad, (name, '共有权重不一致', bad[:3])
        print('[1] %-8s 恒等起步 ok：多出 %d 个键 %s' % (name, len(extra), extra[:3]))
    print('    参数量：基线 %.3f M | +ds %.3f k | +sdf %.3f k'
          % (_n_params(base) / 1e6,
             (_n_params(net_ds) - _n_params(base)) / 1e3,
             (_n_params(net_sdf) - _n_params(base)) / 1e3))

    # ---- [2] 输出契约 ----
    with torch.no_grad():
        out_plain = base(img)
        out_ds = net_ds(img)
        out_aux = net_sdf(img)
    assert isinstance(out_plain, torch.Tensor) and out_plain.shape == (2, 2, 224, 224)
    assert isinstance(out_ds, dict) and 'aux' in out_ds
    assert [tuple(t.shape[-2:]) for t, _ in out_ds['aux']] == [(112, 112), (56, 56)]
    assert [w for _, w in out_ds['aux']] == [0.3, 0.15]
    assert isinstance(out_aux, dict) and 'sdf' in out_aux
    print('[2] 输出契约 ok：ds aux %s 权重 %s；sdf aux 形状 %s'
          % ([tuple(t.shape[-2:]) for t, _ in out_ds['aux']],
             [w for _, w in out_ds['aux']], tuple(out_aux['sdf'].shape)))

    # ---- [3] sdf out：主 logits 就是 ±0.5·sdf，argmax 与零水平集严格等价 ----
    net_out = _make_unet(sdf='out')
    with torch.no_grad():
        o = net_out(img)
        sdf = o['sdf']
        logits = o['out']
        assert torch.allclose(logits[:, 0:1], -0.5 * sdf)
        assert torch.allclose(logits[:, 1:2], 0.5 * sdf)
        assert torch.equal(torch.argmax(logits, 1), (sdf[:, 0] > 0).long())
        prob_max = torch.softmax(logits, 1)[:, 1].max().item()
        prob_min = torch.softmax(logits, 1)[:, 1].min().item()
    print('[3] sdf out ok：argmax 与 sdf>0 逐像素一致；前景概率量程 [%.2e, %.6f]'
          '（= sigmoid(±tau)，窄带外饱和 → 该臂只作机制对照，主臂用 aux）'
          % (prob_min, prob_max))

    # ---- [4] 几何场目标与窄带损失 ----
    tau = 12.0
    t = _sdf_target(label, tau)
    c = (110, 110)          # 方形内部中心
    outside = (10, 10)      # 远离前景
    assert abs(t[0, c[0], c[1]].item() - tau) < 1e-6, t[0, c[0], c[1]].item()
    assert abs(t[0, outside[0], outside[1]].item() + tau) < 1e-6
    band_frac = (t.abs() < tau).float().mean().item()
    print('[4a] 距离场目标 ok：中心 %+.1f / 外部 %+.1f / 窄带占比 %.3f'
          % (t[0, c[0], c[1]].item(), t[0, outside[0], outside[1]].item(), band_frac))

    def sdf_term(sdf_map, tgt, w_eik=0.02):
        band = (tgt.abs() < tau).float()
        cnt = band.sum().clamp(min=1.0)
        l1 = ((sdf_map - tgt).abs() * band).sum() / cnt / tau
        gy, gx = torch.gradient(sdf_map, dim=(2, 3))
        eik = (((gy ** 2 + gx ** 2).sqrt() - 1.0) ** 2 * band).sum() / cnt
        return l1 + w_eik * eik

    tgt = t.unsqueeze(1)
    net_sdf.zero_grad(set_to_none=True)
    out = net_sdf(img)
    l_main = _ce_dice(out['out'], label, ce, dice)
    l_sdf = sdf_term(out['sdf'], tgt)
    (l_main + 1.0 * l_sdf).backward()
    g_sdf = sum(p.grad.abs().sum() for n, p in net_sdf.named_parameters()
                if p.grad is not None and n.startswith('sdf_head'))
    g_main = sum(p.grad.abs().sum() for n, p in net_sdf.named_parameters()
                 if p.grad is not None and not n.startswith('sdf_head'))
    assert torch.isfinite(l_sdf).item() and l_sdf.item() > 0
    assert torch.isfinite(g_sdf).item() and g_sdf.item() > 0
    assert torch.isfinite(g_main).item() and g_main.item() > 0
    print('[4b] 窄带 L1+eik %.4f（主项 %.4f）| 梯度：sdf_head %.3e / 主干 %.3e（辅助项也训主干）'
          % (l_sdf.item(), l_main.item(), g_sdf.item(), g_main.item()))

    # 恒等起步：sdf_w=0 时总损失与不挂版本逐位相同
    net_a = _make_unet()
    net_b = _make_unet(sdf='aux')
    with torch.no_grad():
        la = _ce_dice(net_a(img), label, ce, dice)
        lb = _ce_dice(net_b(img)['out'], label, ce, dice)
    assert abs(la.item() - lb.item()) == 0.0, (la.item(), lb.item())
    print('[4c] sdf_w=0 恒等起步 ok：总损失 %.6f 与基线逐位相同' % la.item())

    # ---- [5] 深监督项 ----
    net_ds.zero_grad(set_to_none=True)
    out = net_ds(img)
    l_total = _ce_dice(out['out'], label, ce, dice)
    parts = []
    for aux_logits, w_aux in out['aux']:
        up = F.interpolate(aux_logits, size=label.shape[-2:], mode='bilinear', align_corners=False)
        l_aux = _ce_dice(up, label, ce, dice)
        parts.append((w_aux, l_aux.item()))
        l_total = l_total + w_aux * l_aux
    l_total.backward()
    g_ds = sum(p.grad.abs().sum() for n, p in net_ds.named_parameters()
               if p.grad is not None and n.startswith('ds_heads'))
    g_rest = sum(p.grad.abs().sum() for n, p in net_ds.named_parameters()
                 if p.grad is not None and not n.startswith('ds_heads'))
    assert torch.isfinite(l_total).item() and torch.isfinite(g_ds).item() and g_ds.item() > 0
    assert g_rest.item() > 0
    print('[5] 深监督 ok：%s | 总损失 %.4f | 梯度 ds_heads %.3e / 其余 %.3e'
          % (['w=%.2f L=%.4f' % p for p in parts], l_total.item(), g_ds.item(), g_rest.item()))

    # ---- [6] 一致性项 ----
    net = _make_unet()
    net.eval()
    with torch.no_grad():
        p1 = torch.softmax(net(img), 1)[:, 1:2]
        p2 = torch.softmax(net(img), 1)[:, 1:2]
        c0 = _cons(p1, p2).item()
        img2 = img * (1.0 + torch.randn_like(img) * 0.08)
        img2 = img2 * (1.0 + (torch.rand(2, 1, 1, 1) * 2 - 1) * 0.1)
        p3 = torch.softmax(net(img2), 1)[:, 1:2]
        c1 = _cons(p1, p3).item()
        flip = ((p1 > 0.5) != (p3 > 0.5)).float().mean().item()
        maxdiff = (p1 - p3).abs().max().item()
    # 同图两次前向在 float32 累加下不是精确 0（实测 2.3e-6），故阈值放到 1e-5
    assert abs(c0) < 1e-5, c0
    assert 0.0 < c1 < 1.0, c1
    assert maxdiff > 0, maxdiff   # 扰动必须真的改变了预测，否则说明扰动没接上
    net.train()
    net.zero_grad(set_to_none=True)
    l1 = _ce_dice(net(img), label, ce, dice)
    p1 = torch.softmax(net(img), 1)[:, 1:2]
    p3 = torch.softmax(net(img2), 1)[:, 1:2]
    (l1 + 0.2 * _cons(p1, p3)).backward()
    g_cons = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
    assert torch.isfinite(g_cons).item() and g_cons.item() > 0
    print('[6] 一致性 ok：同图 %.2e（应 0）/ 扰动后 %.6f | 概率翻转像素 %.2f%%'
          ' | 概率最大差 %.4f | 梯度 %.3e'
          % (c0, c1, 100.0 * flip, maxdiff, g_cons.item()))

    # ---- [7] 空前景 ----
    empty = torch.zeros(2, 224, 224, dtype=torch.long)
    t_empty = _sdf_target(empty, tau).unsqueeze(1)
    with torch.no_grad():
        e = sdf_term(net_sdf(img)['sdf'], t_empty)
    assert torch.isfinite(e).item() and e.item() == 0.0, e
    print('[7] 空前景 ok：窄带为空，几何场项 = 0.0（无 NaN）')

    # ---- [8] R50 侧结构检查 ----
    if not args.no_r50:
        import copy
        from networks.vit_seg_modeling import CONFIGS, VisionTransformer

        def _r50(sdf='none', ds=False):
            # 每次深拷贝配置：VisionTransformer 会就地改 config，共享对象会串味
            cfg = copy.deepcopy(CONFIGS['R50-ViT-B_16'])
            cfg.n_classes = 2
            cfg.n_skip = 3
            cfg.transformer.num_layers = 0
            cfg.patches.grid = (14, 14)
            torch.manual_seed(1234)
            return VisionTransformer(cfg, img_size=224, num_classes=2,
                                     sdf=sdf, sdf_tau=tau, ds=ds)

        r50 = _r50(sdf='aux', ds=True)
        n_sdf = sum(p.numel() for p in r50.sdf_head.parameters())
        n_ds = sum(p.numel() for p in r50.decoder.ds_heads.parameters())
        print('[8a] R50 参数量：%.3f M | sdf 头 +%.3f k | 深监督两头 +%.3f k'
              % (_n_params(r50) / 1e6, n_sdf / 1e3, n_ds / 1e3))
        r50.eval()
        with torch.no_grad():
            o = r50(img[:1])
        assert isinstance(o, dict) and 'sdf' in o and 'aux' in o
        assert [tuple(t.shape[-2:]) for t, _ in o['aux']] == [(56, 56), (112, 112)]
        assert [w for _, w in o['aux']] == [0.15, 0.3]
        assert o['out'].shape == (1, 2, 224, 224) and o['sdf'].shape == (1, 1, 224, 224)
        print('[8b] R50 输出契约 ok：out %s | sdf %s | aux %s 权重 %s'
              % (tuple(o['out'].shape), tuple(o['sdf'].shape),
                 [tuple(t.shape[-2:]) for t, _ in o['aux']],
                 [w for _, w in o['aux']]))

    print('ALL PASS')


if __name__ == '__main__':
    main()
