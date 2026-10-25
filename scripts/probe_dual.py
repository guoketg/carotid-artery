"""R0 探针：双骨干（D11）参数量 / shape 契约 / gamma=0 等价性 / 显存与单步耗时。

用法（仓库根目录）：
    .\\venv\\Scripts\\python.exe scripts\\probe_dual.py
    .\\venv\\Scripts\\python.exe scripts\\probe_dual.py --batch_sizes 4,6,8 --steps 30
    .\\venv\\Scripts\\python.exe scripts\\probe_dual.py --fuse_op gate --levels 1,2,3,4
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.dual_encoder_model import DualEncoderTransUNet
from networks.vit_seg_modeling import CONFIGS
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils import DiceLoss


def build_config(img_size=224, num_classes=2):
    config = CONFIGS['R50-ViT-B_16']
    config.n_classes = num_classes
    config.n_skip = 3
    config.patches.grid = (img_size // 16, img_size // 16)
    return config


def n_param(module):
    return sum(p.numel() for p in module.parameters())


def report_params(net):
    total = n_param(net)
    side = n_param(net.side_res) + n_param(net.side_proj)
    fuse = n_param(net.fuse)
    main = total - side - fuse
    print(f"[params] total = {total/1e6:.2f} M")
    print(f"[params]   main branch (R50-ViT + decoder) = {main/1e6:.2f} M")
    print(f"[params]   side branch (R50 + 1x1 proj)    = {side/1e6:.2f} M")
    print(f"[params]   fusion modules                  = {fuse/1e6:.2f} M")
    for k, m in net.fuse.items():
        print(f"[params]     F{k}: {type(m).__name__:<22s} {n_param(m)/1e6:.3f} M")


@torch.no_grad()
def trace_shapes(net, img_size=224):
    """按 _encode_interleaved 的顺序打印每一级形状。"""
    net.eval()
    x = torch.randn(2, 1, img_size, img_size, device='cuda')
    mres, sres = net.transformer.embeddings.hybrid_model, net.side_res
    from networks.dual_encoder_model import build_detail_channels
    side_in = build_detail_channels(x, img_size)
    print(f"[shape] input            {tuple(x.shape)}   side {tuple(side_in.shape)}")

    a = mres.root(x.repeat(1, 3, 1, 1))
    b = sres.root(side_in)
    print(f"[shape] root             A {tuple(a.shape)}  B {tuple(b.shape)}   -> F1")
    a, b = net._fuse(1, a, b)
    f1 = a
    a, b = net._pool(a), net._pool(b)
    a, b = mres.body[0](a), sres.body[0](b)
    print(f"[shape] block1           A {tuple(a.shape)}  B {tuple(b.shape)}")
    a_f = net._pad_skip(a, img_size // 4)
    b_f = net._pad_skip(b, img_size // 4)
    print(f"[shape] pad to skip      A {tuple(a_f.shape)}  B {tuple(b_f.shape)}   -> F2")
    a_f, b_f = net._fuse(2, a_f, b_f)
    f2 = a_f
    a = a_f[:, :, :a.size(2), :a.size(3)]
    b = b_f[:, :, :b.size(2), :b.size(3)]
    a, b = mres.body[1](a), sres.body[1](b)
    print(f"[shape] block2           A {tuple(a.shape)}  B {tuple(b.shape)}   -> F3")
    a, b = net._fuse(3, a, b)
    f3 = a
    a, b = mres.body[2](a), sres.body[2](b)
    b = net.side_proj(b)
    print(f"[shape] block3           A {tuple(a.shape)}  B {tuple(b.shape)} (proj 768)")
    emb = net.transformer.embeddings
    t = emb.patch_embeddings(a).flatten(2).transpose(-1, -2)
    print(f"[shape] patch embed      {tuple(t.shape)}")
    t = emb.dropout(t + emb.position_embeddings)
    for blk in net.transformer.encoder.layer:
        t, _ = blk(t)
    t = net.transformer.encoder.encoder_norm(t)
    h = int(t.size(1) ** 0.5)
    ta = t.transpose(1, 2).reshape(t.size(0), t.size(2), h, h)
    print(f"[shape] ViT out          {tuple(t.shape)} -> {tuple(ta.shape)}   -> F4")
    ta, b = net._fuse(4, ta, b)
    print(f"[shape] side after F4    {tuple(b.shape)}")
    del a, b, t, ta, f1, f2, f3
    torch.cuda.empty_cache()


@torch.no_grad()
def check_equivalence(args, config, weights):
    """gamma=0 时，dual 应与 baseline 逐元素等价（< 1e-5）。"""
    torch.manual_seed(1234)
    base = ViT_seg(config, img_size=args.img_size, num_classes=config.n_classes).cuda()
    torch.manual_seed(1234)
    dual = DualEncoderTransUNet(config, img_size=args.img_size,
                                num_classes=config.n_classes,
                                levels=args.levels, fuse_op=args.fuse_op,
                                window=args.window, dk_ratio=args.dk_ratio,
                                bidir=bool(args.bidir),
                                side_pretrained=bool(args.side_pretrained),
                                cross_scale=bool(getattr(args, 'cross_scale', 0))).cuda()
    base.load_from(weights=np.load(config.pretrained_path))
    dual.load_from(weights=np.load(config.pretrained_path))

    base.eval()
    dual.eval()

    g = torch.Generator(device='cuda').manual_seed(0)
    x = torch.randn(2, 1, args.img_size, args.img_size, device='cuda', generator=g)
    y_base = base(x)
    y_dual = dual(x)
    diff = (y_base - y_dual).abs().max().item()
    rel = diff / y_base.abs().max().item()
    print(f"[equiv] gamma=0  max|baseline-dual| = {diff:.3e}  (rel {rel:.3e})")
    print(f"[equiv] output {tuple(y_dual.shape)}  -> {'PASS' if diff < 1e-5 else 'FAIL'}")
    del base, dual
    torch.cuda.empty_cache()
    return diff < 1e-5


def _timeit(fn, repeat=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat


def profile_sections(args, config, weights, batch_size=4):
    """逐段计时，找出慢在哪。"""
    from networks.dual_encoder_model import build_detail_channels
    torch.manual_seed(1234)
    net = DualEncoderTransUNet(config, img_size=args.img_size,
                               num_classes=config.n_classes,
                               levels=args.levels, fuse_op=args.fuse_op,
                               window=args.window, dk_ratio=args.dk_ratio,
                               bidir=bool(args.bidir),
                               side_pretrained=bool(args.side_pretrained),
                               cross_scale=bool(getattr(args, 'cross_scale', 0))).cuda()
    net.load_from(weights=weights)
    net.train()
    mres, sres = net.transformer.embeddings.hybrid_model, net.side_res
    x = torch.randn(batch_size, 1, args.img_size, args.img_size, device='cuda')
    s4 = args.img_size // 4

    def fwd_detail():
        with torch.no_grad():
            build_detail_channels(x, args.img_size)

    with torch.no_grad():
        side_in = build_detail_channels(x, args.img_size)
        a1 = mres.root(x.repeat(1, 3, 1, 1))
        b1 = sres.root(side_in)
        a2 = mres.body[0](net._pool(a1))
        b2 = sres.body[0](net._pool(b1))
        a2p = net._pad_skip(a2, s4)
        b2p = net._pad_skip(b2, s4)
        a3 = mres.body[1](a2p)
        b3 = sres.body[1](b2p)
        a4 = mres.body[2](a3)
        b4 = net.side_proj(sres.body[2](b3))
        emb = net.transformer.embeddings
        t0 = emb.dropout(emb.patch_embeddings(a4).flatten(2).transpose(-1, -2)
                         + emb.position_embeddings)
        t1 = net.transformer.encoder.encoder_norm(
            torch.stack([blk(t0)[0] for blk in net.transformer.encoder.layer])[-1])
        h = int(t1.size(1) ** 0.5)
        t1_2d = t1.transpose(1, 2).reshape(t1.size(0), t1.size(2), h, h)

    def run(fn):
        with torch.no_grad():
            return _timeit(fn)

    def row(name, ms):
        print(f'[prof]   {name:<34s}{ms*1000:7.1f} ms')

    print(f'[prof] batch={batch_size}  (forward only, no_grad)')
    row('detail channels', run(fwd_detail))
    row('main root 112^2', run(lambda: mres.root(x.repeat(1, 3, 1, 1))))
    row('side root 112^2', run(lambda: sres.root(side_in)))
    row('F1 window attn 112^2', run(lambda: net._fuse(1, a1, b1)))
    row('main block1', run(lambda: mres.body[0](net._pool(a1))))
    row('side block1', run(lambda: sres.body[0](net._pool(b1))))
    row('F2 window attn 56^2', run(lambda: net._fuse(2, a2p, b2p)))
    row('main block2', run(lambda: mres.body[1](a2p)))
    row('side block2', run(lambda: sres.body[1](b2p)))
    row('F3 global attn 28^2', run(lambda: net._fuse(3, a3, b3)))
    row('main block3', run(lambda: mres.body[2](a3)))
    row('side block3 + proj', run(lambda: net.side_proj(sres.body[2](b3))))
    row('patch embed', run(lambda: emb.patch_embeddings(a4)))
    row('12 x ViT block', run(lambda: [blk(t0) for blk in net.transformer.encoder.layer]))
    row('F4 global attn 14^2', run(lambda: net._fuse(4, t1_2d, b4)))
    row('decoder + head', run(lambda: net.segmentation_head(
        net.decoder(t1, [a3, a2p, a1]))))

    torch.manual_seed(1234)
    base = ViT_seg(config, img_size=args.img_size, num_classes=config.n_classes).cuda()
    base.load_from(weights=weights)
    base.train()

    def base_step():
        out = base(x)
        loss = out.mean()
        loss.backward()
        base.zero_grad(set_to_none=True)

    def dual_step():
        out = net(x)
        loss = out.mean()
        loss.backward()
        net.zero_grad(set_to_none=True)

    print(f'[prof] dual fwd+bwd      {_timeit(dual_step, repeat=3)*1000:7.1f} ms')
    print(f'[prof] baseline fwd+bwd  {_timeit(base_step, repeat=3)*1000:7.1f} ms')
    del net, base
    torch.cuda.empty_cache()


def probe_memory(args, config, weights, batch_size, steps):
    torch.manual_seed(1234)
    net = DualEncoderTransUNet(config, img_size=args.img_size,
                               num_classes=config.n_classes,
                               levels=args.levels, fuse_op=args.fuse_op,
                               window=args.window, dk_ratio=args.dk_ratio,
                               bidir=bool(args.bidir),
                               side_pretrained=bool(args.side_pretrained),
                               cross_scale=bool(getattr(args, 'cross_scale', 0))).cuda()
    net.load_from(weights=np.load(config.pretrained_path))
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    dice = DiceLoss(config.n_classes)

    x = torch.randn(batch_size, 1, args.img_size, args.img_size, device='cuda')
    y = (torch.rand(batch_size, args.img_size, args.img_size, device='cuda') > 0.7).long()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # warmup 2 步：cudnn 自动调优、分配器稳定
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = ce(net(x), y) + dice(net(x), y, softmax=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = net(x)
        loss = ce(out, y) + dice(out, y, softmax=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[mem] batch={batch_size:2d} steps={steps:3d}  peak={peak:5.2f} GB  "
          f"step={dt*1000:6.1f} ms  loss={loss.item():.4f}")
    del net, opt, x, y
    torch.cuda.empty_cache()
    return peak, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--levels', type=str, default='1,2,3,4')
    p.add_argument('--fuse_op', type=str, default='auto')
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--dk_ratio', type=int, default=2)
    p.add_argument('--bidir', type=int, default=1)
    p.add_argument('--side_pretrained', type=int, default=1)
    p.add_argument('--cross_scale', type=int, default=0)
    p.add_argument('--batch_sizes', type=str, default='4,6,8')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--skip_equiv', action='store_true')
    p.add_argument('--profile', action='store_true', help='逐段计时 + baseline 对比')
    p.add_argument('--profile_batch', type=int, default=4)
    args = p.parse_args()

    assert torch.cuda.is_available(), 'CUDA not available'
    print(f"[env] GPU = {torch.cuda.get_device_name(0)}")
    config = build_config(args.img_size)
    weights = np.load(config.pretrained_path)

    torch.manual_seed(1234)
    net = DualEncoderTransUNet(config, img_size=args.img_size,
                               num_classes=config.n_classes,
                               levels=args.levels, fuse_op=args.fuse_op,
                               window=args.window, dk_ratio=args.dk_ratio,
                               bidir=bool(args.bidir),
                               side_pretrained=bool(args.side_pretrained),
                               cross_scale=bool(getattr(args, 'cross_scale', 0))).cuda()
    report_params(net)
    trace_shapes(net, args.img_size)
    del net
    torch.cuda.empty_cache()

    if not args.skip_equiv:
        ok = check_equivalence(args, config, weights)
        if not ok:
            print('[equiv] WARNING: 等价性未通过，先别开夜间队列')

    if args.profile:
        profile_sections(args, config, weights, args.profile_batch)

    for bs in [int(v) for v in args.batch_sizes.split(',') if v.strip()]:
        try:
            probe_memory(args, config, weights, bs, args.steps)
        except torch.cuda.OutOfMemoryError:
            print(f'[mem] batch={bs:2d} OOM')
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
