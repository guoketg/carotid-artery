# coding=utf-8
"""冒烟：编码器 FGR ↔ 解码器 DDBE 跨位置耦合（2026-11-14）。

CPU-only（GPU 让给训练队列）。断言六件：
  1. 恒等起步：同 seed 下「各挂其位组合臂」与「耦合臂」的 logits 都与基线逐位相等
     （DDBE.fuse 零初始化；耦合只是把门乘进频带支，不引入新的零点）。
  2. 参数量：耦合臂与组合臂**逐位相同**（零参数接口）——两臂成对比较不受容量干扰。
  3. 门确实生效：把该级 DDBE 的 fuse 权重置为非零后，换一个门（全 0.5 → 全 2.0）
     输出必须改变；同时 band_gate=None 与不传门的前向逐位一致（历史口径不变）。
  4. 门按分辨率配对：解码第 level 级吃的是编码器同级 FGR 的门，形状 (B,2C,1,1) 与
     该级 DDBE 的 HF/LF 通道数一致（28²/512 ↔ 512、224²/64 ↔ 64）。
  5. 梯度链路：第 0 步 fuse 有梯度、FGR 编码器门 weight 梯度为 0（fuse=0 挡住），
     1 步 SGD 后门 weight 梯度非零——「一条链上只有一个零」不被冻死。
  6. train.py 旗标接线：缺 --ddbe 4 / 缺 --fgr 1 --fgr_pos enc / 换 --arch baseline
     都必须 parser.error（exit 2）。
"""
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch

from networks.ddbe import DDBE
from networks.unet import UNet

SEED = 1234
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')


def build_unet(**kw):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    return UNet(in_channels=1, num_classes=2, **kw)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def check_identity(x):
    arms = [
        ('baseline', {}),
        ('combo_encddbe', dict(ddbe_level=4, fgr=True, fgr_pos='enc')),
        ('coupled', dict(ddbe_level=4, fgr=True, fgr_pos='enc', couple=True)),
    ]
    base = build_unet(**arms[0][1]).eval()
    p0 = n_params(base)
    with torch.no_grad():
        y0 = base(x)
    print('[UNet] baseline params=%d' % p0)
    ok = True
    params = {}
    for name, kw in arms[1:]:
        net = build_unet(**kw).eval()
        p = n_params(net)
        params[name] = p
        with torch.no_grad():
            y = net(x)
        dmax = (y - y0).abs().max().item()
        good = dmax == 0.0
        ok = ok and good
        print('  %-14s params=%d (+%d)  identity=%s  %s'
              % (name, p, p - p0, dmax, 'OK' if good else 'FAIL'))
        del net, y
    same = params['combo_encddbe'] == params['coupled']
    ok = ok and same
    print('  [param] 耦合臂 vs 组合臂 delta=%d  %s'
          % (params['coupled'] - params['combo_encddbe'], 'OK' if same else 'FAIL'))
    del base, y0
    return ok


def module_check():
    """DDBE 的 band_gate 接口：None 时不改数值，非空时确实改输出。"""
    torch.manual_seed(SEED)
    C, H = 8, 8
    x = torch.randn(2, C, H, H)
    m = DDBE(C)
    # 把唯一零点解开，才能观察门的作用（恒等起步时 fuse=0，任何门都被乘成 0）
    torch.nn.init.normal_(m.fuse.weight, std=0.1)
    torch.nn.init.zeros_(m.fuse.bias)
    with torch.no_grad():
        y_plain = m(x)
        y_none = m(x, band_gate=None)
        g1 = torch.full((2, 2 * C, 1, 1), 0.5)
        g2 = torch.full((2, 2 * C, 1, 1), 2.0)
        y_g1 = m(x, band_gate=g1)
        y_g2 = m(x, band_gate=g2)
    d_none = (y_none - y_plain).abs().max().item()
    d_gate = (y_g1 - y_g2).abs().max().item()
    ok = d_none == 0.0 and d_gate > 0.0
    print('[module] DDBE band_gate=None 与不传门: %.1e（应为 0）；两种门输出差: %.3e（应 >0）  %s'
          % (d_none, d_gate, 'OK' if ok else 'FAIL'))
    return ok


def routing_check():
    """门按分辨率投递：解码各级 DDBE 的通道数与对应编码器 FGR 的门通道数配对。"""
    net = build_unet(ddbe_level=4, fgr=True, fgr_pos='enc', couple=True).eval()
    widths = []
    for i, m in enumerate(net.fgr_modules):
        widths.append(m.gate.out_features // 2)     # 门是 2C，前 C 管 hf、后 C 管 lf
    dec = [m.hf_in.out_channels if hasattr(m, 'hf_in') else None for m in net.ddbe_modules]
    # 编码器 FGR 顺序 0..3 = 224²/64、112²/128、56²/256、28²/512
    # 解码器 DDBE 顺序 0..3 = 28²/512、56²/256、112²/128、224²/64
    ok = dec == list(reversed(widths))
    print('[routing] FGR 门通道（浅→深）=%s  DDBE 通道（深→浅）=%s  %s'
          % (widths, dec, 'OK' if ok else 'FAIL'))
    # 端到端跑一遍，确认门真的被投递（改一个编码器门 bias → 输出变化）
    torch.manual_seed(SEED + 1)
    x = torch.randn(1, 1, 224, 224)
    with torch.no_grad():
        y_a = net(x).clone()
        for m in net.ddbe_modules:
            torch.nn.init.normal_(m.fuse.weight, std=0.02)
        y_b = net(x).clone()
        net.fgr_modules[3].gate.bias.add_(1.0)      # 最深一级（28²）的门
        y_c = net(x).clone()
    d_gate = (y_c - y_b).abs().max().item()
    ok = ok and d_gate > 0.0
    print('[routing] 改最深一级 FGR 门 bias 后的输出变化: %.3e（应 >0）  %s'
          % (d_gate, 'OK' if d_gate > 0 else 'FAIL'))
    return ok


def grad_check():
    """梯度链路：fuse 第 0 步即有权重梯度；FGR 门在 1 步之后拿到非零梯度。"""
    torch.manual_seed(SEED)
    net = build_unet(ddbe_level=4, fgr=True, fgr_pos='enc', couple=True)
    x = torch.randn(1, 1, 224, 224)
    params = [p for p in net.parameters()
              if any(p is q for m in list(net.ddbe_modules) + list(net.fgr_modules)
                     for q in m.parameters())]
    opt = torch.optim.SGD(params, lr=0.1)
    fuse_g, gate_g = [], []
    for _ in range(3):
        opt.zero_grad()
        net(x).pow(2).mean().backward()
        fuse_g.append(sum(m.fuse.weight.grad.abs().sum().item()
                          for m in net.ddbe_modules))
        gate_g.append(sum(m.gate.weight.grad.abs().sum().item()
                          for m in net.fgr_modules))
        opt.step()
    print('[grad] DDBE.fuse=%s  FGR.gate=%s'
          % (['%.2e' % v for v in fuse_g], ['%.2e' % v for v in gate_g]))
    ok = fuse_g[0] > 0 and gate_g[0] == 0.0 and gate_g[1] > 0
    print('[grad] 第 0 步 fuse 有梯度 / 门为 0 → 1 步后门非零: %s' % ('OK' if ok else 'FAIL'))
    return ok


def arg_check():
    cases = [
        (['--dataset', 'Carotid', '--arch', 'unet', '--fgr_ddbe_couple', '1',
          '--fgr', '1', '--fgr_pos', 'enc'], 'ddbe'),
        (['--dataset', 'Carotid', '--arch', 'unet', '--fgr_ddbe_couple', '1',
          '--ddbe', '4'], 'fgr'),
        (['--dataset', 'Carotid', '--arch', 'baseline', '--fgr_ddbe_couple', '1',
          '--ddbe', '4', '--fgr', '1', '--fgr_pos', 'enc'], 'unet'),
    ]
    ok = True
    for argv, key in cases:
        r = subprocess.run([PYTHON, 'train.py'] + argv, cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        tail = (r.stderr or r.stdout or '').strip().splitlines()
        tail = tail[-1] if tail else ''
        good = r.returncode == 2 and key in tail
        ok = ok and good
        print('[args] %s -> exit=%d %s  %s'
              % (' '.join(argv[-4:]), r.returncode, tail, 'OK' if good else 'FAIL'))
    return ok


ok = True
x = torch.randn(1, 1, 224, 224)
ok &= check_identity(x)
ok &= module_check()
ok &= routing_check()
ok &= grad_check()
ok &= arg_check()
print('SMOKE OK' if ok else 'SMOKE FAILED')
sys.exit(0 if ok else 1)
