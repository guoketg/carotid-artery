# coding=utf-8
"""冒烟：编码器分带重投影（浅层 / 深层 / 四级 / 跨级门 / 随机频带对照，2026-11-18）。

设计动机见 docs/module-suite-design.md §11 与 docs/results.md §5：
  R50 上唯一有正证据的机制是解码器的「分带 → 拼接 → 1×1 重投影 → 零初始化残差」
  （DDBE 四级 +0.435 / 拉普拉斯 +0.506，plain 对照 +0.060 → 增益来自频率内容）；
  而所有「门控回注」式编码器模块在 R50 上全负（FGR 编码器四级 −1.64）。
  本模块把前者搬到编码器浅层/深层，做位置消融。

断言六项（CPU only，GPU 让给训练队列）：
  1. 恒等起步：各级 BandReproject 与整栈在 fuse 零初始化下 y == x 逐位相等（含 55² 奇数分辨率）。
  2. 参数量：4C²+31C/级，逐级与整栈增量精确；跨级门每级多 1 个标量。
  3. 梯度链路：第 0 步只有 1×1 重投影有梯度（频带支/上下文支/门在零点之后），
     2 步 SGD 后全部解锁 —— 仍是「一条链上只有一个零」。
  4. 随机频带对照臂真的换了内容：把 proj 手动置随机后，对照臂与非对照臂输出必须不同。
  5. train.py 旗标接线：--brp 配 unet、--brp_gate 不带 --brp、--brp_levels 越界都必须 exit 2。
  6. R50 端到端（--arch baseline --vit_layers 0）：浅层两级与深层两级的 logits 与基线逐位相等，
     参数增量与公式一致。
"""
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch

from networks.band_reproject import BandReproject, EncoderBandStack, band_reproject_params

SEED = 1234
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')
SPECS = ((64, 112, 112), (256, 55, 55), (512, 28, 28), (1024, 14, 14))


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def module_check():
    ok = True
    torch.manual_seed(SEED)
    for c, h, w in SPECS:
        x = torch.randn(2, c, h, w)
        m = BandReproject(c, h, w).eval()
        with torch.no_grad():
            d = (m(x) - x).abs().max().item()
        good = d == 0.0
        ok = ok and good
        print('[module] %3dch %3dx%-3d identity=%.1e %s' % (c, h, w, d, 'OK' if good else 'FAIL'))
    # 跨级门形态：浅级 map 下采样到深级分辨率后相乘，仍须恒等
    torch.manual_seed(SEED)
    st = EncoderBandStack(SPECS, (1, 2), use_gate=True).eval()
    x1 = torch.randn(2, 64, 112, 112)
    x2 = torch.randn(2, 256, 55, 55)
    with torch.no_grad():
        y1 = st(1, x1)
        y2 = st(2, x2)
    d1 = (y1 - x1).abs().max().item()
    d2 = (y2 - x2).abs().max().item()
    print('[module] stack(1,2)+gate identity=%.1e/%.1e %s'
          % (d1, d2, 'OK' if (d1 == 0.0 and d2 == 0.0) else 'FAIL'))
    return ok and d1 == 0.0 and d2 == 0.0


def param_check():
    ok = True
    cases = [((1, 2), False), ((3, 4), False), ((1, 2, 3, 4), False), ((1, 2), True)]
    for levels, gate in cases:
        torch.manual_seed(SEED)
        st = EncoderBandStack(SPECS, levels, use_gate=gate)
        expect = sum(band_reproject_params(SPECS[i - 1][0]) for i in levels)
        if gate:
            expect += len(levels)
        got = n_params(st)
        good = got == expect
        ok = ok and good
        print('[param] levels=%s gate=%d total=%d (expect %d) %s'
              % (''.join(str(i) for i in levels), int(gate), got, expect,
                 'OK' if good else 'FAIL'))
    return ok


def grad_check():
    torch.manual_seed(SEED)
    m = BandReproject(32, 16, 16)
    x = torch.randn(2, 32, 16, 16)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    proj_g, ctx_g = [], []
    for _ in range(3):
        opt.zero_grad()
        m(x).pow(2).mean().backward()
        proj_g.append(m.proj.weight.grad.abs().sum().item())
        ctx_g.append(m.ctx_dw.weight.grad.abs().sum().item())
        opt.step()
    print('[grad] proj=%s ctx=%s' % (['%.2e' % v for v in proj_g], ['%.2e' % v for v in ctx_g]))
    good = proj_g[0] > 0 and ctx_g[0] == 0.0 and ctx_g[1] > 0

    # 跨级门：浅级 α 的梯度只能经深级 proj 回流（第 0 步深级 proj 为零 ⇒ α 必为 0），
    # 2 步后必须解锁；同时深级 proj 第 0 步必须有梯度。
    torch.manual_seed(SEED)
    st = EncoderBandStack(SPECS, (1, 2), use_gate=True)
    x1 = torch.randn(2, 64, 112, 112)
    x2 = torch.randn(2, 256, 55, 55)
    opt2 = torch.optim.SGD(st.parameters(), lr=0.1)
    a_g, p2_g = [], []
    for _ in range(3):
        opt2.zero_grad()
        (st(1, x1).pow(2).mean() + st(2, x2).pow(2).mean()).backward()
        a_g.append(st.mods[0].alpha.grad.abs().sum().item())
        p2_g.append(st.mods[1].proj.weight.grad.abs().sum().item())
        opt2.step()
    print('[grad] 浅级α=%s 深级proj=%s' % (['%.2e' % v for v in a_g], ['%.2e' % v for v in p2_g]))
    good = good and a_g[0] == 0.0 and a_g[1] > 0 and p2_g[0] > 0
    return good


def control_check():
    """对照臂必须真的换掉频带内容：proj 手动置随机后两臂输出必须不同。"""
    torch.manual_seed(SEED)
    x = torch.randn(2, 16, 8, 8)
    a = BandReproject(16, 8, 8)
    b = BandReproject(16, 8, 8, control=True)
    torch.manual_seed(999)
    with torch.no_grad():
        for mod in (a, b):
            mod.proj.weight.normal_()
    with torch.no_grad():
        ya, yb = a(x), b(x)
    diff = (ya - yb).abs().max().item()
    print('[control] 随机频带对照 vs 真实频带 max|Δ|=%.3e %s' % (diff, 'OK' if diff > 1e-3 else 'FAIL'))
    return diff > 1e-3


def arg_check():
    cases = [
        (['--dataset', 'Carotid', '--arch', 'unet', '--brp', '1'], 'brp'),
        (['--dataset', 'Carotid', '--arch', 'baseline', '--vit_layers', '0',
          '--brp_gate', '1'], 'brp_gate'),
        (['--dataset', 'Carotid', '--arch', 'baseline', '--vit_layers', '0',
          '--brp', '1', '--brp_levels', '5'], 'brp_levels'),
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


def check_r50():
    import copy
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer

    def build(**kw):
        cfg = copy.deepcopy(CONFIGS['R50-ViT-B_16'])
        cfg.n_classes = 2
        cfg.n_skip = 3
        cfg.transformer.num_layers = 0
        cfg.patches.grid = (int(224 / 16), int(224 / 16))
        torch.manual_seed(SEED)
        return VisionTransformer(cfg, img_size=224, num_classes=2, **kw).eval()

    x = torch.randn(1, 1, 224, 224)
    base = build()
    p0 = n_params(base)
    with torch.no_grad():
        y0 = base(x)
    print('[R50] baseline params=%d' % p0)
    ok = True
    for levels, gate, expect_levels in (('1,2', False, (1, 2)), ('3,4', False, (3, 4))):
        net = build(brp=True, brp_levels=levels, brp_gate=gate)
        p = n_params(net)
        with torch.no_grad():
            y = net(x)
        dmax = (y - y0).abs().max().item()
        expect = sum(band_reproject_params(SPECS[i - 1][0]) for i in expect_levels)
        delta = p - p0
        good = dmax == 0.0 and delta == expect
        ok = ok and good
        print('[R50] brp_levels=%s delta=+%d (expect %d) identity=%.1e %s'
              % (levels, delta, expect, dmax, 'OK' if good else 'FAIL'))
        del net, y
    del base, y0
    return ok


ok = True
ok &= module_check()
ok &= param_check()
ok &= grad_check()
ok &= control_check()
ok &= arg_check()
ok &= check_r50()
print('SMOKE OK' if ok else 'SMOKE FAILED')
sys.exit(0 if ok else 1)
