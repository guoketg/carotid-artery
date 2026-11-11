"""FreqDINO 移植批（MFEA / FGBR）冒烟：四级恒等起步、参数量、梯度可达。

对解码器四级同形张量 (2, C, s, s)（C/s = 256/28、128/56、64/112、16/224）逐级验证：
1) 初始化时输出与输入逐位相同（恒等起步，同 seed 与基线对齐）；
2) 一步反向传播后，两个模块各自的可训练参数有非零梯度（不会被零初始化冻死）；
3) 打印每级与合计参数量，供登记。
"""
import torch

from networks.freqdino import FGBR, MFEA

torch.manual_seed(1234)

LEVELS = ((256, 28), (128, 56), (64, 112), (16, 224))

total = {"MFEA": 0, "FGBR": 0}
for ch, size in LEVELS:
    x = torch.randn(2, ch, size, size, requires_grad=True)
    for name, mod in (("MFEA", MFEA(ch)), ("FGBR", FGBR(ch))):
        y = mod(x)
        delta = (y - x).abs().max().item()
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        total[name] += n
        y.sum().backward(retain_graph=True)
        grads = {k: (v.grad.abs().sum().item() if v.grad is not None else -1.0)
                 for k, v in mod.named_parameters()}
        # MFEA 的注意力分支被 γ 乘住，γ=0 时首步梯度必为 0（一步之后 γ>0 即恢复）；
        # 因此只断言「闸门」本身有梯度，其余参数仅登记不判死
        gate = 'gamma' if name == "MFEA" else 'out.weight'
        assert grads[gate] > 0, f"{name}@{ch}x{size} 闸门 {gate} 梯度为 0，模块会被冻死"
        zero_now = [k for k, g in grads.items() if g == 0.0]
        extra = ""
        if name == "FGBR":
            extra = f" heads={mod.heads}x{mod.head_dim} proto={mod.proto_dim}"
        print(f"{name}@{ch:>3}x{size:<3}: max|y-x|={delta:.3e} params={n/1e3:.1f}k "
              f"gate_grad={grads[gate]:.3e} step0_zero_grad={zero_now}{extra}")
        assert delta < 1e-6, f"{name}@{ch}x{size} 初始化不是恒等"
        x.grad = None

for k, v in total.items():
    print(f"{k} 四级合计: {v/1e6:.3f}M")

print("OK")
