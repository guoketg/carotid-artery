"""FGR：Frequency-Gated Rewriter（2026-11-09 自研规划，与 ddbectl/plain 对照互补）。

动机（results.md §7(3)）：DDBE 的增益可能来自「拼接改写」而非频率内容——
ddbectl/plain 把三支改成普通特征，检验「没有频率内容还剩多少增益」；
FGR 反向检验「只有频带内容 + 轻门控」：门不看频带本身，只看各通道的
DCT 谱描述子，改写只作用于两条频带支。若 FGR ≈ DDBE 且 plain ≈ 0，
则增益确实来自频率信息的利用方式而非拼接结构。

结构（恒等起步，「一条链上只能有一个零」）：
    x ────────────────────────────────┐（残差主干）
    ├─ laplacian_split → hf = |LH|+|HL|+|HH|（双线性上采样回原分辨率）、lf = LL
    ├─ DCT 谱描述子：z_c = <cat[x,hf,lf]_c, base_c>，固定多谱基（借用公开算子：
    │   FcaNet, ICCV 2021 的多谱通道注意力思想，此处按论文口径固定 4×4 频率网格）
    ├─ 门 w = σ(Linear(z))，零权重 + bias 2.0 → 初始 w ≈ σ(2.0) ≈ 0.88（接近全开）
    └─ y = x + fuse(w ⊙ [hf, lf])，fuse 1×1 零初始化（唯一零点）

门只作用于频带支（2C）而不作用于 x 主干：fuse 零初始化下输出逐位 = x，
第 0 步梯度就流向 fuse（直连）与 gate（经 fuse 移动后）——与 DDBE 同型，
不属于「双重零冻死」。

mode='band'（FBGR 分带档，2026-11-10）：R50 上 FGR 与 DDBE-lap 同点串联互覆
（90.632 < 单臂 90.795/90.996），诊断是 single-core 改写把两带内容搅在同一
2C→C 投影里，与 DDBE 的拼接改写形成同型算子互抢。band 档把改写拆成两枚
C→C 零核并联：y = x + fuse_hf(w_hf⊙hf) + fuse_lf(w_lf⊙lf)，高低频各走
各核互不串带；gate 仍是一枚 3C→2C Linear（前 C 列管 hf、后 C 列管 lf），
DCT 谱描述子完全复用，不新增门。参数量 8C²+4C/级（比 whole 档多 C 个 bias）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import laplacian_split_channels


def _dct_bases(channels, height, width, n_freq=16):
    """FcaNet 式固定 DCT 基：(channels, height, width)，每通道一个频率分量。

    频率索引循环取自 4×4 网格：(i, j) = (idx // 4, idx % 4)；
    ω(0,0) = sqrt(1/HW)，其余 sqrt(2/HW)；
    base_c = ω · cos(π(2h+1)i/2H) ⊗ cos(π(2w+1)j/2W)。
    """
    hw = height * width
    bases = torch.zeros(channels, height, width)
    cos_h = [torch.cos(math.pi * (2 * torch.arange(height) + 1) * (idx // 4) / (2 * height))
             for idx in range(n_freq)]
    cos_w = [torch.cos(math.pi * (2 * torch.arange(width) + 1) * (idx % 4) / (2 * width))
             for idx in range(n_freq)]
    for c in range(channels):
        idx = c % n_freq
        omega = (1.0 / hw) ** 0.5 if idx == 0 else (2.0 / hw) ** 0.5
        bases[c] = omega * torch.outer(cos_h[idx], cos_w[idx])
    return bases


class FGR(nn.Module):
    """输入输出同形 (B,C,H,W)，H/W 需为偶数（laplacian_split 要求，28/56/112/224 均满足）。"""

    def __init__(self, channels, height, width, n_freq=16, mode="whole"):
        super().__init__()
        C = channels
        self.mode = mode
        self.gate = nn.Linear(3 * C, 2 * C)
        if mode == "band":
            # FBGR：高低频各一枚 C→C 1×1 零核并联，两带互不串带（见模块 docstring）
            self.fuse_hf = nn.Conv2d(C, C, 1)
            self.fuse_lf = nn.Conv2d(C, C, 1)
        else:
            self.fuse = nn.Conv2d(2 * C, C, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)
        for m in ([self.fuse_hf, self.fuse_lf] if mode == "band" else [self.fuse]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.register_buffer(
            'dct_base', _dct_bases(3 * C, height, width, n_freq), persistent=False)

    def forward(self, x):
        ll, details = laplacian_split_channels(x)
        hf = F.interpolate(details[0].abs() + details[1].abs() + details[2].abs(),
                           size=x.shape[-2:], mode="bilinear", align_corners=False)
        lf = F.interpolate(ll, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gate_in = torch.cat([x, hf, lf], dim=1)
        z = (gate_in * self.dct_base.unsqueeze(0)).sum(dim=(2, 3))     # (B,3C) DCT 谱描述子
        w = torch.sigmoid(self.gate(z)).unsqueeze(-1).unsqueeze(-1)    # (B,2C,1,1)
        if self.mode == "band":
            w_hf, w_lf = w.chunk(2, dim=1)   # gate 前 C 列管 hf、后 C 列管 lf，不新增门
            return x + self.fuse_hf(w_hf * hf) + self.fuse_lf(w_lf * lf)
        return x + self.fuse(w * torch.cat([hf, lf], dim=1))
