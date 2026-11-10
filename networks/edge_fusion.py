"""R-GEF: Reliability-Gated Edge Fusion（可靠性门控边缘融合）。

设计动机（来自本项目已有读数）：
  - U-Net 宿主上边缘输入有明确增益（laplacian +1.35 / sobel +0.97 Dice），说明边缘这一路有信息；
  - 但超声斑点噪声与后方声影会让 Laplacian 在噪声区同样给出强响应；
  - 现有边缘融合（FSS-Net LAEF）是 GAP → FC 的通道级权重，一个通道一个标量，
    无法区分「远壁的真实边缘」与「声影里的假边缘」（二者在同一通道里）。

做法（挂在解码器 skip 融合点，输入 skip、解码特征 x）：
  1) 平滑（固定高斯 depthwise）→ 拉普拉斯（固定旋转不变核，3×3 [-1..8..-1]）→ BN/ReLU/1×1，
     得到边缘特征 E；
  2) 可靠性图 r = σ(Conv1×1([E, x↑]))，逐像素（不是逐通道）；
  3) 残差注入 skip' = skip + γ · (r ⊙ E)。

  两处初始化保证「起点等价基线」：可靠性卷积权重置零、bias = -2（r≈0.12），γ = 0。
  因此训练第 0 步 skip' = skip，增益只能来自模块本身。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 旋转不变 3×3 拉普拉斯核（与 FSS-Net LAEF 相同的经典核）
_LAPLACIAN = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
# 3×3 高斯（归一化 1/16），用于抑制斑点噪声后再求边缘
_GAUSSIAN = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]


class ReliabilityGatedEdgeFusion(nn.Module):
    """逐像素可靠性门控的边缘融合。

    gate_mode:
      - "spatial"：可学习逐像素可靠性（正式模块）
      - "const"  ：r 恒为 0.5（参数完全匹配的对照，用于判断增益是否来自门控）
    """

    def __init__(self, channels, x_channels=None, gate_mode="spatial", init_gamma=0.0, init_bias=-2.0):
        super().__init__()
        self.gate_mode = gate_mode
        x_channels = channels if x_channels is None else x_channels

        self.smooth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.edge = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        g = torch.tensor(_GAUSSIAN, dtype=torch.float32) / 16.0
        lap = torch.tensor(_LAPLACIAN, dtype=torch.float32)
        with torch.no_grad():
            self.smooth.weight.copy_(g.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
            self.edge.weight.copy_(lap.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.smooth.weight.requires_grad_(False)
        self.edge.weight.requires_grad_(False)

        self.norm = nn.BatchNorm2d(channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels + x_channels, channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)

        # γ 初始 0：起点 skip' = skip，与基线逐位等价
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, skip, x):
        e = self.proj(F.relu(self.norm(self.edge(self.smooth(skip)))))
        if self.gate_mode == "const":
            r = torch.full_like(e, 0.5)
        else:
            x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            r = torch.sigmoid(self.gate(torch.cat([e, x_up], dim=1)))
        return skip + self.gamma * (r * e)


class MultiFactorEdgeFusion(nn.Module):
    """MREF：多因子可靠性边缘融合（R-GEF 的加宽版）。

    与 R-GEF 的差别（"多加一些"的部分）：
      1) 边缘支从 1 条扩成 3 条：单尺度拉普拉斯、σ=2 拉普拉斯（高斯两次再拉普拉斯）、
         固定 Sobel 梯度幅值——sobel / laplacian 是本项目在 U-Net 宿主上实测有增益的算子
         （输入级 +0.97 / +1.35 Dice），这里把它们放到特征级并叠加；
      2) 增加一路零参数"斑点噪声图"：局部 std / 均值，描述乘性散斑强度；
      3) 增加一路"跨支不一致度" u = |x↑ − skip|，噪声小且两路一致才允许改写 skip；
      4) 双头门控：边缘注入 r_e 与校正式改写 r_u 各自独立学（不再共用一个门）；
      5) γ1/γ2 初值 0.1 —— G1（γ₀=0）的教训是 γ 会一直贴 0，模块等于没开。

    输出：skip' = skip + γ1·(r_e ⊙ e) + γ2·(r_u ⊙ (x↑ − skip))
    位置：解码器每个 skip 汇合点（与 R-GEF 相同），不动主干。
    """

    def __init__(self, channels, x_channels=None, init_gamma=0.1, init_bias=-2.0):
        super().__init__()
        x_channels = channels if x_channels is None else x_channels

        def _fixed(kind):
            conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            with torch.no_grad():
                if kind == "gauss":
                    k = torch.tensor(_GAUSSIAN, dtype=torch.float32) / 16.0
                elif kind == "lap":
                    k = torch.tensor(_LAPLACIAN, dtype=torch.float32)
                elif kind == "sobel_x":
                    k = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
                else:
                    k = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
                conv.weight.copy_(k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
            conv.weight.requires_grad_(False)
            return conv

        self.gauss = _fixed("gauss")
        self.lap = _fixed("lap")
        self.sob_x = _fixed("sobel_x")
        self.sob_y = _fixed("sobel_y")

        self.norm = nn.BatchNorm2d(channels * 4)
        self.proj = nn.Conv2d(channels * 4, channels, 1)
        self.refine = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        # 解码支的通道数通常不等于 skip（64 vs 128 / 128 vs 256），先对齐到 skip 通道，
        # 门控与不一致度都在这套通道里计算，避免 (x↑ − skip) 维度不匹配
        self.align = nn.Conv2d(x_channels, channels, 1)

        self.gate_e = nn.Conv2d(channels * 2, channels, 1)
        self.gate_u = nn.Conv2d(channels * 2 + 1, channels, 1)
        for conv in (self.gate_e, self.gate_u):
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, init_bias)

        self.gamma1 = nn.Parameter(torch.tensor(float(init_gamma)))
        self.gamma2 = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, skip, x):
        x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x_al = self.align(x_up)

        # 斑点噪声图（零参数）：局部 std / 局部均值
        mu = F.avg_pool2d(skip, 3, 1, 1)
        mu2 = F.avg_pool2d(skip * skip, 3, 1, 1)
        var = (mu2 - mu * mu).clamp_min(0)
        noise = torch.sqrt(var + 1e-6) / (mu.abs() + 1e-3)

        # 三条固定边缘支（多尺度拉普拉斯 + Sobel 幅值）
        lap1 = self.lap(self.gauss(skip))
        lap2 = self.lap(self.gauss(self.gauss(skip)))
        sob = torch.sqrt(self.sob_x(skip) ** 2 + self.sob_y(skip) ** 2 + 1e-6)

        e = self.proj(F.relu(self.norm(torch.cat([lap1, lap2, sob, noise], dim=1))))
        e = F.relu(self.refine(e))

        r_e = torch.sigmoid(self.gate_e(torch.cat([e, x_al], dim=1)))
        u = (x_al - skip).abs().mean(dim=1, keepdim=True)
        r_u = torch.sigmoid(self.gate_u(torch.cat([e, u, x_al], dim=1)))

        return skip + self.gamma1 * (r_e * e) + self.gamma2 * (r_u * (x_al - skip))


class CascadeEdgeFusion(nn.Module):
    """把 depth 个 MREF 串成一条级联（“多加几个模块”，而不是只加分支）。

    - 单个融合块的容量有限（R-GEF 1.02M 实测无增益），所以在一个 skip 汇合点上
      依次串 depth 个块：第 k 块的输入是第 k−1 块改写后的 skip，边缘/噪声/一致性
      都在当前特征上重新估计，等价于 depth 步迭代精炼；
    - 每个块各自带 γ，初值 0.1（0 会让块等于关闭，见 G1 的教训）；
    - DecoderCup 有 3 个 skip 汇合点，depth=4 时全模型共 12 个融合块。

    前向：for blk in self.blocks: skip = blk(skip, x)
    """

    def __init__(self, channels, x_channels=None, depth=4, init_gamma=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MultiFactorEdgeFusion(channels, x_channels=x_channels, init_gamma=init_gamma)
            for _ in range(depth)])

    def forward(self, skip, x):
        for blk in self.blocks:
            skip = blk(skip, x)
        return skip


class FEMFusion(nn.Module):
    """FDFUNet（Neural Networks 2024）的 FEM 搬到解码器 skip 汇合点。

    FEM 由 MFDF（多尺度频域滤波：LN → 通道对半切 → DASConv / FDF 交错）+ ASCA
    （非对称空间-通道注意力）串联而成，原论文把它放在每个 Down/Up 之后；这里按
    本项目既有约定挂在 skip 上，残差注入 skip' = skip + γ·FEM(skip)，γ 初值 0.1
    （0 会让模块等于关闭，见 G1 的教训）。

    FDF 的频域权重形状与输入 H/W 绑定（(C,H,W/2+1,2)），因此构造时必须传入该级
    skip 的真实分辨率；本项目 224 输入下为 28²/56²/112²。
    """

    def __init__(self, channels, height, width, init_gamma=0.1):
        super().__init__()
        from .fdfunet import FEM
        self.fem = FEM(channels, height, width)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, skip, x=None):
        return skip + self.gamma * self.fem(skip)


class AsymBandFusion(nn.Module):
    """ABF（Asymmetric Band Fusion，非对称频带融合）——本项目自研。

    设计依据（本项目 R50 上的实测）：
      - 只在单张 skip 上「改写特征」的融合模块全部≈0，且与容量无关
        （FreqFusion +0.08 / R-GEF +0.13 / MREF×4 12.16M −0.12）；
      - 唯一方向为正的是 DDBE，而 DDBE 恰好是频带解耦（+0.44~+0.51）。
    因此 ABF 不做「精修 skip」，而是做跨路频带搬运：

      Low(·) = 3×3 均值（固定，无参数）
      s_L, s_H = Low(s), s − s_L ;  u_L, u_H = Low(u), u − u_L
      d = mean_C|s − u| / (mean_C|s| + mean_C|u| + ε)     # 跨路分歧 ≈ 边界/斑块
      s' = s + γ ⊙ [ α_L ⊙ (u_L − s_L) + (1 + β·tanh d) ⊙ α_H ⊙ (s_H − u_H) ]

    - 低频向解码器 u 靠（语义可靠、抗斑点），高频留在 skip（定位精确）；
    - α_L 与 α_H 各自独立可学，只有 α_L = −α_H 才退化成单门控 (s−u)，
      因此 tied=True（α_H = −α_L，参数完全匹配）就是它的对照；
    - 不用 sigmoid 门控（G1 的教训：门控会被训练关死，且去门控反而更好）；
    - speckle=True：HF 项按乘性斑点噪声的局部水平衰减
      σ_n = local_std₇ / (local_mean₇ + ε)，HF ← HF / (1 + softplus(κ)·σ_n)；
    - large_kernel=True：LF 修正项再叠一个 depthwise 7×7，补弱骨干的感受野
      （对应 K2 上 MLKM 的 +1.01）。

    强骨干（R50）只用带分 + 独立带权 + 分歧放大；弱骨干（U-Net）再加
    speckle 与 large_kernel。γ 初值 0.1（0 会让模块整个训练期不打开）。
    """

    def __init__(self, channels, x_channels=None, speckle=False, large_kernel=False,
                 tied=False, init_gamma=0.1):
        super().__init__()
        shape = (1, channels, 1, 1)
        self.proj = (None if x_channels in (None, channels)
                     else nn.Conv2d(x_channels, channels, 1, bias=False))
        self.alpha_l = nn.Parameter(torch.ones(shape))
        self.tied = tied
        self.alpha_h = None if tied else nn.Parameter(torch.ones(shape))
        self.gamma = nn.Parameter(torch.full(shape, float(init_gamma)))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.kappa = nn.Parameter(torch.tensor(-2.0)) if speckle else None
        self.dw7 = (nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
                    if large_kernel else None)

    def forward(self, skip, x):
        s = skip
        u = x if x is not None else skip
        if self.proj is not None:
            # DecoderCup 的 x 与 skip 通道数不同（512/256/128 vs 512/256/64）
            u = self.proj(u)
        if u.shape[-2:] != s.shape[-2:]:
            # DecoderCup 里 egf 在 up 之前调用，x 的分辨率比 skip 低（14² vs 28²…）
            u = F.interpolate(u, size=s.shape[-2:], mode='bilinear', align_corners=False)
        s_low = F.avg_pool2d(s, 3, stride=1, padding=1)
        u_low = F.avg_pool2d(u, 3, stride=1, padding=1)
        hf = (s - s_low) - (u - u_low)
        if self.kappa is not None:
            mu = F.avg_pool2d(s, 7, stride=1, padding=3)
            var = (F.avg_pool2d(s * s, 7, stride=1, padding=3) - mu * mu).clamp_min(0.0)
            sn = (var.sqrt() / (mu.abs() + 1e-3)).mean(dim=1, keepdim=True)
            hf = hf / (1.0 + F.softplus(self.kappa) * sn)
        diff = (s - u).abs().mean(dim=1, keepdim=True)
        scale = (s.abs().mean(dim=1, keepdim=True)
                 + u.abs().mean(dim=1, keepdim=True) + 1e-3)
        d = diff / scale
        lf_term = self.alpha_l * (u_low - s_low)
        if self.dw7 is not None:
            lf_term = lf_term + self.dw7(lf_term)
        alpha_h = -self.alpha_l if self.tied else self.alpha_h
        hf_term = alpha_h * (1.0 + self.beta * torch.tanh(d)) * hf
        return skip + self.gamma * (lf_term + hf_term)


