# coding=utf-8
"""U-Net 骨架整体替换版（WMREN, IJCAI 2025）：WRMM 编码 + REM 解码融合。

对照论文（IJCAI-25, "Wavelet Multi-scale Region-Enhanced Network for Medical
Image Segmentation"）第 3.1/3.2 节与 Figure 3/4 实现的骨架级算子，其余积木
（stem / 解码 conv / 上采样）与 `unet.py` 完全一致，可与 U-Net 基线
（86.55% / HD95 0.77）单变量配对：

1. **编码块 → WRMM**（Wavelet Residual Multi-scale Module，Figure 3a）
   四级编码块统一为 `WRDM（下采样）→ MLKM（卷积）`：
   - **WRDM**（Figure 3b / Eq.(1)-(5)）：双支下采样。
     左支 Haar 一级分解 LL/LH/HL/HH → concat → 1×1 + BN + Sigmoid 得 X_HWD；
     右支 3×3 stride2 卷积下采样后接残差块（1×1 → 3×3 → 1×1 + 输入恒等相加）
     得 X_Res；两支按 Eq.(3)-(5) 融合：
     `F1 = Conv3×3(Concat(X_HWD, X_Res))`，
     `W1, W2 = Split(softmax(Conv1×1(Concat(Conv1×1(X_HWD), Conv1×1(X_Res)))))`，
     `F2 = Conv3×3(X_HWD ⊙ W1 + X_Res ⊙ W2)`，`F_down = F1 + F2`。
   - **MLKM**（Multi-scale Large Kernel Module）：论文只引用
     [Wang et al., 2024] 未给细节，此处按该文常见的多尺度大核范式实现
     （见 `MLKM` 的 docstring）。

2. **解码器 skip 融合 → REM**（Region-Enhanced Module，Figure 4a）
   四级统一结构：CBAM(decoder 上一级特征 F) 得 F_CBAM；
   CREM(skip) 同时输出增强特征 X_CREM 与边界权重 W_CREM；
   SAFM(X_CREM, F_CBAM) 得 X_SAFM；`F_fusion = Conv1×1(Concat(X_SAFM, F_CBAM))`；
   最后 `F_decoder = F_fusion ⊙ W_CREM + F_fusion`（Eq.(6)）。

**相对论文的简化（唯一一处，已在 `WRMM` 的 docstring 中注明）**：
最深一级 1024 通道若完整展开 WRDM，融合项 2×18·C² ≈ 38M、整个 WRDM ≈ 60M，
会淹没与基线的单变量对照；因此第四级 WRMM 的 WRDM/MLKM 放在 512 宽隐层，
末端 1×1 升到 1024，该级约 17M（基线该级 down_conv + BasicBlock 约 23.6M）。
三级浅编码块（64→128 / 128→256 / 256→512）完全按论文实现。

参数量：60.3M（U-Net 基线）→ 62.2M（+1.9M），旧的不完整复现版约 76M。
各级增量：WRMM1 0.92M / WRMM2 3.62M / WRMM3 14.38M / WRMM4 16.61M / REM 3.88M。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.unet import (BN_MOMENTUM, BasicBlock, DoubleBasicBlock, conv1x1,
                           conv3x3, init_kaiming, make_downsample, up_conv)


def haar_split(x):
    """一级 Haar 分解：返回 (B, 4C, H/2, W/2)，正交归一化系数 1/2。"""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)


class ResBlock111(nn.Module):
    """WRDM 右支的残差块：1×1 → 3×3 → 1×1 卷积 + 输入恒等相加（论文 3.1 节）。"""

    def __init__(self, channels):
        super(ResBlock111, self).__init__()
        self.conv1 = nn.Sequential(
            conv1x1(channels, channels),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            conv1x1(channels, channels),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv3(self.conv2(self.conv1(x))) + x)


class WRDM(nn.Module):
    """Wavelet-based Residual Downsampling Module（Figure 3b / Eq.(2)-(5)）。

    左支 Haar 小波下采样（X_HWD），右支 3×3 stride2 下采样卷积 + 残差块（X_Res），
    两支按 Eq.(3) 的 F1 与 Eq.(4)(5) 的软加权 F2 相加输出。
    """

    def __init__(self, ch_in, ch_out):
        super(WRDM, self).__init__()
        # 左支：Haar 分解 → 1×1 + BN + Sigmoid，得 X_HWD
        self.wave = nn.Sequential(
            nn.Conv2d(ch_in * 4, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.Sigmoid(),
        )
        # 右支：先卷积 stride2 下采样，再用残差块提特征，得 X_Res
        self.res_down = nn.Sequential(
            conv3x3(ch_in, ch_out, stride=2),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.res_block = ResBlock111(ch_out)
        # Eq.(3)：F1 = Conv3×3(Concat(X_HWD, X_Res))
        self.f1 = nn.Sequential(
            conv3x3(ch_out * 2, ch_out),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
        )
        # Eq.(4)：两支持分别 1×1 变换后 concat，再 1×1 出 2C，softmax 后对半切分
        self.hwd_proj = nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)
        self.res_proj = nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(ch_out * 2, ch_out * 2, kernel_size=1)
        # Eq.(5)：F2 = Conv3×3(X_HWD⊙W1 + X_Res⊙W2)
        self.f2 = nn.Sequential(
            conv3x3(ch_out, ch_out),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
        )

    def forward(self, x):
        f_hwd = self.wave(haar_split(x))
        f_res = self.res_block(self.res_down(x))
        f1 = self.f1(torch.cat([f_hwd, f_res], dim=1))
        weight = torch.softmax(
            self.gate(torch.cat([self.hwd_proj(f_hwd), self.res_proj(f_res)], dim=1)), dim=1)
        w1, w2 = weight.chunk(2, dim=1)
        f2 = self.f2(f_hwd * w1 + f_res * w2)
        return f1 + f2


class MLKM(nn.Module):
    """Multi-scale Large Kernel Module（WRMM 的卷积单元）。

    论文仅引用 [Wang et al., 2024]，未给出 MLKM 内部细节；此处按该文
    "多尺度大核 + 逐点卷积 + 残差" 的常见范式实现一个合理版本：

        1×1(升维/对齐通道) + BN + ReLU
        → 三个并行 depthwise 大核（5×5、7×7 膨胀 2、9×9 膨胀 2，各接 BN + ReLU）
        → concat(3C) → 1×1 融合 + BN
        → 与输入残差相加 → ReLU

    大核用 depthwise 是为了让 28²/512、14²/1024 特征上的感受野扩大而参数可控
    （三个大核合计仅 (25+49+81)·C）。
    """

    def __init__(self, ch_in, ch_out, dilation=2):
        super(MLKM, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        # 多尺度 depthwise 大核：5×5 保持局部、7×7/9×9 膨胀扩感受野
        self.dw5 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=5, padding=2,
                      groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.dw7 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=7, padding=3 * dilation,
                      dilation=dilation, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.dw9 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=9, padding=4 * dilation,
                      dilation=dilation, groups=ch_out, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch_out * 3, ch_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
        )
        self.shortcut = None
        if ch_in != ch_out:
            self.shortcut = nn.Sequential(
                conv1x1(ch_in, ch_out),
                nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        p = self.proj(x)
        out = self.fuse(torch.cat([self.dw5(p), self.dw7(p), self.dw9(p)], dim=1))
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.relu(out + identity)


class WRMM(nn.Module):
    """Wavelet Residual Multi-scale Module = WRDM（下采样）→ MLKM（卷积），Figure 3a。

    `hidden` 为 None 时中间通道即 ch_out（浅三级 64→128 / 128→256 / 256→512）；
    最深一级显式传 hidden=512 后再 1×1 升到 1024 —— 若在 1024 通道上完整展开
    WRDM，其 F1/F2 两个 3×3 融合项各 18·C² ≈ 19M、整级 WRDM ≈ 60M，参数量会
    淹没与 U-Net 基线的单变量对照，故该级只做通道宽度上的等价缩减，模块结构与
    其余三级完全一致（WRDM → MLKM → 1×1 输出）。
    """

    def __init__(self, ch_in, ch_out, hidden=None):
        super(WRMM, self).__init__()
        mid = ch_out if hidden is None else hidden
        self.wrdm = WRDM(ch_in, mid)
        self.mlkm = MLKM(mid, mid)
        if mid == ch_out:
            self.out = nn.Identity()
        else:
            self.out = nn.Sequential(
                conv1x1(mid, ch_out),
                nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.out(self.mlkm(self.wrdm(x)))


class CBAM(nn.Module):
    """通道 + 空间注意力（Figure 4a 中作用在 decoder 上一级特征 F 上）。"""

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        x = x * torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(desc))


class CREM(nn.Module):
    """Contrast Refinement Enhancement Module（Figure 4b / Eq.(7)-(10)）。

    输入 skip 特征 X_skip，输出边界增强特征 X_CREM 与边界权重 W_CREM：
      P0 = Conv1×1(X_skip)，P_i = AvgPool(P_{i-1})（i=1,2,3，级联池化降采样）；
      E_i = P_i − AvgPool(Conv1×1(P_i)) + P_i（Eq.(8)，池化模糊后差分出边界再回加）；
      X_CREM = BN(ReLU(Conv1×1(Concat(P0, E1↑, E2↑, E3↑))))（Eq.(9)，E_i 上采样回 P0 尺寸）；
      W_CREM = σ(Conv1×1(X_CREM − X_skip))（Eq.(10)，单通道空间权重）。
    """

    def __init__(self, channels):
        super(CREM, self).__init__()
        self.p0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        # 每个尺度：1×1 → 同尺寸 AvgPool（边界模糊）
        self.blurs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ) for _ in range(3)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.edge = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x_skip):
        p0 = self.p0(x_skip)
        edges = []
        p = p0
        for blur in self.blurs:
            p = self.down(p)
            edges.append(p - blur(p) + p)
        size = p0.shape[-2:]
        edges = [F.interpolate(e, size=size, mode='bilinear', align_corners=False)
                 for e in edges]
        x_crem = self.fuse(torch.cat([p0] + edges, dim=1))
        w_crem = torch.sigmoid(self.edge(x_crem - x_skip))
        return x_crem, w_crem


class SAFM(nn.Module):
    """Spatial Adaptive Fusion Module（Figure 4c / Eq.(11)-(13)）。

    F_MUL = X_CREM ⊙ σ(F_CBAM) + F_CBAM ⊙ σ(X_CREM)（互为门控，只保留共同显著区域）；
    W_SAFM = σ(ReLU(BN(Conv1×1(F_MUL))))；
    F_SAFM = X_CREM ⊙ W_SAFM + X_CREM（残差保留浅层信息）。
    """

    def __init__(self, channels):
        super(SAFM, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_crem, f_cbam):
        f_mul = x_crem * torch.sigmoid(f_cbam) + f_cbam * torch.sigmoid(x_crem)
        w_safm = torch.sigmoid(self.out(f_mul))
        return x_crem * w_safm + x_crem


class REM(nn.Module):
    """Region-Enhanced Module（Figure 4a / Eq.(6)），四级统一结构。

    F_CBAM = CBAM(F)（F 为 decoder 上一级特征）；
    X_CREM, W_CREM = CREM(X_skip)；
    X_SAFM = SAFM(X_CREM, F_CBAM)；
    F_fusion = Conv1×1(Concat(X_SAFM, F_CBAM))；
    F_decoder = F_fusion ⊙ W_CREM + F_fusion。
    """

    def __init__(self, channels):
        super(REM, self).__init__()
        self.cbam = CBAM(channels)
        self.crem = CREM(channels)
        self.safm = SAFM(channels)
        self.fuse = nn.Sequential(
            conv1x1(channels * 2, channels),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
        )

    def forward(self, x_skip, dec):
        f_cbam = self.cbam(dec)
        x_crem, w_crem = self.crem(x_skip)
        x_safm = self.safm(x_crem, f_cbam)
        f_fusion = self.fuse(torch.cat([x_safm, f_cbam], dim=1))
        return f_fusion * w_crem + f_fusion


class UNetWMREN(nn.Module):
    """U-Net 骨架的 WMREN 版本（单输入单输出，(B,1,224,224)）。

    严格按论文 Figure 2 的拓扑（编码 4 个 WRMM / 解码 3 个 REM）：

        stem 224²/64 → WRMM1 112²/128 → WRMM2 56²/256 → WRMM3 28²/512
        → WRMM4 14²/1024
        解码：up14→28 → REM(x4, up) → DBB → up→56 → REM(x3, up) → DBB
              → up→112 → REM(x2, up) → DBB → up→224 → 1×1 出 logits

    - 只有 WRMM1/2/3 的输出做 skip；最深一级 WRMM4 的输出直接进解码器底部；
    - 解码器只有三级 REM（28²/512、56²/256、112²/128），最后一级 224²/64
      只上采样 + 1×1，与 Figure 2 里 REM 的数量一致；
    - 论文 "skip 上 1×1 调通道" 的职责由 CREM 的 P0 = Conv1×1(X_skip) 承担
      （本实现中 skip 通道与解码器通道本就一致，故不额外加 1×1）；
    - REM 的输出 F_decoder 已包含 concat + 1×1 融合（Eq.6），所以解码块用
      单输入 DoubleBasicBlock(C, C)，不再外面再 cat 一次。
    """

    def __init__(self, in_channels=1, num_classes=2):
        super(UNetWMREN, self).__init__()
        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024
        # stem：Figure 2 从 224²/64 起画（WRMM1 的 skip 标 H/2 × W/2 × C1），
        # 该级不参与 skip，只做浅层特征提纯
        self.stem = nn.Sequential(
            conv3x3(in_channels, l1c), nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            conv3x3(l1c, l1c), nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        # 四个 WRMM = WRDM（下采样）→ MLKM（卷积）
        self.wrmm1 = WRMM(l1c, l2c)
        self.wrmm2 = WRMM(l2c, l3c)
        self.wrmm3 = WRMM(l3c, l4c)
        # 最深一级通道爆炸（见 WRMM docstring），WRDM/MLKM 放 512 宽隐层后 1×1 升到 1024
        self.wrmm4 = WRMM(l4c, l5c, hidden=l4c)
        # 解码：三级 REM（28²/512、56²/256、112²/128）+ DBB；最后一级只上采样
        self.rems = nn.ModuleList([REM(l4c), REM(l3c), REM(l2c)])
        self.d5_1 = up_conv(l5c, l4c)
        self.d4_1 = DoubleBasicBlock(l4c, l4c)
        self.d4_2 = up_conv(l4c, l3c)
        self.d3_1 = DoubleBasicBlock(l3c, l3c)
        self.d3_2 = up_conv(l3c, l2c)
        self.d2_1 = DoubleBasicBlock(l2c, l2c)
        self.d2_2 = up_conv(l2c, l1c)
        self.d1_2 = nn.Conv2d(l1c, num_classes, kernel_size=1)
        init_kaiming(self)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x1 = self.stem(x)                              # 224²/64（不参与 skip）
        x2 = self.wrmm1(x1)                            # 112²/128
        x3 = self.wrmm2(x2)                            # 56²/256
        x4 = self.wrmm3(x3)                            # 28²/512
        x5 = self.wrmm4(x4)                            # 14²/1024
        up4 = self.d5_1(x5)
        d4 = self.d4_1(self.rems[0](x4, up4))          # 28²/512
        up3 = self.d4_2(d4)
        d3 = self.d3_1(self.rems[1](x3, up3))          # 56²/256
        up2 = self.d3_2(d3)
        d2 = self.d2_1(self.rems[2](x2, up2))          # 112²/128
        return self.d1_2(self.d2_2(d2))                # 224²/64 → logits

    def load_from(self, weights):
        return
