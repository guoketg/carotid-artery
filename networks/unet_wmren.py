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

参数量（实测）：60.32M（U-Net 基线）→ 58.11M（−2.21M）。旧的不完整版（W1）70.58M。
各级增量：WRMM1 0.92M / WRMM2 3.62M / WRMM3 14.38M / WRMM4 16.61M / REM 3.88M。

**消融开关**（`ablate=`，原位替换，用于模块归因；其余结构完全不动）：

  - `unet_enc`：**四个 WRMM 整体换成 U-Net 编码块**（down_conv + DBB/BasicBlock），
    解码 REM 不动 —— 配合 `unet_dec` 与完整版/标准 U-Net 构成 2×2 阶梯；
  - `unet_dec`：**三级 REM 整体换成 U-Net 的 cat + DBB 融合**，编码 WRMM 不动；
  - 更细的留一开关（可选，默认不跑）：
    `no_wave`（WRDM 只留残差右支）、`no_gate`（softmax 软选择换固定 0.5）、
    `no_mlkm`（去掉多尺度大核）、`no_rem`（REM 退化为 1×1 融合）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.unet import (BN_MOMENTUM, BasicBlock, DoubleBasicBlock, conv1x1,
                           conv3x3, down_conv, init_kaiming, make_downsample, up_conv)
from networks.vit_seg_modeling_resnet_skip import ResNetV2, np2th
from networks.wavelet import laplacian_pyramid_channels, laplacian_split_channels


def pyr_split(x):
    """一级拉普拉斯金字塔分解：返回 (B, 4C, H/2, W/2)，与 `haar_split` 同形。

    与 `lap_split` 的区别：不做 abs（保留符号），且三个细节槽是三个真实尺度
    （全分辨率 / H/2 / H/4 的 DoG 带通），不是同一幅幅值图复制三份。
    """
    ll, details = laplacian_pyramid_channels(x)
    return torch.cat([ll] + list(details), dim=1)


def lap_split(x):
    """一级拉普拉斯分解：返回 (B, 4C, H/2, W/2)，与 `haar_split` 同形。

    高频基复用 §1.6 边缘算子测试的同一口径（4 邻域 Laplacian → abs → 逐样本
    标准化），只做 depthwise 化 + 半分辨率池化；LL 沿用 Haar 低通。
    """
    ll, details = laplacian_split_channels(x)
    return torch.cat([ll] + list(details), dim=1)


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


def _split(x, op):
    """wave_op → 一级频率分解；三种算子输出与 `haar_split` 逐位同形、零参数。"""
    if op == 'lap':
        return lap_split(x)
    if op == 'pyr':
        return pyr_split(x)
    return haar_split(x)


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

    消融开关（留一变量，其余结构不动）：
      - `use_wave=False`：只保留残差右支，去掉 Haar 左支、F1 与 Eq.(4)(5) 门控；
      - `use_gate=False`：两支都在，Eq.(4)(5) 换成固定等权 0.5，只去掉可学习软选择；
      - `wave_op='lap'`：左支的 Haar 4 子带换成同一口径的拉普拉斯分解（零参数，
        接口逐位不变），用于 §1.12 的算子替换。
      - `wave_op='pyr'`：左支换成拉普拉斯金字塔（DoG，3 个真实尺度、保留符号），
        作为 §1.12 的「正统多尺度拉普拉斯分解」臂。
      - `stride=1`：分辨率不变的同级精炼模式（`enc='r50wrmm'` 用）。
        右支 stride 改 1；左支一级分解天然到 H/2，再插值回 H 后融合，
        因此四个输出形状与输入逐级相同，可原样挂在 R50 的四级特征上。
    """

    def __init__(self, ch_in, ch_out, use_wave=True, use_gate=True, wave_op='haar',
                 stride=2):
        super(WRDM, self).__init__()
        self.stride = stride
        if wave_op not in ('haar', 'lap', 'pyr'):
            raise ValueError('unknown wave_op: %r' % (wave_op,))
        self.use_wave = use_wave
        self.use_gate = use_gate and use_wave
        self.wave_op = wave_op
        if use_wave:
            # 左支：Haar 分解 → 1×1 + BN + Sigmoid，得 X_HWD
            self.wave = nn.Sequential(
                nn.Conv2d(ch_in * 4, ch_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
                nn.Sigmoid(),
            )
        # 右支：先卷积 stride2 下采样，再用残差块提特征，得 X_Res
        self.res_down = nn.Sequential(
            conv3x3(ch_in, ch_out, stride=stride),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.res_block = ResBlock111(ch_out)
        if use_wave:
            # Eq.(3)：F1 = Conv3×3(Concat(X_HWD, X_Res))
            self.f1 = nn.Sequential(
                conv3x3(ch_out * 2, ch_out),
                nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            )
        if self.use_gate:
            # Eq.(4)：两支持分别 1×1 变换后 concat，再 1×1 出 2C，softmax 后对半切分
            self.hwd_proj = nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)
            self.res_proj = nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)
            self.gate = nn.Conv2d(ch_out * 2, ch_out * 2, kernel_size=1)
        if use_wave:
            # Eq.(5)：F2 = Conv3×3(X_HWD⊙W1 + X_Res⊙W2)
            self.f2 = nn.Sequential(
                conv3x3(ch_out, ch_out),
                nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            )

    def forward(self, x):
        f_res = self.res_block(self.res_down(x))
        if not self.use_wave:
            return f_res
        f_hwd = self.wave(_split(x, self.wave_op))
        if self.stride == 1:
            # 精炼模式：一级分解落在 H/2，插值回右支尺寸后再融合
            f_hwd = F.interpolate(f_hwd, size=f_res.shape[-2:], mode='bilinear',
                                  align_corners=False)
        f1 = self.f1(torch.cat([f_hwd, f_res], dim=1))
        if self.use_gate:
            weight = torch.softmax(
                self.gate(torch.cat([self.hwd_proj(f_hwd), self.res_proj(f_res)], dim=1)), dim=1)
            w1, w2 = weight.chunk(2, dim=1)
        else:
            w1 = w2 = 0.5
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

    def __init__(self, ch_in, ch_out, hidden=None, ablate=None, wave_op='haar',
                 stride=2):
        super(WRMM, self).__init__()
        mid = ch_out if hidden is None else hidden
        self.wrdm = WRDM(ch_in, mid, use_wave=(ablate != 'no_wave'),
                         use_gate=(ablate != 'no_gate'), wave_op=wave_op,
                         stride=stride)
        # no_mlkm：只保留下采样，去掉多尺度大核卷积单元
        self.mlkm = None if ablate == 'no_mlkm' else MLKM(mid, mid)
        if mid == ch_out:
            self.out = nn.Identity()
        else:
            self.out = nn.Sequential(
                conv1x1(mid, ch_out),
                nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.wrdm(x)
        if self.mlkm is not None:
            x = self.mlkm(x)
        return self.out(x)


class UNetEncBlock(nn.Module):
    """`unet_enc` 消融：用 U-Net 的编码块原位替换 WRMM。

    浅三级 = `down_conv`（3×3 stride2）+ `DoubleBasicBlock`，最深一级 = `down_conv`
    + `BasicBlock`，与 `networks/unet.py` 的 l1_2/l2_1 … l4_2/l5_1 逐层一致，
    只换编码块，REM 解码与其余结构不动。
    """

    def __init__(self, ch_in, ch_out, deepest=False):
        super(UNetEncBlock, self).__init__()
        self.down = down_conv(ch_in, ch_out)
        block = BasicBlock if deepest else DoubleBasicBlock
        self.block = block(ch_out, ch_out)

    def forward(self, x):
        return self.block(self.down(x))


class UNetFusion(nn.Module):
    """`unet_dec` 消融：用 U-Net 的 skip 融合原位替换 REM。

    解码器上一级特征（此处对应 U-Net 的 up）与 skip 直接 concat 后接
    `DoubleBasicBlock(2C→C)` + 1×1 残差 shortcut，即 U-Net 的 d4_1/d3_1/d2_1。
    替换后 WRMM 编码、上采样算子都不动。
    """

    def __init__(self, channels):
        super(UNetFusion, self).__init__()
        self.fuse = DoubleBasicBlock(channels * 2, channels,
                                     make_downsample(channels * 2, channels))

    def forward(self, x_skip, dec):
        return self.fuse(torch.cat([dec, x_skip], dim=1))


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

    def __init__(self, channels, use_region=True):
        super(REM, self).__init__()
        self.use_region = use_region
        # no_rem：退化为 U-Net 式 Concat + 1×1 + BN（不含 CBAM/CREM/SAFM/边界权重）
        self.fuse = nn.Sequential(
            conv1x1(channels * 2, channels),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
        )
        if not use_region:
            return
        self.cbam = CBAM(channels)
        self.crem = CREM(channels)
        self.safm = SAFM(channels)

    def forward(self, x_skip, dec):
        if not self.use_region:
            return self.fuse(torch.cat([x_skip, dec], dim=1))
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

    def __init__(self, in_channels=1, num_classes=2, ablate=None, wave_op='haar',
                 no_wave=False, no_gate=False, no_mlkm=False, enc='wrmm'):
        super(UNetWMREN, self).__init__()
        if ablate not in (None, '', 'no_wave', 'no_mlkm', 'no_rem', 'no_gate',
                          'unet_enc', 'unet_dec'):
            raise ValueError('unknown ablate: %r' % (ablate,))
        if wave_op not in ('haar', 'lap', 'pyr'):
            raise ValueError('unknown wave_op: %r' % (wave_op,))
        if ablate == 'unet_enc' and wave_op != 'haar':
            raise ValueError('wave_op only applies to WRMM, not to unet_enc')
        if enc not in ('wrmm', 'r50', 'r50deep', 'r50wrmm', 'r50seq'):
            raise ValueError('unknown enc: %r' % (enc,))
        if enc in ('r50', 'r50wrmm', 'r50seq') and ablate == 'unet_enc':
            raise ValueError('enc=%s 的浅层是 R50，没有 WRMM 可换' % enc)
        if enc == 'r50' and (ablate in ('no_wave', 'no_gate', 'no_mlkm')
                             or no_wave or no_gate or no_mlkm):
            raise ValueError('enc=r50 没有 WRMM，不能与 WRMM 级开关共用')
        if enc == 'r50deep' and (ablate == 'unet_enc' or no_wave or no_gate or no_mlkm):
            raise ValueError('enc=r50deep 的后两级是 R50 block2/3，WRMM 级开关只作用于浅两级，语义不一致')
        if sum(bool(v) for v in (no_wave, no_gate, no_mlkm)) > 1:
            raise ValueError('no_wave / no_gate / no_mlkm are mutually exclusive')
        if (no_wave or no_gate or no_mlkm) and ablate in ('no_wave', 'no_gate', 'no_mlkm', 'unet_enc'):
            raise ValueError('no_wave/no_gate/no_mlkm flags conflict with ablate=%r' % (ablate,))
        self.ablate = ablate or None
        self.wave_op = wave_op
        self.enc_name = enc
        # WRMM 级留一开关：结构性的 ablate（unet_enc/unet_dec/no_rem）不进 WRMM，
        # 内部的 no_wave/no_gate/no_mlkm 才往下传；布尔参数优先，便于在 K2 宿主上叠加控制臂。
        if no_wave:
            wr_ablate = 'no_wave'
        elif no_gate:
            wr_ablate = 'no_gate'
        elif no_mlkm:
            wr_ablate = 'no_mlkm'
        elif self.ablate in ('no_wave', 'no_mlkm', 'no_gate'):
            wr_ablate = self.ablate
        else:
            wr_ablate = None
        self.wr_ablate = wr_ablate
        self.no_wave = no_wave
        self.no_gate = no_gate
        self.no_mlkm = no_mlkm
        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024
        if enc == 'r50':
            # α 臂：R50（ImageNet21k 的 conv_root + block1~3）整体当编码器，WRMM 全不建。
            # R50 出 112²/64、56²/256、28²/512、14²/1024，与 wrmm1~4 的输出逐级同形，
            # 只需一个 1×1 把 112² 的 64 通道对齐到 128，解码器（REM/DBB 或 unet_dec）完全不动。
            self.stem = nn.Identity()
            self.r50 = ResNetV2(block_units=(3, 4, 6), width_factor=1, in_channels=3)
            self.r50_proj = nn.Conv2d(l1c, l2c, kernel_size=1, bias=False)
            self.wrmm1 = self.wrmm2 = self.wrmm3 = self.wrmm4 = None
        elif enc == 'r50wrmm':
            # R50 全段（conv_root + block1~3，3/4/6，不截断）+ 四级 WRMM 同级精炼：
            #   root → 112²/64（1×1 升到 128）→ WRMM1 → 112²/128
            #   maxpool + block1 → 56²/256      → WRMM2 → 56²/256
            #   block2 → 28²/512                → WRMM3 → 28²/512
            #   block3 → 14²/1024               → WRMM4 → 14²/1024
            # 四个 WRMM 都跑 stride=1 的精炼模式，尺度/通道与 R50 输出逐级同形，
            # 解码器（unet_dec）完全不动；R50 的 ImageNet21k 权重可逐位加载。
            self.stem = nn.Identity()
            self.r50 = ResNetV2(block_units=(3, 4, 6), width_factor=1, in_channels=3)
            self.r50_proj = nn.Conv2d(l1c, l2c, kernel_size=1, bias=False)
            self.wrmm1 = WRMM(l2c, l2c, ablate=wr_ablate, wave_op=wave_op, stride=1)
            self.wrmm2 = WRMM(l3c, l3c, ablate=wr_ablate, wave_op=wave_op, stride=1)
            self.wrmm3 = WRMM(l4c, l4c, ablate=wr_ablate, wave_op=wave_op, stride=1)
            self.wrmm4 = WRMM(l5c, l5c, hidden=l4c, ablate=wr_ablate,
                              wave_op=wave_op, stride=1)
        elif enc == 'r50seq':
            # R50 + WRMM 串行（与 r50wrmm 的差别只在接线）：
            #   r50wrmm：R50 四级并联，各级 WRMM 只精炼「本级 skip」，R50 的下一级
            #            看到的仍是原始 R50 特征（WRMM 输出不进 R50）。
            #   r50seq ：WRMM 插进 R50 的前向主链——WRMM1 的输出既做 112² skip，
            #            又送进 block1；WRMM2 的输出送进 block2；依此类推。
            # 通道对齐：R50 的 block1 入口是 64 通道，故 WRMM1 必须是 64→64（隐层 64），
            # 112² 的 skip 再单独用 1×1 升到 128 供解码器使用；后三级 WRMM 与 R50
            # 各级输出同形（256/512/1024），可直接串行。
            self.stem = nn.Identity()
            self.r50 = ResNetV2(block_units=(3, 4, 6), width_factor=1, in_channels=3)
            self.r50_proj = nn.Conv2d(l1c, l2c, kernel_size=1, bias=False)
            self.wrmm1 = WRMM(l1c, l1c, ablate=wr_ablate, wave_op=wave_op, stride=1)
            self.wrmm2 = WRMM(l3c, l3c, ablate=wr_ablate, wave_op=wave_op, stride=1)
            self.wrmm3 = WRMM(l4c, l4c, ablate=wr_ablate, wave_op=wave_op, stride=1)
            self.wrmm4 = WRMM(l5c, l5c, hidden=l4c, ablate=wr_ablate,
                              wave_op=wave_op, stride=1)
        else:
            # stem：Figure 2 从 224²/64 起画（WRMM1 的 skip 标 H/2 × W/2 × C1），
            # 该级不参与 skip，只做浅层特征提纯
            self.stem = nn.Sequential(
                conv3x3(in_channels, l1c), nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                conv3x3(l1c, l1c), nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            # 四个 WRMM = WRDM（下采样）→ MLKM（卷积）；`unet_enc` 时原位换成 U-Net 编码块
            if self.ablate == 'unet_enc':
                self.wrmm1 = UNetEncBlock(l1c, l2c)
                self.wrmm2 = UNetEncBlock(l2c, l3c)
                self.wrmm3 = UNetEncBlock(l3c, l4c)
                self.wrmm4 = UNetEncBlock(l4c, l5c, deepest=True)
            else:
                self.wrmm1 = WRMM(l1c, l2c, ablate=wr_ablate, wave_op=wave_op)
                self.wrmm2 = WRMM(l2c, l3c, ablate=wr_ablate, wave_op=wave_op)
                if enc == 'r50deep':
                    # 后两级（56²/256→28²/512、28²/512→14²/1024）原位换成 R50 block2/block3
                    self.wrmm3 = self.wrmm4 = None
                else:
                    self.wrmm3 = WRMM(l3c, l4c, ablate=wr_ablate, wave_op=wave_op)
                    # 最深一级通道爆炸（见 WRMM docstring），WRDM/MLKM 放 512 宽隐层后 1×1 升到 1024
                    self.wrmm4 = WRMM(l4c, l5c, hidden=l4c, ablate=wr_ablate, wave_op=wave_op)
            if enc == 'r50deep':
                # 不调 ResNetV2.forward（它带 root/maxpool/补齐逻辑），只取 block2/block3 两级：
                #   56²/256 --block2(stride2)--> 28²/512 --block3(stride2)--> 14²/1024
                # 与 wrmm3/wrmm4 的输入输出逐级同形，因此浅两级 WRMM（含小波支与 MLKM）原样保留。
                # block_units[0]=1 会顺带建出 conv_root/block1（约 0.08M，前向不用）。
                self.r50_deep = ResNetV2(block_units=(1, 4, 6), width_factor=1, in_channels=3)
        # 解码：三级 REM（28²/512、56²/256、112²/128）+ DBB；最后一级只上采样
        # `unet_dec` 时 REM 原位换成 U-Net 融合块（已含 DBB），其后的 DBB 置空避免重复卷积
        if self.ablate == 'unet_dec':
            self.rems = nn.ModuleList([UNetFusion(c) for c in (l4c, l3c, l2c)])
            self.d4_1 = nn.Identity()
            self.d3_1 = nn.Identity()
            self.d2_1 = nn.Identity()
        else:
            self.rems = nn.ModuleList([REM(c, use_region=(self.ablate != 'no_rem'))
                                       for c in (l4c, l3c, l2c)])
            self.d4_1 = DoubleBasicBlock(l4c, l4c)
            self.d3_1 = DoubleBasicBlock(l3c, l3c)
            self.d2_1 = DoubleBasicBlock(l2c, l2c)
        self.d5_1 = up_conv(l5c, l4c)
        self.d4_2 = up_conv(l4c, l3c)
        self.d3_2 = up_conv(l3c, l2c)
        self.d2_2 = up_conv(l2c, l1c)
        self.d1_2 = nn.Conv2d(l1c, num_classes, kernel_size=1)
        init_kaiming(self)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.enc_name == 'r50':
            # R50 的 conv_root 是 3 通道预训练权重（与 A0/V-N1 同一份 ImageNet21k），
            # 单通道灰度图直接 repeat 成 3 通道，不做通道折叠，保持权重逐位可比。
            xr = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
            x5, feats = self.r50(xr)                       # 14²/1024；feats = [28²/512, 56²/256, 112²/64]
            x2 = self.r50_proj(feats[2])                   # 112²/64 → 112²/128
            x3, x4 = feats[1], feats[0]                    # 56²/256、28²/512
        elif self.enc_name == 'r50wrmm':
            # R50 全段出四级特征（ResNetV2.forward 已做 55²→56² 补齐），
            # 每级先过 WRMM 的 stride=1 精炼再接解码器
            xr = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
            x5, feats = self.r50(xr)                       # 14²/1024；[28²/512, 56²/256, 112²/64]
            x2 = self.wrmm1(self.r50_proj(feats[2]))       # 112²/128
            x3 = self.wrmm2(feats[1])                      # 56²/256
            x4 = self.wrmm3(feats[0])                      # 28²/512
            x5 = self.wrmm4(x5)                            # 14²/1024
        elif self.enc_name == 'r50seq':
            # R50 主干逐级手动推进，WRMM 串在两级之间（WRMM 输出既是 skip 又是下一级输入）。
            # 与 ResNetV2.forward 等价：maxpool 后 112²→55²，block1 是 stride1 仍 55²，
            # 原实现用零填充补到 56²，这里用 F.pad 复现同一补齐。
            xr = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
            r = self.r50.root(xr)                          # 112²/64
            w1 = self.wrmm1(r)                             # 112²/64
            x2 = self.r50_proj(w1)                         # 112²/128（skip）
            p = F.max_pool2d(r, kernel_size=3, stride=2, padding=0)   # 55²/64
            y = self.r50.body[0](p)                        # block1 → 55²/256
            pad = 56 - y.size(2)
            if pad > 0:
                y = F.pad(y, (0, pad, 0, pad))
            x3 = self.wrmm2(y)                             # 56²/256（skip → block2）
            y = self.r50.body[1](x3)                       # block2 → 28²/512
            x4 = self.wrmm3(y)                             # 28²/512（skip → block3）
            y = self.r50.body[2](x4)                       # block3 → 14²/1024
            x5 = self.wrmm4(y)                             # 14²/1024（瓶颈）
        else:
            x1 = self.stem(x)                              # 224²/64（不参与 skip）
            x2 = self.wrmm1(x1)                            # 112²/128
            x3 = self.wrmm2(x2)                            # 56²/256
            if self.enc_name == 'r50deep':
                x4 = self.r50_deep.body.block2(x3)         # 56²/256 → 28²/512（ImageNet21k）
                x5 = self.r50_deep.body.block3(x4)         # 28²/512 → 14²/1024（ImageNet21k）
            else:
                x4 = self.wrmm3(x3)                        # 28²/512
                x5 = self.wrmm4(x4)                        # 14²/1024
        up4 = self.d5_1(x5)
        d4 = self.d4_1(self.rems[0](x4, up4))          # 28²/512
        up3 = self.d4_2(d4)
        d3 = self.d3_1(self.rems[1](x3, up3))          # 56²/256
        up2 = self.d3_2(d3)
        d2 = self.d2_1(self.rems[2](x2, up2))          # 112²/128
        return self.d1_2(self.d2_2(d2))                # 224²/64 → logits

    def load_from(self, weights):
        return

    def load_r50(self, weights):
        """只取 ImageNet21k npz 里的 R50 部分（conv_root + block1~3），与 ccfr 同一套键。

        `enc='r50'`（整段编码器）、`enc='r50deep'`（只嫁接 block2/block3）、
        `enc='r50wrmm'`（R50 全段四级 + WRMM 精炼）共用；
        后两者顺带建的其余 block 不参与前向，加载它们无副作用。
        `load_from` 对 wrmm 臂保持空操作（WRMM 无可迁移形状）。
        """
        enc = getattr(self, 'r50', None)
        if enc is None:
            enc = getattr(self, 'r50_deep', None)
        with torch.no_grad():
            conv_w = np2th(weights['conv_root/kernel'], conv=True)
            tgt = enc.root.conv.weight
            if conv_w.shape[1] != tgt.shape[1]:
                pad = torch.zeros(tgt.shape[0], tgt.shape[1] - conv_w.shape[1],
                                  *conv_w.shape[2:])
                conv_w = torch.cat([conv_w, pad], dim=1)
            tgt.copy_(conv_w)
            enc.root.gn.weight.copy_(np2th(weights['gn_root/scale']).view(-1))
            enc.root.gn.bias.copy_(np2th(weights['gn_root/bias']).view(-1))
            for bname, block in enc.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)
        return True
