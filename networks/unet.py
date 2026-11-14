# coding=utf-8
"""纯卷积 U-Net 基线 + XNet/GobletNet 同族积木（忠实移植两篇论文的官方 block）。

积木取自作者官方仓库（github.com/Yanfeng-Zhou/XNet、/GobletNet）：

- `down_conv`：3×3 stride2 + BN + ReLU（下采样）；
- `up_conv`：双线性 ×2 + 3×3 + BN + ReLU（上采样）；
- `same_conv`：3×3 同尺度 + BN + ReLU（融合用的同尺度变换）；
- `transition_conv`：1×1 + BN + ReLU（融合后的通道压合）；
- `BasicBlock`：ResNet 式 3×3+BN+ReLU ×2，残差相加后再 ReLU
  （注意官方写法是 `bn2(conv2) + identity` 再 relu，不是标准 pre-add BN）；
- `DoubleBasicBlock`：两个 BasicBlock 串联，第一块可带 1×1+BN 降采样 shortcut。
  XNet 官方两块都以 ReLU 结尾；GobletNet 官方第二块默认不带末端 ReLU
  （`if_relu=False`），两个端口各自保持原样。

UNet 基线（本文件）：与两篇论文同构的 5 级 U-Net —— l1 224²/64 →
l2 112²/128 → l3 56²/256 → l4 28²/512 → l5 14²/1024，解码对称，
skip 为 concat 后接 DoubleBasicBlock（1×1+BN 降维），末端 1×1 卷积出
num_classes 类 logits。作为 XNet/GobletNet 的容量对齐基线：两篇论文的
网络都以这种 U-Net 为底座，只加频率分支与融合模块，所以基线也必须是
这套积木的纯 U-Net，而不是项目原有的 TransUNet（容量不同没有可比性）。

基线不做 GobletNet 的 if_relu 消融变体（stem/DBB 末端 ReLU 全开），
即教科书式 ResUNet；两篇论文端口在各自文件里保持官方 if_relu 原样。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.asr import ASR
from networks.ddbe import ACAM, DDBE
from networks.fgr import EFF, FGR
from networks.wavelet import build_edge_map


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


BN_MOMENTUM = 0.1  # 官方仓库统一 BN 动量


class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class asr_up(nn.Module):
    """ASR 版上采样（WeaveSeg ICCV 2025）：1×1+BN+ReLU 通道对齐 → ASR(feat, skip)

    输出 (B, ch_out, 2H, 2W)，与 skip 同形状同通道，后续 cat + DBB 不变。
    """

    def __init__(self, ch_in, ch_out):
        super(asr_up, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.asr = ASR(ch_out)

    def forward(self, feat, skip):
        return self.asr(self.proj(feat), skip)


class same_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(same_conv, self).__init__()
        self.same = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.same(x)


class transition_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(transition_conv, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.transition(x)


class BasicBlock(nn.Module):
    """官方 BasicBlock：conv3×3→BN→ReLU→conv3×3→BN，加 identity 后再 ReLU。

    if_relu=False 时输出不带末端 ReLU（GobletNet 的 stem/l2-l4 DBB 用）。
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, if_relu=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.if_relu = if_relu

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.bn2(out) + identity
        if self.if_relu:
            out = self.relu(out)
        return out


class DoubleBasicBlock(nn.Module):
    """两个 BasicBlock 串联；第一块可带 1×1+BN 降采样 shortcut。

    if_relu 控制第二块末端是否有 ReLU：XNet 官方恒为 True；
    GobletNet 官方默认 False（编码 l2-l4 与解码各 DBB 都不带）。
    """

    def __init__(self, inplanes, planes, downsample=None, if_relu=True):
        super(DoubleBasicBlock, self).__init__()
        self.DBB = nn.Sequential(
            BasicBlock(inplanes=inplanes, planes=planes, downsample=downsample),
            BasicBlock(inplanes=planes, planes=planes, if_relu=if_relu),
        )

    def forward(self, x):
        return self.DBB(x)


def make_downsample(in_planes, out_planes):
    """DoubleBasicBlock 的 1×1+BN 残差降采样 shortcut（官方写法）"""
    return nn.Sequential(
        conv1x1(in_planes, out_planes),
        nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
    )


def init_kaiming(model):
    """官方初始化：卷积 Kaiming(fan_out)、BN weight=1/bias=0"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class UNetDecoder(nn.Module):
    """与主解码器逐层同构的独立解码器（边缘分支用）：14²/1024 瓶颈 + 四级 skip → 224²。

    双解码器实验要求「共享编码器 + 两个对等解码器」：这里参数形状与主解码器完全
    一致（约 +9M），主解码器保持原样不重构，避免改变历史 checkpoint 的 state_dict 键。
    skips 按浅→深 (x1,x2,x3,x4) 传入。
    """

    def __init__(self, num_classes=2):
        super(UNetDecoder, self).__init__()
        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024
        self.d5_1 = up_conv(l5c, l4c)
        self.d4_1 = DoubleBasicBlock(l4c * 2, l4c, make_downsample(l4c * 2, l4c))
        self.d4_2 = up_conv(l4c, l3c)
        self.d3_1 = DoubleBasicBlock(l3c * 2, l3c, make_downsample(l3c * 2, l3c))
        self.d3_2 = up_conv(l3c, l2c)
        self.d2_1 = DoubleBasicBlock(l2c * 2, l2c, make_downsample(l2c * 2, l2c))
        self.d2_2 = up_conv(l2c, l1c)
        self.d1_1 = DoubleBasicBlock(l1c * 2, l1c, make_downsample(l1c * 2, l1c))
        self.d1_2 = nn.Conv2d(l1c, num_classes, kernel_size=1)

    def forward(self, x5, skips):
        d4 = self.d4_1(torch.cat([self.d5_1(x5), skips[3]], dim=1))
        d3 = self.d3_1(torch.cat([self.d4_2(d4), skips[2]], dim=1))
        d2 = self.d2_1(torch.cat([self.d3_2(d3), skips[1]], dim=1))
        d1 = self.d1_1(torch.cat([self.d2_2(d2), skips[0]], dim=1))
        return self.d1_2(d1)


class UNet(nn.Module):
    """容量对齐的纯卷积 U-Net 基线（单输入单输出，输入 (B,1,224,224)）。

    编码 l1(l1_1 stem)→l5 与 GobletNet 语义支完全同构；解码为标准两路
    skip（up + encoder skip → DBB），与 GobletNet 的三路 skip 解码相比只
    少了高频支那份 skip —— 这正是两篇论文声称的增量来源。

    asr=True 时接入 WeaveSeg (ICCV 2025) 的 ASR：把解码器每级的 up_conv
    换成 1×1 通道对齐 + ASR(F_high, skip)，其后的 cat + DBB 结构原样不动，
    所以下游参数形状不变、可做单变量对照。默认只在解码器前三级
    （28²/512、56²/256、112²/128）启用，224²/64 那一级保留原 up_conv：
    该级 F_low 是 224²×64，自适应核 + 相似度图在这个分辨率上显存代价最高，
    而 ASR 主要解决的是深层语义特征上采样时的边界细节丢失。
    asr=False（默认）时不新增任何模块，state_dict 与老 checkpoint 逐键一致。
    """

    def __init__(self, in_channels=1, num_classes=2, edge_in=False, edge_op="wavelet",
                 wavelet_basis="haar", ddbe_level=0, acam=False, asr=False,
                 ddbe_hf_op="haar", fgr=False, fgr_mode="whole", fgr_pos="dec",
                 eff=False, eff_mode="whole", eff_lfdiff=False, edge_branch=False,
                 edge_fuse="none"):
        super(UNet, self).__init__()
        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024
        # edge_in：stem 多吃一张边缘图（与 TransUNet edge_in 同一算子、同一 λ=0 口径）
        self.edge_in = edge_in
        self.edge_op = edge_op
        # wavelet_basis：小波基选择实验（只作用于 edge_op=wavelet/wavelet_lam）
        self.wavelet_basis = wavelet_basis
        stem_in = in_channels + 1 if edge_in else in_channels
        # encoder：l1 stem（两层 3×3 + 残差块），l2-l4 各一个 DBB，l5 一个 BasicBlock
        self.l1_1 = nn.Sequential(
            conv3x3(stem_in, l1c), conv3x3(l1c, l1c), BasicBlock(l1c, l1c))
        self.l1_2 = down_conv(l1c, l2c)
        self.l2_1 = DoubleBasicBlock(l2c, l2c)
        self.l2_2 = down_conv(l2c, l3c)
        self.l3_1 = DoubleBasicBlock(l3c, l3c)
        self.l3_2 = down_conv(l3c, l4c)
        self.l4_1 = DoubleBasicBlock(l4c, l4c)
        self.l4_2 = down_conv(l4c, l5c)
        self.l5_1 = BasicBlock(l5c, l5c)
        # decoder：每级 cat(up, skip) → DBB(2C→C) → up；末端 1×1 出 logits
        self.d5_1 = up_conv(l5c, l4c)
        self.d4_1 = DoubleBasicBlock(l4c * 2, l4c, make_downsample(l4c * 2, l4c))
        self.d4_2 = up_conv(l4c, l3c)
        self.d3_1 = DoubleBasicBlock(l3c * 2, l3c, make_downsample(l3c * 2, l3c))
        self.d3_2 = up_conv(l3c, l2c)
        self.d2_1 = DoubleBasicBlock(l2c * 2, l2c, make_downsample(l2c * 2, l2c))
        self.d2_2 = up_conv(l2c, l1c)
        self.d1_1 = DoubleBasicBlock(l1c * 2, l1c, make_downsample(l1c * 2, l1c))
        self.d1_2 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # BDFR-Net 式解码器改造（与 baseline 的 --ddbe/--acam 同语义）：
        # 解码器四级按 28²/512 → 56²/256 → 112²/128 → 224²/64 编号，
        # ddbe_level=1~3 只挂单级，4 = 四级全挂；acam 对每级 skip 做跨级对齐。
        # DDBE/FGR 模块在下方 init_kaiming 之后创建（RNG 对齐纪律：pitfalls 3
        # 「DDBE 挂载时机太早」——同 seed 基础权重与不挂载版本逐位一致）。
        dec_chs = (l4c, l3c, l2c, l1c)
        self.ddbe_modules = None
        self.fgr_modules = None
        self.fgr_pos = fgr_pos
        self.eff_modules = None
        self.acams = nn.ModuleList([ACAM(c, c) for c in dec_chs]) if acam else None
        # 解码器 skip 汇合点的融合模块（ABF）：由 train.py 在权重初始化之后挂，
        # 保证同 seed 的基础权重与基线逐位对齐
        self.egf = None
        # ASR（WeaveSeg ICCV 2025）：把解码器前三级的上采样算子原地替换成
        # 1×1+BN+ReLU 通道对齐 → ASR(低分辨率特征, 同通道 skip)，模块名不变。
        self.asr_levels = ()
        if asr:
            # (ch_in, ch_out) 与四级 up_conv 一一对应：28²/512 → 56²/256 → 112²/128 → 224²/64
            dec_pairs = ((l5c, l4c), (l4c, l3c), (l3c, l2c), (l2c, l1c))
            up_names = ('d5_1', 'd4_2', 'd3_2', 'd2_2')
            self.asr_levels = (0, 1, 2)
            for level in self.asr_levels:
                c_in, c_out = dec_pairs[level]
                setattr(self, up_names[level], asr_up(c_in, c_out))
        init_kaiming(self)
        # 挂载模块在全部基础权重初始化之后建（RNG 对齐纪律，与 baseline 侧 L572 同型）：
        # DDBE 内部 conv 保留 PyTorch 默认初始化（与 R50 侧 DDBE 一致），fuse 本身
        # 在 DDBE.__init__ 已零初始化且不再被 init_kaiming 覆盖，恒等起步自然成立。
        # FGR 同理：fuse 零初始化、gate bias 2.0，挂载次序与 baseline（ddbe→fgr）一致。
        if ddbe_level:
            if ddbe_level == 4:
                self.ddbe_modules = nn.ModuleList([DDBE(c, hf_op=ddbe_hf_op) for c in dec_chs])
            else:
                modules = [nn.Identity() for _ in dec_chs]
                modules[ddbe_level - 1] = DDBE(dec_chs[ddbe_level - 1], hf_op=ddbe_hf_op)
                self.ddbe_modules = nn.ModuleList(modules)
        if fgr:
            # fgr_mode：whole = 单核改写 [hf,lf]（原 FGR）；band = 双零核并联分带（FBGR）
            # fgr_pos：dec = 解码器四级输出（历史位置）；enc = 编码器主干四级输出
            # （改写后的 x_i 继续下采样，同时作 skip）；skip = 只改写送进解码器的
            # skip 副本（编码器主干用原特征，对照「是否只需改 skip 内容」）。
            if fgr_pos == "dec":
                chs, sizes = dec_chs, (28, 56, 112, 224)
            else:
                chs, sizes = (l1c, l2c, l3c, l4c), (224, 112, 56, 28)
            self.fgr_modules = nn.ModuleList(
                [FGR(c, s, s, mode=fgr_mode) for c, s in zip(chs, sizes)])
        if eff:
            # EFF：编码器跨尺度频率融合阶梯（自顶向下级联，见 fgr.py EFF）
            # eff_mode='band' / eff_lfdiff：EFF-v2 的两条独立改动（默认关 = 首版）
            self.eff_modules = nn.ModuleList([
                EFF(c, s, s, dc, mode=eff_mode, lf_diff=eff_lfdiff)
                for c, dc, s in zip((l1c, l2c, l3c, l4c),
                                    (l2c, l3c, l4c, l5c), (224, 112, 56, 28))])
        # 边缘监督分支（共享编码器 + 独立的对称边缘解码器）：最后创建，
        # 保证同 seed 的基础权重与不挂载版本逐位对齐（RNG 对齐纪律）。
        self.edge_dec = None
        if edge_branch:
            self.edge_dec = UNetDecoder(num_classes)
            init_kaiming(self.edge_dec)  # 与主解码器同初始化口径
        # edge_fuse：把边缘概率以零初始化标量 γ 残差加到主解码器最浅层（224²）。
        # γ=0 时与不注入逐位一致，是否采用边缘证据交给训练自己决定。
        self.edge_fuse_scale = (nn.Parameter(torch.zeros(1))
                                if edge_fuse == "add" and edge_branch else None)

    def _up(self, level, feat, skip):
        """单级上采样：该级挂了 ASR 就用 ASR(feat, skip)，否则用原 up_conv"""
        ups = (self.d5_1, self.d4_2, self.d3_2, self.d2_2)
        if level in self.asr_levels:
            return ups[level](feat, skip)
        return ups[level](feat)

    def attach_edge_fusion(self, kind="abf", init_gamma=0.1):
        """在四级 skip 汇合点挂 ABF（非对称频带融合）。

        弱骨干版配置：speckle=True（斑点噪声衰减）+ large_kernel=True（depthwise 7×7
        补感受野）。kind='abftied' 为参数完全匹配的对照（α_H = −α_L，退化单门控）。
        """
        from .edge_fusion import AsymBandFusion
        chans = (512, 256, 128, 64)   # 28²/512、56²/256、112²/128、224²/64
        self.egf = nn.ModuleList([
            AsymBandFusion(c, speckle=True, large_kernel=True,
                           tied=(kind == "abftied"), init_gamma=init_gamma)
            for c in chans])

    def _decode(self, up, skip, fuse, level):
        """单级解码：ABF → ACAM 对齐 skip → concat → DBB → DDBE → FGR（与 baseline 同序）。

        FGR 只在 fgr_pos='dec' 时挂在这里；enc/skip 位置在 forward 里各自应用。
        """
        if self.egf is not None:
            skip = self.egf[level](skip, up)
        if self.acams is not None:
            skip = self.acams[level](skip, up)
        out = fuse(torch.cat([up, skip], dim=1))
        if self.ddbe_modules is not None:
            out = self.ddbe_modules[level](out)
        if self.fgr_modules is not None and self.fgr_pos == "dec":
            out = self.fgr_modules[level](out)
        return out

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if self.edge_in:
            x = torch.cat([x, build_edge_map(x, self.edge_op,
                                             basis=self.wavelet_basis)], dim=1)
        x1 = self.l1_1(x)                    # 224²/64
        if self.fgr_pos == "enc" and self.fgr_modules is not None:
            x1 = self.fgr_modules[0](x1)
        x2 = self.l2_1(self.l1_2(x1))        # 112²/128
        if self.fgr_pos == "enc" and self.fgr_modules is not None:
            x2 = self.fgr_modules[1](x2)
        x3 = self.l3_1(self.l2_2(x2))        # 56²/256
        if self.fgr_pos == "enc" and self.fgr_modules is not None:
            x3 = self.fgr_modules[2](x3)
        x4 = self.l4_1(self.l3_2(x3))        # 28²/512
        if self.fgr_pos == "enc" and self.fgr_modules is not None:
            x4 = self.fgr_modules[3](x4)
        x5 = self.l5_1(self.l4_2(x4))        # 14²/1024
        if self.eff_modules is not None:
            # EFF 自顶向下级联：y4←(x4,x5)、y3←(x3,y4)、y2←(x2,y3)、y1←(x1,y2)
            y4 = self.eff_modules[3](x4, x5)
            y3 = self.eff_modules[2](x3, y4)
            y2 = self.eff_modules[1](x2, y3)
            y1 = self.eff_modules[0](x1, y2)
            skips = (y1, y2, y3, y4)
        else:
            skips = (x1, x2, x3, x4)
        if self.fgr_pos == "skip" and self.fgr_modules is not None:
            # 只改写送进解码器的 skip 副本，编码器主干仍用原特征
            skips = tuple(m(s) for m, s in zip(self.fgr_modules, skips))
        if self.edge_dec is not None:
            # 边缘分支只吃共享编码器的瓶颈与原始 skip（浅→深）
            edge = self.edge_dec(x5, (x1, x2, x3, x4))
        d4 = self._decode(self._up(0, x5, skips[3]), skips[3], self.d4_1, 0)
        d3 = self._decode(self._up(1, d4, skips[2]), skips[2], self.d3_1, 1)
        d2 = self._decode(self._up(2, d3, skips[1]), skips[1], self.d2_1, 2)
        d1 = self._decode(self._up(3, d2, skips[0]), skips[0], self.d1_1, 3)
        if self.edge_fuse_scale is not None:
            # 边缘概率（第 1 类）按标量 γ 残差注入主路最浅层；γ=0 起步
            e = torch.softmax(edge, dim=1)[:, 1:2]
            e = F.interpolate(e, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = d1 + self.edge_fuse_scale * e
        out = self.d1_2(d1)
        if self.edge_dec is not None:
            return {"out": out, "edge": edge}
        return out

    def load_from(self, weights):
        """train.py 统一调用的预训练加载入口；卷积 U-Net 无 ImageNet 权重可载，直接跳过"""
        return
