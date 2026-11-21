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
from networks.ddbe import DDBE
from networks.fgr import EFF, FGR
from networks.sdf import SDFHead
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

    def forward(self, x5, skips, return_levels=False):
        d4 = self.d4_1(torch.cat([self.d5_1(x5), skips[3]], dim=1))
        d3 = self.d3_1(torch.cat([self.d4_2(d4), skips[2]], dim=1))
        d2 = self.d2_1(torch.cat([self.d3_2(d3), skips[1]], dim=1))
        d1 = self.d1_1(torch.cat([self.d2_2(d2), skips[0]], dim=1))
        if return_levels:
            # 双向门控臂逐级交错解码时复用（避免把四级 DBB 再跑一遍）
            return self.d1_2(d1), (d4, d3, d2, d1)
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
                 wavelet_basis="haar", ddbe_level=0, asr=False,
                 ddbe_hf_op="haar", fgr=False, fgr_mode="whole", fgr_pos="dec",
                 couple=False,
                 eff=False, eff_mode="whole", eff_lfdiff=False, edge_branch=False,
                 edge_fuse="none", edge_gate="none", sdf="none", sdf_tau=12.0, ds=False,
                 dec_wrmm=0, dec_wrmm_op="haar", dec_wrmm_mlkm=0):
        super(UNet, self).__init__()
        # couple：跨位置耦合——编码器四级 FGR 的逐通道谱门按分辨率投递到解码器
        # 同级 DDBE 的 HF/LF 两支（零参数接口，见 networks/ddbe.py 的 band_gate）。
        # 只在「编码器 FGR + 解码器 DDBE 四级」这一组合下有意义：UNet 各级编码器与
        # 解码器通道数一一配对（64/128/256/512），门可以直接复用、不必再学一个投影。
        if couple and not (fgr and fgr_pos == "enc" and ddbe_level == 4):
            raise ValueError('couple=True 需要 fgr=True + fgr_pos="enc" + ddbe_level=4'
                             '（编码器 FGR 门 → 解码器同级 DDBE 频带支）')
        self.fgr_ddbe_couple = bool(couple)
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
        # BDFR-Net 式解码器改造（与 baseline 的 --ddbe 同语义）：
        # 解码器四级按 28²/512 → 56²/256 → 112²/128 → 224²/64 编号，
        # ddbe_level=1~3 只挂单级，4 = 四级全挂；ACAM 已于 2026-11-20 移除。
        # DDBE/FGR 模块在下方 init_kaiming 之后创建（RNG 对齐纪律：pitfalls 3
        # 「DDBE 挂载时机太早」——同 seed 基础权重与不挂载版本逐位一致）。
        dec_chs = (l4c, l3c, l2c, l1c)
        self.ddbe_modules = None
        self.fgr_modules = None
        self.fgr_pos = fgr_pos
        self.eff_modules = None
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
        self.edge_gate = "none"
        self.edge_proj = None
        self.edge_heads = None
        self.edge_gate_w = None
        if edge_branch:
            self.edge_dec = UNetDecoder(num_classes)
            init_kaiming(self.edge_dec)  # 与主解码器同初始化口径
            self.edge_gate = edge_gate
        elif edge_gate != "none":
            raise ValueError('edge_gate 需要 edge_branch=True（没有边缘支就没有可交互的解码器）')
        if self.edge_gate == "bi":
            # 双向逐级门控（ba-net/gscnn 式）：主解码每级特征零初始化投影进边缘支，
            # 边缘支每级边界响应经 sigmoid 门控回主解码。全部零初始化（α=0、Π=0），
            # 第 0 步与「只挂边缘支、两条解码器互不相通」的臂逐位等价。
            # 必须在 init_kaiming 之后创建，保证同 seed 的基础权重与历史臂逐位一致。
            chs = (l4c, l3c, l2c, l1c)   # 28²/512、56²/256、112²/128、224²/64
            self.edge_proj = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chs])
            self.edge_heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in chs])
            for mod in list(self.edge_proj) + list(self.edge_heads):
                nn.init.zeros_(mod.weight)
                nn.init.zeros_(mod.bias)
            self.edge_gate_w = nn.Parameter(torch.zeros(4))
        # edge_fuse：把边缘概率以零初始化标量 γ 残差加到主解码器最浅层（224²）。
        # γ=0 时与不注入逐位一致，是否采用边缘证据交给训练自己决定。
        self.edge_fuse_scale = (nn.Parameter(torch.zeros(1))
                                if edge_fuse == "add" and edge_branch else None)
        # 连续界面表示（--sdf，与 baseline 侧同语义）：距离场头挂在主解码器
        # 224²/64 上，最后创建以保证同 seed 的基础权重与不挂载版本逐位对齐。
        #   out = 分割输出改成带符号距离场（logits = ±sharp·sdf，零水平集给掩膜）
        #   aux = 只作窄带辅助监督，分类头仍是主输出
        self.sdf_mode = sdf
        self.sdf_head = None
        self.sdf_sharp = 0.5
        if sdf != "none":
            self.sdf_head = SDFHead(l1c, tau=float(sdf_tau))
        # 多尺度深监督（--ds）：112²/128、56²/256 两级解码中间特征各挂一个 1×1
        # 分类头，监督权重 0.3 / 0.15（浅层更大，与 Syn-Net 深监督同向）。
        # 推理只取主输出，辅助头不参与评测链路。
        self.ds_heads = None
        if ds:
            self.ds_heads = nn.ModuleDict({
                "h112": nn.Conv2d(l2c, num_classes, 1),
                "h56": nn.Conv2d(l3c, num_classes, 1),
            })
        # 解码器 WRMM（2026-11-20）：把 `unet_wmren` 编码器的 WRDM 镜像搬到 U-Net
        # 解码器，stride=1 同级精炼，挂在每个解码级 DBB 输出之后：
        #   28²/512 → 56²/256 → 112²/128 → 224²/64。
        # 动机：WRMM 此前只出现在编码器（K2 / r50wrmm），而 r50wrmm 对 R50 基线是
        # −0.75~−0.84，判为「预训练编码器已编码的低频被再分一次频 = 冗余」。解码器
        # 上采样路径是编码器完全没碰过的区域，本开关就是把这层频域精炼挪到那里。
        # 档位与 wmren 的 `--wmren_dec_wrmm` 一致：1=四级全挂，2=只挂浅两级（112/224），
        # 3=只挂 224²/64。最后创建（在 init_kaiming 之后），保证同 seed 的基础权重
        # 与不挂载版本逐位一致（RNG 对齐纪律）。
        self.dec_wrmm = int(dec_wrmm)
        self.dec_wrmms = None
        if self.dec_wrmm:
            if self.dec_wrmm not in (1, 2, 3):
                raise ValueError('dec_wrmm must be 0/1/2/3, got %r' % (dec_wrmm,))
            # 延迟导入：`unet_wmren` 反过来 import 本文件的积木，模块级导入会成环
            from networks.unet_wmren import WRDM, MLKM
            levels = {1: (0, 1, 2, 3), 2: (2, 3), 3: (3,)}[self.dec_wrmm]
            full = bool(dec_wrmm_mlkm)   # 1 = 完整 WRMM（WRDM → MLKM），0 = 只有 WRDM

            def _blk(c):
                if not full:
                    return WRDM(c, c, wave_op=dec_wrmm_op, stride=1)
                return nn.Sequential(WRDM(c, c, wave_op=dec_wrmm_op, stride=1), MLKM(c, c))

            mods = [_blk(c) if i in levels else nn.Identity() for i, c in enumerate(dec_chs)]
            self.dec_wrmms = nn.ModuleList(mods)

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

    def _enc_fgr(self, idx, x):
        """编码器主链上的 FGR 改写（idx 顺序 0=224²/64 … 3=28²/512）。

        耦合臂（couple=True）顺带导出该级谱门 w (B,2C,1,1) 供解码器同级 DDBE 使用，
        非耦合臂返回 (x, None)，与改动之前逐位一致。
        """
        if self.fgr_modules is None or self.fgr_pos != "enc":
            return x, None
        if self.fgr_ddbe_couple:
            return self.fgr_modules[idx](x, return_gate=True)
        return self.fgr_modules[idx](x), None

    def _edge_level(self, level, prev, skip):
        """双向门控臂的单级边缘解码：up_conv → cat(skip) → DBB（与 UNetDecoder 同构）。

        level 与主解码一致：0=28²/512 … 3=224²/64；prev 为该级上一层的边缘特征
        （level 0 时为编码器瓶颈 x5）。
        """
        ups = (self.edge_dec.d5_1, self.edge_dec.d4_2, self.edge_dec.d3_2, self.edge_dec.d2_2)
        blocks = (self.edge_dec.d4_1, self.edge_dec.d3_1, self.edge_dec.d2_1, self.edge_dec.d1_1)
        return blocks[level](torch.cat([ups[level](prev), skip], dim=1))

    def _decode(self, up, skip, fuse, level, band_gate=None):
        """单级解码：ABF → concat skip → DBB → DDBE → FGR（与 baseline 同序）。

        FGR 只在 fgr_pos='dec' 时挂在这里；enc/skip 位置在 forward 里各自应用。
        band_gate 非空时（couple 臂）作为编码器同级 FGR 的谱门送进该级 DDBE。
        """
        if self.egf is not None:
            skip = self.egf[level](skip, up)
        out = fuse(torch.cat([up, skip], dim=1))
        if self.ddbe_modules is not None:
            out = (self.ddbe_modules[level](out, band_gate=band_gate)
                   if band_gate is not None else self.ddbe_modules[level](out))
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
        x1, g1 = self._enc_fgr(0, x1)
        x2 = self.l2_1(self.l1_2(x1))        # 112²/128
        x2, g2 = self._enc_fgr(1, x2)
        x3 = self.l3_1(self.l2_2(x2))        # 56²/256
        x3, g3 = self._enc_fgr(2, x3)
        x4 = self.l4_1(self.l3_2(x3))        # 28²/512
        x4, g4 = self._enc_fgr(3, x4)
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
        bi = (self.edge_dec is not None and self.edge_gate == "bi")
        if self.edge_dec is not None and not bi:
            # 边缘分支只吃共享编码器的瓶颈与原始 skip（浅→深）
            edge = self.edge_dec(x5, (x1, x2, x3, x4))
        # 耦合臂：解码器第 level 级（28²→224²）吃编码器同级 FGR 的谱门，级号按
        # 分辨率配对（解码 level0=28²/512 ← FGR idx3=28²/512，其余同理）。
        gates = (g1, g2, g3, g4)
        d4 = self._decode(self._up(0, x5, skips[3]), skips[3], self.d4_1, 0, gates[3])
        if self.dec_wrmms is not None:
            d4 = self.dec_wrmms[0](d4)
        if bi:
            e4 = self._edge_level(0, x5, skips[3])
            e4 = e4 + self.edge_proj[0](d4)
            d4 = d4 * (1 + self.edge_gate_w[0] * torch.sigmoid(self.edge_heads[0](e4)))
        d3 = self._decode(self._up(1, d4, skips[2]), skips[2], self.d3_1, 1, gates[2])
        if self.dec_wrmms is not None:
            d3 = self.dec_wrmms[1](d3)
        if bi:
            e3 = self._edge_level(1, e4, skips[2])
            e3 = e3 + self.edge_proj[1](d3)
            d3 = d3 * (1 + self.edge_gate_w[1] * torch.sigmoid(self.edge_heads[1](e3)))
        d2 = self._decode(self._up(2, d3, skips[1]), skips[1], self.d2_1, 2, gates[1])
        if self.dec_wrmms is not None:
            d2 = self.dec_wrmms[2](d2)
        if bi:
            e2 = self._edge_level(2, e3, skips[1])
            e2 = e2 + self.edge_proj[2](d2)
            d2 = d2 * (1 + self.edge_gate_w[2] * torch.sigmoid(self.edge_heads[2](e2)))
        d1 = self._decode(self._up(3, d2, skips[0]), skips[0], self.d1_1, 3, gates[0])
        if self.dec_wrmms is not None:
            d1 = self.dec_wrmms[3](d1)
        if bi:
            e1 = self._edge_level(3, e2, skips[0])
            e1 = e1 + self.edge_proj[3](d1)
            d1 = d1 * (1 + self.edge_gate_w[3] * torch.sigmoid(self.edge_heads[3](e1)))
            edge = self.edge_dec.d1_2(e1)
        if self.edge_fuse_scale is not None:
            # 边缘概率（第 1 类）按标量 γ 残差注入主路最浅层；γ=0 起步
            e = torch.softmax(edge, dim=1)[:, 1:2]
            e = F.interpolate(e, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = d1 + self.edge_fuse_scale * e
        # 深监督辅助头吃解码中间特征（112²/128、56²/256），与主输出同网格比例
        ds = None
        if self.ds_heads is not None:
            ds = [(self.ds_heads["h112"](d2), 0.3),
                  (self.ds_heads["h56"](d3), 0.15)]
        out = self.d1_2(d1)
        sdf = None
        if self.sdf_head is not None:
            sdf = self.sdf_head(d1)
            if self.sdf_mode == "out":
                out = torch.cat([-self.sdf_sharp * sdf,
                                 self.sdf_sharp * sdf], dim=1)
        if self.edge_dec is not None or sdf is not None or ds is not None:
            res = {"out": out}
            if self.edge_dec is not None:
                res["edge"] = edge
            if sdf is not None:
                res["sdf"] = sdf
            if ds is not None:
                res["aux"] = ds
            return res
        return out

    def load_from(self, weights):
        """train.py 统一调用的预训练加载入口；卷积 U-Net 无 ImageNet 权重可载，直接跳过"""
        return


# ---------------------------------------------------------------------------
# 标准 U-Net（2026-11-21 起成为 --arch unet 的默认宿主，残差版退到 --unet_style res）
# ---------------------------------------------------------------------------
class _PlainDoubleConv(nn.Module):
    """原版 U-Net 的双卷积块：conv3×3 + BN + ReLU，连续两次（无残差捷径）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _PlainDown(nn.Module):
    """原版下采样：MaxPool2d(2) + 双卷积（零参数池化，不是 stride-2 卷积）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _PlainDoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class _PlainUp(nn.Module):
    """原版上采样：双线性插值到 skip 尺寸 → 1×1 降维（无 BN、无激活）→ concat → 双卷积。"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv = _PlainDoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.conv(torch.cat([x, skip], dim=1))


class PlainUNet(nn.Module):
    """标准 U-Net（Ronneberger 2015 拓扑）+ 与残差版逐点对齐的模块挂载接口。

    通道按原版论文的 64→1024：base_channel=64 时 64/128/256/512/1024，参数量
    28.95 M；base_channel=32 时 32/64/128/256/512，参数量 7.24 M（与同组研究生
    的 PlainUNet 逐位同构，可用于交叉核对）。

    与 `UNet`（残差版）的差别只在基础积木：MaxPool 下采样、纯双卷积、无残差、
    双线性上采样 + 1×1 降维。模块挂载点与调用顺序照抄残差版，保证同一份队列
    脚本换宿主后仍是单变量对照：
        编码器 l1..l4 输出 → FGR（fgr_pos=enc 时）
        解码器四级输出     → DDBE → FGR（fgr_pos=dec 时）→ dec_wrmm(+MLKM)
    不支持的开关（asr/eff/edge_in/edge_branch/sdf/ds/edge_fuse/edge_gate）直接
    抛错，避免静默跑成另一个模型。
    """

    def __init__(self, in_channels=1, num_classes=2, base_channel=64,
                 ddbe_level=0, ddbe_hf_op="haar",
                 fgr=False, fgr_mode="whole", fgr_pos="dec", couple=False,
                 dec_wrmm=0, dec_wrmm_op="haar", dec_wrmm_mlkm=0,
                 asr=False, eff=False, edge_in=False, edge_branch=False,
                 edge_fuse="none", edge_gate="none", sdf="none", ds=False,
                 **_ignored):
        super().__init__()
        unsupported = {"asr": asr, "eff": eff, "edge_in": edge_in,
                       "edge_branch": edge_branch, "sdf": sdf != "none", "ds": ds,
                       "edge_fuse": edge_fuse != "none", "edge_gate": edge_gate != "none"}
        bad = [k for k, v in unsupported.items() if v]
        if bad:
            raise NotImplementedError(
                'PlainUNet（标准 U-Net 宿主）暂未接这些开关：%s；需要它们请用 --unet_style res'
                % ", ".join(bad))

        self.in_channels = int(in_channels)
        self.base_channel = int(base_channel)
        c0, c1, c2, c3, c4 = (self.base_channel * m for m in (1, 2, 4, 8, 16))

        # 编码器：stem 双卷积 → 4 次 MaxPool+双卷积
        self.l1_1 = _PlainDoubleConv(in_channels, c0)
        self.l1_2 = _PlainDown(c0, c1)
        self.l2_1 = _PlainDown(c1, c2)
        self.l3_1 = _PlainDown(c2, c3)
        self.l4_1 = _PlainDown(c3, c4)

        # 解码器：与残差版同名同序（d5_1 先上采样，d4_1 出该级特征，依此类推）
        self.d5_1 = _PlainUp(c4, c3, c3)
        self.d4_1 = _PlainUp(c3, c2, c2)
        self.d3_1 = _PlainUp(c2, c1, c1)
        self.d2_1 = _PlainUp(c1, c0, c0)
        self.d1_2 = nn.Conv2d(c0, num_classes, 1)

        dec_chs = (c3, c2, c1, c0)                 # 28²/512 → 224²/64
        self.ddbe_modules = None
        self.fgr_modules = None
        self.fgr_pos = fgr_pos
        self.fgr_ddbe_couple = bool(couple)
        if couple and not (fgr and fgr_pos == "enc" and ddbe_level == 4):
            raise ValueError('couple=True 需要 fgr=True + fgr_pos="enc" + ddbe_level=4')
        if ddbe_level:
            if ddbe_level == 4:
                self.ddbe_modules = nn.ModuleList([DDBE(c, hf_op=ddbe_hf_op) for c in dec_chs])
            else:
                mods = [nn.Identity() for _ in dec_chs]
                mods[ddbe_level - 1] = DDBE(dec_chs[ddbe_level - 1], hf_op=ddbe_hf_op)
                self.ddbe_modules = nn.ModuleList(mods)
        if fgr:
            if fgr_pos == "dec":
                chs, sizes = dec_chs, (28, 56, 112, 224)
            else:
                chs, sizes = (c0, c1, c2, c3), (224, 112, 56, 28)
            self.fgr_modules = nn.ModuleList(
                [FGR(c, s, s, mode=fgr_mode) for c, s in zip(chs, sizes)])
        self.dec_wrmm = int(dec_wrmm)
        self.dec_wrmms = None
        if self.dec_wrmm:
            if self.dec_wrmm not in (1, 2, 3):
                raise ValueError('dec_wrmm must be 0/1/2/3, got %r' % (dec_wrmm,))
            from networks.unet_wmren import WRDM, MLKM
            levels = {1: (0, 1, 2, 3), 2: (2, 3), 3: (3,)}[self.dec_wrmm]
            full = bool(dec_wrmm_mlkm)

            def _blk(c):
                if not full:
                    return WRDM(c, c, wave_op=dec_wrmm_op, stride=1)
                return nn.Sequential(WRDM(c, c, wave_op=dec_wrmm_op, stride=1), MLKM(c, c))

            self.dec_wrmms = nn.ModuleList(
                [_blk(c) if i in levels else nn.Identity() for i, c in enumerate(dec_chs)])

    def attach_edge_fusion(self, kind="abf", init_gamma=0.1):
        raise NotImplementedError('PlainUNet 不支持 --edge_fusion；请用 --unet_style res')

    def _enc_fgr(self, idx, x):
        if self.fgr_modules is None or self.fgr_pos != "enc":
            return x, None
        if self.fgr_ddbe_couple:
            return self.fgr_modules[idx](x, return_gate=True)
        return self.fgr_modules[idx](x), None

    def _decode(self, up, skip, block, level, band_gate=None):
        out = block(up, skip)
        if self.ddbe_modules is not None:
            out = (self.ddbe_modules[level](out, band_gate=band_gate)
                   if band_gate is not None else self.ddbe_modules[level](out))
        if self.fgr_modules is not None and self.fgr_pos == "dec":
            out = self.fgr_modules[level](out)
        return out

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x1, g1 = self._enc_fgr(0, self.l1_1(x))          # 224²/64
        x2, g2 = self._enc_fgr(1, self.l1_2(x1))         # 112²/128
        x3, g3 = self._enc_fgr(2, self.l2_1(x2))         # 56²/256
        x4, g4 = self._enc_fgr(3, self.l3_1(x3))         # 28²/512
        x5 = self.l4_1(x4)                               # 14²/1024

        skips = (x1, x2, x3, x4)
        if self.fgr_pos == "skip" and self.fgr_modules is not None:
            skips = tuple(m(s) for m, s in zip(self.fgr_modules, skips))
        gates = (g1, g2, g3, g4)

        d4 = self._decode(x5, skips[3], self.d5_1, 0, gates[3])
        if self.dec_wrmms is not None:
            d4 = self.dec_wrmms[0](d4)
        d3 = self._decode(d4, skips[2], self.d4_1, 1, gates[2])
        if self.dec_wrmms is not None:
            d3 = self.dec_wrmms[1](d3)
        d2 = self._decode(d3, skips[1], self.d3_1, 2, gates[1])
        if self.dec_wrmms is not None:
            d2 = self.dec_wrmms[2](d2)
        d1 = self._decode(d2, skips[0], self.d2_1, 3, gates[0])
        if self.dec_wrmms is not None:
            d1 = self.dec_wrmms[3](d1)
        return self.d1_2(d1)

    def load_from(self, weights):
        return
