# coding=utf-8
"""GobletNet（TMI 2024）端口：语义编码器 + 高频编码器，FAM 四级融合。

论文：Zhou et al., GobletNet: Wavelet-Based High-Frequency Fusion Network for
Semantic Segmentation of Electron Microscopy Images (TMI 2024)。
结构逐行对照作者官方实现（github.com/Yanfeng-Zhou/GobletNet，models/GobletNet.py）。

与官方的适配点（全部在 forward 入口，网络本体不改）：

1. 官方把 HF 图作为离线小波预处理（pywt db2，H = HL+LH+HH+λ·LL，λ=0.1/0.3）
   存盘，训练时读两张图（input1 原图、input2 HF 图）。这里改为 forward 内从
   1 通道原图现算 —— 复用 networks/wavelet.py 的 1 级 Haar
   `build_hf_image(log_domain=False, lf_weight=λ)`，train/eval 自动一致，
   不依赖离线文件。小波基从 db2 换 Haar 是已记录的偏离（论文自证影响小）。
2. 官方 `forward(input1, input2)` 双输入，本端口改为 `forward(x)` 单输入
   （原图），内部现算 HF 图喂语义/HF 两支，训练脚本无需感知第二输入。
3. 单输出（与 XNet 不同，GobletNet 只有一个 1×1 分割头），trainer 无需
   双输出处理。

结构要点（224 输入，l1..l5 = 64/128/256/512/1024 @ 224/112/56/28/14）：

- 双编码器权不共享：语义支 b1_* 吃原图，HF 支 b2_* 吃 HF 图；l1 stem 均为
  3×3+3×3+BasicBlock(if_relu=False)，l2-l4 各一个 DoubleBasicBlock
  （官方第二块 if_relu=False），l5 各一个 BasicBlock；
- FAM（Fusion_Attention_Module）挂在 l1-l4 每级出口：cat[两支特征] →
  1×1+BN+ReLU 融合 → 两路独立 1×1+BN+Sigmoid 注意力图 → 输出
  relu(semantic·A1)、relu(HF·A2)，FAM 结果既送入本支下一级、也作解码 skip；
- 最深 l5 融合：cat[x1_5, x2_5] → BasicBlock(2048→1024, 1×1+BN downsample)
  （官方 fa_l5）；
- 解码每级 cat[up, sem_fa, det_fa] → DoubleBasicBlock(3C→C, 1×1+BN ds) → up，
  末端 1×1 出 num_classes 类 logits。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.unet import (BasicBlock, BN_MOMENTUM, DoubleBasicBlock, conv1x1,
                           conv3x3, down_conv, init_kaiming, up_conv)
from networks.wavelet import build_hf_image

L1C, L2C, L3C, L4C, L5C = 64, 128, 256, 512, 1024


class FusionAttentionModule(nn.Module):
    """官方 Fusion_Attention_Module：融合两支同级特征，各自生成空间注意力图。

    cat[x1, x2] → 1×1+BN+ReLU 融合 → 两路独立 1×1+BN+Sigmoid 注意力，
    输出 relu(x1·A1)、relu(x2·A2)（无残差、无 gamma，论文原样）。
    """

    def __init__(self, in_channels, out_channels):
        super(FusionAttentionModule, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.Sigmoid(),
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        x = self.fusion(torch.cat((x1, x2), dim=1))
        out1 = F.relu(x1 * self.attention1(x))
        out2 = F.relu(x2 * self.attention2(x))
        return out1, out2


def _dbb(cin, cout, downsample=False, if_relu=False):
    """编码 l2-l4 / 解码用 DBB：第二块无末端 ReLU（官方 GobletNet 默认）"""
    ds = nn.Sequential(
        conv1x1(cin, cout), nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
    ) if downsample else None
    return DoubleBasicBlock(cin, cout, ds, if_relu=if_relu)


class GobletNet(nn.Module):
    """GobletNet 端口：输入 (B,1,H,W) 原图 → 单个 (B,num_classes,H,W) logits。

    HF 图在 forward 内现算（见文件头适配点 1）；λ 默认 0.3（论文 Table VI
    最优档），可用 hf_lambda 覆盖。H/W 必须为偶数（224 满足）。
    """

    def __init__(self, in_channels=1, num_classes=2, hf_lambda=0.3):
        super(GobletNet, self).__init__()
        self.hf_lambda = hf_lambda
        # ---- 编码器（b1 = 语义支吃原图，b2 = HF 支吃高频图，权不共享）
        # l1 stem：3×3+3×3+BasicBlock，末端无 ReLU
        self.b1_l1_1 = nn.Sequential(
            conv3x3(in_channels, L1C), conv3x3(L1C, L1C),
            BasicBlock(L1C, L1C, if_relu=False))
        self.b2_l1_1 = nn.Sequential(
            conv3x3(1, L1C), conv3x3(L1C, L1C),
            BasicBlock(L1C, L1C, if_relu=False))
        self.fa_l1 = FusionAttentionModule(L1C * 2, L1C)
        self.b1_l1_2 = down_conv(L1C, L2C)
        self.b2_l1_2 = down_conv(L1C, L2C)
        # l2
        self.b1_l2_1 = _dbb(L2C, L2C)
        self.b2_l2_1 = _dbb(L2C, L2C)
        self.fa_l2 = FusionAttentionModule(L2C * 2, L2C)
        self.b1_l2_2 = down_conv(L2C, L3C)
        self.b2_l2_2 = down_conv(L2C, L3C)
        # l3
        self.b1_l3_1 = _dbb(L3C, L3C)
        self.b2_l3_1 = _dbb(L3C, L3C)
        self.fa_l3 = FusionAttentionModule(L3C * 2, L3C)
        self.b1_l3_2 = down_conv(L3C, L4C)
        self.b2_l3_2 = down_conv(L3C, L4C)
        # l4
        self.b1_l4_1 = _dbb(L4C, L4C)
        self.b2_l4_1 = _dbb(L4C, L4C)
        self.fa_l4 = FusionAttentionModule(L4C * 2, L4C)
        self.b1_l4_2 = down_conv(L4C, L5C)
        self.b2_l4_2 = down_conv(L4C, L5C)
        # l5：各过 BasicBlock 后 cat → BasicBlock(2048→1024, 1×1 ds) 深融合
        self.b1_l5_1 = BasicBlock(L5C, L5C)
        self.b2_l5_1 = BasicBlock(L5C, L5C)
        self.fa_l5 = BasicBlock(
            L5C * 2, L5C,
            downsample=nn.Sequential(
                conv1x1(L5C * 2, L5C),
                nn.BatchNorm2d(L5C, momentum=BN_MOMENTUM)))
        # ---- 解码器：每级 cat[up, sem_fa, det_fa] → DBB(3C→C) → up
        self.d5_1 = up_conv(L5C, L4C)
        self.d4_1 = _dbb(L4C * 3, L4C, downsample=True)
        self.d4_2 = up_conv(L4C, L3C)
        self.d3_1 = _dbb(L3C * 3, L3C, downsample=True)
        self.d3_2 = up_conv(L3C, L2C)
        self.d2_1 = _dbb(L2C * 3, L2C, downsample=True)
        self.d2_2 = up_conv(L2C, L1C)
        self.d1_1 = _dbb(L1C * 3, L1C, downsample=True)
        self.d1_2 = nn.Conv2d(L1C, num_classes, kernel_size=1, stride=1, padding=0)
        init_kaiming(self)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        raw = x[:, 0:1]
        # 图像域 1 级 Haar：H = HL+LH+HH+λ·LL（论文 Eq.5，λ 由 hf_lambda 控制）
        det = build_hf_image(raw, log_domain=False, lf_weight=self.hf_lambda)
        # ---- 编码（FAM 结果既送下一级、也留作解码 skip）
        x1_1 = self.b1_l1_1(raw)
        x2_1 = self.b2_l1_1(det)
        x1_1_fa, x2_1_fa = self.fa_l1(x1_1, x2_1)
        x1_2 = self.b1_l2_1(self.b1_l1_2(x1_1_fa))
        x2_2 = self.b2_l2_1(self.b2_l1_2(x2_1_fa))
        x1_2_fa, x2_2_fa = self.fa_l2(x1_2, x2_2)
        x1_3 = self.b1_l3_1(self.b1_l2_2(x1_2_fa))
        x2_3 = self.b2_l3_1(self.b2_l2_2(x2_2_fa))
        x1_3_fa, x2_3_fa = self.fa_l3(x1_3, x2_3)
        x1_4 = self.b1_l4_1(self.b1_l3_2(x1_3_fa))
        x2_4 = self.b2_l4_1(self.b2_l3_2(x2_3_fa))
        x1_4_fa, x2_4_fa = self.fa_l4(x1_4, x2_4)
        x1_5 = self.b1_l5_1(self.b1_l4_2(x1_4_fa))
        x2_5 = self.b2_l5_1(self.b2_l4_2(x2_4_fa))
        # 最深融合（官方 fa_l5）
        x5 = self.fa_l5(torch.cat((x1_5, x2_5), dim=1))
        # ---- 解码（skip 用 FAM 输出）
        x4 = self.d4_1(torch.cat((self.d5_1(x5), x1_4_fa, x2_4_fa), dim=1))
        x3 = self.d3_1(torch.cat((self.d4_2(x4), x1_3_fa, x2_3_fa), dim=1))
        x2 = self.d2_1(torch.cat((self.d3_2(x3), x1_2_fa, x2_2_fa), dim=1))
        x1 = self.d1_1(torch.cat((self.d2_2(x2), x1_1_fa, x2_1_fa), dim=1))
        return self.d1_2(x1)

    def load_from(self, weights):
        """train.py 统一调用的预训练加载入口；独立复现网络无 R50 权重可载，跳过"""
        return
