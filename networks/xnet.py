# coding=utf-8
"""XNet（ICCV 2023）全监督版端口：小波低/高频双分支 + X 形分离-融合。

论文：Zhou et al., XNet: Wavelet-Based Low and High Frequency Fusion Networks
for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images.
结构逐行对照作者官方实现（github.com/Yanfeng-Zhou/XNet，及同作者
GobletNet 仓库 models/xnet.py，二者一致）。

与官方的适配点（全部在 forward 入口，网络本体不改）：

1. 官方把 LF/HF 图作为离线小波预处理（pywt db2，tools/wavelet2D.py）存盘，
   训练时读两张图。这里改为 forward 内从 1 通道原图现算 —— 复用项目
   networks/wavelet.py 的 1 级 Haar（LF = 只保留 LL 的 IDWT 重建，
   HF = HL+LH+HH 置零 LL 的 IDWT 重建），train/eval 自动一致，不依赖
   离线文件。小波基从 db2 换 Haar 是已记录的偏离（论文自证小波基影响小）。
2. 官方 branch1 吃 LF 图、branch2 吃 HF 图，二者都是 1 通道，本项目同构。
3. 官方返回 (out1, out2) 双输出：训练时两支都监督（本项目口径
   0.5CE+0.5Dice 各算一遍再平均，见 trainer.py）；评估统一取第一支
   （低频主支），不在测试集上挑更好的一支。
4. X 形融合无门控、无零初始化 —— 论文原样，不适用本项目的恒等起步纪律
   （该纪律只约束"在已有 baseline 上插新模块"的实验，独立复现网络除外）。

结构要点（224 输入，l1..l5 = 64/128/256/512/1024 @ 224/112/56/28/14）：

- 每支编码：l1 stem（3×3+3×3+BasicBlock）→ l2/l3 各 DoubleBasicBlock
  → l4 两个 DBB（第二个提供融合用的 same/down 特征）→ l5 DBB；
- X 形融合只在最深两级（论文消融最优 = 2LF+2HF）：
    第 5 级 cat[本支 same5(1024), 对方 same5(1024), 对方 l4 下采样(512)]
        → transition 1×1 压回 1024 → DBB → up 到 28²/512；
    第 4 级 cat[本支 same4(512), 对方 same4(512), 对方 l5 上采样(1024)]
        → transition 1×1 压回 512 → DBB，再与本支第 5 级融合结果
        cat → DBB(512+512→512) → up 到 56²/256；
  即每支的融合模块吃的是「本支同尺度特征 + 对方同尺度特征 + 对方
  上/下采样特征」，融合结果只回流进本支解码器，形成 X 形交叉；
- 解码：l3/l2/l1 各 cat[本支 encoder skip, 上采样] → DBB(2C→C) → up，
  末端 1×1 出 logits；encoder skip 不被融合改写（官方如此）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.unet import (BasicBlock, DoubleBasicBlock, conv3x3, down_conv,
                           init_kaiming, make_downsample, same_conv,
                           transition_conv, up_conv)
from networks.wavelet import _standardize, build_hf_image, haar_dwt

L1C, L2C, L3C, L4C, L5C = 64, 128, 256, 512, 1024


def _build_ll_image(x, size):
    """LF 图：图像域 1 级 Haar 取 LL 子带，上采样回 size 并逐样本标准化。

    Haar 下「只保留 LL 的 IDWT 重建」等价于 LL 逐 2 倍上采样；与 HF 路
    build_hf_image(log_domain=False) 同域、同标准化口径。
    """
    if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
        x = F.interpolate(x, size=(x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2),
                          mode="bilinear", align_corners=False)
    ll, _ = haar_dwt(x)
    ll = F.interpolate(ll, size=(size, size), mode="bilinear", align_corners=False)
    return _standardize(ll)


class _XNetBranch(nn.Module):
    """XNet 单支：编码 l1..l5 + 本支流内的 X 形融合模块 + 解码 l3..l1。

    两支结构完全对称、权不共享，仅 stem 输入通道不同（LF 支 in_channels、
    HF 支恒 1）。融合模块命名 m4/m5（官方 b*_5_3/b*_4_4 系列）。
    """

    def __init__(self, in_channels, num_classes):
        super(_XNetBranch, self).__init__()
        # ---- 编码
        self.l1_1 = nn.Sequential(
            conv3x3(in_channels, L1C),
            conv3x3(L1C, L1C),
            BasicBlock(L1C, L1C))
        self.l1_2 = down_conv(L1C, L2C)
        self.l2_1 = DoubleBasicBlock(L2C, L2C)
        self.l2_2 = down_conv(L2C, L3C)
        self.l3_1 = DoubleBasicBlock(L3C, L3C)
        self.l3_2 = down_conv(L3C, L4C)
        self.l4_1 = DoubleBasicBlock(L4C, L4C)   # x4（l5 的来源）
        self.l4_2 = DoubleBasicBlock(L4C, L4C)   # 融合特征的来源（same4/down4）
        self.l4_3_same = same_conv(L4C, L4C)
        self.l4_3_down = down_conv(L4C, L4C)
        self.l4_2_down = down_conv(L4C, L5C)
        self.l5_1 = DoubleBasicBlock(L5C, L5C)
        self.l5_2_same = same_conv(L5C, L5C)
        self.l5_2_up = up_conv(L5C, L5C)
        # ---- X 形融合（本支流内）
        self.m5_transition = transition_conv(L5C + L5C + L4C, L5C)
        self.m5_block = DoubleBasicBlock(L5C, L5C)
        self.m5_up = up_conv(L5C, L4C)
        self.m4_transition = transition_conv(L4C + L4C + L5C, L4C)
        self.m4_block = DoubleBasicBlock(L4C, L4C)
        self.m4_skip = DoubleBasicBlock(L4C + L4C, L4C, make_downsample(L4C + L4C, L4C))
        self.m4_up = up_conv(L4C, L3C)
        # ---- 解码
        self.d3_block = DoubleBasicBlock(L3C + L3C, L3C, make_downsample(L3C + L3C, L3C))
        self.d3_up = up_conv(L3C, L2C)
        self.d2_block = DoubleBasicBlock(L2C + L2C, L2C, make_downsample(L2C + L2C, L2C))
        self.d2_up = up_conv(L2C, L1C)
        self.d1_block = DoubleBasicBlock(L1C + L1C, L1C, make_downsample(L1C + L1C, L1C))
        self.head = nn.Conv2d(L1C, num_classes, kernel_size=1, stride=1, padding=0)

    def encode(self, x):
        """返回解码 skip 与融合所需的同尺度/跨尺度特征（对照官方 forward 变量名）"""
        x1 = self.l1_1(x)                        # 224²/64
        x2 = self.l2_1(self.l1_2(x1))            # 112²/128
        x3 = self.l3_1(self.l2_2(x2))            # 56²/256
        x4 = self.l4_1(self.l3_2(x3))            # 28²/512（l5 从这里下采样，非 x4b）
        x4b = self.l4_2(x4)
        same4 = self.l4_3_same(x4b)              # 28²/512 同尺度
        down4 = self.l4_3_down(x4b)              # 14²/512 下采样（供对方第 5 级融合）
        x5 = self.l5_1(self.l4_2_down(x4))       # 14²/1024
        same5 = self.l5_2_same(x5)               # 14²/1024 同尺度
        up5 = self.l5_2_up(x5)                   # 28²/1024（供对方第 4 级融合）
        return {"x1": x1, "x2": x2, "x3": x3,
                "same4": same4, "down4": down4, "same5": same5, "up5": up5}

    def merge_decode(self, e_self, e_other):
        """X 形融合 + 解码出本支 logits（官方 merge/decode 段逐行对应）"""
        # 第 5 级：本支同尺度 + 对方同尺度 + 对方 l4 下采样
        m5 = self.m5_transition(torch.cat(
            [e_self["same5"], e_other["same5"], e_other["down4"]], dim=1))
        m5 = self.m5_up(self.m5_block(m5))       # 28²/512
        # 第 4 级：本支同尺度 + 对方同尺度 + 对方 l5 上采样，再并入本支第 5 级结果
        m4 = self.m4_transition(torch.cat(
            [e_self["same4"], e_other["same4"], e_other["up5"]], dim=1))
        m4 = self.m4_block(m4)
        m4 = self.m4_skip(torch.cat([m4, m5], dim=1))
        # 解码：只用本支的 pristine encoder skip
        d3 = self.d3_block(torch.cat([e_self["x3"], self.m4_up(m4)], dim=1))
        d2 = self.d2_block(torch.cat([e_self["x2"], self.d3_up(d3)], dim=1))
        d1 = self.d1_block(torch.cat([e_self["x1"], self.d2_up(d2)], dim=1))
        return self.head(d1)


class XNet(nn.Module):
    """XNet 全监督端口：输入 (B,1,H,W) 原图 → (logits_LF, logits_HF)。

    LF/HF 图在 forward 内现算（见文件头适配点 1）；H/W 必须为偶数
    （224 满足），wavelet.py 内部会兜底裁偶。
    """

    def __init__(self, in_channels=1, num_classes=2):
        super(XNet, self).__init__()
        self.branch_lf = _XNetBranch(in_channels, num_classes)
        self.branch_hf = _XNetBranch(1, num_classes)
        init_kaiming(self)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        raw = x[:, 0:1]
        # 图像域 1 级 Haar（论文为图像域 db2 DWT，不走本项目别处的对数域约定）
        lf = _build_ll_image(raw, size=raw.shape[-1])
        hf = build_hf_image(raw, log_domain=False, lf_weight=0.0)
        e_lf = self.branch_lf.encode(lf)
        e_hf = self.branch_hf.encode(hf)
        out_lf = self.branch_lf.merge_decode(e_lf, e_hf)
        out_hf = self.branch_hf.merge_decode(e_hf, e_lf)
        return out_lf, out_hf

    def load_from(self, weights):
        """train.py 统一调用的预训练加载入口；独立复现网络无 R50 权重可载，跳过"""
        return
