# coding=utf-8
"""带符号距离场（SDF）回归头：把分割从"逐像素分类"换成"连续界面表示"。

为什么与数据集无关（CLAUDE.md 第 17 条要求的自查）：
- 距离场目标由每张 GT 掩膜的距离变换**逐样本**生成，不含任何群体统计量
  （没有面积/圆度/宽高比/形状先验），换数据集只需 GT 是二值掩膜；
- 不做推理期后处理：掩膜直接由 sdf > 0 给出，零水平集就是界面，
  亚像素位置保存在连续值里，不是靠阈值扫描或形态学修边；
- tanh 饱和：|sdf| 超过 tau 后梯度为 0，监督自动聚焦在界面窄带，
  这是"窄带监督"，与图像尺度、设备、裁块方式都无关。

输出单位为像素（原始 224 网格），范围 (-tau, tau)。
"""
import torch
import torch.nn as nn


class SDFHead(nn.Module):
    def __init__(self, in_ch, hidden=32, tau=12.0):
        super(SDFHead, self).__init__()
        self.tau = float(tau)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        # 第一次前向时 x 的分辨率即输出网格（TransUNet 解码器为 224²），不做额外插值
        return torch.tanh(self.body(x)) * self.tau
