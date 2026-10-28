# coding=utf-8
"""GobletNet 式轻量高频特征支 + FAM（批次 2 复现）。

论文：GobletNet。高频图 H = HL+LH+HH+λ·LL（Eq.5），FAM 用通道注意力双向
重标定两支特征，FA 后的特征继续进下一层（贴官方实现顺序）。

移植策略（不翻主干容量，单 R50+ViT 编码器）：
  - 高频图在 forward 内从 1 通道原图现算（train/eval 自动一致），λ 为固定
    超参，不进 state_dict；
  - 轻量 HF 支三级细节特征与主干 skip 分辨率对齐：32@112² / 64@56² / 128@28²；
  - FAM 逐级把 skip（sem）与 HF 细节（det）互标定，FA 后的 det 继续下采样，
    编排见 VisionTransformer.forward；
  - 官方 decoder 每级 cat(x_up, x1_fa, x2_fa)，此处最小移植为
    skip = x1_fa + inject(x2_fa)，inject 为 1×1 零初始化卷积；
  - 恒等起步纪律：FAM.gamma=0（输出=x1/x2）、inject 权重=0 → 整条 HF 路
    初始时对 features 逐位恒等，同 seed 下基础权重与 baseline 完全对齐。

官方 FAM 两支通道相同（都是 lNc），此处 sem/det 通道不等，做通道适配：
fusion 把 concat 压到 sem_ch，att1 输出 sem_ch，att2 输出 det_ch。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# HF 支三级细节通道（浅→深），与 skip 特征 64@112 / 256@56 / 512@28 对齐
HF_DET_CHANNELS = (32, 64, 128)


def _conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class HFBranch(nn.Module):
    """轻量 HF 特征支：高频图 → stem(32@112²) → down(64@56²) → down(128@28²)。

    stem 首个卷积 s2 把 224² 高频图降到 112²（与最浅 skip 同分辨率）。
    downs 只在 FAM 重标定之后调用（交错编排见 VisionTransformer.forward），
    forward() 给出的是未经 FAM 的三级原始细节特征（冒烟/调试用）。
    """

    def __init__(self, det_channels=HF_DET_CHANNELS):
        super().__init__()
        c0, c1, c2 = det_channels
        self.det_channels = det_channels
        self.stem = nn.Sequential(_conv_bn_relu(1, c0, stride=2),
                                  _conv_bn_relu(c0, c0))
        self.downs = nn.ModuleList([
            nn.Sequential(_conv_bn_relu(c0, c1, stride=2), _conv_bn_relu(c1, c1)),
            nn.Sequential(_conv_bn_relu(c1, c2, stride=2), _conv_bn_relu(c2, c2)),
        ])

    def forward(self, hf_img):
        det = self.stem(hf_img)
        dets = [det]
        for down in self.downs:
            det = down(det)
            dets.append(det)
        return dets


class FAM(nn.Module):
    """Fusion Attention Module（通道不等适配版）。

    fusion：concat(sem, det) → 1×1 压到 sem_ch + BN + ReLU，得到融合特征；
    att1/att2：两个独立 1×1+BN+Sigmoid 注意力支（官方结构）；
    输出 sem + gamma·relu(sem·a1)、det + gamma·relu(det·a2)（官方为
    relu(x·a) 直接替换，这里加 gamma 门控残差实现恒等起步）。
    """

    def __init__(self, sem_ch, det_ch):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(sem_ch + det_ch, sem_ch, 1, bias=False),
            nn.BatchNorm2d(sem_ch), nn.ReLU(inplace=True))
        self.att1 = nn.Sequential(
            nn.Conv2d(sem_ch, sem_ch, 1, bias=False),
            nn.BatchNorm2d(sem_ch), nn.Sigmoid())
        self.att2 = nn.Sequential(
            nn.Conv2d(sem_ch, det_ch, 1, bias=False),
            nn.BatchNorm2d(det_ch), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, sem, det):
        fused = self.fusion(torch.cat([sem, det], dim=1))
        a1, a2 = self.att1(fused), self.att2(fused)
        return (sem + self.gamma * F.relu(sem * a1),
                det + self.gamma * F.relu(det * a2))
