"""BDFR-Net 式 DDBE：双域解耦三分支特征融合（单级，挂 TransUNet 解码器某一级输出）。

出处（教师指定论文）：docs/paper-reading.md「BDFR-Net」一节 / pdf\\3612 Submission.pdf。

结构（与论文对齐，残差与零初始化为工程选择）：
    ECA 抑斑点
    ├─ HF 支：2D-DWT 取 LH/HL/HH → concat(3C) → 1×1 → 3×3+BN → 转置卷积上采样 → 通道注意力
    ├─ LF 支：LL → 5×5+BN → 双线性上采样 → 通道注意力
    └─ CTX 支：7×7+BN → 自门控（sigmoid 门乘回）
    → concat(3C) → 1×1（**零初始化**）→ CBAM → 残差加回输入

零初始化只放一处（最后的 1×1）：初始时整支路恒等，logits 与不挂 DDBE 的基线逐位相等，
同 seed 可直接做等价性冒烟（status.md 3.3「一条链上只能有一个零」）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import _HAAR


def dwt_haar(x):
    """多通道 1 级 Haar DWT：(B,C,H,W) → (ll, lh, hl, hh)，各 (B,C,H/2,W/2)，无参数。

    wavelet.haar_dwt 只支持单通道，这里按通道分组做 depthwise 2×2 卷积：
    groups=C 时输出通道排布为 [ch0 的 4 个子带, ch1 的 4 个子带, ...]。
    H/W 需为偶数——解码器四级输出分辨率 28/56/112/224 均满足。
    """
    B, C, H, W = x.shape
    w = _HAAR.to(device=x.device, dtype=x.dtype).view(4, 1, 2, 2).repeat(C, 1, 1, 1)
    bands = F.conv2d(x, w, stride=2, groups=C)               # (B, 4C, H/2, W/2)
    bands = bands.view(B, C, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4)
    return bands[:, 0], bands[:, 1], bands[:, 2], bands[:, 3]


class ECA(nn.Module):
    """1D 卷积通道注意力（Woo et al.），先于三分支抑制斑点。"""

    def __init__(self, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3)).unsqueeze(1)                   # (B,1,C)
        y = self.sigmoid(self.conv(y)).squeeze(1)              # (B,C)
        return x * y.unsqueeze(-1).unsqueeze(-1)


class ChannelAttention(nn.Module):
    """SE 式通道注意力，HF/LF 支上采样之后使用（论文只写「通道注意力」，取 SE）。"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.stack([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=2)  # (B,C,2)
        y = self.sigmoid(self.fc(y).mean(dim=2))                          # (B,C)
        return x * y.unsqueeze(-1).unsqueeze(-1)


class CBAM(nn.Module):
    """通道 + 空间注意力（Woo et al.），三分支融合后的最后一步。"""

    def __init__(self, channels, reduction=8, kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.stack([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=2)  # (B,C,2)
        y = self.sigmoid(self.mlp(y).mean(dim=2))                          # (B,C)
        x = x * y.unsqueeze(-1).unsqueeze(-1)                              # 通道
        s = torch.cat([x.mean(dim=1, keepdim=True),
                       x.amax(dim=1, keepdim=True)], dim=1)        # (B,2,H,W)
        return x * self.sigmoid(self.spatial(s))                   # 空间


class DDBE(nn.Module):
    """双域解耦三分支：输入输出同形 (B,C,H,W)，H/W 为偶数。"""

    def __init__(self, channels):
        super().__init__()
        C = channels
        self.eca = ECA()
        # HF 支
        self.hf_in = nn.Conv2d(3 * C, C, 1)
        self.hf_conv = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.hf_bn = nn.BatchNorm2d(C)
        self.hf_up = nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1)
        self.hf_ca = ChannelAttention(C)
        # LF 支
        self.lf_in = nn.Conv2d(C, C, 5, padding=2, bias=False)
        self.lf_bn = nn.BatchNorm2d(C)
        self.lf_ca = ChannelAttention(C)
        # CTX 支
        self.ctx_conv = nn.Conv2d(C, C, 7, padding=3, bias=False)
        self.ctx_bn = nn.BatchNorm2d(C)
        self.ctx_gate = nn.Conv2d(C, C, 1)
        # 融合（唯一零初始化点）
        self.fuse = nn.Conv2d(3 * C, C, 1)
        self.cbam = CBAM(C)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        f = self.eca(x)
        ll, lh, hl, hh = dwt_haar(f)
        # HF：细节子带上采样回原分辨率
        hf = torch.cat([lh, hl, hh], dim=1)
        hf = self.hf_up(self.hf_bn(self.hf_conv(self.hf_in(hf))))
        hf = self.hf_ca(hf)
        # LF：近似子带双线性上采样
        lf = self.lf_bn(self.lf_in(ll))
        lf = F.interpolate(lf, scale_factor=2, mode="bilinear", align_corners=False)
        lf = self.lf_ca(lf)
        # CTX：大核上下文 + 自门控
        ctx = self.ctx_bn(self.ctx_conv(f))
        ctx = ctx * torch.sigmoid(self.ctx_gate(ctx))
        # 融合：初始 fuse=0 → 输出 = x（恒等起步）
        return x + self.cbam(self.fuse(torch.cat([hf, lf, ctx], dim=1)))


class ACAM(nn.Module):
    """BDFR-Net 式跨级语义对齐（论文 p.3「ACAM for cross-level semantic alignment」）。

    深层解码器特征上采样到 skip 分辨率，与 skip 各自 GAP 后 concat 做通道注意力，
    对 skip 重标定并以零初始化残差加回（恒等起步，保同 seed 等价性）。
    输入：skip (B, skip_ch, Hs, Ws)，dec (B, dec_ch, Hd, Wd)——当前级未上采样特征。
    """

    def __init__(self, skip_ch, dec_ch):
        super().__init__()
        hidden = max(skip_ch // 4, 8)
        self.excite = nn.Sequential(
            nn.Linear(skip_ch + dec_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, skip_ch))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, skip, dec):
        d = F.interpolate(dec, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        pooled = torch.cat([skip.mean(dim=(2, 3)), d.mean(dim=(2, 3))], dim=1)
        w = torch.sigmoid(self.excite(pooled))[..., None, None]
        return skip + self.gamma * (w * skip)
