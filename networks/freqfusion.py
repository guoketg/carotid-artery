"""NeurIPS 2024 FreqFusion 核心复现（只保留能直接挂到本项目解码器的部分）。

论文：Frequency-aware Feature Fusion for Dense Image Prediction (NeurIPS 2024)
范围：Eq.(6)–(13) 的三个生成器（ALPF / offset / AHPF）与初始融合增强；
     不含论文里的检测/全景分割头、CARAFE/IndexNet/A2U/FADE/SAPA 等对比算子。
接口：ff(x_high, x_low) -> cat([Ỹ, X̃])，通道 = c_high + c_low，
     与原解码器 up + cat 的输出通道对齐（论文是相加，这里保留 concat 以便直接替换）。

与论文的两处工程化偏差（均已记录，便于复现时对齐）：
1. offset 在高层特征分辨率 H×W 上预测，再对 2× 上采样后的特征做重采样；
   论文只在文字里给出 (i+u, j+v) 的采样点，未给 offset 的像素尺度换算。
2. 论文未给出通道压缩率 r 的数值，这里做成参数 reduction（默认 4）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _local_cos_sim(z):
    """Eq.(9)：每个像素与 8 邻域的余弦相似度，返回 (B,8,H,W)。"""
    b, c, h, w = z.shape
    u = F.unfold(z, 3, padding=1).view(b, c, 9, h * w)
    u = F.normalize(u, dim=1, eps=1e-6)
    center = u[:, :, 4:5]
    s = (u * center).sum(dim=1)                      # (B,9,H*W)
    s = s[:, [0, 1, 2, 3, 5, 6, 7, 8]]               # 去掉中心像素
    return s.view(b, 8, h, w)


def _kernel_softmax(w, k):
    """按位置对 K×K 核做 kernel-wise softmax；w: (B, n*k*k, H, W)。"""
    b, nk, h, wd = w.shape
    n = nk // (k * k)
    return w.view(b, n, k * k, h * wd).softmax(dim=2).view(b, nk, h, wd)


def _dyn_filter(x, w, k):
    """逐位置动态卷积（通道共享核）：x (B,C,H,W)，w (B,N,k*k,H,W) -> (B,N*C,H,W)。

    对应 Eq.(8) 的 Σ_{p,q∈Ω} W·Y 与 Eq.(13) 的残差高通滤波。
    """
    b, c, h, wd = x.shape
    n = w.shape[1] // (k * k)
    xu = F.unfold(x, k, padding=k // 2).view(b, 1, c, k * k, h * wd)
    wv = w.flatten(2).view(b, n, 1, k * k, h * wd)
    out = (xu * wv).sum(dim=3)                        # (B,N,C,H*W)
    return out.view(b, n * c, h, wd)


class _ALPF(nn.Module):
    """Eq.(7)–(8)：自适应低通滤波生成器，把高层特征低通上采样到 2×。

    输入低层压缩特征 z_low（2H×2W）预测 K̄² 个核，经 pixel unshuffle 拆成 4 组，
    每组用各自核滤波高层特征 y_high（H×W），最后 PixelShuffle 回 2H×2W。
    """

    def __init__(self, mid, k=5, groups=4):
        super().__init__()
        self.k = k
        self.groups = groups
        self.conv = nn.Conv2d(mid, k * k, 3, padding=1)

    def forward(self, z_low, y_high):
        b, c, h, wd = y_high.shape
        w = self.conv(z_low)                          # (B,k²,2H,2W)
        w = F.pixel_unshuffle(w, 2)                   # (B,4k²,H,W)
        w = _kernel_softmax(w, self.k)                # 4 组低通核，全正且和为 1
        y = _dyn_filter(y_high, w, self.k)            # (B,4C,H,W)
        return F.pixel_shuffle(y, 2)                  # (B,C,2H,2W)


class _AHPF(nn.Module):
    """Eq.(12)–(13)：自适应高通滤波生成器，E − softmax 得到高通核，残差增强。"""

    def __init__(self, mid, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(mid, k * k, 3, padding=1)
        eye = torch.zeros(k * k)
        eye[(k * k) // 2] = 1.0
        self.register_buffer('identity', eye.view(1, k * k, 1, 1))

    def forward(self, z, x):
        w = _kernel_softmax(self.conv(z), self.k)
        w = self.identity - w                          # 高通 = 单位核 − 低通核
        return x + _dyn_filter(x, w, self.k)


class _OffsetGen(nn.Module):
    """Eq.(10)：D = Conv(cat(Z,S))，A = sigmoid(Conv(cat(Z,S)))，O = D·A。"""

    def __init__(self, mid, groups=4):
        super().__init__()
        self.groups = groups
        self.conv_d = nn.Conv2d(mid + 8, 2 * groups, 3, padding=1)
        self.conv_a = nn.Conv2d(mid + 8, 2 * groups, 3, padding=1)

    def forward(self, z, s):
        inp = torch.cat([z, s], dim=1)
        return self.conv_d(inp) * torch.sigmoid(self.conv_a(inp))


def _resample(y, off, groups):
    """按 offset 对 (B,C,2H,2W) 特征分组重采样；off 为 (B,2G,H,W)，单位是高层像素。"""
    b, c, h2, w2 = y.shape
    g = groups if c % groups == 0 else 1
    cg = c // g
    # offset 场先按最近邻铺到上采样分辨率，采样网格与特征同尺寸
    off = F.interpolate(off, size=(h2, w2), mode='nearest')
    ys = torch.arange(h2, device=y.device, dtype=y.dtype) / (h2 / 2.0) - 1.0
    xs = torch.arange(w2, device=y.device, dtype=y.dtype) / (w2 / 2.0) - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0)   # (1,2H,2W,2)
    outs = []
    for i in range(g):
        # 1 个高层像素 = 2 个上采样像素 = 归一化 2/W（或 2/H）
        dx = off[:, 2 * i:2 * i + 1].squeeze(1) * (2.0 / (w2 / 2.0))
        dy = off[:, 2 * i + 1:2 * i + 2].squeeze(1) * (2.0 / (h2 / 2.0))
        grid = base + torch.stack([dx, dy], dim=-1)
        outs.append(F.grid_sample(y[:, i * cg:(i + 1) * cg], grid,
                                  mode='bilinear', padding_mode='border',
                                  align_corners=False))
    return torch.cat(outs, dim=1)


class FreqFusion(nn.Module):
    """论文 Eq.(6)–(13) 的完整融合算子（初始融合增强 + 最终融合）。"""

    def __init__(self, c_high, c_low, reduction=4, low_k=5, high_k=3, offset_groups=4,
                 use_offset=True, use_low_pass=True, use_high_pass=True):
        super().__init__()
        mid = max(8, c_low // reduction)
        self.mid = mid
        self.offset_groups = offset_groups
        self.use_offset = use_offset
        self.use_low_pass = use_low_pass
        self.use_high_pass = use_high_pass
        # Eq.(6)：两支各自 1×1 压到 C/r
        self.compress_high = nn.Conv2d(c_high, mid, 1)
        self.compress_low = nn.Conv2d(c_low, mid, 1)
        # 初始融合用的两条支（ALPF 用低层预测的核上采样高层；AHPF 增强低层）
        self.lp_init = _ALPF(mid, low_k)
        self.hp_init = _AHPF(mid, high_k)
        # 最终融合的三个生成器
        self.lp = _ALPF(mid, low_k)
        self.hp = _AHPF(mid, high_k)
        self.off = _OffsetGen(mid, offset_groups)
        # 输出回到原通道：高层支压到 C/r 后再投回，低层支保留原通道（Eq.13 的 X̃）
        self.proj_high = nn.Conv2d(mid, c_high, 1)

    def forward(self, x_high, x_low):
        z_high = self.compress_high(x_high)                # C/r @ H×W
        z_low = self.compress_low(x_low)                   # C/r @ 2H×2W
        # 4.2 节：初始融合的上采样改用 ALPF，而不是双线性插值
        y0 = self.lp_init(z_low, z_high) if self.use_low_pass else \
            F.interpolate(z_high, scale_factor=2, mode='bilinear', align_corners=False)
        x0 = self.hp_init(z_low, z_low) if self.use_high_pass else z_low
        z = y0 + x0
        # Eq.(5)：Ỹ = F_up(F_lp(Y))；X̃ = X + F_hp(X)
        y = self.lp(z, z_high) if self.use_low_pass else \
            F.interpolate(z_high, scale_factor=2, mode='bilinear', align_corners=False)
        xe = self.hp(z, x_low) if self.use_high_pass else x_low
        if self.use_offset:
            s = _local_cos_sim(z_high)
            off = self.off(F.avg_pool2d(z, 2), s)           # 与 S 同分辨率
            y = _resample(y, off, self.offset_groups)
        return torch.cat([self.proj_high(y), xe], dim=1)
