# coding=utf-8
"""WeaveSeg (ICCV 2025) 的 ASR：Adaptive Spectral Detail Refinement 模块复现。

论文形式化（原文 Eq.1-4，Figure 3(a)）：

    F_LP_high = PixelShuffle(W_LP ⊛ F_high)                      (1)
    O         = Conv3×3([F_comb, S]),  S = 局部余弦相似度图(8ch)  (2)
    F_R_high(i, j) = F_LP_high(i + u, j + v)                     (3)
    F_out     = F_R_high + F_HF_low                              (4)

输入输出契约（与论文一致）：
    F_high : (B, C, H, W)     —— 更深一层 decoder 的低分辨率特征；
    F_low  : (B, C, 2H, 2W)   —— 更浅一层 skip 的高分辨率特征；
    F_out  : (B, C, 2H, 2W)。

实现要点：
  - 低频支：在 F_high 上按通道预测空间可变低通核 W_LP（每个通道 K² 个系数），
    做逐通道空间可变卷积后 1×1 升到 4C 再 PixelShuffle(×2) 搬到 2H×2W，
    得到 F_LP_high（Eq.1）；
  - LSGR：F_comb = 融合(bilinear_up(F_high), F_low)，其 8 邻域局部余弦相似度
    图 S 与 F_comb 拼接后预测重采样偏移 O（Eq.2），offset 按组给（论文
    O ∈ R^{2G×2H×2W}），用 grid_sample 重采样 F_LP_high 得 F_R_high（Eq.3）；
  - 高频支：在与低频支对称的位置（F_low）上再预测一组逐通道核，高通 = 恒等 −
    学到的低通，即 F_HF_low = F_low − W_HP ⊛ F_low（Eq.4 的第二项）；
  - 输出：F_out = F_R_high + F_HF_low（Eq.4）。

v2（2026-11-02）相对 v1 的改动 —— 去掉三处降低自由度，使模块与 Eq.1-4 等价：
  1) 自适应核：v1 把通道分成 8 组、组内共享同一组 K² 系数；v2 改为逐通道
     （每个通道各自一组 K² 系数），自由度从 8·K² 提到 C·K²；
  2) 核作用分辨率：v1 先把 F_high 1×1 升到 4C 再在 H×W 上做卷积，核要多覆盖
     4C 个通道；v2 先在 C 通道的 F_high 上做逐通道空间可变卷积，再 1×1 升到
     4C + PixelShuffle。两者输出同形，v2 的核数量少 4 倍（省显存），且没有
     「4 个亚像素共享一组系数」的隐式约束；
  3) 重采样偏移：v1 硬 clamp 到 ±MAX_OFFSET（超出即零梯度）；v2 改为
     MAX_OFFSET·tanh(o/MAX_OFFSET)，同样有界但处处可导。

显存：
    逐通道核张量为 (B, C·K², H, W)。在本项目使用到的三级（C=512@14²、
    C=256@28²、C=128@56²）上单样本约 3.6/7.2/14.5 MB；高频支的核在 2H×2W 上，
    C=128@112² 时约 58 MB/样本，是峰值项。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1   # 与 networks/unet.py 统一 BN 动量
MAX_OFFSET = 4.0    # 重采样偏移上限（像素），tanh 软约束
EPS = 1e-6


def _shift(x, dy, dx, pad):
    """把 x 平移 (dy, dx) 像素，边界复制（取 3×3 邻域用）"""
    xp = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    H, W = x.shape[-2:]
    return xp[:, :, pad + dy:pad + dy + H, pad + dx:pad + dx + W]


def _adaptive_conv(x, kernel, k, groups):
    """逐通道空间可变卷积（逐 K² 平移相乘累加）。

    x      : (B, C, H, W)
    kernel : (B, groups·K², H, W)，groups = C 时即逐通道（每个通道一组 K² 系数）。

    等价于 unfold+einsum 的分组深度卷积，但把 K² 次乘加拆成循环，避免物化
    (B, C, K², H·W) 的展开张量（这是本模块显存可控的关键）。
    """
    B, C, H, W = x.shape
    cg = C // groups
    kg = kernel.view(B, groups, k * k, H, W)
    xp = F.pad(x, (k // 2,) * 4, mode='replicate')
    out = x.new_zeros(B, groups, cg, H, W)
    idx = 0
    for i in range(k):
        for j in range(k):
            # (B,groups,1,H,W) 逐组广播到组内每个通道
            wk = kg[:, :, idx].view(B, groups, 1, H, W)
            xs = xp[:, :, i:i + H, j:j + W].view(B, groups, cg, H, W)
            out += xs * wk
            idx += 1
    return out.view(B, C, H, W)


def _local_cosine_map(x):
    """局部余弦相似度图 S ∈ R^{8×H×W}：中心特征与 3×3 邻域（去掉中心）的 cos"""
    xn = F.normalize(x, dim=1, eps=EPS)
    outs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            outs.append((xn * _shift(xn, dy, dx, 1)).sum(dim=1, keepdim=True))
    return torch.cat(outs, dim=1)


def _base_grid(H, W, device, dtype):
    """align_corners=False 的恒等采样网格 (1, H·W, 2)，末维顺序为 (x, y)"""
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (2.0 / H) - 1.0
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (2.0 / W) - 1.0
    gx = xs.view(1, W).expand(H, W)
    gy = ys.view(H, 1).expand(H, W)
    return torch.stack([gx, gy], dim=-1).view(1, H * W, 2)


class ASR(nn.Module):
    """自适应光谱细节精炼：低分辨率高语义特征 + 高分辨率 skip → 细节丰富的融合特征。

    参数：
        channels      : 通道数（F_high / F_low / 输出同通道）；
        kernel_size   : 空间可变卷积核边长 K（默认 3）；
        offset_groups : 重采样偏移的分组数（论文 O 按组给；核本身逐通道，不受此参数影响）。
    """

    def __init__(self, channels, kernel_size=3, offset_groups=8):
        super(ASR, self).__init__()
        self.channels = channels
        self.k = kernel_size
        self.offset_groups = offset_groups if channels % offset_groups == 0 else 1
        hid = max(channels // 4, 8)
        # 低频支（作用于 F_high）：逐通道预测空间可变低通核 W_LP（C·K² 个系数）
        self.kernel_lp = nn.Sequential(
            nn.Conv2d(channels, hid, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels * kernel_size ** 2, 1, bias=True),
        )
        # Eq.1 的亚像素升维：逐通道低通结果 1×1 → 4C，再 PixelShuffle 到 2H×2W
        self.expand_lp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1, bias=False),
            nn.PixelShuffle(2),
        )
        # 高频支（作用于 F_low）：与低频支对称的逐通道核预测，高通 = 恒等 − 低通
        self.kernel_hf = nn.Sequential(
            nn.Conv2d(channels, hid, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels * kernel_size ** 2, 1, bias=True),
        )
        # F_comb：上采样后的 F_high 与 F_low 融合（论文 F_comb，通道回到 C）
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        # LSGR：[F_comb, S] → Conv3×3 → 每组 2 个偏移（dx, dy）
        self.offset = nn.Conv2d(channels + 8, self.offset_groups * 2, 3, padding=1, bias=True)

    def forward(self, F_high, F_low):
        B, C, H, W = F_high.shape
        H2, W2 = F_low.shape[-2:]
        go = self.offset_groups
        cg = C // go
        # F_comb：F_high 上采样到 2H×2W 后与 F_low 融合
        up = F.interpolate(F_high, size=(H2, W2), mode='bilinear', align_corners=False)
        f_comb = self.fuse(torch.cat([up, F_low], dim=1))

        # 低频支 Eq.1：逐通道 W_LP ⊛ F_high → 1×1 升 4C → PixelShuffle(×2)
        f_lp = _adaptive_conv(F_high, self.kernel_lp(F_high), self.k, C)
        f_lp_high = self.expand_lp(f_lp)                    # (B, C, 2H, 2W)

        # LSGR Eq.2-3：局部相似度图 → 偏移 → 对 F_LP_high 内容感知重采样
        s = _local_cosine_map(f_comb)                       # (B, 8, 2H, 2W)
        o = self.offset(torch.cat([f_comb, s], dim=1))      # (B, 2G, 2H, 2W)
        # 像素单位，tanh 软约束到 ±MAX_OFFSET（有界且处处可导）
        o = MAX_OFFSET * torch.tanh(o / MAX_OFFSET)
        o = o.view(B, go, 2, H2, W2).permute(0, 1, 3, 4, 2).reshape(B * go, H2 * W2, 2)
        # 像素偏移 → align_corners=False 下的归一化坐标，逐组加在恒等网格上
        scale = torch.tensor([2.0 / W2, 2.0 / H2], device=o.device, dtype=o.dtype)
        grid = _base_grid(H2, W2, o.device, o.dtype) + o * scale
        f_r_high = F.grid_sample(f_lp_high.view(B * go, cg, H2, W2),
                                 grid.view(B * go, H2, W2, 2), mode='bilinear',
                                 padding_mode='border', align_corners=False)
        f_r_high = f_r_high.view(B, C, H2, W2)

        # 高频支：对称的"学习高通"（恒等 − 学到的低通），逐通道核
        k_hf = self.kernel_hf(F_low)
        f_hf_low = F_low - _adaptive_conv(F_low, k_hf, self.k, C)

        # Eq.4
        return f_r_high + f_hf_low
