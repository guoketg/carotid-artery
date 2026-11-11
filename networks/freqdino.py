"""FreqDINO（arXiv 2512.11335，超声分割）两模块移植：MFEA + FGBR。

论文原始设置：DINOv3-Large 冻结 + 轻 adapter，可训练部分只有 MFEA/FGBR/MBGD 与
adapter，消融基线 82.35 Dice（BUSI）。本移植不换骨干、不换数据集，把两个频域模块
挂到 R50-TransUNet 解码器**全部四级输出**（28²/256 → 56²/128 → 112²/64 → 224²/16），
与 DDBE 四级全挂同口径 —— 单点只挂最深层的接法在以往批次里普遍拿不到增益
（DDBE 单级 ≈ 噪声、四级全挂才是唯一过线的那次）。

MFEA（Multi-scale Frequency Extraction and Alignment）
    论文：Haar 两级分解出低频结构 F_L 与多尺度高频 F_Hf，分别经轻量网络产出
    边界注意力 A_b 与结构注意力 A_s，α=β=0.5，F_enh = F_spatial + λ·(F_spatial ⊙
    (αA_b + βA_s))，λ=0.3。
    移植：λ=0.3、α=β=0.5 照抄；多加一枚标量 γ（零初始化）把调制项乘起来，
    使起点与不挂的基线逐位等价（本仓统一「恒等起步」纪律）。γ 的梯度 = λ·x⊙s
    非零，不会被冻死（不同 FGR 的双零问题）。

FGBR（Frequency-Guided Boundary Refinement）
    论文：从拼接的高频特征里逐级降维出 64 维边界原型；Q 来自 F_enh，K/V 来自
    该原型，8 头 × 每头 128 维；F_refined = F_enh + ω·W_O(Attn(Q,K,V))，ω=0.2。
    移植：原型维 64、8 头、每头 128 维、ω=0.2 全部照抄；W_O 零初始化恒等起步。
    注意论文每个头只有 1 个原型 token，softmax 在单 token 上恒为 1，因此本模块
    实际等价于「8 路全局边界原型的低秩加性调制」——这正是它区别于我们已判负的
    全图跨支 QKV 的地方（容量被强制压进 8×64 的瓶颈）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import laplacian_split_channels


def _haar_two_scale(x):
    """两级 Haar：返回 (hf1, hf2, lf)，均已双线性插值回输入分辨率。

    hf1 = 一级细节 |LH|+|HL|+|HH|，hf2 = 二级细节（低频再分解），
    lf = 二级近似 LL。三者通道数均等于 x 的通道数。
    """
    ll1, d1 = laplacian_split_channels(x)
    ll2, d2 = laplacian_split_channels(ll1)
    size = x.shape[-2:]
    hf1 = F.interpolate(d1[0].abs() + d1[1].abs() + d1[2].abs(), size, mode="bilinear",
                        align_corners=False)
    hf2 = F.interpolate(d2[0].abs() + d2[1].abs() + d2[2].abs(), size, mode="bilinear",
                        align_corners=False)
    lf = F.interpolate(ll2, size, mode="bilinear", align_corners=False)
    return hf1, hf2, lf


class _ChannelAttn(nn.Module):
    """轻量通道注意力：GAP → 1×1 → ReLU → 1×1 → σ（论文「lightweight networks」）。"""

    def __init__(self, in_channels, channels, r=16):
        super().__init__()
        hidden = max(channels // r, 8)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        z = F.adaptive_avg_pool2d(x, 1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(z))))


class MFEA(nn.Module):
    """输入输出同形 (B,C,H,W)，H/W 需为偶数（Haar 一级分解要求）。"""

    def __init__(self, channels, lam=0.3, alpha=0.5, beta=0.5):
        super().__init__()
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.attn_b = _ChannelAttn(2 * channels, channels)   # 边界注意力吃两级高频
        self.attn_s = _ChannelAttn(channels, channels)       # 结构注意力吃低频
        # 恒等起步闸门：y = x + λ·γ·(x ⊙ (αA_b + βA_s))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        hf1, hf2, lf = _haar_two_scale(x)
        a_b = self.attn_b(torch.cat([hf1, hf2], dim=1))
        a_s = self.attn_s(lf)
        m = self.alpha * a_b + self.beta * a_s
        return x + self.lam * self.gamma * (x * m)


class FGBR(nn.Module):
    """边界原型引导的空间特征精修（输入输出同形 (B,C,H,W)）。

    四级全挂时按层宽自适应：head_dim = min(128, C)、proto_dim = min(64, C)。
    深两层（256/128）与论文 8×128 + 64 维原型逐字相同；浅两层（64/16）按通道
    收缩 —— 否则 C=16 时 K/V 的 8×64×128 基（131k 参数）会把 16 通道特征完全
    压死，且 224²×8×128 的 Q 张量在 12GB 卡上直接 OOM。
    """

    def __init__(self, channels, heads=8, head_dim=None, proto_dim=None, omega=0.2):
        super().__init__()
        C = channels
        head_dim = min(128, C) if head_dim is None else head_dim
        proto_dim = min(64, C) if proto_dim is None else proto_dim
        self.heads = heads
        self.head_dim = head_dim
        self.proto_dim = proto_dim
        self.omega = omega
        # 逐级降维：2C（两级高频拼接）→ C → C（深度可分离）→ GAP → heads×64 原型
        self.reduce = nn.Sequential(
            nn.Conv2d(2 * C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )
        self.proto = nn.Linear(C, heads * proto_dim)
        # Q 来自空间特征；K/V 来自原型（每头一枚 64 维 token）
        self.q = nn.Linear(C, heads * head_dim)
        self.k = nn.Parameter(torch.zeros(heads, proto_dim, head_dim))
        self.v = nn.Parameter(torch.zeros(heads, proto_dim, head_dim))
        nn.init.xavier_uniform_(self.k)
        nn.init.xavier_uniform_(self.v)
        self.out = nn.Linear(heads * head_dim, C)
        nn.init.zeros_(self.out.weight)   # 唯一零点：恒等起步
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        hf1, hf2, _ = _haar_two_scale(x)
        feat = self.reduce(torch.cat([hf1, hf2], dim=1))
        p = self.proto(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        p = p.view(B, self.heads, self.proto_dim)                       # (B,h,64)
        k = torch.einsum("bhp,hpd->bhd", p, self.k)                     # (B,h,dh)
        v = torch.einsum("bhp,hpd->bhd", p, self.v)                     # (B,h,dh)
        q = self.q(x.flatten(2).transpose(1, 2))                        # (B,N,h*dh)
        q = q.view(B, H * W, self.heads, self.head_dim).permute(0, 2, 1, 3)  # (B,h,N,dh)
        # 单 token softmax（论文每个头只有一枚原型，故权重恒为 1）
        attn = torch.softmax(torch.einsum("bhnd,bhd->bhn", q, k) / math.sqrt(self.head_dim),
                             dim=-1).unsqueeze(-1)
        y = (attn * v.unsqueeze(2)).reshape(B, self.heads, H * W, self.head_dim)
        y = y.permute(0, 2, 1, 3).reshape(B, H * W, self.heads * self.head_dim)
        y = self.out(y).transpose(1, 2).view(B, C, H, W)
        return x + self.omega * y
