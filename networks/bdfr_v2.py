"""BDFR-Net（3612 Submission）Eq.(1)–(10) 的逐式实现。

与 v1（networks/ddbe.py 的近似移植）的差异：
- ECA 用 GAP/GMP 双路独立 1×1 bottleneck 相加（Eq.1），不是 ECA-Net 的 1D 卷积；
- HF/LF 支补齐 δ，上采样统一为转置卷积，顺序为 φ1→B→δ→φ3→U→CE（Eq.3）/ φ5→B→δ→U→CE（Eq.4）；
- CTX 门控改为全局通道注意力 CE（Eq.5），不是逐像素 1×1；
- 融合为 (F_hf + F_lf + F_ctx) → φ1(C→C) → CBAM（Eq.6，⊕ 按论文 Eq.1 的定义取逐元素相加），无残差；
- ACAM 用空间 softmax 聚合 + LayerNorm + 加性融合替代 concat（Eq.7–10）。

论文未指定的参数：bottleneck 收缩比 r=16（默认）；小波基 Haar（1 级分解）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import haar_dwt_channels, laplacian_split_channels, laplacian_pyramid_channels


def _bottleneck(channels, reduction):
    """两层 1×1 bottleneck（C→C/r→C），Eq.1/3/4/5/9 中的 φ1(δ(φ1(·)))"""
    hidden = max(channels // reduction, 8)
    return nn.Sequential(
        nn.Conv2d(channels, hidden, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, channels, 1, bias=False),
    )


class ECAv2(nn.Module):
    """Eq.(1)：F ⊙ σ(φ1(δ(φ1(Pavg))) ⊕ φ1(δ(φ1(Pmax))))"""

    def __init__(self, channels, reduction=16):
        super(ECAv2, self).__init__()
        self.gap_mlp = _bottleneck(channels, reduction)
        self.gmp_mlp = _bottleneck(channels, reduction)

    def forward(self, x):
        avg = self.gap_mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.gmp_mlp(x.amax(dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class ChannelAttentionV2(nn.Module):
    """Eq.(5) 的 CE(X) = σ(φ1(δ(φ1(Pavg(X)))))：只做 GAP 的通道注意力"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttentionV2, self).__init__()
        self.mlp = _bottleneck(channels, reduction)

    def gate(self, x):
        """只返回 sigmoid 门控 W，不乘回输入（Eq.5 里 CE 只当门控用）"""
        return torch.sigmoid(self.mlp(x.mean(dim=(2, 3), keepdim=True)))

    def forward(self, x):
        """通道注意力模块形态：X ⊙ W（Eq.3/4 的 CE）"""
        return x * self.gate(x)


class CBAMv2(nn.Module):
    """Eq.(6) 的 M_sp(M_ch(·))：标准 CBAM 通道 + 空间两段"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMv2, self).__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        att = self.mlp(x.mean(dim=(2, 3), keepdim=True)) + self.mlp(x.amax(dim=(2, 3), keepdim=True))
        x = x * torch.sigmoid(att)
        edge = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(edge))


class DDBEv2(nn.Module):
    """DDBE 块 = ECA（Eq.1）+ 双域解耦与融合（Eq.2–6）。

    注意 ECA 不是独立组件：论文 Eq.1 先对特征做 ECA 通道重标定，再进入高低频分解，
    因此本模块的开关必须整体开关，不能只去掉分解而保留 ECA。
    """

    def __init__(self, channels, reduction=16, wave_op='haar'):
        super(DDBEv2, self).__init__()
        # wave_op：把论文的 Haar 分解换成别的基，只改 HF/LF 的取法，其余 Eq.(1)–(6) 不变
        if wave_op not in ('haar', 'lap', 'pyr'):
            raise ValueError('wave_op 只支持 haar/lap/pyr，收到 %r' % (wave_op,))
        self.wave_op = wave_op
        self.eca = ECAv2(channels, reduction)
        # HF 支：cat(LH,HL,HH) 3C→C → B → δ → 3×3 → 转置卷积 → CE
        self.hf_proj = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.hf_bn = nn.BatchNorm2d(channels)
        self.hf_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.hf_up = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False)
        self.hf_ce = ChannelAttentionV2(channels, reduction)
        # LF 支：LL → 5×5 → B → δ → 转置卷积 → CE
        self.lf_conv = nn.Conv2d(channels, channels, 5, padding=2, bias=False)
        self.lf_bn = nn.BatchNorm2d(channels)
        self.lf_up = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False)
        self.lf_ce = ChannelAttentionV2(channels, reduction)
        # CTX 支：7×7 → B → δ → CE → ⊙ F_ECA
        self.ctx_conv = nn.Conv2d(channels, channels, 7, padding=3, bias=False)
        self.ctx_bn = nn.BatchNorm2d(channels)
        self.ctx_ce = ChannelAttentionV2(channels, reduction)
        # Eq.(6)：F_hf ⊕ F_lf ⊕ F_ctx（⊕ = 逐元素相加，论文 Eq.1 已定义）→ φ1(C→C) → CBAM
        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)
        self.cbam = CBAMv2(channels, reduction)

    def forward(self, x):
        f_eca = self.eca(x)
        if self.wave_op == 'lap':
            ll, details = laplacian_split_channels(f_eca)
        elif self.wave_op == 'pyr':
            ll, details = laplacian_pyramid_channels(f_eca)
        else:
            ll, details = haar_dwt_channels(f_eca)

        hf = self.hf_proj(torch.cat(details, dim=1))
        hf = F.relu(self.hf_bn(hf), inplace=True)
        hf = self.hf_conv(hf)
        hf = self.hf_ce(self.hf_up(hf))

        lf = F.relu(self.lf_bn(self.lf_conv(ll)), inplace=True)
        lf = self.lf_ce(self.lf_up(lf))

        ctx = F.relu(self.ctx_bn(self.ctx_conv(f_eca)), inplace=True)
        # Eq.5：F_ECA ⊙ CE(δ(B(φ7(F_ECA))))，CE 在此只当通道门控，不再乘回自身
        ctx = f_eca * self.ctx_ce.gate(ctx)

        return self.cbam(self.fuse(hf + lf + ctx))


class ACAMv2(nn.Module):
    """Eq.(7)–(10)：两流各自空间 softmax 聚合 → LN → 两层 1×1 → sigmoid 门控 → 加性融合。

    g（编码 skip）与 x（解码特征）通道数不同时，用 1×1 把 g 投到 Cx（论文对称解码器下无此步）。
    """

    def __init__(self, g_channels, x_channels):
        super(ACAMv2, self).__init__()
        self.proj_g = nn.Conv2d(g_channels, x_channels, 1, bias=False) \
            if g_channels != x_channels else nn.Identity()
        self.att_g = nn.Conv2d(x_channels, 1, 1)
        self.att_x = nn.Conv2d(x_channels, 1, 1)
        # Eq.(9) σ(φ1(δ(LN(φ1(c_f)))))：内层 φ1 是 C→C（等价 1×1，作用在通道向量上），
        # LN 作用在通道维，再 δ → φ1。用 Linear 实现是为避免 (B,C) 向量还要 reshape 成 4D
        self.in_g = nn.Linear(x_channels, x_channels, bias=False)
        self.in_x = nn.Linear(x_channels, x_channels, bias=False)
        self.ln_g = nn.LayerNorm(x_channels)
        self.ln_x = nn.LayerNorm(x_channels)
        self.out_g = nn.Linear(x_channels, x_channels, bias=False)
        self.out_x = nn.Linear(x_channels, x_channels, bias=False)

    @staticmethod
    def _modulate(f, att, inner, ln, outer):
        b, c, h, w = f.shape
        alpha = torch.softmax(att(f).flatten(2), dim=-1)          # Eq.(7) (B,1,HW)
        ctx = torch.einsum('bnh,bch->bc', alpha, f.flatten(2))     # Eq.(8) (B,C)
        ctx = ln(inner(ctx))                                       # Eq.(9) φ1 → LN
        wgt = torch.sigmoid(outer(F.relu(ctx)))                    # Eq.(9) δ → φ1 → σ
        return wgt.view(b, c, 1, 1)

    def forward(self, g, x):
        g = self.proj_g(g)
        return self._modulate(g, self.att_g, self.in_g, self.ln_g, self.out_g) * g + \
            self._modulate(x, self.att_x, self.in_x, self.ln_x, self.out_x) * x
