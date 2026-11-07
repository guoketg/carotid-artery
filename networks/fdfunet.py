"""FDFUNet 复现（Multi-scale frequency domain filtering network, Neural Networks 2024）

逐条对应论文：

- Eq.(1) / Fig.5(e)  DRDAC：double residual dilated atrous convolution
      X1 = Conv3x3(X)
      X2 = X1 + PWConv( X1 + ASSConv(X1) )
      Y  = Conv1x1( Concat(X, X2) )          # 外侧"残差"是 concat+1x1，不是相加
- Eq.(2)(3) / Fig.6(a) MFDF：LN → 通道对半切 → 空间支 DASConv / 频率支 FDF
      四路（X11 DASConv, X12 FDF, X21 DASConv, X22 FDF）concat 后加输入残差
- Eq.(4)  FDF：Y = IFFT( W ⊙ FFT(X) )，W 为 (C, H, W/2+1) 可学习复数权重
- Eq.(5) / Fig.6(b)  ASCA：双轴池化取代表值 (2C,1,1) → MLP → Conv1d → Sigmoid → 乘回
- Fig.7  整体：5 级 UNet*（通道 64/128/256/512/1024）+ skip 直加 + 双线性上采样，
      每个 Down 后挂 FEM，前 4 个 Up 后挂 FEM（最后一个 Up 只做上采样）

论文未写明的实现细节（FDF 初始化、ASCA 池化顺序、MLP 结构等）在
docs/study/fdfunet-repro.md 的"歧义决策表"里逐项记录。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 基础算子
class ASSConv(nn.Module):
    """式(3) 中的空洞可分离卷积本体：depthwise dilated conv（groups=C, rate=3）。"""

    def __init__(self, channels, rate=3):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate,
                            groups=channels, bias=True)

    def forward(self, x):
        return self.dw(x)


class DASConv(nn.Module):
    """式(3)：DASConv = PWConv ∘ ASSConv = W_PW(W_ASS·X + B_ASS)。"""

    def __init__(self, channels, rate=3):
        super().__init__()
        self.ass = ASSConv(channels, rate)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.pw(self.ass(x))


class DRDAC(nn.Module):
    """式(1) / Fig.5(e)：双残差空洞卷积（只用在编码器，Table 9 的结论）。"""

    def __init__(self, in_ch, out_ch, rate=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.ass = ASSConv(out_ch, rate)
        self.pw = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.fuse = nn.Conv2d(in_ch + out_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = x1 + self.pw(x1 + self.ass(x1))   # 内残差出现两次（Fig.5(e) 的两个 ⊕）
        return self.fuse(torch.cat([x, x2], dim=1))


class FDF(nn.Module):
    """式(4)：频域滤波 Y = IFFT(W ⊙ FFT(X))。

    W 为复数参数，按 GFNet 惯例用 randn×0.02 初始化（论文未给，见歧义表 A4）。
    """

    def __init__(self, channels, height, width, init_scale=0.02):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(channels, height, width // 2 + 1, 2) * init_scale)

    def forward(self, x):
        h, w = x.shape[-2:]
        xf = torch.fft.rfft2(x.float(), norm='ortho')
        wc = torch.view_as_complex(self.weight.contiguous())
        y = torch.fft.irfft2(xf * wc, s=(h, w), norm='ortho')
        return y.type_as(x)


class MFDF(nn.Module):
    """式(2) / Fig.6(a)：多尺度频率域滤波（LN 在通道维，逐像素归一化）。"""

    def __init__(self, channels, height, width):
        super().__init__()
        assert channels % 4 == 0, 'MFDF 需要通道数能被 4 整除（两次对半切）'
        c_half = channels // 2
        c_quad = c_half // 2
        self.ln_w = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.ln_b = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.spatial = DASConv(c_half)          # X1 -> X1'
        self.das11 = DASConv(c_quad)            # 空间支前 1/4
        self.fdf12 = FDF(c_quad, height, width)  # 空间支后 1/4（频率）
        self.fdf2 = FDF(c_half, height, width)   # X2 整支频率滤波
        self.das21 = DASConv(c_quad)            # 频率支前 1/4（空间）
        self.fdf22 = FDF(c_quad, height, width)  # 频率支后 1/4

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        h = (x - mu) * torch.rsqrt(var + 1e-6) * self.ln_w + self.ln_b
        x1, x2 = h.chunk(2, dim=1)
        x1 = self.spatial(x1)
        x11, x12 = x1.chunk(2, dim=1)
        x2 = self.fdf2(x2)
        x21, x22 = x2.chunk(2, dim=1)
        out = torch.cat([self.das11(x11), self.fdf12(x12),
                         self.das21(x21), self.fdf22(x22)], dim=1)
        return x + out


class ASCA(nn.Module):
    """式(5) / Fig.6(b)：自适应空间通道注意力。

    H 轴：AvgPool(沿 W) → MaxPool(沿 H)；W 轴：MaxPool(沿 H) → AvgPool(沿 W)。
    该非对称顺序取自 Table 14 作者最终选用的配置（歧义表 A5 记录与图 6(b) 的差异）。
    """

    def __init__(self, channels, k=3, variant='row5'):
        super().__init__()
        self.variant = variant
        self.mlp = nn.Linear(2 * channels, channels)
        self.conv = nn.Conv1d(channels, channels, k, padding=k // 2)

    def forward(self, x):
        hgt, wid = x.shape[-2:]
        if self.variant == 'row5':
            y1 = F.adaptive_avg_pool2d(x, (hgt, 1))      # (B,C,H,1)
            r1 = F.adaptive_max_pool2d(y1, 1)            # (B,C,1,1)
            y2 = F.adaptive_max_pool2d(x, (1, wid))      # (B,C,1,W)
            r2 = F.adaptive_avg_pool2d(y2, 1)            # (B,C,1,1)
        else:                                            # 'symmetric'：正文/图 6(b) 的 Avg→Max
            r1 = F.adaptive_max_pool2d(F.adaptive_avg_pool2d(x, (hgt, 1)), 1)
            r2 = F.adaptive_max_pool2d(F.adaptive_avg_pool2d(x, (1, wid)), 1)
        v = torch.cat([r1, r2], dim=1).flatten(1)        # (B,2C)
        v = F.relu(self.mlp(v))
        v = self.conv(v.unsqueeze(-1)).squeeze(-1)       # (B,C)
        return x * torch.sigmoid(v)[:, :, None, None]


class FEM(nn.Module):
    """Fig.6/7 的 FEM：MFDF 与 ASCA 串联。"""

    def __init__(self, channels, height, width, use_mfdf=True, use_asca=True,
                 asca_variant='row5'):
        super().__init__()
        self.mfdf = MFDF(channels, height, width) if use_mfdf else nn.Identity()
        self.asca = ASCA(channels, variant=asca_variant) if use_asca else nn.Identity()

    def forward(self, x):
        return self.asca(self.mfdf(x))


# ---------------------------------------------------------------- 主干
class DownBlock(nn.Module):
    """Fig.7：DRDAC → BN → MaxPool → ReLU（DOUBLE residual 块的主体）。"""

    def __init__(self, in_ch, out_ch, use_drdac=True):
        super().__init__()
        self.body = DRDAC(in_ch, out_ch) if use_drdac else \
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.norm = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return F.relu(self.pool(self.norm(self.body(x))), inplace=True)


class UpBlock(nn.Module):
    """Fig.7：Conv → BN → Interpolate(双线性) → ReLU。"""

    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.norm = nn.BatchNorm2d(out_ch)
        self.upsample = upsample

    def forward(self, x):
        x = self.norm(self.conv(x))
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return F.relu(x, inplace=True)


class FDFUNet(nn.Module):
    """5 级 UNet* 主干 + DRDAC + 每级 FEM。

    结构依据 Fig.7 的分辨率/通道自洽性推导（见歧义表 A1）：
      编码：224 →112 →56 →28 →14 →7（每个 Down 末尾都池化，FEM 挂在池化后）
      解码：7 →14 →28 →56 →112 →224（前 4 个 Up 后接 FEM，且 skip 直加在 FEM 之前）
    因此 skip 在 14/28/56/112 四处，通道正好等于该级 Up 的输出通道。

    use_drdac/use_mfdf/use_asca 全关时即为论文的 UNet* 基线。
    """

    def __init__(self, in_channels=1, num_classes=2, img_size=224,
                 channels=(64, 128, 256, 512, 1024),
                 use_drdac=True, use_mfdf=True, use_asca=True,
                 asca_variant='row5'):
        super().__init__()
        assert img_size % 32 == 0, '5 级主干要求输入尺寸能被 32 整除'
        self.channels = tuple(channels)
        # 编码器中各个 FEM 所在的分辨率（池化后）
        sizes = [img_size // (2 ** (i + 1)) for i in range(len(channels))]
        self.sizes = sizes

        self.downs = nn.ModuleList()
        in_ch = in_channels
        for c in channels:
            self.downs.append(DownBlock(in_ch, c, use_drdac=use_drdac))
            in_ch = c
        self.fems_down = nn.ModuleList([
            FEM(c, s, s, use_mfdf=use_mfdf, use_asca=use_asca, asca_variant=asca_variant)
            for c, s in zip(channels, sizes)])
        # 解码：前 4 个 Up 后接 FEM，最后一个 Up 只上采样
        self.ups = nn.ModuleList()
        self.fems_up = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.ups.append(UpBlock(channels[i], channels[i - 1]))
            # Up 之后的分辨率是上一层（sizes[i-1]），FEM 挂在上采样后的特征上
            self.fems_up.append(
                FEM(channels[i - 1], sizes[i - 1], sizes[i - 1],
                    use_mfdf=use_mfdf, use_asca=use_asca, asca_variant=asca_variant))
        self.up_last = UpBlock(channels[0], channels[0])
        self.final = nn.Conv2d(channels[0], num_classes, 1, bias=True)

    def forward(self, x):
        feats = []
        for down, fem in zip(self.downs, self.fems_down):
            x = down(x)
            feats.append(fem(x))
        u = feats[-1]
        for idx, (up, fem) in enumerate(zip(self.ups, self.fems_up)):
            u = up(u)
            u = u + feats[len(feats) - 2 - idx]   # Fig.7 的 skip 直加（⊕）
            u = fem(u)
        u = self.up_last(u)
        return self.final(u)

    def load_from(self, weights):
        """与其他复现网络保持同一接口：FDFUNet 不使用任何预训练权重。"""
        return 0
