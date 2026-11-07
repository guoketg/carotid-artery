"""CCFR 复现（ESWA-D-25-34301）：边缘感知粗分割 + 不确定性引导 rectified flow 精修。

两阶段：
  Stage 1  EA-Net：U-Net 编码器/解码器，解码每级做空间（窗内 QKV）注意力，
           多尺度编码特征上采样融合成边缘图 E；区域 logit Sc 与边缘 logit Se 双输出，
           监督为 Dice + BCE + 0.4·Edge。
  Stage 2  FlowHead：在 logit 空间做 rectified flow，X0=logit(Pc)、X1=logit(Pgt)，
           速度场 Vθ(Xt,t) 由 U-Net 编解码给出，条件为 F_cond = Concat[Fg, FP, FE]，
           四级结构先验 Uℓ 经跨注意注入，t 经 FiLM 调制瓶颈；推理用 K 步 Euler，
           每步增量乘以不确定性门 M = f(Pc, E)。

论文的 EA-Net 用灰度复制 3 通道喂 ImageNet 预训练 ResNet 骨干（原文 "This setting
adapts the input requirement of pretrained ResNet backbones"）。本文件给两个 Stage 1：
  EANet      从零训练、1 通道输入的纯卷积 U-Net（统一口径版本）
  EANetR50   论文保真版：3 通道输入 + ImageNet21k 预训练 R50 编码器
CCFR(pretrain=True) 走 EANetR50，其余（解码级空间注意力、边缘支、priors、Stage 2）一致。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.vit_seg_modeling_resnet_skip import ResNetV2, np2th


def _window_partition(x, w):
    """(B,C,H,W) -> (B*nW, C, w*w)，要求 H、W 被 w 整除"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // w, w, W // w, w).permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(-1, C, w * w)


def _window_reverse(x, w, H, W):
    """(B*nW, C, w*w) -> (B,C,H,W)"""
    B = int(x.shape[0] // ((H // w) * (W // w)))
    C = x.shape[1]
    x = x.view(B, H // w, W // w, C, w, w).permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, H, W)


class SpatialAttention(nn.Module):
    """解码级内空间自注意力：窗内 QKV + 残差（γ 零初始化）。

    224² 直接用全图注意力矩阵会爆显存（12544²），故与非整除分辨率分别退化为
    窗内（window=7，正好整除 224/112/56/28/14）与 28² 池化全局注意力。
    """

    def __init__(self, c, ratio=4, window=7):
        super().__init__()
        d = max(8, c // ratio)
        self.q = nn.Conv2d(c, d, 1)
        self.k = nn.Conv2d(c, d, 1)
        self.v = nn.Conv2d(c, d, 1)
        self.proj = nn.Conv2d(d, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = d ** -0.5
        self.window = window
        self.pool = 28

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        if H % self.window == 0 and W % self.window == 0 and H >= self.window:
            w = self.window
            q = _window_partition(q, w)              # (B*nW, d, w²)
            k = _window_partition(k, w)
            v = _window_partition(v, w)
            attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
            out = torch.bmm(v, attn.transpose(1, 2))
            out = _window_reverse(out, w, H, W)
        else:
            q = F.adaptive_avg_pool2d(q, self.pool)
            k = F.adaptive_avg_pool2d(k, self.pool)
            v = F.adaptive_avg_pool2d(v, self.pool)
            d = q.shape[1]
            q = q.view(B, d, -1)
            k = k.view(B, d, -1)
            v = v.view(B, d, -1)
            attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
            out = torch.bmm(v, attn.transpose(1, 2)).view(B, d, self.pool, self.pool)
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return x + self.gamma * self.proj(out)


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(ConvBNReLU(cin, cout), ConvBNReLU(cout, cout))

    def forward(self, x):
        return self.block(x)


class EANet(nn.Module):
    """Stage 1：边缘感知 U-Net。

    forward 返回 dict：
      region  区域 logit（1 通道）
      edge    边缘 logit（1 通道）
      priors  [f1(112), f2(56), f3(28), bott(14)]，供 Stage 2 跨注意注入
    """

    def __init__(self, in_ch=1, base=32, attn=True):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.stem = DoubleConv(in_ch, c1)                                  # 224
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))    # 112
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))    # 56
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4))    # 28
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c4, c4 * 2))  # 14

        self.up3 = DoubleConv(c4 * 2 + c4, c4)
        self.att3 = SpatialAttention(c4) if attn else nn.Identity()
        self.up2 = DoubleConv(c4 + c3, c3)
        self.att2 = SpatialAttention(c3) if attn else nn.Identity()
        self.up1 = DoubleConv(c3 + c2, c2)
        self.att1 = SpatialAttention(c2) if attn else nn.Identity()
        self.up0 = DoubleConv(c2 + c1, c1)
        self.att0 = SpatialAttention(c1) if attn else nn.Identity()

        # 多尺度边缘支：各级编码特征 → 1 通道 → 上采样拼接
        self.e3 = nn.Conv2d(c4, 1, 1)
        self.e2 = nn.Conv2d(c3, 1, 1)
        self.e1 = nn.Conv2d(c2, 1, 1)
        self.e0 = nn.Conv2d(c1, 1, 1)
        self.edge_fuse = ConvBNReLU(4, base)
        self.edge_out = nn.Conv2d(base, 1, 1)
        self.head = nn.Sequential(ConvBNReLU(c1 + base, base), nn.Conv2d(base, 1, 1))

    def forward(self, x):
        hw = x.shape[-2:]
        f0 = self.stem(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        b = self.bottleneck(f3)

        def up(t, size):
            return F.interpolate(t, size=size, mode="bilinear", align_corners=False)

        d3 = self.att3(self.up3(torch.cat([up(b, f3.shape[-2:]), f3], 1)))
        d2 = self.att2(self.up2(torch.cat([up(d3, f2.shape[-2:]), f2], 1)))
        d1 = self.att1(self.up1(torch.cat([up(d2, f1.shape[-2:]), f1], 1)))
        d0 = self.att0(self.up0(torch.cat([up(d1, f0.shape[-2:]), f0], 1)))

        ef = self.edge_fuse(torch.cat([up(self.e3(f3), hw), up(self.e2(f2), hw),
                                       up(self.e1(f1), hw), up(self.e0(f0), hw)], 1))
        return {
            "region": self.head(torch.cat([d0, ef], 1)),
            "edge": self.edge_out(ef),
            "priors": [f1, f2, f3, b],
        }


class EANetR50(nn.Module):
    """Stage 1 的论文保真版：ImageNet21k 预训练 R50 编码器 + 轻量解码器。

    论文 EA-Net 把灰度复制成 RGB 再喂预训练 ResNet 骨干，此处照此实现：
    输入 1 通道时内部 repeat 成 3 通道；编码器用 ResNetV2（= TransUNet 的 R50，
    三段瓶颈，14² 停，与 A0 同一份权重文件），解码级仍保留空间注意力，
    边缘支与区域/边缘双输出、priors 顺序均与 EANet 一致。
    """

    def __init__(self, in_ch=3, block_units=(3, 4, 6), width_factor=1, attn=True):
        super().__init__()
        self.in_ch = in_ch
        self.enc = ResNetV2(block_units=block_units, width_factor=width_factor,
                            in_channels=in_ch)
        w = int(64 * width_factor)
        c112, c56, c28, c14 = w, w * 4, w * 8, w * 16          # 64 / 256 / 512 / 1024
        self.up3 = ConvBNReLU(c14 + c28, 256)                  # 14 → 28
        self.att3 = SpatialAttention(256) if attn else nn.Identity()
        self.up2 = ConvBNReLU(256 + c56, 128)                  # 28 → 56
        self.att2 = SpatialAttention(128) if attn else nn.Identity()
        self.up1 = ConvBNReLU(128 + c112, 64)                  # 56 → 112
        self.att1 = SpatialAttention(64) if attn else nn.Identity()
        self.up0 = ConvBNReLU(64, 64)                          # 112 → 224
        self.att0 = SpatialAttention(64) if attn else nn.Identity()

        self.e14 = nn.Conv2d(c14, 1, 1)
        self.e28 = nn.Conv2d(c28, 1, 1)
        self.e56 = nn.Conv2d(c56, 1, 1)
        self.e112 = nn.Conv2d(c112, 1, 1)
        self.edge_fuse = ConvBNReLU(4, 32)
        self.edge_out = nn.Conv2d(32, 1, 1)
        self.head = nn.Sequential(ConvBNReLU(64 + 32, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        if x.shape[1] != self.in_ch:
            x = x.repeat(1, self.in_ch, 1, 1)
        hw = x.shape[-2:]
        # ResNetV2 只返回 3 级（28²/512、56²/256、112²/64），14²/1024 就是 b 本身
        b, feats = self.enc(x)
        f28, f56, f112 = feats[0], feats[1], feats[2]
        f14 = b

        def up(t, size):
            return F.interpolate(t, size=size, mode="bilinear", align_corners=False)

        d3 = self.att3(self.up3(torch.cat([up(f14, f28.shape[-2:]), f28], 1)))
        d2 = self.att2(self.up2(torch.cat([up(d3, f56.shape[-2:]), f56], 1)))
        d1 = self.att1(self.up1(torch.cat([up(d2, f112.shape[-2:]), f112], 1)))
        d0 = self.att0(self.up0(up(d1, hw)))

        ef = self.edge_fuse(torch.cat([up(self.e14(f14), hw), up(self.e28(f28), hw),
                                       up(self.e56(f56), hw), up(self.e112(f112), hw)], 1))
        return {
            "region": self.head(torch.cat([d0, ef], 1)),
            "edge": self.edge_out(ef),
            "priors": [f112, f56, f28, f14],
        }

    def load_pretrained(self, weights):
        """只取 ImageNet21k npz 里的 R50 部分（与 VisionTransformer.load_from 同一套键）。"""
        with torch.no_grad():
            conv_w = np2th(weights["conv_root/kernel"], conv=True)
            tgt = self.enc.root.conv.weight
            if conv_w.shape[1] != tgt.shape[1]:
                pad = torch.zeros(tgt.shape[0], tgt.shape[1] - conv_w.shape[1],
                                  *conv_w.shape[2:])
                conv_w = torch.cat([conv_w, pad], dim=1)
            tgt.copy_(conv_w)
            self.enc.root.gn.weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
            self.enc.root.gn.bias.copy_(np2th(weights["gn_root/bias"]).view(-1))
            n = 0
            for bname, block in self.enc.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)
                    n += 1
        return n


class PriorCrossAttn(nn.Module):
    """结构先验跨注意：当前精修特征作 Q，Stage-1 的 Uℓ 作 K/V，残差注入。

    token 数超过 28² 时先做自适应池化再算注意力，输出上采样回原分辨率，
    避免 112²=12544 token 的 O(N²) 显存开销。
    """

    def __init__(self, c, pc, heads=4, dk=32):
        super().__init__()
        self.heads = heads
        self.dk = dk
        self.q = nn.Conv2d(c, heads * dk, 1)
        self.k = nn.Conv2d(pc, heads * dk, 1)
        self.v = nn.Conv2d(pc, heads * dk, 1)
        self.proj = nn.Conv2d(heads * dk, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pool = 28

    def forward(self, x, prior):
        B, _, H, W = x.shape
        q, k, v = self.q(x), self.k(prior), self.v(prior)
        if H * W > self.pool * self.pool:
            q = F.adaptive_avg_pool2d(q, self.pool)
            k = F.adaptive_avg_pool2d(k, self.pool)
            v = F.adaptive_avg_pool2d(v, self.pool)
        h, w = q.shape[-2:]
        q = q.view(B, self.heads, self.dk, -1)
        k = k.view(B, self.heads, self.dk, -1)
        v = v.view(B, self.heads, self.dk, -1)
        attn = torch.softmax(torch.einsum("bhdn,bhdm->bhnm", q, k) * self.dk ** -0.5, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(B, self.heads * self.dk, h, w)
        out = self.proj(out)
        if (h, w) != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return x + self.gamma * out


def _time_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class FlowHead(nn.Module):
    """Stage 2：时间相关速度场 Vθ(Xt,t)。

    cond=True 时输入 [Xt, F_cond] 并在四级做结构先验跨注意 + 瓶颈 FiLM；
    cond=False（w/o condition）时只输入 Xt，无先验跨注意、无 FiLM 条件注入。
    """

    def __init__(self, cond_ch, prior_chs=(64, 128, 256, 512), base=32, cond=True, t_dim=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.cond = cond
        self.stem = DoubleConv(1 + (cond_ch if cond else 0), c1)           # 224
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))    # 112
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))    # 56
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4))    # 28
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c4, c4 * 2))  # 14

        self.film = None
        self.xa_enc = nn.ModuleList()
        self.xa_dec = nn.ModuleList()
        if cond:
            self.film = nn.Sequential(nn.Linear(t_dim, c4 * 2), nn.SiLU(),
                                      nn.Linear(c4 * 2, c4 * 4))
            # 编码 112/56/28/14 与解码 28/56/112/224 各一次
            self.xa_enc = nn.ModuleList([
                PriorCrossAttn(c2, prior_chs[0]), PriorCrossAttn(c3, prior_chs[1]),
                PriorCrossAttn(c4, prior_chs[2]), PriorCrossAttn(c4 * 2, prior_chs[3]),
            ])
            self.xa_dec = nn.ModuleList([
                PriorCrossAttn(c4, prior_chs[2]), PriorCrossAttn(c3, prior_chs[1]),
                PriorCrossAttn(c2, prior_chs[0]), PriorCrossAttn(c1, prior_chs[0]),
            ])

        self.up3 = DoubleConv(c4 * 2 + c4, c4)
        self.up2 = DoubleConv(c4 + c3, c3)
        self.up1 = DoubleConv(c3 + c2, c2)
        self.up0 = DoubleConv(c2 + c1, c1)
        self.out = nn.Sequential(ConvBNReLU(c1, c1), nn.Conv2d(c1, 1, 1))
        # 输出零初始化：训练起点速度场恒为 0，即 X_K ≡ X0 = logit(Pc)，
        # Stage 2 从「与 Stage 1 完全一致」出发，只学能降低 P_K 损失的增量。
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, xt, t, cond=None, priors=None):
        x0 = torch.cat([xt, cond], 1) if self.cond else xt
        f0 = self.stem(x0)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        b = self.bottleneck(f3)
        if self.cond:
            f1 = self.xa_enc[0](f1, priors[0])
            f2 = self.xa_enc[1](f2, priors[1])
            f3 = self.xa_enc[2](f3, priors[2])
            b = self.xa_enc[3](b, priors[3])
            s, sh = self.film(_time_embedding(t, 32)).chunk(2, dim=1)
            b = b * (1 + s[:, :, None, None]) + sh[:, :, None, None]

        def up(x, size):
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        d3 = self.up3(torch.cat([up(b, f3.shape[-2:]), f3], 1))
        d2 = self.up2(torch.cat([up(d3, f2.shape[-2:]), f2], 1))
        d1 = self.up1(torch.cat([up(d2, f1.shape[-2:]), f1], 1))
        d0 = self.up0(torch.cat([up(d1, f0.shape[-2:]), f0], 1))
        if self.cond:
            d3 = self.xa_dec[0](d3, priors[2])
            d2 = self.xa_dec[1](d2, priors[1])
            d1 = self.xa_dec[2](d1, priors[0])
            d0 = self.xa_dec[3](d0, priors[0])
        return self.out(d0)


class UncertaintyGate(nn.Module):
    """不确定性门控特征 Fg：高不确定处保留局部细节，高置信处换全局统计。

    输入 Stage-1 第一级编码特征（112²），局部特征与全局均值按空间门 g 插值，
    再上采样到 224² 供 Stage-2 拼接。
    """

    def __init__(self, c_in, c_out=32, size=224):
        super().__init__()
        self.size = size
        self.local = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in), nn.SiLU(),
            nn.Conv2d(c_in, c_out, 1),
        )

    def forward(self, feat, gate):
        fl = self.local(feat)
        fg = fl.mean(dim=(2, 3), keepdim=True).expand_as(fl)
        g = F.interpolate(gate, size=fl.shape[-2:], mode="bilinear", align_corners=False)
        out = g * fl + (1 - g) * fg
        if out.shape[-1] != self.size:
            out = F.interpolate(out, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return out


def logit(p, eps=1e-3):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log(1 - p)


class CCFR(nn.Module):
    """两阶段拼装体。

    forward(x) 走完整推理：EA-Net → 不确定性门 → K 步 Euler → 2 通道 logits
    （cat[0, X_K] 与单通道 BCE 完全等价，便于复用项目 CE+Dice 评估/训练管线）。
    """

    def __init__(self, num_classes=2, base=32, flow_base=32, steps=5,
                 cond=True, unc=True, alpha=1.0, m_floor=0.1, m_edge_w=0.5,
                 img_size=224, pretrain=False, noattn=False):
        super().__init__()
        self.num_classes = num_classes
        self.steps = steps
        self.cond = cond
        self.unc = unc
        self.alpha = alpha
        self.m_floor = m_floor
        self.m_edge_w = m_edge_w
        self.pretrain = pretrain
        if pretrain:
            # 论文保真版：预训练 R50 编码器，先验通道由 R50 决定（64/256/512/1024）
            self.ea = EANetR50(in_ch=3, attn=not noattn)
            c_gate = 64
            prior_chs = (64, 256, 512, 1024)
        else:
            self.ea = EANet(in_ch=1, base=base, attn=not noattn)
            c_gate = base * 2
            prior_chs = (base * 2, base * 4, base * 8, base * 16)
        c_cond = 32 * 3
        if cond:
            self.gate = UncertaintyGate(c_gate, 32)
            self.proj_p = nn.Conv2d(1, 32, 1)
            self.proj_e = nn.Conv2d(1, 32, 1)
            if not unc:
                # w/o uncertainty guidance：保留结构条件，去掉空间门控
                self.gate_fallback = nn.Conv2d(c_gate, 32, 1)
        else:
            c_cond = 0
        self.flow = FlowHead(cond_ch=c_cond, prior_chs=prior_chs, base=flow_base, cond=cond)
        self.img_size = img_size

    # ---------- Stage 1 ----------
    def stage1(self, image):
        out = self.ea(image)
        pc = torch.sigmoid(out["region"])
        e = torch.sigmoid(out["edge"])
        return {"region": out["region"], "edge": out["edge"], "pc": pc, "e": e,
                "priors": out["priors"]}

    # ---------- 条件特征 ----------
    def build_mask(self, pc, e):
        """不确定性门 M（论文 Eq.5）：像素不确定性 u = Pc(1−Pc) 与归一化边缘强度 E
        自适应加权后取下限。论文未给加权系数，这里用 m_edge_w 线性组合。"""
        if not self.unc:
            return torch.ones_like(pc)
        u = pc * (1.0 - pc)
        m = (1.0 - self.m_edge_w) * u + self.m_edge_w * e
        return m.clamp(min=self.m_floor)

    def build_cond(self, pc_logit, e, priors, mask):
        if not self.cond:
            return None
        if self.unc:
            fg = self.gate(priors[0], mask)
        else:
            fg = self.gate_fallback(priors[0])
            fg = F.interpolate(fg, size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False)
        fp = self.proj_p(pc_logit)
        fe = self.proj_e(e)
        return torch.cat([fg, fp, fe], 1)

    def velocity(self, xt, t, cond, priors):
        return self.flow(xt, t, cond, priors)

    # ---------- 完整推理 ----------
    def forward(self, image, steps=None):
        steps = steps or self.steps
        s1 = self.stage1(image)
        mask = self.build_mask(s1["pc"], s1["e"])
        cond = self.build_cond(s1["region"], s1["e"], s1["priors"], mask)
        x = logit(s1["pc"])
        B = x.shape[0]
        for k in range(steps):
            t = torch.full((B,), k / steps, device=x.device, dtype=x.dtype)
            x = x + (1.0 / steps) * mask * self.velocity(x, t, cond, s1["priors"])
        prob = torch.sigmoid(x)
        return {"out": torch.cat([torch.zeros_like(x), x], dim=1),
                "coarse": s1["pc"], "prob": prob, "mask": mask}
