"""WeaveSeg (ICCV 2025) 完整结构复现：ASR + ICW + 五项损失所需的三个上下文头。

论文：Li et al., "WeaveSeg: Iterative Contrast-weaving and Spectral Feature-refining
for Nuclei Instance Segmentation", ICCV 2025（正文见 docs/papers_txt/）。

本文件只做**结构**，损失在 trainer.py 的 WeaveSegFiveTermLoss 里。

与论文的对应关系
----------------
- 编码器：与 networks/unet.py 完全同构（l1 224²/64 → l5 14²/1024），
  目的是让 Baseline 行仍然是项目里那条 U-Net（86.55），四行消融只差模块。
- 解码器：每一级都是 cat(上采样结果, skip) → DoubleBasicBlock（与 unet.py 逐键一致）；
  启用 ASR 的那一级只把 up_conv 换成 `1×1 通道对齐 + ASR(F_high, skip)`：
      ASR 级：  cat( ASR(1×1投影(F_high), F_low), F_low ) → DoubleBasicBlock
      非 ASR 级：cat( up_conv(F_high),        F_low ) → DoubleBasicBlock
  这样四行消融的唯一变量是模块本身，不会因为"ASR 顶掉 DBB"而少掉约 17M 参数。
  注：论文 Eq.(4) 的 F_out 也可以读成"ASR 已经完成融合、后面不再 cat"，
  但那是另一种接线，会让 ASR 行比 Baseline 少 17M 参数，无法做单变量归因，
  所以这里明确选"只换上采样算子"这一读法（与 unet.py 的 --asr 口径一致）。
- ICW：接在解码器末端 224²/64 的融合特征上（1×1 出 logits 之前），
  T 次迭代（Table 2：T=3 最优），Eq.(5)–(10)。
  上下文图 Sf/Sb/Su 由三个 1×1 头从同一特征解耦（论文第二条通路的
  "decouples the encoder features into foreground, background, and uncertain
  region components"），分别被 L_fg / L_bg / L_uncert 监督。

正文没给、只能选定的量（全部集中记在这里，避免散落）
----------------------------------------------------
1. ICW 局部窗口 K：论文只写 "local neighborhood of size K×K"，未给数值；
   这里取 K=7（细胞核尺度经验值），可由参数改。
2. DenseBlock 的层数/增长率：论文只画了 DenseBlock(·)，未给通道；
   这里 4 层、growth=16，末端 1×1 压回 C。
3. ASR↔ICW 的"closed-loop iteration"：正文没有给外层次数，也没有给
   ICW 输出回灌 ASR 的接入点，只有 ICW 内部的 T（Table 2）。因此这里
   实现为**单次 ASR→ICW 前向**，不做外层回灌；这是正文可确定的部分。
4. 三个上下文头的位置：论文只说是"第二条通路解耦编码器特征"，未给层级；
   这里挂在 ICW 输入特征上（与 ICW 同分辨率），保证 ICW 拿到的就是被监督的图。
5. ASR 的启用层级：Eq.4 对每一级解码都成立，默认四级全开；
   224² 一级显存最贵，可用 asr_levels 参数关掉（会作为偏差记录）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.asr import ASR
from networks.unet import (BN_MOMENTUM, BasicBlock, DoubleBasicBlock, conv1x1,
                           conv3x3, down_conv, init_kaiming, make_downsample, up_conv)


class DenseBlock(nn.Module):
    """ICW 里的 DenseBlock(·)：逐层 concat 的稠密连接，末端 1×1 压回 C。"""

    def __init__(self, channels, layers=4, growth=16):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        c_in = channels
        for _ in range(layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(c_in, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, growth, 3, padding=1, bias=False),
            ))
            c_in += growth
        self.proj = nn.Sequential(
            nn.Conv2d(c_in, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return self.proj(torch.cat(feats, dim=1))


def _neighbors(x, k):
    """把 (B,C,H,W) 展开成 K×K 邻域堆叠 (B,C·K²,H,W)，边缘 replicate。

    逐平移循环而不是 unfold，避免 (B,C,K²,H·W) 的展开张量（224² 上会是 GB 级）。
    """
    B, C, H, W = x.shape
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode='replicate')
    outs = []
    for i in range(k):
        for j in range(k):
            outs.append(xp[:, :, i:i + H, j:j + W])
    return torch.cat(outs, dim=1)


class ICW(nn.Module):
    """Iterative Contrast Weaving（Eq.5–10）。

        F_dense = DenseBlock(F^(t-1))                       Eq.(5)
        α(x,y)  = softmax_{K×K}( wf·Sf↓ − wb·Sb↓ )          Eq.(6)-(8)
        V       = Wv · F_dense
        F_CA(x,y) = Σ_{K×K} α · V                           Eq.(6)
        F_gated = F_CA ⊙ σ(κ·F_CA + λ)                      Eq.(9) AGLU
        F^(t)   = F^(t-1) + F_gated                         Eq.(10)
    """

    def __init__(self, channels, iters=3, k=7, dense_layers=4, growth=16):
        super(ICW, self).__init__()
        self.iters = iters
        self.k = k
        self.dense = DenseBlock(channels, layers=dense_layers, growth=growth)
        self.wv = nn.Conv2d(channels, channels, 1, bias=False)
        # wf / wb：上下文图上的可学习权重（单通道图 → 标量）
        self.wf = nn.Parameter(torch.tensor(1.0))
        self.wb = nn.Parameter(torch.tensor(1.0))
        # AGLU 的两个可学习参数（Figure 8 显示它们训练中自适应变化）
        self.kappa = nn.Parameter(torch.tensor(1.0))
        self.lam = nn.Parameter(torch.tensor(0.0))

    def forward(self, feat, sf, sb):
        k = self.k
        # α 只依赖上下文图，与迭代轮次无关（Eq.6 里的 Sf↓/Sb↓ 是固定的），算一次即可
        score = self.wf * sf - self.wb * sb                  # (B,1,H,W)
        alpha = F.softmax(_neighbors(score, k), dim=1)       # (B,K²,H,W)
        B, C, H, W = feat.shape
        p = k // 2
        for _ in range(self.iters):
            v = self.wv(self.dense(feat))                    # (B,C,H,W)
            vp = F.pad(v, (p, p, p, p), mode='replicate')
            # Eq.(6) 的 K×K 加权聚合：逐平移累加，避免拼 (B,K²,C,H,W)
            # （224² 上那个张量单次就 1.2 GB，是显存爆掉的主因）
            f_ca = torch.zeros_like(v)
            for i in range(k):
                for j in range(k):
                    idx = i * k + j
                    f_ca = f_ca + alpha[:, idx:idx + 1] * vp[:, :, i:i + H, j:j + W]
            f_gated = f_ca * torch.sigmoid(self.kappa * f_ca + self.lam)  # Eq.(9)
            feat = feat + f_gated                            # Eq.(10)
        return feat


class ASRMerge(nn.Module):
    """一级上采样算子：1×1 通道对齐 + ASR(F_high, F_low)，输出 2H×2W / C 通道。"""

    def __init__(self, c_in, c_out):
        super(ASRMerge, self).__init__()
        self.proj = nn.Sequential(
            conv1x1(c_in, c_out),
            nn.BatchNorm2d(c_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.asr = ASR(c_out)

    def forward(self, feat, skip):
        return self.asr(self.proj(feat), skip)


class WeaveSeg(nn.Module):
    """完整 WeaveSeg（U-Net 底座 + ASR + ICW + 上下文头）。

    参数：
        asr        : 是否启用 ASR（False = 论文 Table 1 的 "w/o ASR" / Baseline 那一侧）
        icw        : 是否启用 ICW（False = "w/o ICW"）
        asr_levels : 启用 ASR 的解码层级 (0,1,2,3) = 28²/512、56²/256、112²/128、224²/64
        iters / k  : ICW 迭代次数 T（Table 2 取 3）与局部窗口 K
    """

    def __init__(self, in_channels=1, num_classes=2, asr=True, icw=True,
                 asr_levels=(0, 1, 2, 3), iters=3, k=7):
        super(WeaveSeg, self).__init__()
        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024
        # ---- 编码器：与 unet.py 逐键同构 ----
        # 与 unet.py 的 stem 逐键一致：两个裸 3×3 卷积 + BasicBlock（无 BN/ReLU 夹层）
        self.l1_1 = nn.Sequential(
            conv3x3(in_channels, l1c), conv3x3(l1c, l1c), BasicBlock(l1c, l1c))
        self.l1_2 = down_conv(l1c, l2c)
        self.l2_1 = DoubleBasicBlock(l2c, l2c)
        self.l2_2 = down_conv(l2c, l3c)
        self.l3_1 = DoubleBasicBlock(l3c, l3c)
        self.l3_2 = down_conv(l3c, l4c)
        self.l4_1 = DoubleBasicBlock(l4c, l4c)
        self.l4_2 = down_conv(l4c, l5c)
        self.l5_1 = BasicBlock(l5c, l5c)
        # ---- 解码器 ----
        dec_pairs = ((l5c, l4c), (l4c, l3c), (l3c, l2c), (l2c, l1c))
        self.asr_levels = tuple(asr_levels) if asr else ()
        self.icw = ICW(l1c, iters=iters, k=k) if icw else None
        # 每一级都保留 cat + DoubleBasicBlock（与 unet.py 基线逐键一致），
        # ASR 只替换该级的 up_conv 上采样算子 → 四行消融的唯一变量是模块本身，
        # 不会因为"ASR 顶掉 DBB"而偷偷少掉 ~17M 参数。
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for level, (c_in, c_out) in enumerate(dec_pairs):
            self.up.append(ASRMerge(c_in, c_out) if level in self.asr_levels
                           else up_conv(c_in, c_out))
            self.fuse.append(DoubleBasicBlock(
                c_out * 2, c_out, make_downsample(c_out * 2, c_out)))
        # ---- 上下文头（第二条通路的解耦）+ 输出 ----
        self.head_sf = nn.Conv2d(l1c, 1, 1)
        self.head_sb = nn.Conv2d(l1c, 1, 1)
        self.head_su = nn.Conv2d(l1c, 1, 1)
        self.out_conv = nn.Conv2d(l1c, num_classes, 1)
        init_kaiming(self)

    def _dec(self, level, feat, skip):
        up = (self.up[level](feat, skip) if level in self.asr_levels
              else self.up[level](feat))
        return self.fuse[level](torch.cat([up, skip], dim=1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x1 = self.l1_1(x)                    # 224²/64
        x2 = self.l2_1(self.l1_2(x1))        # 112²/128
        x3 = self.l3_1(self.l2_2(x2))        # 56²/256
        x4 = self.l4_1(self.l3_2(x3))        # 28²/512
        x5 = self.l5_1(self.l4_2(x4))        # 14²/1024
        d4 = self._dec(0, x5, x4)
        d3 = self._dec(1, d4, x3)
        d2 = self._dec(2, d3, x2)
        d1 = self._dec(3, d2, x1)
        # 上下文图（Sf/Sb/Su）在 ICW 之前解耦，并被 Eq.(12) 的 L_fg/L_bg/L_uncert 监督
        sf = self.head_sf(d1)
        sb = self.head_sb(d1)
        su = self.head_su(d1)
        if self.icw is not None:
            d1 = self.icw(d1, torch.sigmoid(sf), torch.sigmoid(sb))
        return {'out': self.out_conv(d1), 'sf': sf, 'sb': sb, 'su': su}

    def load_from(self, weights):
        """train.py 统一入口：卷积网络无 ImageNet 权重可载，直接跳过"""
        return
