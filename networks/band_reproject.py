# coding=utf-8
"""编码器分带重投影（Band Reprojection, BRP）—— 2026-11-18 设计，用户明确要求先跑 ResNet。

要解决的问题
------------
R50 + DecoderCup（--arch baseline --vit_layers 0，A0 = 90.4903 / 0.14473 / HD 0.7062）上，
所有「注入式编码器改写」全负：FGR 四级门控回注 −1.64、FDFUNet 残差频域支 +0.03、
多尺度融合 −0.37，FEM / R-GEF / MREF / ABF / FreqFusion 也全部判负
（见 docs/results.md §5(2)(5)-(15)、§12）。而同一批机制挂在**解码器**上是正的：
DDBE 四级 Haar +0.435 / HD 0.6292、单尺度拉普拉斯 +0.506 / HD 0.6581，
且「去分解只留拼接」的 plain 对照仅 +0.060 → 增益来自频率内容而不是拼接结构本身。

所以本模块只做一件事：把解码器上唯一有 R50 正证据的那套形态
（分带 → 拼接 → 1×1 重投影 → 零初始化残差）原样搬到编码器，并用位置消融
（浅层两级 / 深层两级 / 四级 / 跨级门 / 随机频带对照）检验它是被「位置」否掉的，
还是被「门控回注这种形态」否掉的。

结构（每级）
------------
    hf  = |Laplacian| 细节（半分辨率，z-score 后上采样回原尺寸）
    lf  = Haar LL 低频（同上）
    ctx = 5×5 depthwise → GroupNorm → 1×1        # 覆盖本数据集散斑相关尺度
    y   = x + proj([hf, lf, ctx])                # proj 权重与 bias 全零初始化 ⇒ 恒等起步

零参数量之外的纪律：
  * 恒等起步只能走「残差 + 零初始化 1×1」，不能用 eye 恒等卷积 —— TF32 下 cuDNN 会把
    激活舍入，逐位恒等会破（docs/pitfalls.md 有记录，FGR/DDBE 同款）。
  * 改写保形（通道数与分辨率都不动），因此可以直接继续下采样、也可直接当 skip。
  * 三个支路**共用一枚 1×1 重投影**（不做逐带单核）：逐带单核是 R50 上「两个改写机器
    抢同一份频率内容」的互覆形态之一（§5(17) 同点串联比两个单挂各低 0.29/0.16）。

参数（每级）：4C² + 28C
------------------------
    proj 1×1 (3C→C)  3C² + C（bias 也零初始化）
    ctx  5×5 depthwise 25C，1×1 C²（无 bias），GroupNorm 2C
逐级（R50 实测形状）：64ch/112² → 18,176；256ch/55² → 269,312；
                      512ch/28² → 1,062,912；1024ch/14² → 4,222,976。
浅层两级合计 +287,552（+2 个门标量），深层两级 +5,286,144，四级 +5,574,272。

跨级门（--brp_gate，默认关）：浅级门 → 深级门，单向、只传门不传内容
------------------------------------------------------------------
浅级把自己的「频带强度图」sigmoid(α·mean_c(hf)) 双线性缩放到深级分辨率，作为深级
hf/lf 的**空间门**。α 零初始化 ⇒ 起步整幅 0.5 的常数门（不破坏恒等，因为 proj 是零）。
只传门不传内容是刻意的：EFF 那种「跨尺度低频内容搬运」在 R50 上低 1.14（§5(13)），
而门只调「在本级哪些位置相信频带」，不引入跨尺度的低频共线。
最深层若没有下游消费者，其 α 是惰性标量（1 个参数，无梯度），不影响任何读数。

随机频带对照（--brp_control）：把 hf/lf 换成同形状固定随机缓冲（无参数）
------------------------------------------------------------------
只换内容不换容量，用来回答「涨的是频率内容还是多出来的 4C² 参数容量」。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import laplacian_split_channels


def band_reproject_params(channels):
    """单级参数：proj 3C²+C，ctx 25C + C²，GroupNorm 2C ⇒ 4C²+28C。"""
    return 4 * channels * channels + 28 * channels


class BandReproject(nn.Module):
    """单级分带重投影，保形残差。"""

    def __init__(self, channels, height, width, control=False, gate=False):
        super().__init__()
        c = channels
        self.channels = c
        self.height = height
        self.width = width
        self.control = bool(control)
        self.gate = bool(gate)
        self.band_map = None

        self.ctx_dw = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        # R50 的 hybrid model 全程用 GroupNorm(32) + eps 1e-6，这里对齐同一口径
        self.ctx_norm = nn.GroupNorm(32, c, eps=1e-6) if c % 32 == 0 else nn.GroupNorm(8, c, eps=1e-6)
        self.ctx_pw = nn.Conv2d(c, c, 1, bias=False)

        self.proj = nn.Conv2d(3 * c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        if self.gate:
            # 零初始化 ⇒ 起步是整幅 0.5 常数的门；梯度经下一级的 proj 回流后解锁
            self.alpha = nn.Parameter(torch.zeros(1))

        if self.control:
            # 固定随机缓冲：同形状、零参数，只把「频带内容」换成无关噪声
            g = torch.Generator().manual_seed(20261118 + c)
            self.register_buffer('ctl_hf', torch.randn(1, c, height, width, generator=g),
                                 persistent=False)
            self.register_buffer('ctl_lf', torch.randn(1, c, height, width, generator=g),
                                 persistent=False)

    def forward(self, x, spatial_gate=None, return_map=False):
        ll, details = laplacian_split_channels(x)
        # 三个细节槽位共享同一幅幅值（见 wavelet.laplacian_split_channels），取均值而非求和
        h = (details[0] + details[1] + details[2]) / 3.0
        hf = F.interpolate(h, size=x.shape[-2:], mode='bilinear', align_corners=False)
        lf = F.interpolate(ll, size=x.shape[-2:], mode='bilinear', align_corners=False)
        ctx = self.ctx_pw(self.ctx_norm(self.ctx_dw(x)))

        if self.control:
            hf = self.ctl_hf.to(dtype=hf.dtype).expand_as(hf)
            lf = self.ctl_lf.to(dtype=lf.dtype).expand_as(lf)
        if spatial_gate is not None:
            hf = hf * spatial_gate
            lf = lf * spatial_gate

        y = x + self.proj(torch.cat([hf, lf, ctx], dim=1))
        if self.gate:
            self.band_map = torch.sigmoid(self.alpha * hf.mean(dim=1, keepdim=True))
        return (y, self.band_map) if return_map else y


class EncoderBandStack(nn.Module):
    """编码器四级上的挂载容器：只建被选中的级，并维护「浅 → 深」单向门。

    级号 1..4（1-based，与命令行 --brp_levels 一致）：
        1 → 112²/64（stem），2 → 55²/256（block1，padding hack 之前），
        3 → 28²/512（block2），4 → 14²/1024（block3 输出）
    """

    def __init__(self, specs, levels, use_gate=False, control=False):
        super().__init__()
        self.levels = tuple(int(i) for i in levels)
        self.use_gate = bool(use_gate)
        self.slot = {}
        mods = []
        for stage, (c, h, w) in enumerate(specs, start=1):
            if stage in self.levels:
                self.slot[stage] = len(mods)
                mods.append(BandReproject(c, h, w, control=control, gate=use_gate))
        self.mods = nn.ModuleList(mods)
        self._prev_map = None
        self._prev_stage = None

    def forward(self, stage, x):
        pos = self.slot.get(stage)
        if pos is None:
            return x
        # 阶段号不递增 = 新一次前向，清掉上一条链的门（容器被反复复用）
        if self._prev_stage is None or stage <= self._prev_stage:
            self._prev_map = None
        self._prev_stage = stage

        gate_map = None
        if self.use_gate and self._prev_map is not None:
            gate_map = F.interpolate(self._prev_map, size=x.shape[-2:],
                                     mode='bilinear', align_corners=False)
        y = self.mods[pos](x, spatial_gate=gate_map)
        if self.use_gate:
            self._prev_map = self.mods[pos].band_map
        return y
