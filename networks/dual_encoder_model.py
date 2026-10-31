"""双骨干（R50-ViT + 独立 R50）按 stage 交错 + 四级双向跨支融合。

对应 docs/technical-design.md D11：

- 主路只吃 C1（复制成 3 通道），侧路只吃 1 级 Haar 的 LH/HL/HH 方向子带；
- 四个接点：★F1 112²/64、★F2 56²/256、★F3 28²/512、★F4 14²/768；
- 两个骨干**交错执行**：每级先各自出特征、双向融合，融合结果再回到两条骨干往下走；
- 融合单元 `A' = A + γ_A·GN(Attn(Q=A, K=V=B))`，**只有 γ 标量零初始化**（输出投影正常
  初始化，否则两个零相乘会互相锁死梯度）→ 第 0 步与 baseline 逐元素等价（误差 < 1e-5）；
- 最深接点默认 `chain=True`：先更新侧路 B，主路再拿更新后的 B 重取 K/V（串联双向）。
  并行双向时 B 没有下游消费者，`gamma_b` 恒为 0、该方向整支悬空。

侧路直接复用 `ResNetV2`（其 PreActBottleneck 内部本来就是 GroupNorm，不需要 BN→GN 改写），
并可复用主路同一份 ImageNet21k R50 权重初始化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math

from networks.vit_seg_modeling import VisionTransformer
from networks.vit_seg_modeling_resnet_skip import ResNetV2, np2th
from networks.wavelet import haar_dwt, build_hf_image


# ---------------------------------------------------------------- 输入侧


def _standardize(x, eps=1e-6):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std

# 获得高频图，先拼接，然后插值到224
def build_detail_channels(x, size=224, log_domain=True):
    """(B,1,H,W) 归一化原图 → (B,3,size,size) 的 LH / HL / HH 方向子带。

    侧路只吃方向子带：不含低频，与主路 C1 的信息不重叠。
    子带在 112² 上算完再双线性上采样回 size，避免侧路比主路低两个层级。
    """
    if log_domain:
        x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
        x = torch.log1p(x_pos)
    _, details = haar_dwt(x)
    outs = []
    for d in details:
        d = F.interpolate(d, size=(size, size), mode="bilinear", align_corners=False)
        outs.append(_standardize(d))
    return torch.cat(outs, dim=1)

# 获得低频图，插值到224
def build_ll_image(x, size=224, log_domain=True):
    """(B,1,H,W) 归一化原图 → (B,1,size,size) 的 LL 低频近似图。

    与 build_detail_channels 同域（log1p 后做 1 级 Haar），取近似子带 LL（112²）
    双线性上采样回 size，逐样本标准化。Haar 下「只保留 LL 的重建」就是逐 2 倍
    上采样，所以这里直接上采样等价于 IDWT（高频置零）的低频图。
    """
    # 使用对数预处理
    if log_domain:
        x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
        x = torch.log1p(x_pos)
    approx, _ = haar_dwt(x)
    ll = F.interpolate(approx, size=(size, size), mode="bilinear", align_corners=False)
    return _standardize(ll)

# ---------------------------------------------------------------- 窗口工具


def window_partition(x, ws):
    """(B,C,H,W) → (B*gh*gw, ws*ws, C)"""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // ws, ws, w // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.reshape(-1, ws * ws, c)


def window_reverse(x, ws, gh, gw, b, c):
    """(B*gh*gw, ws*ws, C) → (B,C,H,W)"""
    x = x.reshape(b, gh, gw, ws, ws, c).permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.reshape(b, c, gh * ws, gw * ws)


# ---------------------------------------------------------------- 融合算子


class BidirCrossAttention(nn.Module):
    """双向跨支注意力。

    mode='window'：按 ws×ws 窗做注意力（每个窗内 token 数 = ws²）；
    mode='global'：全图 token 注意力。
    两个方向共享同一份 qkv 投影（A 算一份、B 算一份，然后互喂），
    但输出投影各一套；`bidir=False` 时只改写 A（R7 单向对照）。
    `chain=True` 时改为串联：先更新 B，再用更新后的 B 重取 K/V 更新 A。
    """

    def __init__(self, dim, heads=4, dk_ratio=2, window=8, mode="window", bidir=True,chain=False):
        super().__init__()
        d = max(heads, dim // dk_ratio)
        d = max(heads, (d // heads) * heads)
        self.dim, self.d, self.heads = dim, d, heads
        self.window, self.mode, self.bidir = window, mode, bidir
        self.chain = chain and bidir

        self.qkv_a = nn.Conv2d(dim, 3 * d, 1)
        self.qkv_b = nn.Conv2d(dim, 3 * d, 1)
        self.proj_a = nn.Conv2d(d, dim, 1)
        self.proj_b = nn.Conv2d(d, dim, 1)
        self.norm_a = nn.GroupNorm(1, dim)   # GN 在残差支路内
        self.norm_b = nn.GroupNorm(1, dim)
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))

    def _split_heads(self, t):
        # (N, L, d) → (N, heads, L, d/heads)。N 是 batch×窗数，L 是 token 数
        # （窗内 = ws²，全局 = H*W），d 是 qkv 投影后的通道数
        n, l, d = t.shape
        return t.reshape(n, l, self.heads, d // self.heads).transpose(1, 2).contiguous()

    def _merge_heads(self, t):
        # (N, heads, L, d/heads) → (N, L, d)，多头结果拼回原通道顺序
        n, h, l, dh = t.shape
        return t.transpose(1, 2).reshape(n, l, h * dh)

    def _attend(self, q, k, v):
        # 缩放点积注意力，缩放因子 1/sqrt(d/heads) 由 PyTorch 内部处理；
        # 不开 causal mask、不做 dropout —— 这里是全连接的空间注意力
        out = F.scaled_dot_product_attention(
            self._split_heads(q), self._split_heads(k), self._split_heads(v))
        return self._merge_heads(out)

    def forward(self, A, B):
        qa, ka, va = self.qkv_a(A).chunk(3, dim=1)
        qb, kb, vb = self.qkv_b(B).chunk(3, dim=1)
        b_, _, h, w = A.shape

        if self.mode == "window":
            # ---- 窗内注意力：112²/56² 用。全图 token 太多，切成 ws×ws 窗各自算
            ws = self.window
            gh, gw = h // ws, w // ws          # 窗的行/列数，h、w 需能被 ws 整除
            to_tok = lambda *ts: tuple(window_partition(t, ws) for t in ts)
            back = lambda t: window_reverse(t, ws, gh, gw, b_, self.d)
        else:
            # ---- 全局注意力：28²/14² 用，token 数（784 / 196）可承受全图自注意力
            to_tok = lambda *ts: tuple(t.flatten(2).transpose(1, 2) for t in ts)
            back = lambda t: t.transpose(1, 2).reshape(b_, self.d, h, w)

        # (B,3d,H,W) → (B*gh*gw, ws², 3d) 或 (B,H*W,3d)
        qa, ka, va = to_tok(qa, ka, va)
        qb, kb, vb = to_tok(qb, kb, vb)

        # 这里不能出现两个零乘因子，否则 Proj 和 gamma 的输出全都是 0，梯度被截断
        if self.chain:
            # 串联双向：先用融合前的 B 更新侧路，主路再拿更新后的 B 重取 K/V。
            # 并行写法下最深接点的 B 没有任何下游消费者，gamma_b 恒为 0、整支悬空。
            B2 = B + self.gamma_b * self.norm_b(self.proj_b(back(self._attend(qb, ka, va))))
            kb, vb = to_tok(*self.qkv_b(B2).chunk(3, dim=1)[1:])
            out_a = back(self._attend(qa, kb, vb))
            A2 = A + self.gamma_a * self.norm_a(self.proj_a(out_a))
            return A2, B2

        out_a = back(self._attend(qa, kb, vb))
        A2 = A + self.gamma_a * self.norm_a(self.proj_a(out_a))
        if not self.bidir:
            return A2, B
        B2 = B + self.gamma_b * self.norm_b(self.proj_b(back(self._attend(qb, ka, va))))
        return A2, B2


class BidirGate(nn.Module):
    """L1 档：双向逐像素门控（只调幅度、不重组内容），R6 对照。"""

    def __init__(self, dim, kernel=7, bidir=True):
        super().__init__()
        self.conv_a = nn.Conv2d(2, 1, kernel, padding=kernel // 2)
        self.conv_b = nn.Conv2d(2, 1, kernel, padding=kernel // 2)
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))
        self.bidir = bidir
        nn.init.zeros_(self.conv_a.weight)
        nn.init.zeros_(self.conv_a.bias)
        nn.init.zeros_(self.conv_b.weight)
        nn.init.zeros_(self.conv_b.bias)

    @staticmethod
    def _spatial(x):
        return torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)

    def forward(self, A, B):
        w_a = torch.sigmoid(self.conv_a(self._spatial(A)))
        w_b = torch.sigmoid(self.conv_b(self._spatial(B)))
        A2 = A + self.gamma_a * (A * w_b)   # 用对方权重门控自己
        if not self.bidir:
            return A2, B
        return A2, B + self.gamma_b * (B * w_a)


class BidirConcat(nn.Module):
    """L0 档：concat + 1×1 回投影，用来区分「跨支交互」与「多一路特征」。"""

    def __init__(self, dim, bidir=True):
        super().__init__()
        self.conv_a = nn.Conv2d(2 * dim, dim, 1, bias=False)
        self.conv_b = nn.Conv2d(2 * dim, dim, 1, bias=False)
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))
        self.bidir = bidir

    def forward(self, A, B):
        A2 = A + self.gamma_a * self.conv_a(torch.cat([A, B], dim=1))
        if not self.bidir:
            return A2, B
        return A2, B + self.gamma_b * self.conv_b(torch.cat([B, A], dim=1))


class BidirDeform(nn.Module):
    """L3 档：可变形大核跨支聚合（D-LKA 思路）。

    一支特征先用 1×1 降到 d 维，再由 3×3 预测 k×k 采样偏移，在**另一支**上做
    可变形卷积。采样点跟着解剖边界走，而不是固定网格 —— 这直接针对旧版三通道＋
    小波引导注意力「固定网格 + 只在 14² 一点融合」两个缺陷。

    两个方向共用降维投影（A→低维既作自身的偏移预测源，又作对方的内容源），
    因此参数量约为双向各建一套的一半。偏移卷积零初始化 → 初始等价于固定网格
    （等价于零偏移的常规卷积，输出非零）；输出投影正常初始化，只靠 γ=0 保证
    第 0 步与 baseline 逐元素相同。
    """

    def __init__(self, dim, dk_ratio=2, k=5, bidir=True):
        super().__init__()
        d = max(32, min(128, dim // dk_ratio))
        self.d, self.k, self.bidir = d, k, bidir
        self.red_a = nn.Conv2d(dim, d, 1)
        self.red_b = nn.Conv2d(dim, d, 1)
        self.off_a = nn.Conv2d(d, 2 * k * k, 3, padding=1)
        self.off_b = nn.Conv2d(d, 2 * k * k, 3, padding=1)
        self.def_a = nn.Conv2d(d, d, k, padding=k // 2)
        self.def_b = nn.Conv2d(d, d, k, padding=k // 2)
        self.proj_a = nn.Conv2d(d, dim, 1)
        self.proj_b = nn.Conv2d(d, dim, 1)
        self.norm_a = nn.GroupNorm(1, dim)
        self.norm_b = nn.GroupNorm(1, dim)
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))
        for m in (self.off_a, self.off_b):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _dir(self, orig, src, dst, off, dfm, proj, norm, gamma):
        o = off(src)
        y = deform_conv2d(dst, o, dfm.weight, dfm.bias, padding=self.k // 2)
        return orig + gamma * norm(proj(y))

    def forward(self, A, B):
        a, b = self.red_a(A), self.red_b(B)
        A2 = self._dir(A, a, b, self.off_a, self.def_a, self.proj_a, self.norm_a, self.gamma_a)
        if not self.bidir:
            return A2, B
        B2 = self._dir(B, b, a, self.off_b, self.def_b, self.proj_b, self.norm_b, self.gamma_b)
        return A2, B2


class _MSCAM(nn.Module):
    """AFF 的多尺度通道注意力（arXiv 2009.14082 的 MS-CAM）。

    原版用 BN，这里局部支路换 GroupNorm（batch 4 下 BN 统计不稳）。
    全局支路 GAP 后空间为 1×1，GroupNorm 校验 batch×空间>1 会在
    eval（batch=1）时崩溃 → 全局支路不带归一化（SE 式 MLP，带 bias）。
    局部支路保留空间分布，全局支路用 GAP 抓通道显著性，相加后 sigmoid。
    """

    def __init__(self, dim, r=8):
        super().__init__()
        c = max(8, dim // r)
        g = 8 if c % 8 == 0 else 1
        self.local = nn.Sequential(
            nn.Conv2d(dim, c, 1, bias=False), nn.GroupNorm(g, c), nn.ReLU(inplace=True),
            nn.Conv2d(c, dim, 1, bias=False), nn.GroupNorm(1, dim))
        self.glb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, c, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, dim, 1))
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.local(x) + self.glb(x))


class BidirAFF(nn.Module):
    """AFF / iAFF：先把两支相加做「初始整合」，再用 MS-CAM 重加权。

    针对 AFF 论文的核心论点 —— 融合瓶颈在初始整合而非后续精化。与 L0 concat
    的差别是：不扩通道、不做逐点线性混合，两支各自保留原值，只按 MS-CAM 的
    逐通道空间权重互相压低/抬高。iters=2 即 iAFF。
    """

    def __init__(self, dim, r=8, iters=1, bidir=True):
        super().__init__()
        self.mscam = _MSCAM(dim, r)
        self.iters, self.bidir = iters, bidir
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))

    def forward(self, A, B):
        m = self.mscam(A + B)
        A2 = A + self.gamma_a * (A * m)
        if not self.bidir:
            return A2, B
        B2 = B + self.gamma_b * (B * (1.0 - m))
        if self.iters > 1:
            m2 = self.mscam(A2 + B2)
            A2 = A2 + self.gamma_a * (A2 * m2)
            B2 = B2 + self.gamma_b * (B2 * (1.0 - m2))
        return A2, B2


def _dct_vec(k, n, device, dtype):
    """1D 正交 DCT-II 基的第 k 个频率分量（长度 n）"""
    i = torch.arange(n, device=device, dtype=dtype)
    ref = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
    return ref * torch.cos(math.pi * k * (2.0 * i + 1.0) / (2.0 * n))


# FcaNet（Qin et al., ICCV 2021）16 组多光谱通道注意力的固定频点表。
# 直接取 FcaNet 官方 get_freq_indices('top16')：all_top_indices_x[:16] ×
# all_top_indices_y[:16]，频点坐标定义在 7×7 DCT tile 上（FcaNet 的 dct_h=dct_w=7）。
_FCANET_TOP16 = (
    (0, 0), (0, 1), (6, 0), (0, 5), (0, 2), (1, 0), (1, 2), (4, 0),
    (5, 0), (1, 6), (3, 0), (0, 4), (0, 6), (0, 3), (3, 5), (2, 3),
)
_FCANET_TILE = 7


class BidirFrC(nn.Module):
    """Syn-Net（JBHI 2025）FrCFusion 的移植，替换接点处的融合算子（严格按原文）。

    论文 Eq.5-11 三步：
    1) 通道分 G 组，第 g 组用固定 2D-DCT 频点 (u_g, v_g) 做「特征×基」空间求和，
       得到每通道频率描述子，按通道拼成 C 维；
    2) 描述子 → FC → sigmoid 得通道权重，分别重加权两条支路；
    3) 另一路空间支：两支逐元素相乘 → 1×1 压通道 → 1×1 出单通道 → sigmoid 得
       空间注意力图；最后 `Freqout = proj(cat[FreS, FreR, A_spatial])`。

    严格化要点（相对第一版移植的三处改动）：
    - 频点表：改用 FcaNet 官方 top-16 频点（7×7 DCT tile，见 `_FCANET_TOP16`），
      特征先自适应池化到 7×7 再取描述子 —— FcaNet 的原设定，不再用 4×4 zig-zag；
    - 回写：双向 `A2 = A + γ_A·Freqout`、`B2 = B + γ_B·Freqout`（原文两支互相更新）；
      `bidir=False` 只回写 A（单点融合下 B 若没有下游消费者时用）；
    - 两个 γ 均零初始化 → 第 0 步与未融合的两支逐元素等价。
    """

    def __init__(self, dim, bidir=True, n_groups=16, r=8):
        super().__init__()
        assert dim % n_groups == 0, f"FrC: dim={dim} 不能被 n_groups={n_groups} 整除"
        self.dim, self.n_groups, self.bidir = dim, n_groups, bidir
        c_s = max(8, dim // r)
        self.fc_a = nn.Sequential(nn.Linear(dim, c_s), nn.ReLU(inplace=True), nn.Linear(c_s, dim))
        self.fc_b = nn.Sequential(nn.Linear(dim, c_s), nn.ReLU(inplace=True), nn.Linear(c_s, dim))
        self.sp = nn.Sequential(
            nn.Conv2d(dim, c_s, 1), nn.ReLU(inplace=True), nn.Conv2d(c_s, 1, 1))
        self.proj = nn.Conv2d(2 * dim + 1, dim, 1)
        self.gamma_a = nn.Parameter(torch.zeros(1))
        self.gamma_b = nn.Parameter(torch.zeros(1))
        self._basis_cache = {}

    def _basis(self, device, dtype):
        key = (device, dtype)
        if key not in self._basis_cache:
            rows = []
            for u, v in _FCANET_TOP16[:self.n_groups]:
                rows.append(_dct_vec(u, _FCANET_TILE, device, dtype)[:, None]
                            * _dct_vec(v, _FCANET_TILE, device, dtype)[None, :])
            self._basis_cache[key] = torch.stack(rows, dim=0)   # (G,7,7)
        return self._basis_cache[key]

    def _desc(self, x):
        """(B,C,H,W) → (B,C) 的逐通道频率描述子（池化到 FcaNet 的 7×7 tile）"""
        b, c = x.shape[:2]
        if x.shape[-2:] != (_FCANET_TILE, _FCANET_TILE):
            x = F.adaptive_avg_pool2d(x, _FCANET_TILE)
        basis = self._basis(x.device, x.dtype)
        f = x.reshape(b, self.n_groups, c // self.n_groups, _FCANET_TILE, _FCANET_TILE)
        d = torch.einsum("bgchw,ghw->bgc", f, basis)
        return d.reshape(b, c)

    def forward(self, A, B):
        wa = torch.sigmoid(self.fc_a(self._desc(A)))[:, :, None, None]
        wb = torch.sigmoid(self.fc_b(self._desc(B)))[:, :, None, None]
        m = torch.sigmoid(self.sp(A * B))
        out = self.proj(torch.cat([A * wa, B * wb, m], dim=1))
        A2 = A + self.gamma_a * out
        if not self.bidir:
            return A2, B
        return A2, B + self.gamma_b * out


def make_fusion(kind, dim, heads, dk_ratio, window, bidir, chain=False):
    if kind == "window":
        return BidirCrossAttention(dim, heads, dk_ratio, window, mode="window",
                                   bidir=bidir, chain=chain)
    if kind == "global":
        return BidirCrossAttention(dim, heads, dk_ratio, 1, mode="global",
                                   bidir=bidir, chain=chain)
    if kind == "gate":
        return BidirGate(dim, bidir=bidir)
    if kind == "concat":
        return BidirConcat(dim, bidir=bidir)
    if kind == "deform":
        return BidirDeform(dim, dk_ratio=dk_ratio, bidir=bidir)
    if kind == "aff":
        return BidirAFF(dim, bidir=bidir)
    if kind == "frc":
        return BidirFrC(dim, bidir=bidir)
    raise ValueError(f"unknown fusion kind: {kind}")


# ---------------------------------------------------------------- 主模型


class DualEncoderTransUNet(VisionTransformer):
    """主路 R50-ViT + 侧路 R50，stage 交错 + 四级双向融合。

    levels：`"1,2,3,4"` 的子集，对应 112²/64、56²/256、28²/512、14²/768。
    fuse_op：auto（小分辨率窗内 + 大分辨率全局）/ window / global / gate / concat。
    """

    def __init__(self, config, img_size=224, num_classes=2, levels="1,2,3,4",
                 fuse_op="auto", window=8, dk_ratio=2, bidir=True,
                 side_pretrained=True, cross_scale=False, main_input="raw",
                 edge_in=False):
        super().__init__(config, img_size, num_classes, edge_in=edge_in)
        self.img_size = img_size
        self.levels = [int(v) for v in str(levels).split(",") if str(v).strip()]
        self.fuse_op = fuse_op
        self.bidir = bidir
        self.side_pretrained = side_pretrained
        self.cross_scale = cross_scale
        self.main_input=main_input

        width = int(64 * config.resnet.width_factor)
        hidden = config.hidden_size

        # 侧路：与主路 hybrid R50 同构（内部已是 GroupNorm）
        self.side_res = ResNetV2(block_units=config.resnet.num_layers,
                                 width_factor=config.resnet.width_factor)
        self.side_proj = nn.Conv2d(width * 16, hidden, 1)

        # 根据层查参数
        res_of = {1: img_size // 2, 2: img_size // 4, 3: img_size // 8, 4: img_size // 16}
        dim_of = {1: width, 2: width * 4, 3: width * 8, 4: hidden}
        win_of = {1: window, 2: 7, 3: 7, 4: 7}
        heads_of = {1: 4, 2: 8, 3: 8, 4: 8}

        self.fuse = nn.ModuleDict()
        for lv in self.levels:
            res = res_of[lv]
            if fuse_op == "auto":
                # 112²/56² 用窗内（全局在 12544 token 上会爆），28²/14² 用全局
                kind = "window" if res >= 56 else "global"
            elif fuse_op == "window":
                # level-4 只有 196 token，全局严格更好且几乎免费
                kind = "global" if lv == 4 else "window"
            elif fuse_op == "global":
                # 112² 全图注意力在 batch 8 下会爆，强制退回窗内
                kind = "window" if res >= 112 else "global"
            else:
                kind = fuse_op
            # 最深的那个接点用串联双向：并行写法下该层的 B 没有下游消费者
            chain = bool(bidir) and lv == max(self.levels) and kind in ("window", "global")
            self.fuse[str(lv)] = make_fusion(
                kind, dim_of[lv], heads_of[lv], dk_ratio, win_of[lv], bidir, chain=chain)

        self._pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # 跨尺度 top-down 回流：14² → 28² → 56² → 112²，补 BiFPN 的跨级通路
        if cross_scale:
            self.cs43 = nn.Conv2d(hidden, width * 8, 1)
            self.cs32 = nn.Conv2d(width * 8, width * 4, 1)
            self.cs21 = nn.Conv2d(width * 4, width, 1)
            self.cs_gamma = nn.Parameter(torch.zeros(3))

    # ------------------------------------------------------------ 工具

    def _has(self, lv):
        return str(lv) in self.fuse

    def _fuse(self, lv, A, B):
        if not self._has(lv):
            return A, B
        return self.fuse[str(lv)](A, B)

    @staticmethod
    def _pad_skip(x, size):
        """复刻 ResNetV2.forward 里 55→56 的右下零填充（只用于 skip 与融合）。"""
        if x.size(2) == size and x.size(3) == size:
            return x
        feat = torch.zeros((x.size(0), x.size(1), size, size), device=x.device, dtype=x.dtype)
        feat[:, :, 0:x.size(2), 0:x.size(3)] = x
        return feat

    # ------------------------------------------------------------ 编码

    def _encode_interleaved(self, main_in, side_in):
        mres, sres = self.transformer.embeddings.hybrid_model, self.side_res
        s2 = self.img_size // 2
        s4 = self.img_size // 4

        # ---- ★F1：112² / 64
        a = mres.root(main_in)
        b = sres.root(side_in)
        a, b = self._fuse(1, a, b)
        f1 = a

        # ---- 下采样 + block1 → 55²（原实现 maxpool(3,2,0) 得 55，skip 补零到 56）
        a = self._pool(a)
        b = self._pool(b)
        
        # 这里的body是ResnetV2的三个瓶颈
        a = mres.body[0](a)
        b = sres.body[0](b)
        a_f = self._pad_skip(a, s4)
        b_f = self._pad_skip(b, s4)

        # ---- ★F2：56² / 256，融合后裁回 55² 继续走主干（与 baseline 几何一致）
        a_f, b_f = self._fuse(2, a_f, b_f)
        f2 = a_f
        a = a_f[:, :, :a.size(2), :a.size(3)]
        b = b_f[:, :, :b.size(2), :b.size(3)]

        a = mres.body[1](a)
        b = sres.body[1](b)

        # ---- ★F3：28² / 512
        a, b = self._fuse(3, a, b)
        f3 = a

        a = mres.body[2](a)          # 14² / 1024
        b = sres.body[2](b)          # 14² / 1024
        # 1024-》768对齐
        b = self.side_proj(b)        # 14² / 768，与 ViT token 维度对齐

        # ---- patch embedding + 12 层 ViT（主路独有）
        emb = self.transformer.embeddings
        t = emb.patch_embeddings(a)
        t = t.flatten(2).transpose(-1, -2)
        t = emb.dropout(t + emb.position_embeddings)
        for blk in self.transformer.encoder.layer:
            t, _ = blk(t)
        t = self.transformer.encoder.encoder_norm(t)

        # ---- ★F4：14² / 768
        bsz, n_tok, ch = t.shape
        h = int(n_tok ** 0.5)
        ta = t.transpose(1, 2).reshape(bsz, ch, h, h)

        ta, b = self._fuse(4, ta, b)
        t = ta.reshape(bsz, ch, n_tok).transpose(1, 2).contiguous()

        # ---- 跨尺度 top-down 回流（可选）：深层语义逐级上送到浅层 skip
        if self.cross_scale:
            f3 = f3 + self.cs_gamma[0] * F.interpolate(
                self.cs43(ta), size=f3.shape[-2:], mode="bilinear", align_corners=False)
            f2 = f2 + self.cs_gamma[1] * F.interpolate(
                self.cs32(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
            f1 = f1 + self.cs_gamma[2] * F.interpolate(
                self.cs21(f2), size=f1.shape[-2:], mode="bilinear", align_corners=False)

        # DecoderCup 的 skip 顺序固定为 [28², 56², 112²]
        return t, [f3, f2, f1]

    def forward(self, x):

        # 输入的修改
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        if x.size(1) == 1:
            raw = x

            if self.main_input == "ll":
                # XNet 式：主路只吃低频近似图，高频信息全在侧路
                main_in = build_ll_image(raw, self.img_size).repeat(1, 3, 1, 1)
            else:
                main_in = x.repeat(1, 3, 1, 1)
            if self.edge_in:
                # 边缘特征融合·输入级：主路 stem 再吃一张小波高频重建图
                main_in = torch.cat([main_in, build_hf_image(raw)], dim=1)
        else:
            raw = x[:, 0:1]
            main_in = x[:, 0:3]
        side_in = build_detail_channels(raw, self.img_size)

        encoded, features = self._encode_interleaved(main_in, side_in)
        out = self.decoder(encoded, features)
        return self.segmentation_head(out)

    # ------------------------------------------------------------ 权重

    def load_from(self, weights):
        """主路照常加载；侧路复用同一份 ImageNet21k R50 权重。"""
        super().load_from(weights)
        if not self.side_pretrained:
            return
        with torch.no_grad():
            self.side_res.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            self.side_res.root.gn.weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
            self.side_res.root.gn.bias.copy_(np2th(weights["gn_root/bias"]).view(-1))
            for bname, block in self.side_res.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)

    def extra_repr(self):
        return f"levels={self.levels}, fuse_op={self.fuse_op}, bidir={self.bidir}"
