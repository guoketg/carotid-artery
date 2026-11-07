"""Haar 小波预处理：对数域软阈值去噪 + 细节能量图，构造三通道输入。

不引入第三方小波库，用固定 2x2 卷积核实现，可 GPU、无参数、可逆。
"""
import torch
import torch.nn.functional as F

# 分析滤波器（4, 1, 2, 2）：LL / LH / HL / HH
_HAAR = torch.tensor(
    [
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [-0.5, -0.5]],
        [[0.5, -0.5], [0.5, -0.5]],
        [[0.5, -0.5], [-0.5, 0.5]],
    ],
    dtype=torch.float32,
)


def _filters(device):
    return _HAAR.to(device=device).unsqueeze(1)


def haar_dwt(x):
    """x: (B,1,H,W) -> (LL, [LH, HL, HH])，H/W 为偶数"""
    w = _filters(x.device)
    bands = F.conv2d(x, w, stride=2)
    return bands[:, 0:1], [bands[:, 1:2], bands[:, 2:3], bands[:, 3:4]]


def haar_dwt_channels(x):
    """x: (B,C,H,W) -> (LL, [LH, HL, HH])，逐通道 Haar 分解（depthwise，H/W 为偶数）"""
    c = x.shape[1]
    # 逐通道复制 4 个滤波器：group i 的 4 个核 → (4C, 1, 2, 2)
    w = _HAAR.to(device=x.device, dtype=x.dtype).unsqueeze(1).repeat(c, 1, 1, 1, 1).view(4 * c, 1, 2, 2)
    bands = F.conv2d(x, w, stride=2, groups=c)
    b, _, h, wd = bands.shape
    bands = bands.view(b, c, 4, h, wd)
    return bands[:, :, 0], [bands[:, :, 1], bands[:, :, 2], bands[:, :, 3]]


def haar_idwt(ll, details):
    """Haar 逆变换：四个子带各自转置卷积（depthwise）后相加"""
    w = _filters(ll.device).permute(0, 1, 2, 3)  # (4, 1, 2, 2)
    bands = torch.cat([ll] + list(details), dim=1)
    return F.conv_transpose2d(bands, w, stride=2, groups=4).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------- 多小波基支持
# 系数取自 PyWavelets 的标准滤波器表（归一化到 sum = sqrt(2)），硬编码进本文件，
# 运行时不依赖 pywt。分析/合成都用 torch 的可分离卷积实现，可 GPU、无参数。
# 约定：分析用 dec_lo / dec_hi（torch 互相关），合成用 rec_lo / rec_hi 的反序
# （见 _basis_pair），该组合在 224×224 上 PR 误差 < 1e-6（Temp/probe_wavelet_basis.py）。
# Haar 仍走上面的 haar_dwt / haar_idwt，保证历史实验逐位一致。
BASIS_FILTERS = {
    "db2": {
        "dec_lo": [-0.1294095225512604, 0.2241438680420134,
                   0.8365163037378079, 0.4829629131445342],
        "dec_hi": [-0.4829629131445342, 0.8365163037378079,
                   -0.2241438680420134, -0.1294095225512604],
        "rec_lo": [0.4829629131445342, 0.8365163037378079,
                   0.2241438680420134, -0.1294095225512604],
        "rec_hi": [-0.1294095225512604, -0.2241438680420134,
                   0.8365163037378079, -0.4829629131445342],
    },
    "db4": {
        "dec_lo": [-0.010597401785069, 0.0328830116668852,
                   0.0308413818355608, -0.1870348117190931,
                   -0.0279837694168599, 0.6308807679298589,
                   0.7148465705529157, 0.2303778133088965],
        "dec_hi": [-0.2303778133088965, 0.7148465705529157,
                   -0.6308807679298589, -0.0279837694168599,
                   0.1870348117190931, 0.0308413818355608,
                   -0.0328830116668852, -0.010597401785069],
        "rec_lo": [0.2303778133088965, 0.7148465705529157,
                   0.6308807679298589, -0.0279837694168599,
                   -0.1870348117190931, 0.0308413818355608,
                   0.0328830116668852, -0.010597401785069],
        "rec_hi": [-0.010597401785069, -0.0328830116668852,
                   0.0308413818355608, 0.1870348117190931,
                   -0.0279837694168599, -0.6308807679298589,
                   0.7148465705529157, -0.2303778133088965],
    },
    "sym4": {
        "dec_lo": [-0.0757657147892733, -0.0296355276459985,
                   0.4976186676320155, 0.8037387518059161,
                   0.2978577956052774, -0.0992195435768472,
                   -0.0126039672620378, 0.0322231006040427],
        "dec_hi": [-0.0322231006040427, -0.0126039672620378,
                   0.0992195435768472, 0.2978577956052774,
                   -0.8037387518059161, 0.4976186676320155,
                   0.0296355276459985, -0.0757657147892733],
        "rec_lo": [0.0322231006040427, -0.0126039672620378,
                   -0.0992195435768472, 0.2978577956052774,
                   0.8037387518059161, 0.4976186676320155,
                   -0.0296355276459985, -0.0757657147892733],
        "rec_hi": [-0.0757657147892733, 0.0296355276459985,
                   0.4976186676320155, -0.8037387518059161,
                   0.2978577956052774, 0.0992195435768472,
                   -0.0126039672620378, -0.0322231006040427],
    },
    "coif1": {
        "dec_lo": [-0.015655728135792, -0.0727326195125265,
                   0.3848648468648578, 0.8525720202116004,
                   0.3378976624574818, -0.0727326195125265],
        "dec_hi": [0.0727326195125265, 0.3378976624574818,
                   -0.8525720202116004, 0.3848648468648578,
                   0.0727326195125265, -0.015655728135792],
        "rec_lo": [-0.0727326195125265, 0.3378976624574818,
                   0.8525720202116004, 0.3848648468648578,
                   -0.0727326195125265, -0.015655728135792],
        "rec_hi": [-0.015655728135792, 0.0727326195125265,
                   0.3848648468648578, -0.8525720202116004,
                   0.3378976624574818, 0.0727326195125265],
    },
    "bior1.5": {
        "dec_lo": [0.0165728151840597, -0.0165728151840597,
                   -0.1215339780164379, 0.1215339780164379,
                   0.7071067811865476, 0.7071067811865476,
                   0.1215339780164379, -0.1215339780164379,
                   -0.0165728151840597, 0.0165728151840597],
        "dec_hi": [-0.0, 0.0, -0.0, 0.0,
                   -0.7071067811865476, 0.7071067811865476,
                   -0.0, 0.0, -0.0, 0.0],
        "rec_lo": [0.0, 0.0, 0.0, 0.0,
                   0.7071067811865476, 0.7071067811865476,
                   0.0, 0.0, 0.0, 0.0],
        "rec_hi": [0.0165728151840597, 0.0165728151840597,
                   -0.1215339780164379, -0.1215339780164379,
                   0.7071067811865476, -0.7071067811865476,
                   0.1215339780164379, 0.1215339780164379,
                   -0.0165728151840597, -0.0165728151840597],
    },
}

WAVELET_BASES = ("haar",) + tuple(BASIS_FILTERS.keys())


def _basis_pair(basis, device, dtype):
    """返回 (dec_lo, dec_hi, rec_lo, rec_hi) 四个 1D 张量。

    rec_* 取 PyWavelets 合成滤波器的反序：torch 的 conv 是互相关、conv_transpose
    的索引方向与其相反，反序后才能满足 PR（probe 里逐个基验证过）。
    """
    if basis not in BASIS_FILTERS:
        raise ValueError(f"unknown wavelet basis: {basis}")
    f = BASIS_FILTERS[basis]
    dec_lo = torch.tensor(f["dec_lo"], device=device, dtype=dtype)
    dec_hi = torch.tensor(f["dec_hi"], device=device, dtype=dtype)
    rec_lo = torch.tensor(f["rec_lo"], device=device, dtype=dtype).flip(0)
    rec_hi = torch.tensor(f["rec_hi"], device=device, dtype=dtype).flip(0)
    return dec_lo, dec_hi, rec_lo, rec_hi


def dwt2(x, basis="haar"):
    """1 级二维离散小波分解，返回 (LL, [LH, HL, HH])。

    basis="haar" 直接走 haar_dwt（历史口径逐位一致）；其余基用可分离卷积：
    先横向 lo/hi，再纵向 lo/hi，得到 LL/LH/HL/HH 四个子带（各 H/2 × W/2）。
    边界 replicate 延拓，重建时裁掉延拓部分。
    """
    if basis == "haar":
        return haar_dwt(x)
    dec_lo, dec_hi, _, _ = _basis_pair(basis, x.device, x.dtype)
    n = dec_lo.numel()
    # 延拓长度取整个滤波器长度 L（而非 L/2）：滤波器越长，边界需要的余量越大，
    # 否则 x 的最外圈几行几列无法精确重建（probe 里 db4/sym4 只差边界）。
    p = n
    xp = F.pad(x, (p, p, p, p), mode="replicate")

    # 横向：通道 0 = 低通行，通道 1 = 高通行
    kh = torch.stack([dec_lo, dec_hi]).view(2, 1, 1, n)
    y = F.conv2d(xp, kh, stride=(1, 2))

    # 纵向：groups=2 逐通道（低通行 / 高通行）各出 lo、hi 两路
    kv = torch.stack([torch.stack([dec_lo, dec_hi]),
                      torch.stack([dec_lo, dec_hi])]).view(4, 1, n, 1)
    z = F.conv2d(y, kv, stride=(2, 1), groups=2)   # (B,4,H/2,W/2)
    return z[:, 0:1], [z[:, 1:2], z[:, 2:3], z[:, 3:4]]


def idwt2(ll, details, basis="haar"):
    """dwt2 的逆变换，输出与原图同尺寸。"""
    if basis == "haar":
        return haar_idwt(ll, details)
    _, _, rec_lo, rec_hi = _basis_pair(basis, ll.device, ll.dtype)
    n = rec_lo.numel()
    p = n
    lh, hl, hh = details

    kv_lo = rec_lo.view(1, 1, n, 1)
    kv_hi = rec_hi.view(1, 1, n, 1)
    kh_lo = rec_lo.view(1, 1, 1, n)
    kh_hi = rec_hi.view(1, 1, 1, n)

    # 纵向合成：低通行 = LL⊗lo + LH⊗hi，高通行 = HL⊗lo + HH⊗hi
    y0 = F.conv_transpose2d(ll, kv_lo, stride=(2, 1)) + \
        F.conv_transpose2d(lh, kv_hi, stride=(2, 1))
    y1 = F.conv_transpose2d(hl, kv_lo, stride=(2, 1)) + \
        F.conv_transpose2d(hh, kv_hi, stride=(2, 1))

    # 横向合成后裁掉两侧延拓
    x = F.conv_transpose2d(y0, kh_lo, stride=(1, 2)) + \
        F.conv_transpose2d(y1, kh_hi, stride=(1, 2))
    return x[:, :, p:x.shape[2] - p, p:x.shape[3] - p]


def soft_threshold(x, t):
    """软阈值，t 为 (B,1,1,1) 或标量"""
    return torch.sign(x) * torch.relu(x.abs() - t)


def visushrink_threshold(details):
    """每个样本用 MAD 估计噪声 sigma，T = sigma * sqrt(2 log N)"""
    flat = torch.cat([d.abs().reshape(d.shape[0], -1) for d in details], dim=1)
    sigma = torch.median(flat, dim=1).values / 0.6745
    n = flat.shape[1]
    t = sigma * (2.0 * torch.log(torch.tensor(float(n), device=flat.device))).sqrt()
    return t.reshape(-1, 1, 1, 1)


def _standardize(x, eps=1e-6):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def build_hf_image(x, log_domain=True, lf_weight=0.0, basis="haar"):
    """输入 (B,1,H,W) 归一化原图，输出 (B,1,H,W) 的小波高频重建图。

    对数域做 1 级分解（默认 Haar，可由 basis 换基），把近似子带 LL 置零后 IDWT
    重建 —— 得到的是与原图同分辨率、只含高频成分的「边缘图」。这是老师定的
    边缘图算子 (technical-design.md 频率注入位置一节)：Sobel/Laplacian 只作
    后续对照。basis 只换滤波器组，分解级数、对数域、置零策略、标准化全部不变。

    lf_weight = λ（GobletNet Eq.5：H = HL+LH+HH+λ·LL）：>0 时高频图掺回少量
    低频，缓解纯高频图信息量过稀（论文 Table VI：λ∈[0,0.3] 最优）。
    默认 0.0，与 edge_in 的纯高频行为逐位一致。

    为什么不直接用细节能量图（sqrt(LH²+HL²+HH²)）：能量图丢了方向与符号，
    只剩「哪里有高频」，没有「高频往哪个方向走」，边缘朝向信息丢光。
    """
    if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
        x = F.interpolate(x, size=(x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2),
                          mode="bilinear", align_corners=False)

    if log_domain:
        x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
        xl = torch.log1p(x_pos)
    else:
        xl = x

    ll, details = dwt2(xl, basis=basis)
    hf = idwt2(lf_weight * ll, details, basis=basis)   # λ=0 → LL 置零只剩高频；λ>0 → GobletNet 式回灌
    if log_domain:
        hf = torch.expm1(hf)
    return _standardize(hf)


# ---------------------------------------------------------------- 边缘算子注册表


def _sobel_mag(x):
    """x: (B,1,H,W) -> Sobel 梯度幅值（与 scripts/probe_edge_signal.py 同名实现一致）"""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def _laplacian_mag(x):
    """x: (B,1,H,W) -> 4 邻域 Laplacian 幅值"""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(x, k, padding=1).abs()


def laplacian_split_channels(x):
    """模块内可替换 Haar 的拉普拉斯分解：(B,C,H,W) -> (LL, [D,D,D])。

    接口与 `haar_dwt_channels` 逐位对齐（4 个子带、半分辨率、同通道数）：
      - 高频基**直接复用** `_laplacian_mag` + `_standardize`，即 §1.6 边缘算子
        测试的同一口径（4 邻域 Laplacian → abs → 逐样本 z-score），只做两处
        适配：depthwise 到 C 通道；abs+标准化后 2×2 均值池化到半分辨率。
      - LL 仍返回 Haar 低通，保证单变量替换只改高频基，不动低频通路。
      - 单核三份填空是刻意为之：输出 3 个细节槽位共享同一幅幅值图，等价于
        「只要一个方向无关的拉普拉斯响应」，不做方向伪造。
    零参数。
    """
    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        x = F.interpolate(x, size=(x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2),
                          mode='bilinear', align_corners=False)
    c = x.shape[1]
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    m = F.conv2d(x, k.repeat(c, 1, 1, 1), padding=1, groups=c).abs()
    m = _standardize(m)
    d = F.avg_pool2d(m, 2)
    ll, _ = haar_dwt_channels(x)
    return ll, [d, d, d]


def laplacian_pyramid_channels(x, levels=3):
    """拉普拉斯金字塔分解：(B,C,H,W) -> (LL, [D1, D2, D3])，全部半分辨率。

    与 `haar_dwt_channels` / `laplacian_split_channels` 逐位同形（4 个子带、
    半分辨率、同通道数），但这是**真正的多尺度拉普拉斯分解**，且保留符号：

        G1 = LL(x)          # 2×2 box 低通 + 抽取 → H/2
        G2 = LL(G1)         # H/4
        G3 = LL(G2)         # H/8
        L1 = x  − up(G1)    # 全分辨率最细细节
        L2 = G1 − up(G2)    # H/2
        L3 = G2 − up(G3)    # H/4
        D1/D2/D3 = 各细节带双线性对齐到 H/2

    与 `laplacian_split_channels` 的区别（§1.12 的「更正统」臂）：
      - 不做 `abs`，保留符号（负响应 = 亮暗方向相反，原版金字塔就能区分）；
      - 三个细节槽是**三个真实尺度**，不是同一幅图复制三份；
      - 低通逐级抽取，等价于 DoG 带通，无需方向伪造。
    零参数（低通用 Haar LL 的 2×2 均值，上采样用 nearest，均无可学权重）。
    """
    if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
        x = F.interpolate(x, size=(x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2),
                          mode='bilinear', align_corners=False)
    gauss = [x]
    for _ in range(levels):
        prev = gauss[-1]
        if min(prev.shape[-2], prev.shape[-1]) < 4:
            gauss.append(prev)
        else:
            ll, _ = haar_dwt_channels(prev)
            gauss.append(ll)
    details = []
    for i in range(1, levels + 1):
        low = F.interpolate(gauss[i], size=gauss[i - 1].shape[-2:], mode='nearest')
        details.append(gauss[i - 1] - low)
    size_ll = gauss[1].shape[-2:]
    details = [F.interpolate(d, size=size_ll, mode='bilinear', align_corners=False)
               if d.shape[-2:] != size_ll else d for d in details]
    return gauss[1], details


def _detail_energy(x):
    """x: (B,1,H,W) -> 细节能量图 sqrt(LH²+HL²+HH²)，半分辨率插值回原尺寸。

    旧方案：只保留「哪里有高频」，丢方向与符号。
    """
    x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
    _, details = haar_dwt(torch.log1p(x_pos))
    e = torch.sqrt(sum(d * d for d in details) + 1e-8)
    return F.interpolate(e, size=x.shape[-2:], mode="bilinear", align_corners=False)


EDGE_OPS = ("wavelet", "wavelet_lam", "energy", "sobel", "laplacian")


def build_edge_map(x, op="wavelet", lf_weight=0.3, basis="haar"):
    """五个候选边缘算子统一入口，输出都是逐样本标准化的 (B,1,H,W)。

    - wavelet     : 1 级 Haar，LL 置零后 IDWT 重建（项目现用口径，λ=0）
    - wavelet_lam : 同上但掺回 λ·LL（GobletNet Eq.5，默认 λ=0.3）
    - energy      : 小波细节能量图 sqrt(LH²+HL²+HH²)（旧方案）
    - sobel       : Sobel 梯度幅值
    - laplacian   : 4 邻域 Laplacian 幅值

    basis 只作用于 wavelet / wavelet_lam（基选择实验），其余算子忽略。
    """
    if op == "wavelet":
        return build_hf_image(x, log_domain=True, lf_weight=0.0, basis=basis)
    if op == "wavelet_lam":
        return build_hf_image(x, log_domain=True, lf_weight=lf_weight, basis=basis)
    if op == "energy":
        return _standardize(_detail_energy(x))
    if op == "sobel":
        return _standardize(_sobel_mag(x))
    if op == "laplacian":
        return _standardize(_laplacian_mag(x))
    raise ValueError(f"unknown edge op: {op}")


def build_three_channel(x, threshold=True):
    """输入 (B,1,H,W) 已归一化图像，输出 (B,3,H,W)：原图 / 小波去噪重建 / 细节能量。

    细节能量只取第 1 级，作为边界与钙化强反射的显著性图。
    """
    if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
        x = F.interpolate(x, size=(x.shape[-2] // 2 * 2, x.shape[-1] // 2 * 2),
                          mode="bilinear", align_corners=False)

    # 对数域：把超声乘性斑点转为加性，再做阈值才有理论依据
    x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
    xl = torch.log1p(x_pos)

    ll, details = haar_dwt(xl)
    energy = torch.sqrt(sum(d * d for d in details) + 1e-8)

    if threshold:
        t = visushrink_threshold(details)
        details_t = [soft_threshold(d, t) for d in details]
    else:
        details_t = details

    rec = haar_idwt(ll, details_t)
    c2 = torch.expm1(rec)

    # C2 拉回到与 C1 同量级；C3 本身是能量图，单独标准化
    c1 = x
    c2 = _standardize(c2) * c1.std(dim=(1, 2, 3), keepdim=True).mean() \
         + c1.mean(dim=(1, 2, 3), keepdim=True).mean()
    c3 = F.interpolate(_standardize(energy), size=c1.shape[-2:],
                       mode="bilinear", align_corners=False)

    return torch.cat([c1, c2, c3], dim=1)
