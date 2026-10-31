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


def haar_idwt(ll, details):
    """Haar 逆变换：四个子带各自转置卷积（depthwise）后相加"""
    w = _filters(ll.device).permute(0, 1, 2, 3)  # (4, 1, 2, 2)
    bands = torch.cat([ll] + list(details), dim=1)
    return F.conv_transpose2d(bands, w, stride=2, groups=4).sum(dim=1, keepdim=True)


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


def build_hf_image(x, log_domain=True, lf_weight=0.0):
    """输入 (B,1,H,W) 归一化原图，输出 (B,1,H,W) 的小波高频重建图。

    对数域做 1 级 Haar 分解，把近似子带 LL 置零后 IDWT 重建 —— 得到的是与原图
    同分辨率、只含高频成分的「边缘图」。这是老师定的边缘图算子
    (technical-design.md 频率注入位置一节)：Sobel/Laplacian 只作后续对照。

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

    ll, details = haar_dwt(xl)
    hf = haar_idwt(lf_weight * ll, details)   # λ=0 → LL 置零只剩高频；λ>0 → GobletNet 式回灌
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


def _detail_energy(x):
    """x: (B,1,H,W) -> 细节能量图 sqrt(LH²+HL²+HH²)，半分辨率插值回原尺寸。

    旧方案：只保留「哪里有高频」，丢方向与符号。
    """
    x_pos = x - x.amin(dim=(2, 3), keepdim=True) + 1e-3
    _, details = haar_dwt(torch.log1p(x_pos))
    e = torch.sqrt(sum(d * d for d in details) + 1e-8)
    return F.interpolate(e, size=x.shape[-2:], mode="bilinear", align_corners=False)


EDGE_OPS = ("wavelet", "wavelet_lam", "energy", "sobel", "laplacian")


def build_edge_map(x, op="wavelet", lf_weight=0.3):
    """五个候选边缘算子统一入口，输出都是逐样本标准化的 (B,1,H,W)。

    - wavelet     : 1 级 Haar，LL 置零后 IDWT 重建（项目现用口径，λ=0）
    - wavelet_lam : 同上但掺回 λ·LL（GobletNet Eq.5，默认 λ=0.3）
    - energy      : 小波细节能量图 sqrt(LH²+HL²+HH²)（旧方案）
    - sobel       : Sobel 梯度幅值
    - laplacian   : 4 邻域 Laplacian 幅值
    """
    if op == "wavelet":
        return build_hf_image(x, log_domain=True, lf_weight=0.0)
    if op == "wavelet_lam":
        return build_hf_image(x, log_domain=True, lf_weight=lf_weight)
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
