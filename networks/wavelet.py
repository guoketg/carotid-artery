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
