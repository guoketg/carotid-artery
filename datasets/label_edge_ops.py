# -*- coding: utf-8 -*-
"""标签侧边缘算子（在 224 二值掩膜上计算），供 CarotidDataset 的 --edge_mode 调用。

背景：3×3 差分族（Prewitt/Sobel/Scharr/Robinson/Frei-Chen/Kirsch/Laplacian8）在二值图上
输出点集与 3×3 形态学梯度**逐位相同**（Jaccard 0.998~1.000），5×5 族同样退化为 k=5，
所以这里只保留点集真正不同的四个：Roberts（2×2，更窄）、Laplacian 4 邻域（更窄）、
Canny（NMS + 双阈值滞后，最细且带连通）、多尺度 log-Gabor 能量（相位一致性近似）。

输入输出：float32 的 (H,W)，值为 0/1。
"""
import numpy as np
import scipy.ndimage as ndi


def _bin(score, thr):
    return (score > thr).astype(np.float32)


def roberts(label):
    """2×2 Roberts 交叉梯度，>0 置 1。比 3×3 族窄，会漏掉仅对角相邻的边界点。"""
    a = ndi.convolve(label, np.array([[1., 0.], [0., -1.]], np.float32), mode="nearest")
    b = ndi.convolve(label, np.array([[0., 1.], [-1., 0.]], np.float32), mode="nearest")
    return _bin(np.hypot(a, b), 1e-6)


def lap4(label):
    """4 邻域 Laplacian（不含对角），|响应|>0 置 1。"""
    k = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], np.float32)
    return _bin(np.abs(ndi.convolve(label, k, mode="nearest")), 1e-6)


def canny(label, sigma=1.0, low_ratio=0.4, high_pct=90.0):
    """σ 平滑 → Sobel → 非极大值抑制 → 双阈值 + 8 连通滞后。输出约 1 px 细线。"""
    g = ndi.gaussian_filter(label, sigma, mode="nearest")
    gy = ndi.sobel(g, axis=0, mode="nearest")
    gx = ndi.sobel(g, axis=1, mode="nearest")
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

    nms = np.zeros_like(mag)
    for sel, s1, s2 in [
        ((ang < 22.5) | (ang >= 157.5), (0, 1), (0, -1)),
        ((ang >= 22.5) & (ang < 67.5), (-1, 1), (1, -1)),
        ((ang >= 67.5) & (ang < 112.5), (1, 0), (-1, 0)),
        ((ang >= 112.5) & (ang < 157.5), (-1, -1), (1, 1)),
    ]:
        a = np.roll(mag, s1, axis=(0, 1))
        b = np.roll(mag, s2, axis=(0, 1))
        keep = sel & (mag >= a) & (mag >= b)
        nms[keep] = mag[keep]

    nz = nms[nms > 0]
    if nz.size == 0:
        return np.zeros_like(nms, np.float32)
    high = np.percentile(nz, high_pct)
    strong = nms >= high
    weak = (nms >= high * low_ratio) & ~strong
    lab, _ = ndi.label(strong | weak, ndi.generate_binary_structure(2, 2))
    ids = np.unique(lab[strong])
    return np.isin(lab, ids[ids > 0]).astype(np.float32)


def pc(label, nscale=4, norient=6, min_wl=3.0, mult=2.0, sigma_onf=0.55, rel_thr=0.60):
    """多尺度 log-Gabor 方向能量（相位一致性的简化近似）。

    对二值阶跃边缘，log-Gabor 的响应集中在边界两侧，rel_thr 控制带宽
    （0.35 → 约 9% 像素，0.60 → 约 3%~4%，与 k=5 同量级，便于单变量对照）。
    """
    H, W = label.shape
    f = np.fft.fft2(label)
    u = np.fft.fftfreq(W)[None, :]
    v = np.fft.fftfreq(H)[:, None]
    radius = np.sqrt(u ** 2 + v ** 2)
    radius[0, 0] = 1e-3
    theta = np.arctan2(v, u)
    en = np.zeros((H, W))
    for s in range(nscale):
        fo = 1.0 / (min_wl * mult ** s)
        lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2))
        lg[0, 0] = 0
        for o in range(norient):
            ang = np.pi * o / norient
            dtheta = np.abs(np.angle(np.exp(1j * (theta - ang))))
            ds = np.exp(-(dtheta ** 2) / (2 * (np.pi / norient) ** 2))
            en += np.abs(np.real(np.fft.ifft2(f * lg * ds)))
    thr = rel_thr * en.max()
    return _bin(en, thr)


def label_edge_op(label, mode):
    """统一入口：label 为 (H,W) float32 0/1，返回同尺寸 0/1 边缘带。"""
    if mode == "roberts":
        return roberts(label)
    if mode == "lap4":
        return lap4(label)
    if mode == "canny":
        return canny(label)
    if mode == "pc":
        return pc(label)
    raise ValueError("unknown label edge op: %s" % mode)
