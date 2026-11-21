import csv
import os
import random

import cv2
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from datasets.label_edge_ops import label_edge_op

DATA_ROOT = r"E:\mkl\code\datasets\Carotid2D"
# 两个子集必须**分开**训练与评估，不能混合
SUBSETS = {"SPARC": "PlaqueDataNii_SPARC_S", "Zhongnan": "PlaqueDataNii_Zhongnan_S"}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST_DIR = os.path.join(PROJECT_ROOT, "lists", "lists_Carotid2D")


def read_list(list_file):
    """读取样本相对路径列表，统一成正斜杠并去掉开头的 ./"""
    with open(list_file, "r", encoding="utf-8") as f:
        items = [line.strip().replace("\\", "/") for line in f if line.strip()]
    return [item[2:] if item.startswith("./") else item for item in items]


def patient_key(rel_path):
    """由相对路径提取病例键，跨子集加前缀避免 ID 冲突"""
    parts = rel_path.split("/")
    subset, case = parts[0], parts[1]
    prefix = "SPARC" if "SPARC" in subset else "ZN"
    pid = case.split("_")[0]
    return f"{prefix}:{pid}"


def label_path_of(image_path):
    return image_path.replace(".nii.gz", "_label.nii.gz")


def load_weight_table(csv_path):
    """读取样本权重表（列：rel,w），键与列表文件中的相对路径一致。"""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["rel"].replace("\\", "/"): float(r["w"]) for r in csv.DictReader(f)}


def normalize_image(volume):
    """1%~99% 分位截断后按自身均值方差标准化"""
    lo, hi = np.percentile(volume, 1), np.percentile(volume, 99)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    volume = np.clip(volume, lo, hi)
    mean, std = volume.mean(), volume.std()
    if std < 1e-6:
        std = 1.0
    return (volume - mean) / std

# 图像和标签全部resize
def resize_pair(image, label, size):
    factors = (size[0] / image.shape[0], size[1] / image.shape[1])
    image = ndimage.zoom(image, factors, order=3, mode="nearest")

    label = ndimage.zoom(label.astype(np.float32), factors, order=0, mode="nearest")
    # 标签二值化
    return image.astype(np.float32), (label > 0.5).astype(np.float32)


# ---------------- 2026-11-20 对齐口径（--align_proto 1）----------------
# 同组研究生 UNet 复现配置的预处理链：1%~99% 分位截断后 min-max 到 [0,1]，
# 再用 cv2.INTER_CUBIC 缩放到 224；mask 用 INTER_NEAREST 后二值化。
# 与既有口径的差别有两处：① 归一化公式是 min-max 不是 z-score；② 先归一化再 resize
# （既有口径是先 augment → z-score → resize，插值走 scipy zoom(order=3)）。
def percentile_minmax(volume):
    """1%~99% 分位截断后 min-max 拉伸到 [0,1]"""
    lo, hi = np.percentile(volume, 1), np.percentile(volume, 99)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    volume = np.clip(volume, lo, hi)
    return ((volume - lo) / (hi - lo)).astype(np.float32)


def resize_pair_cv2(image, label, size):
    """cv2 口径 resize：图像 INTER_CUBIC，mask INTER_NEAREST 后二值化"""
    image = cv2.resize(image.astype(np.float32), (int(size[1]), int(size[0])),
                       interpolation=cv2.INTER_CUBIC)
    label = cv2.resize(label.astype(np.float32), (int(size[1]), int(size[0])),
                       interpolation=cv2.INTER_NEAREST)
    return image, (label > 0.5).astype(np.float32)


def resize_a0_pair(image, label, size):
    """A0 口径预处理：先 scipy zoom(order=1 双线性) resize，再 1%~99% 分位 min-max 到 [0,1]。

    与 compare 口径（--align_proto 1）的区别：顺序是 resize 先、插值是双线性不是 cv2 三次、
    归一化是 min-max 不是 z-score。
    """
    factors = (size[0] / image.shape[0], size[1] / image.shape[1])
    image = ndimage.zoom(image.astype(np.float32), factors, order=1)
    label = ndimage.zoom(label.astype(np.float32), factors, order=0)
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo + 1e-6:
        lo, hi = float(image.min()), float(image.max())
    image = ((np.clip(image, lo, hi) - lo) / (hi - lo + 1e-6)).astype(np.float32)
    return image, (label > 0.5).astype(np.float32)


def augment_a0(image, label):
    """A0 口径的温和增强：hflip 0.5 / vflip 0.2 / 仿射 0.4（±10°、±5%、±3%）/
    gamma 0.5（0.8~1.2）/ 轻对比度 0.25（0.92~1.08 + 亮度 ±0.08）。

    与 compare 口径（AugmentAlign：±30°、CLAHE、偏置场、噪声）刻意区分；
    README 明确要求 A0 不要再加 CLAHE。
    """
    height, width = image.shape[:2]
    if random.random() < 0.5:
        image, label = image[:, ::-1], label[:, ::-1]
    if random.random() < 0.2:
        image, label = image[::-1], label[::-1]
    if random.random() < 0.4:
        angle = float(np.deg2rad(random.uniform(-10.0, 10.0)))
        scale = random.uniform(0.95, 1.05)
        shift_y = random.uniform(-0.03, 0.03) * height
        shift_x = random.uniform(-0.03, 0.03) * width
        cosine, sine = float(np.cos(angle)), float(np.sin(angle))
        forward = scale * np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        inverse = np.linalg.inv(forward)
        center = np.array([(height - 1) / 2.0, (width - 1) / 2.0], dtype=np.float64)
        offset = center - inverse @ (center + np.array([shift_y, shift_x], dtype=np.float64))
        image = ndimage.affine_transform(image, inverse, offset=offset, output_shape=image.shape,
                                         order=1, mode="constant", cval=0.0, prefilter=False)
        label = ndimage.affine_transform(label, inverse, offset=offset, output_shape=label.shape,
                                         order=0, mode="constant", cval=0.0, prefilter=False)
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        label = (label >= 0.5).astype(np.float32)
    if random.random() < 0.5:
        image = np.power(np.clip(image, 0.0, 1.0), random.uniform(0.8, 1.2)).astype(np.float32)
    if random.random() < 0.25:
        contrast = random.uniform(0.92, 1.08)
        brightness = random.uniform(-0.08, 0.08)
        image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label, dtype=np.float32)


def augment_align(image, label):
    """对齐口径的强增强（同组研究生配置）：水平翻转 + 重仿射 + 光度扰动 + 噪声 + 偏置场。

    逐项概率：翻转 0.5 / 仿射 0.5（旋转 ±30°、缩放 ±15%、平移 ±8%）/
    亮度对比度 0.8 / gamma 0.55 / CLAHE 0.35 / 模糊或锐化 0.45 /
    高斯噪声 σ=0.02 0.35 / 偏置场 0.30（乘性 ±15% 平滑场）。
    输入输出都在 [0,1]，mask 全程 nearest 且最后一次二值化。
    """
    h, w = image.shape[:2]
    if random.random() < 0.5:
        image, label = image[:, ::-1], label[:, ::-1]
    if random.random() < 0.5:
        angle = random.uniform(-30.0, 30.0)
        scale = random.uniform(0.85, 1.15)
        tx = random.uniform(-0.08, 0.08) * w
        ty = random.uniform(-0.08, 0.08) * h
        M = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101)
        label = (label > 0.5).astype(np.float32)
    if random.random() < 0.8:
        alpha = random.uniform(0.8, 1.2)          # 对比度
        beta = random.uniform(-0.1, 0.1)         # 亮度
        image = np.clip(image * alpha + beta, 0.0, 1.0)
    if random.random() < 0.55:
        g = random.uniform(0.7, 1.4)
        image = np.clip(np.power(np.clip(image, 0.0, 1.0), g), 0.0, 1.0)
    if random.random() < 0.35:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(np.clip(image * 255.0, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0
    if random.random() < 0.45:
        if random.random() < 0.5:
            k = random.choice((3, 5))
            image = cv2.GaussianBlur(image, (k, k), 0)
        else:
            sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
            image = cv2.filter2D(image, -1, sharp)
        image = np.clip(image, 0.0, 1.0)
    if random.random() < 0.35:
        image = np.clip(image + np.random.normal(0.0, 0.02, image.shape).astype(np.float32), 0.0, 1.0)
    if random.random() < 0.30:
        low = np.random.normal(0.0, 1.0, (h, w)).astype(np.float32)
        low = cv2.GaussianBlur(low, (0, 0), sigmaX=max(h, w) * 0.12, sigmaY=max(h, w) * 0.12)
        span = float(low.max() - low.min())
        low = (low - float(low.min())) / (span if span > 1e-6 else 1.0)
        image = np.clip(image * (0.85 + 0.30 * low), 0.0, 1.0)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label, dtype=np.float32)


def sobel_edges(label):
    """原始尺寸二值标签过 Sobel：梯度 > 0 置 1，其余 0（旧实验口径）。

    注意与 trainer._edge_target 的区别：这里**不 resize**，边缘在原始小 crop
    （如 42×63）上算出来，再随图像一起 nearest 放大到 224，边缘带宽度会被
    放大同样的倍数（≈5 倍），前景占比从 ~2% 变成 ~12%。
    """
    gx = ndimage.sobel(label, axis=0, mode="nearest")
    gy = ndimage.sobel(label, axis=1, mode="nearest")
    return (np.hypot(gx, gy) > 0).astype(np.float32)


class CarotidDataset(Dataset):
    """颈动脉斑块 2D 超声切片数据集，返回单通道图像与二值标签"""

    def __init__(self, root, list_file, split="train", img_size=224, augment=True, use_wavelet=False,
                 gamma_aug=False, weight_file=None, edge_mode="", align_proto=False, a0_proto=False):
        self.root = root
        self.split = split
        # 2026-11-20：对齐同组研究生的 UNet 复现口径（min-max + cv2 cubic + 强增强）。
        # 默认 False，既有实验的数据管线逐位不变。
        self.align_proto = align_proto
        # 2026-11-21：A0 复现口径（resize 先 + 双线性 + 1~99 分位 min-max + 温和增强）。
        # 同样默认 False；A0 与 compare 两种口径不能混用（README §6）。
        self.a0_proto = bool(a0_proto)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.augment = augment and split == "train"
        self.use_wavelet = use_wavelet
        # BDFR-Net 论文的 augmentation 含 gamma 扰动；默认关，保持既有实验口径不变
        self.gamma_aug = gamma_aug
        # 边缘监督分支的目标来源：'' = 不返回边缘（沿用 trainer 内的 224 形态学梯度）；
        # 'sobel_orig' = 原始尺寸二值标签过 Sobel（>0→1）后再随图像放大到 224
        self.edge_mode = edge_mode
        self.samples = read_list(list_file)
        if not self.samples:
            raise RuntimeError(f"列表为空: {list_file}")
        # 逐切片损失权重表（可选）：缺失切片回退到 1.0，保证与无权重口径可对照
        self.weights = load_weight_table(weight_file) if weight_file else {}

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, label):
        k = random.randint(0, 3)
        if k:
            image, label = np.rot90(image, k), np.rot90(label, k)
        if random.random() < 0.5:
            image, label = np.fliplr(image), np.fliplr(label)
        if random.random() < 0.5:
            image, label = np.flipud(image), np.flipud(label)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            image = ndimage.rotate(image, angle, order=3, reshape=False, mode="nearest")
            label = ndimage.rotate(label, angle, order=0, reshape=False, mode="nearest")
        if self.gamma_aug and random.random() < 0.5:
            # gamma 增强（论文 augmentation 之一）：在 [0,1] 上做幂变换再映回原灰度范围
            lo, hi = float(image.min()), float(image.max())
            if hi > lo:
                g = random.uniform(0.7, 1.4)
                image = np.power((image - lo) / (hi - lo), g) * (hi - lo) + lo
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def __getitem__(self, index):
        rel = self.samples[index]
        image = nib.load(os.path.join(self.root, rel)).get_fdata(dtype=np.float32)
        label = nib.load(os.path.join(self.root, label_path_of(rel))).get_fdata(dtype=np.float32)
        # 兼容意外出现 3D 的情况：取中间层
        if image.ndim == 3:
            image, label = image[..., image.shape[-1] // 2], label[..., label.shape[-1] // 2]

        label = (label > 0.5).astype(np.float32)
        if self.a0_proto and self.edge_mode == "":
            # A0 口径：resize(双线性) → 分位 min-max → 温和增强
            image, label = resize_a0_pair(image, label, self.img_size)
            if self.augment:
                image, label = augment_a0(image, label)
            image = torch.from_numpy(image).unsqueeze(0).float()
            if self.use_wavelet:
                from networks.wavelet import build_three_channel
                image = build_three_channel(image.unsqueeze(0))[0]
            return {
                "image": image,
                "label": torch.from_numpy(label).long(),
                "case": rel,
                "weight": torch.tensor(self.weights.get(rel, 1.0), dtype=torch.float32),
            }
        if self.align_proto and self.edge_mode == "":
            # 对齐口径：min-max → cv2 cubic resize → 强增强（不做 z-score，不改先 resize 的顺序）
            image = percentile_minmax(image)
            image, label = resize_pair_cv2(image, label, self.img_size)
            if self.augment:
                image, label = augment_align(image, label)
            image = torch.from_numpy(image).unsqueeze(0).float()
            if self.use_wavelet:
                from networks.wavelet import build_three_channel
                image = build_three_channel(image.unsqueeze(0))[0]
            return {
                "image": image,
                "label": torch.from_numpy(label).long(),
                "case": rel,
                "weight": torch.tensor(self.weights.get(rel, 1.0), dtype=torch.float32),
            }
        if self.augment:
            image, label = self._augment(image, label)
        edge = sobel_edges(label) if self.edge_mode == "sobel_orig" else None
        image = normalize_image(image)
        if edge is None:
            image, label = resize_pair(image, label, self.img_size)
            if self.edge_mode in ("roberts", "lap4", "canny", "pc"):
                # 标签侧算子族：在 224 网格上对掩膜算，点集与形态学梯度真正不同
                edge = label_edge_op(label, self.edge_mode)
        else:
            # 与 resize_pair 同一组 factors，边缘用 order=0 nearest 一起放大
            factors = (self.img_size[0] / image.shape[0], self.img_size[1] / image.shape[1])
            image, label = resize_pair(image, label, self.img_size)
            edge = ndimage.zoom(edge, factors, order=0, mode="nearest")
            edge = (edge > 0.5).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0).float()
        if self.use_wavelet:
            from networks.wavelet import build_three_channel
            image = build_three_channel(image.unsqueeze(0))[0]

        batch = {
            "image": image,
            "label": torch.from_numpy(label).long(),
            "case": rel,
            "weight": torch.tensor(self.weights.get(rel, 1.0), dtype=torch.float32),
        }
        if edge is not None:
            batch["edge"] = torch.from_numpy(edge).long()
        return batch


def build_splits(seed=1234, test_ratio=0.2):
    """每个子集独立按病例划分 train/test（不设验证集），同一病例不跨 split"""
    for short, folder in SUBSETS.items():
        list_file = os.path.join(DATA_ROOT, folder, "image_Niilist.txt")
        samples = read_list(list_file)

        groups = {}
        for rel in samples:
            groups.setdefault(patient_key(rel), []).append(rel)

        keys = sorted(groups.keys())
        random.Random(seed).shuffle(keys)
        n_test = max(1, int(len(keys) * test_ratio))
        split_keys = {
            "test": keys[:n_test],
            "train": keys[n_test:],
        }

        out_dir = os.path.join(LIST_DIR, short)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{short}] {len(samples)} 张 / {len(keys)} 例")
        for split, ks in split_keys.items():
            rows = sorted(rel for k in ks for rel in groups[k])
            with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(rows))
            print(f"  {split}: {len(ks)} 例 / {len(rows)} 张")


if __name__ == "__main__":
    build_splits()
