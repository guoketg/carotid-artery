import csv
import os
import random

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

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
                 gamma_aug=False, weight_file=None, edge_mode=""):
        self.root = root
        self.split = split
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
        if self.augment:
            image, label = self._augment(image, label)
        edge = sobel_edges(label) if self.edge_mode == "sobel_orig" else None
        image = normalize_image(image)
        if edge is None:
            image, label = resize_pair(image, label, self.img_size)
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
