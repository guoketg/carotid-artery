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


class CarotidDataset(Dataset):
    """颈动脉斑块 2D 超声切片数据集，返回单通道图像与二值标签"""

    def __init__(self, root, list_file, split="train", img_size=224, augment=True, use_wavelet=False):
        self.root = root
        self.split = split
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.augment = augment and split == "train"
        self.use_wavelet = use_wavelet
        self.samples = read_list(list_file)
        if not self.samples:
            raise RuntimeError(f"列表为空: {list_file}")

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
        image = normalize_image(image)
        image, label = resize_pair(image, label, self.img_size)
        image = torch.from_numpy(image).unsqueeze(0).float()
        if self.use_wavelet:
            from networks.wavelet import build_three_channel
            image = build_three_channel(image.unsqueeze(0))[0]

        return {
            "image": image,
            "label": torch.from_numpy(label).long(),
            "case": rel,
        }


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
