"""把 Ar-PlaqSegm1（公开数据集，541 对 800x800 png）转成本项目的 nii.gz 口径。

原始包：C:\\Users\\ranzh\\Downloads\\Ar-PlaqSegm1 First Argentine database of B-mode ul.zip
结构：test/48 对 + train/493 对，每对为 `<id>.png` 与 `<id>_labeled.png`。

处理规则：
1. 掩膜二值化（>0），RGB 取任一通道；整幅为空的样本跳过（README 口径未剔除健康样本，
   实测 test 24/48、train 177/493 为空）。
2. 输出 `PlaqueDataNii_ArPlaq_S/<split>_<id>/<split>_<id>.nii.gz` + `_label.nii.gz`。
3. spacing 记 (1.0, 1.0) mm：公开图已统一缩放到 800x800，真实 mm/px 未知，
   因此本子集的 HD95/ASSD 数值单位是「像素」，只用于子集内部比较。
4. 写 lists/lists_Carotid2D/ArPlaqSegm1/{train,test}.txt。
"""

import os
import zipfile

import numpy as np
import nibabel as nib
from PIL import Image

ZIP_PATH = r"C:\Users\ranzh\Downloads\Ar-PlaqSegm1 First Argentine database of B-mode ul.zip"
DATA_ROOT = r"E:\mkl\code\datasets\Carotid2D"
OUT_SUBDIR = "PlaqueDataNii_ArPlaq_S"
LIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lists", "lists_Carotid2D", "ArPlaqSegm1")


def load_png(zf, name):
    with zf.open(name) as f:
        arr = np.array(Image.open(f))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def save_nii(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), path)


def main():
    zf = zipfile.ZipFile(ZIP_PATH)
    root_name = zf.namelist()[0].split("/")[0]
    stats = {}
    rows = {"train": [], "test": []}

    for split in ("train", "test"):
        prefix = f"{root_name}/{split}/"
        names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png")]
        stems = sorted({os.path.basename(n)[:-4].replace("_labeled", "") for n in names})
        empty = 0
        for stem in stems:
            img_name = f"{prefix}{stem}.png"
            lbl_name = f"{prefix}{stem}_labeled.png"
            if img_name not in zf.namelist() or lbl_name not in zf.namelist():
                print(f"[warn] 缺一对: {stem}")
                continue
            image = load_png(zf, img_name)
            label = (load_png(zf, lbl_name) > 0).astype(np.float32)
            if label.sum() == 0:
                empty += 1
                continue
            case = f"{split}_{stem}"
            out_dir = os.path.join(DATA_ROOT, OUT_SUBDIR, case)
            os.makedirs(out_dir, exist_ok=True)
            save_nii(os.path.join(out_dir, f"{case}.nii.gz"), image)
            save_nii(os.path.join(out_dir, f"{case}_label.nii.gz"), label)
            rows[split].append(f"./{OUT_SUBDIR}/{case}/{case}.nii.gz")

        stats[split] = (len(stems), empty, len(rows[split]))
        os.makedirs(LIST_DIR, exist_ok=True)
        with open(os.path.join(LIST_DIR, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows[split]) + "\n")

    for split, (total, empty, kept) in stats.items():
        print(f"[{split}] 总 {total} 对，空掩膜 {empty}，可用 {kept}")

    # 复核：可用样本斑块面积（原图 800x800 与 224 重采样后）
    for split in ("train", "test"):
        ratios, px224 = [], []
        for rel in rows[split]:
            case = rel.split("/")[2]
            lbl = nib.load(os.path.join(DATA_ROOT, OUT_SUBDIR, case, f"{case}_label.nii.gz"))
            m = lbl.get_fdata() > 0
            ratios.append(m.mean())
            step = m.shape[0] / 224
            small = m[:: max(1, int(round(step))), :: max(1, int(round(step)))]
            px224.append(int(small.sum()))
        ratios, px224 = np.array(ratios), np.array(px224)
        print(f"[{split}] 斑块占图 中位 {np.median(ratios)*100:.2f}%，"
              f"224² 斑块像素 中位 {np.median(px224):.0f}，最小 {px224.min()}")


if __name__ == "__main__":
    main()
