import os

import nibabel as nib
import numpy as np
from medpy.metric.binary import assd, dc, hd, hd95
from scipy import ndimage as ndi


def _surface_fallback(mask_shape, spacing=None):
    """一侧为空时用图像对角线长度作为距离指标的最坏值（传 spacing 时单位 mm）"""
    sy, sx = spacing if spacing is not None else (1.0, 1.0)
    return float(np.sqrt((mask_shape[0] * sy) ** 2 + (mask_shape[1] * sx) ** 2))


def compute_case_metrics(pred, gt, spacing=None):
    """pred/gt 为二值 2D 数组；spacing=(行间距, 列间距) mm/px，来自 NIfTI 头"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    sy, sx = spacing if spacing is not None else (1.0, 1.0)
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    delta = (pred_area - gt_area) * sy * sx
    fallback = _surface_fallback(pred.shape, spacing)

    if pred_area == 0 and gt_area == 0:
        return {"dice": 1.0, "assd": 0.0, "hd95": 0.0, "hd": 0.0, "delta_area": 0}
    if pred_area == 0 or gt_area == 0:
        return {"dice": 0.0, "assd": fallback, "hd95": fallback, "hd": fallback,
                "delta_area": delta}

    return {
        "dice": float(dc(pred, gt)),
        "assd": float(assd(pred, gt, voxelspacing=spacing)),
        "hd95": float(hd95(pred, gt, voxelspacing=spacing)),
        # HD(max)：论文 Table 1 的 HD 口径，与 HD95 共用同一张表面距离矩阵
        "hd": float(hd(pred, gt, voxelspacing=spacing)),
        "delta_area": delta,
    }


def summarize(records):
    """records 为 dict 列表，返回均值、标准差与绝对值均值"""
    keys = ["dice", "assd", "hd95", "hd", "delta_area"]
    summary = {}
    for key in keys:
        values = np.array([r[key] for r in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    delta = np.array([abs(r["delta_area"]) for r in records], dtype=np.float64)
    summary["abs_delta_area"] = {"mean": float(delta.mean()), "std": float(delta.std(ddof=1))}
    summary["n_cases"] = len(records)
    return summary

def original_grid_pair(pred224, root, rel):
    """把 224 网格预测重采样回原分辨率网格，并读取原网格 GT 与头文件 spacing。
    全部为库调用；返回 (pred_orig, gt_orig, spacing)，spacing 直接来自 NIfTI 头。"""
    img_path = os.path.join(root, rel)
    hdr = nib.load(img_path).header
    oh, ow = hdr.get_data_shape()[:2]
    zooms = hdr.get_zooms()[:2]
    gt = nib.load(img_path.replace(".nii.gz", "_label.nii.gz")).get_fdata(dtype=np.float32)
    if gt.ndim == 3:  # 与 dataset_carotid 一致：取中间层
        gt = gt[..., gt.shape[-1] // 2]
    pred = ndi.zoom(pred224.astype(np.float32),
                    (oh / pred224.shape[0], ow / pred224.shape[1]),
                    order=0, mode="nearest") > 0.5
    if pred.shape != (oh, ow):  # 浮点取整极端情况，裁剪/补零对齐
        pred = np.pad(pred[:oh, :ow],
                      [(0, max(0, oh - pred.shape[0])), (0, max(0, ow - pred.shape[1]))])
    return pred, gt > 0.5, (float(zooms[0]), float(zooms[1]))

