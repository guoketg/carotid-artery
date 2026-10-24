import numpy as np
from medpy.metric.binary import assd, dc, hd95


def _surface_fallback(mask_shape):
    """一侧为空时用图像对角线长度作为距离指标的最坏值"""
    return float(np.sqrt(mask_shape[0] ** 2 + mask_shape[1] ** 2))


def compute_case_metrics(pred, gt):
    """pred/gt 为二值 2D 数组，返回 dice / assd / hd95 / delta_area"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    delta = pred_area - gt_area
    fallback = _surface_fallback(pred.shape)

    if pred_area == 0 and gt_area == 0:
        return {"dice": 1.0, "assd": 0.0, "hd95": 0.0, "delta_area": 0}
    if pred_area == 0 or gt_area == 0:
        return {"dice": 0.0, "assd": fallback, "hd95": fallback, "delta_area": delta}

    return {
        "dice": float(dc(pred, gt)),
        "assd": float(assd(pred, gt)),
        "hd95": float(hd95(pred, gt)),
        "delta_area": delta,
    }


def summarize(records):
    """records 为 dict 列表，返回均值、标准差与绝对值均值"""
    keys = ["dice", "assd", "hd95", "delta_area"]
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
