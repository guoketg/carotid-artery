"""cvartail 逐例配对分析（零训练，只读已有 metrics.csv）。

回答：R50 上判负的 −0.05 mm 是不是"真没效应"，还是"有稳定小效应但均值口径看不出来"。
做法：同 seed、同协议的两个 run（A0 与 arm）按 case 逐个配对，算逐例 ΔHD/ΔDice/ΔASSD，
再用配对 bootstrap 给均值差的 95% CI、Wilcoxon 符号秩检验，并按基线 HD 分层看"是否只
在难例上起作用"。全部为现成 checkpoint 的评测产物，不训练、不碰 GPU。

对照（同一 seed 内配对）：
  R50  s1234 : A0 90.4903 / 0.14473 / 0.7062  vs cvartail 90.50 / 0.142 / 0.65
  R50  s2024 : A0 90.3567 / 0.14449 / 0.6977  vs cvartail 90.09 / 0.149 / 0.68
  UNet s1234 : A0 86.5455 / 0.22500 / 1.13618 vs cvartail 87.37 / 0.200 / 0.96（参照臂）

用法：
  .\\venv\\Scripts\\python.exe scripts\\analyze_cvar_paired.py
"""
import csv
import os
import sys

import numpy as np

try:                                  # Windows 控制台默认 GBK，表格里的 − 号会炸
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = r"E:\mkl\code\python_code\model\TU_Carotid224"
OUT_DIR = os.path.join(BASE_DIR, "logs", "analysis")

R50 = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224"
U50 = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
CVAR = "lcvartail_W0.1Q0.1A0.05"

PAIRS = [
    ("R50 s1234", f"{R50}_SPARC_c2d", f"{R50}_SPARC_{CVAR}_c2d"),
    ("R50 s2024", f"{R50}_s2024_SPARC_c2d", f"{R50}_s2024_SPARC_{CVAR}_c2d"),
    ("UNet s1234", f"{U50}_SPARC_unet_c2d", f"{U50}_SPARC_{CVAR}_unet_c2d"),
]
METRICS = ["hd", "hd95", "assd", "dice", "delta_area"]
N_BOOT = 10000
RNG = np.random.default_rng(20261115)


def read_metrics(snapshot):
    path = os.path.join(MODEL_DIR, snapshot, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("case"):
                continue
            rows[r["case"]] = {m: float(r[m]) for m in METRICS}
    return rows


def boot_ci(x, n=N_BOOT):
    """配对 bootstrap：对逐例差值重采样，取均值的 2.5/97.5 分位。"""
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(x):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(x, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def analyse(label, base_snap, arm_snap, lines):
    base = read_metrics(base_snap)
    arm = read_metrics(arm_snap)
    cases = sorted(set(base) & set(arm))
    if len(cases) < 20:
        lines.append(f"### {label}\n\n共同病例仅 {len(cases)} 例，跳过。\n")
        return None

    arr = {m: np.array([arm[c][m] - base[c][m] for c in cases]) for m in METRICS}
    # delta_area 在 metrics.csv 里是有符号的（预测面积−GT 面积）/GT，表里按绝对值口径报
    arr["abs_delta_area"] = np.abs([arm[c]["delta_area"] for c in cases]) \
        - np.abs([base[c]["delta_area"] for c in cases])
    base_hd = np.array([base[c]["hd"] for c in cases])
    arm_hd = np.array([arm[c]["hd"] for c in cases])
    d_hd = arr["hd"]

    lo, hi = boot_ci(d_hd)
    # 稳健性：剔除 |ΔHD| 最大的 3 例（HD 是 max 型，个别崩例能主导均值）
    drop = np.argsort(-np.abs(d_hd))[3:]
    lo_r, hi_r = boot_ci(d_hd[drop])

    lines.append(f"### {label}\n")
    lines.append("| 指标 | A0 均值 | 臂均值 | 配对 Δ 均值 | Δ sd | Δ 中位 | 95% CI(配对 bootstrap) | 改善/变差例数 | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    spec = [("hd", "HD", -1, 1.0), ("hd95", "HD95", -1, 1.0),
            ("assd", "ASSD", -1, 1.0), ("dice", "Dice", +1, 100.0),
            ("abs_delta_area", "|ΔArea|", -1, 1.0)]
    for m, name, good_sign, scale in spec:
        x = arr[m]
        m_lo, m_hi = boot_ci(x)
        better = int((np.sign(x) == good_sign).sum())
        worse = int((np.sign(x) == -good_sign).sum())
        p = wilcoxon_p(x * good_sign)
        if m == "abs_delta_area":
            b_mean = np.mean([abs(base[c]["delta_area"]) for c in cases]) * scale
            a_mean = np.mean([abs(arm[c]["delta_area"]) for c in cases]) * scale
        else:
            b_mean = np.mean([base[c][m] for c in cases]) * scale
            a_mean = np.mean([arm[c][m] for c in cases]) * scale
        lines.append("| %s | %.4f | %.4f | %+.4f | %.4f | %+.4f | [%+.4f, %+.4f] | %d/%d | %.4g |"
                     % (name, b_mean, a_mean,
                        x.mean() * scale, x.std(ddof=1) * scale,
                        np.median(x) * scale, m_lo * scale, m_hi * scale,
                        better, worse, p))
    lines.append("")
    lines.append(f"- 剔除 |ΔHD| 最大的 3 例后：ΔHD 均值 {d_hd[drop].mean():+.4f} mm，95% CI "
                 f"[{lo_r:+.4f}, {hi_r:+.4f}]（原 {d_hd.mean():+.4f} / [{lo:+.4f}, {hi:+.4f}]）")
    lines.append(f"- 逐例 HD 相关：ΔHD 与 A0 HD 的 Pearson r = "
                 f"{np.corrcoef(base_hd, d_hd)[0, 1]:+.3f}；与 A0 HD 的 Spearman rho = "
                 f"{_spearman(base_hd, d_hd):+.3f}")

    # 按 A0 的 HD 四分位分层：看效应是否集中在本来最难的病例上
    q = np.quantile(base_hd, [0.25, 0.5, 0.75])
    grp = np.digitize(base_hd, q)
    lines.append("")
    lines.append("| A0 HD 分层 | 例数 | A0 HD 均值 | 臂 HD 均值 | ΔHD 均值 | ΔHD 中位 | 改善例数 |")
    lines.append("|---|---|---|---|---|---|---|")
    for g, name in enumerate(["Q1(最好)", "Q2", "Q3", "Q4(最难)"]):
        m = grp == g
        if m.sum() == 0:
            continue
        lines.append("| %s | %d | %.4f | %.4f | %+.4f | %+.4f | %d |"
                     % (name, int(m.sum()), base_hd[m].mean(), arm_hd[m].mean(),
                        d_hd[m].mean(), np.median(d_hd[m]), int((d_hd[m] < 0).sum())))
    lines.append("")
    return {"label": label, "n": len(cases), "d_hd": d_hd.mean(),
            "ci": (lo, hi), "p": wilcoxon_p(d_hd)}


def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ["# cvartail 逐例配对分析（零训练）\n",
             "口径：同 seed、同协议的 A0 与 cvartail 逐例 metrics.csv 配对；",
             "Δ = 臂 − A0（HD/ASSD/|ΔArea| 为负=变好，Dice 为负=变差）；",
             f"配对 bootstrap {N_BOOT} 次。\n"]
    summary = []
    for label, b, a in PAIRS:
        try:
            res = analyse(label, b, a, lines)
            if res:
                summary.append(res)
        except FileNotFoundError as e:
            lines.append(f"### {label}\n\n缺文件：{e}\n")

    lines.append("## 汇总\n")
    lines.append("| 配对 | 例数 | ΔHD 均值 | 95% CI | 显著性 |")
    lines.append("|---|---|---|---|---|")
    for r in summary:
        sig = "CI 不含 0" if (r["ci"][0] > 0 or r["ci"][1] < 0) else "CI 含 0"
        lines.append("| %s | %d | %+.4f | [%+.4f, %+.4f] | %s |"
                     % (r["label"], r["n"], r["d_hd"], r["ci"][0], r["ci"][1], sig))
    text = "\n".join(lines)
    out = os.path.join(OUT_DIR, "cvar_paired.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
