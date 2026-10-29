"""Aggregate the single-variable dev-screening runs into one comparison table.

Scans every snapshot dir ending with `_dev` under the model root, reads the
per-case `metrics.csv` (case,dice,assd,hd95,delta_area) and the optional
`test_curve.csv` (epoch,dice,assd,hd95,abs_delta_area), and prints a table
sorted by Dice.  Ranking only -- dev subset numbers are NOT paper numbers.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\summarize_dev.py
    .\\venv\\Scripts\\python.exe scripts\\summarize_dev.py --model_root F:\\code\\MyPython\\model\\TU_Carotid224
"""
import argparse
import csv
import os
import re

STRIP_PREFIX = "TU_pretrain_R50-ViT-B_16_skip3_epo"


def short_name(dirname):
    """Turn a snapshot dir name into a readable label.

    ..._224_SPARC_dual_L1234_auto_dev          -> SPARC dual L1234 auto
    ..._224_SPARC_dev                          -> SPARC baseline
    """
    name = re.sub(r"^.*?_224_", "", dirname)
    name = re.sub(r"_dev$", "", name)
    if "_dual_" not in name:
        return name + " baseline"
    sub, rest = name.split("_dual_", 1)
    return sub + " dual " + rest.replace("_", " ")


def read_metrics(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if k != "case"})
            except (TypeError, ValueError):
                continue
    if not rows:
        return None
    n = len(rows)
    out = {k: sum(r[k] for r in rows) / n for k in rows[0]}
    out["abs_delta_area"] = sum(abs(r["delta_area"]) for r in rows) / n
    out["n_case"] = n
    return out


def read_curve(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
    if not rows:
        return None, None
    best = max(rows, key=lambda r: r["dice"])
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", type=str,
                    default=r"E:\mkl\code\python_code\model\TU_Carotid224")
    ap.add_argument("--suffix", type=str, default="_dev")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    records = []
    for d in sorted(os.listdir(args.model_root)):
        full = os.path.join(args.model_root, d)
        if not os.path.isdir(full) or not d.endswith(args.suffix):
            continue
        mpath = os.path.join(full, "metrics.csv")
        if not os.path.exists(mpath):
            records.append({"name": short_name(d), "dir": d, "status": "running/no metrics"})
            continue
        m = read_metrics(mpath)
        if m is None:
            continue
        curve, best = read_curve(os.path.join(full, "test_curve.csv"))
        rec = {
            "name": short_name(d), "dir": d, "status": "done",
            "dice": m["dice"], "assd": m["assd"], "hd95": m["hd95"],
            "abs_delta_area": m["abs_delta_area"], "n_case": int(m["n_case"]),
            "best_ep": int(best["epoch"]) if best else -1,
            "best_dice": best["dice"] if best else float("nan"),
            "curve": curve,
        }
        records.append(rec)

    done = [r for r in records if r["status"] == "done"]
    done.sort(key=lambda r: -r["dice"])

    lines = []
    lines.append("# Dev 子集单变量快筛汇总 (SPARC, 175 训练切片 / 46 例, test=官方 holdout 82 切片 / 28 例)")
    lines.append("")
    lines.append("> 仅用于**方案排序**，不作论文数字；60 epoch 未收敛，绝对值不可与全量 120 epoch 结果比较。")
    lines.append("")
    if done:
        base = next((r for r in done if r["dir"].endswith("_dev") and "_dual" not in r["dir"]), None)
        base_dice = base["dice"] if base else None
        lines.append("| # | 方案 | Dice | ASSD | HD95 | |ΔArea| | ΔDice vs base | best_ep | cases |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(done, 1):
            dd = ("%+.2f" % ((r["dice"] - base_dice) * 100)) if base_dice is not None else "-"
            lines.append("| %d | %s | %.2f%% | %.2f | %.2f | %.1f | %s | %d | %d |" % (
                i, r["name"], r["dice"] * 100, r["assd"], r["hd95"],
                r["abs_delta_area"], dd, r["best_ep"], r["n_case"]))
    pending = [r for r in records if r["status"] != "done"]
    if pending:
        lines.append("")
        lines.append("未出结果: " + ", ".join(r["name"] for r in pending))
    text = "\n".join(lines)
    print(text)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "logs", "queue", "dev_screening_summary.md")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n[saved] " + out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
