"""Build a small case-level dev subset for fast ablation screening.

Train split: sample a fixed fraction of *cases* from the full train split.
Test split: copy the full test split unchanged, so dev numbers are directly
comparable to full-data numbers (same holdout, same metric code).

Usage:
    python scripts/build_dev_subset.py                 # SPARC, 40% of train cases
    python scripts/build_dev_subset.py --subset Zhongnan --ratio 0.3
"""

import argparse
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "lists", "lists_Carotid")
DST_ROOT = os.path.join(PROJECT_ROOT, "lists", "lists_Carotid_dev")


def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def case_key(rel):
    return rel.split("/")[1].split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=str, default="SPARC", choices=["SPARC", "Zhongnan"])
    ap.add_argument("--ratio", type=float, default=0.4, help="fraction of train CASES kept")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default=DST_ROOT)
    args = ap.parse_args()

    src_dir = os.path.join(SRC_ROOT, args.subset)
    dst_dir = os.path.join(args.out, args.subset)
    os.makedirs(dst_dir, exist_ok=True)

    # train: case-level subsample (never slice-level -> no leakage)
    rows = read_list(os.path.join(src_dir, "train.txt"))
    groups = {}
    for rel in rows:
        groups.setdefault(case_key(rel), []).append(rel)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n_keep = max(1, int(round(len(keys) * args.ratio)))
    keep = sorted(keys[:n_keep])
    dev_train = sorted(rel for k in keep for rel in groups[k])
    with open(os.path.join(dst_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(dev_train))

    # test: keep the official holdout untouched
    test_rows = read_list(os.path.join(src_dir, "test.txt"))
    with open(os.path.join(dst_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_rows))

    print(f"[{args.subset}] dev train: {len(dev_train)} 张 / {n_keep} 例 "
          f"(原 {len(rows)} 张 / {len(keys)} 例, ratio={args.ratio})")
    print(f"[{args.subset}] dev test : {len(test_rows)} 张 / "
          f"{len({case_key(r) for r in test_rows})} 例 (与全量一致)")
    print(f"[{args.subset}] -> {dst_dir}")


if __name__ == "__main__":
    main()
