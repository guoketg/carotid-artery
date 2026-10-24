import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from datasets.dataset_carotid import DATA_ROOT, LIST_DIR, SUBSETS, CarotidDataset, build_splits
from metrics_carotid import compute_case_metrics, summarize
from networks.vit_seg_modeling import CONFIGS, VisionTransformer as ViT_seg
from utils import DiceLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument("--list_dir", default=LIST_DIR)
    parser.add_argument("--subset", default="SPARC", choices=list(SUBSETS.keys()),
                        help="两个子集分开训练，不能混合")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=1, help="Windows 内存有限，最大 1")
    parser.add_argument("--tag", default="a0_baseline")
    parser.add_argument("--limit", type=int, default=0, help="仅用前 N 个样本做冒烟测试")
    parser.add_argument("--rebuild_splits", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_model(num_classes=2, img_size=224):
    config = CONFIGS["R50-ViT-B_16"]
    config.n_classes = num_classes
    config.n_skip = 3
    if img_size % 16 == 0:
        config.patches.grid = (img_size // 16, img_size // 16)
    model = ViT_seg(config, img_size=img_size, num_classes=num_classes)
    weights_path = os.path.join(config.pretrained_path)
    if os.path.isfile(weights_path):
        model.load_from(weights=np.load(weights_path))
    return model


def poly_lr(base_lr, step, total_steps, power=0.9):
    return base_lr * (1.0 - min(step / max(total_steps, 1), 1.0)) ** power


def train_one_epoch(model, loader, optimizer, scaler, criterion, dice_loss, scheduler, epoch, logger):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        image = batch["image"].cuda(non_blocking=True)
        label = batch["label"].cuda(non_blocking=True)

        for group in optimizer.param_groups:
            group["lr"] = scheduler(step)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(image)
            loss = criterion(logits, label) + dice_loss(logits, label, softmax=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if (step + 1) % 40 == 0:
            logger(f"epoch {epoch} iter {step + 1}/{len(loader)} loss {loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def test(model, loader):
    model.eval()
    records = []
    for batch in loader:
        image = batch["image"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(image)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        gt = batch["label"].numpy()[0]
        record = compute_case_metrics(pred, gt)
        record["case"] = batch["case"][0]
        records.append(record)
    return records


def main():
    args = parse_args()
    set_seed(args.seed)

    split_dir = os.path.join(args.list_dir, args.subset)
    if args.rebuild_splits or not os.path.isfile(os.path.join(split_dir, "train.txt")):
        build_splits(seed=args.seed)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{args.tag}_{args.subset}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "log.txt"))

    train_set = CarotidDataset(args.root, os.path.join(split_dir, "train.txt"), "train", args.img_size)
    test_set = CarotidDataset(args.root, os.path.join(split_dir, "test.txt"), "test", args.img_size)
    if args.limit > 0:
        train_set.samples = train_set.samples[: args.limit]

    logger(f"subset = {args.subset}  train/test = {len(train_set)}/{len(test_set)}")
    logger(f"run_dir = {run_dir}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.workers)

    model = build_model(img_size=args.img_size).cuda()
    logger(f"params(M) = {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    criterion = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=2)

    total_steps = args.epochs * max(len(train_loader), 1)
    global_step = 0

    def scheduler(_step):
        return poly_lr(args.lr, global_step, total_steps)

    last_path = os.path.join(run_dir, "last.pth")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                               dice_loss, scheduler, epoch, logger)
        global_step += len(train_loader)
        logger(f"epoch {epoch} loss {loss:.4f} "
               f"lr {optimizer.param_groups[0]['lr']:.5f} time {time.time() - start:.0f}s")

    torch.save(model.state_dict(), last_path)
    logger(f"training done, saved {last_path}")

    records = test(model, test_loader)
    summary = summarize(records)

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dice", "assd", "hd95", "delta_area"])
        writer.writeheader()
        writer.writerows(records)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"subset": args.subset, "args": vars(args), "test": summary}, f, ensure_ascii=False, indent=2)

    logger("test summary:")
    for key in ["dice", "assd", "hd95", "delta_area", "abs_delta_area"]:
        item = summary[key]
        logger(f"  {key}: {item['mean']:.4f} +/- {item['std']:.4f}")
    logger(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
