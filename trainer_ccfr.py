"""CCFR 两阶段训练（ESWA-D-25-34301 复现）。

Stage 1：EA-Net，Dice + BCE + 0.4·Edge，Adam(1e-3, wd 0.01)，ep200 取终点。
Stage 2：冻结 EA-Net，训练 flow 精修头，速度损失按不确定性门 M 加权，
         K=5 步 Euler 得到 P_K，再对 P_K 加 BCE + Dice（权重 5:1:5），
         AdamW(1e-4, wd 0.01)，20 epoch。
口径与 §0 一致：batch 4 × grad_accum 2，每 5 epoch 在 SPARC test 160 张上评估。
"""

import csv
import functools
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics_carotid import compute_case_metrics, original_grid_pair, summarize
from networks.ccfr import logit
from trainer import _worker_init


def _edge_target(label, k=3):
    """由 GT 掩膜显式提取边缘标签（论文只写 "edge map from the ground-truth mask"，
    未给算子，此处固定为可复现口径）：

      1) 掩膜二值化（> 0.5）：数据集返回的是 resize 后的 long 标签，先转 float 并阈值化；
      2) k×k 形态学梯度 dilate − erode：k=3 时得到 1 像素内环 + 1 像素外环的二值边缘带；
      3) clamp 到 [0,1] 作为 BCE 目标。

    注意它作用在 224² 网格上（标签已被 dataset resize 到 img_size），因此边缘带在原始
    800²/512² 图上对应更宽的物理带宽，这与 Stage 2 里 M 使用的预测边缘强度同网格。
    """
    m = (label.float().unsqueeze(1) > 0.5).float()
    dil = F.max_pool2d(m, k, 1, k // 2)
    ero = -F.max_pool2d(-m, k, 1, k // 2)
    return (dil - ero).clamp(0.0, 1.0)


def _split_by_case(samples, ratio, seed):
    """按病例（相对路径第二级目录）划分 train/val，避免同一斑块的相邻切片跨 split 泄漏。"""
    cases = sorted({s.split('/')[1] for s in samples})
    random.Random(seed).shuffle(cases)
    n_val = int(round(len(cases) * ratio))
    val_cases = set(cases[:n_val])
    tr = [s for s in samples if s.split('/')[1] not in val_cases]
    va = [s for s in samples if s.split('/')[1] in val_cases]
    return tr, va


def _soft_dice(prob, target):
    p = prob.flatten(1)
    g = target.flatten(1)
    inter = (p * g).sum(1)
    return 1 - ((2 * inter + 1e-5) / (p.sum(1) + g.sum(1) + 1e-5)).mean()


def _build_loaders(args, snapshot_path=None):
    from datasets.dataset_carotid import CarotidDataset, read_list
    split_dir = os.path.join(args.list_dir, args.subset)
    train_list = os.path.join(split_dir, "train.txt")
    val_list = None
    ratio = float(getattr(args, 'stage1_val_ratio', 0.0) or 0.0)
    if ratio > 0 and snapshot_path is not None:
        # 论文只说 Stage 2 用 "best-performing Stage-1 model"，未说如何选；这里从训练集
        # 按病例切出 val（不碰 test），Stage-1 每 eval_test_every 轮在 val 上选最优权重。
        tr, va = _split_by_case(read_list(train_list), ratio, args.seed)
        with open(os.path.join(snapshot_path, 'stage1_train.txt'), 'w', encoding='utf-8') as f:
            f.write("\n".join("./" + s for s in tr) + "\n")
        train_list = os.path.join(snapshot_path, 'stage1_train.txt')
        if va:
            with open(os.path.join(snapshot_path, 'stage1_val.txt'), 'w', encoding='utf-8') as f:
                f.write("\n".join("./" + s for s in va) + "\n")
            val_list = os.path.join(snapshot_path, 'stage1_val.txt')
    train_set = CarotidDataset(root=args.root_path, list_file=train_list,
                               split="train", img_size=args.img_size, use_wavelet=False)
    test_set = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "test.txt"),
                              split="test", img_size=args.img_size, use_wavelet=False)
    val_set = None
    if val_list:
        val_set = CarotidDataset(root=args.root_path, list_file=val_list,
                                 split="test", img_size=args.img_size, use_wavelet=False)
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0,
                             worker_init_fn=functools.partial(_worker_init, seed=args.seed))
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return trainloader, val_loader, testloader, len(train_set)


@torch.no_grad()
def _eval(model, testloader, root_path, stage):
    """stage=1 只跑 EA-Net；stage=2 跑完整 K 步精修。"""
    was_training = model.training
    model.eval()
    records = []
    for batch in testloader:
        if stage == 1:
            r = model.stage1(batch['image'].cuda())
            prob = torch.sigmoid(r['region'])
        else:
            prob = model(batch['image'].cuda())['prob']
        pred = (prob[:, 0] > 0.5).cpu().numpy()[0].astype(np.uint8)
        pred_o, gt_o, spacing = original_grid_pair(pred, root_path, batch['case'][0])
        record = compute_case_metrics(pred_o, gt_o, spacing=spacing)
        record['case'] = batch['case'][0]
        records.append(record)
    if was_training:
        model.train()
    return records, summarize(records)


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _curve_row(epoch, summary):
    return {"epoch": epoch,
            "dice": summary["dice"]["mean"],
            "assd": summary["assd"]["mean"],
            "hd95": summary["hd95"]["mean"],
            "hd": summary["hd"]["mean"],
            "abs_delta_area": summary["abs_delta_area"]["mean"]}


def _log_curve(tag, row):
    logging.info("%s epoch %d test curve: dice=%.2f%% assd=%.3f hd=%.2f hd95=%.2f"
                 % (tag, row["epoch"], row["dice"] * 100, row["assd"], row["hd"], row["hd95"]))


def trainer_ccfr(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(snapshot_path + '/log')
    trainloader, val_loader, testloader, n_train = _build_loaders(args, snapshot_path)
    logging.info("train samples: %d (val samples: %d)"
                 % (n_train, len(val_loader.dataset) if val_loader is not None else 0))
    logging.info("params: total=%.2fM ea=%.2fM flow=%.2fM" % (
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in model.ea.parameters()) / 1e6,
        sum(p.numel() for p in model.flow.parameters()) / 1e6))

    stage = str(getattr(args, 'ccfr_stage', 'both'))
    stage1_ckpt = None

    # ---------------- Stage 1: EA-Net ----------------
    if stage in ('both', '1'):
        s1_epochs = args.stage1_epochs
        if getattr(args, 'ccfr_pretrain', 0):
            # 预训练编码器单独用低 lr（AdamW），解码器/边缘支仍用 stage1_lr，避免 200 轮把
            # ImageNet 特征冲掉；非预训练路径保持原有 Adam 单组行为，读数可比。
            enc = [p for n, p in model.ea.named_parameters() if n.startswith('enc.')]
            rest = [p for n, p in model.ea.named_parameters() if not n.startswith('enc.')]
            opt = torch.optim.AdamW([
                {'params': enc, 'lr': args.stage1_lr_enc},
                {'params': rest, 'lr': args.stage1_lr},
            ], weight_decay=args.stage1_wd)
            base_lrs = [args.stage1_lr_enc, args.stage1_lr]
        else:
            opt = torch.optim.Adam(model.ea.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_wd)
            base_lrs = [args.stage1_lr] * len(opt.param_groups)
        curve, curve_rows = os.path.join(snapshot_path, 'stage1_curve.csv'), []
        best_dice, best_ep = -1.0, 0
        iters_per_epoch = max(1, (len(trainloader) + args.grad_accum - 1) // args.grad_accum)
        total_iters = s1_epochs * iters_per_epoch
        iterator = tqdm(range(s1_epochs * len(trainloader)), ncols=70)
        iter_num = 0
        for epoch_num in range(s1_epochs):
            model.train()
            opt.zero_grad()
            for i, batch in enumerate(trainloader):
                image, label = batch['image'].cuda(), batch['label'].cuda().long()
                out = model.stage1(image)
                l_bce = F.binary_cross_entropy_with_logits(out['region'].squeeze(1), label.float())
                l_dice = _soft_dice(torch.sigmoid(out['region'].squeeze(1)), label.float())
                l_edge = F.binary_cross_entropy_with_logits(
                    out['edge'], _edge_target(label, args.edge_kernel))
                loss = l_bce + l_dice + args.edge_w * l_edge
                (loss / args.grad_accum).backward()
                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.ea.parameters(), 12.0)
                    opt.step()
                    opt.zero_grad()
                    lr_ = 1.0 - iter_num / total_iters
                    for g, b in zip(opt.param_groups, base_lrs):
                        g['lr'] = b * lr_ ** 0.9
                    iter_num += 1
                if iter_num % 20 == 0:
                    writer.add_scalar('stage1/loss', loss.item(), iter_num)
                    writer.add_scalar('stage1/loss_edge', l_edge.item(), iter_num)
                iterator.update(1)
            logging.info("stage1 epoch %d: loss=%.4f bce=%.4f dice=%.4f edge=%.4f"
                         % (epoch_num + 1, loss.item(), l_bce.item(), l_dice.item(), l_edge.item()))

            ep = epoch_num + 1
            if args.eval_test_every > 0 and (ep % args.eval_test_every == 0 or ep == s1_epochs):
                _, summary = _eval(model, testloader, args.root_path, stage=1)
                row = _curve_row(ep, summary)
                curve_rows.append(row)
                _write_csv(curve, curve_rows, ["epoch", "dice", "assd", "hd95", "hd", "abs_delta_area"])
                _log_curve("stage1", row)
                for k in ("dice", "assd", "hd"):
                    writer.add_scalar('stage1_curve/' + k, row[k], ep)
                # 选模只看 val（不碰 test），对应论文 "best-performing Stage-1 model"；
                # 无 val 时退回 test 曲线（会在日志里标注，读数偏乐观）
                if val_loader is not None:
                    _, sel = _eval(model, val_loader, args.root_path, stage=1)
                    sel_dice = sel["dice"]["mean"]
                    writer.add_scalar('stage1_val/dice', sel_dice, ep)
                    logging.info("stage1 epoch %d val curve: dice=%.2f%%" % (ep, sel_dice * 100))
                else:
                    sel_dice = row["dice"]
                if sel_dice > best_dice:
                    best_dice, best_ep = sel_dice, ep
                    p = os.path.join(snapshot_path, 'stage1_best.pth')
                    torch.save(model.ea.state_dict(), p)
                    logging.info("save stage1 best (val dice=%.2f%% @ep%d) to %s"
                                 % (sel_dice * 100, ep, p))
                if str(args.stage1_pick) != 'best' and ep == int(args.stage1_pick):
                    p = os.path.join(snapshot_path, 'stage1_ep%d.pth' % ep)
                    torch.save(model.ea.state_dict(), p)
                    logging.info("save stage1 to %s" % p)
        iterator.close()
        for g in opt.param_groups:
            del g
        if str(args.stage1_pick) == 'best':
            stage1_ckpt = os.path.join(snapshot_path, 'stage1_best.pth')
            logging.info("stage2 will use best stage1: %s (val dice=%.2f%% @ep%d)"
                         % (stage1_ckpt, best_dice * 100, best_ep))
        else:
            stage1_ckpt = os.path.join(snapshot_path, 'stage1_ep%d.pth' % int(args.stage1_pick))
            if not os.path.exists(stage1_ckpt):
                stage1_ckpt = os.path.join(snapshot_path, 'stage1_ep%d.pth' % s1_epochs)

    # ---------------- Stage 2: rectified flow 精修 ----------------
    if stage in ('both', '2'):
        if stage == '2':
            stage1_ckpt = args.ccfr_stage1_ckpt
        if not stage1_ckpt or not os.path.exists(stage1_ckpt):
            logging.error("stage1 checkpoint 不存在: %s" % stage1_ckpt)
            writer.close()
            return
        model.ea.load_state_dict(torch.load(stage1_ckpt, map_location='cuda'))
        for p in model.ea.parameters():
            p.requires_grad_(False)
        logging.info("load stage1 from %s, frozen" % stage1_ckpt)

        params = [p for n, p in model.named_parameters() if not n.startswith('ea.')]
        opt = torch.optim.AdamW(params, lr=args.stage2_lr, weight_decay=args.stage2_wd)
        K = args.flow_steps
        curve, curve_rows = os.path.join(snapshot_path, 'stage2_curve.csv'), []
        for epoch_num in range(args.stage2_epochs):
            model.train()
            model.ea.eval()  # 冻结 BN 统计，避免 stage1 特征漂移
            opt.zero_grad()
            for i, batch in enumerate(trainloader):
                image, label = batch['image'].cuda(), batch['label'].cuda().long()
                gt = label.float().unsqueeze(1)
                with torch.no_grad():
                    s1 = model.stage1(image)
                mask = model.build_mask(s1['pc'], s1['e'])
                cond = model.build_cond(s1['region'], s1['e'], s1['priors'], mask)
                x0 = logit(s1['pc'])
                x1 = logit(gt)

                B = image.shape[0]
                t = torch.rand(B, device=image.device)
                xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
                v = model.velocity(xt, t, cond, s1['priors'])
                # 论文 Eq.(10)：Lv = E‖M ⊙ (Vθ − α(X1−X0))‖²，M 在范数内 → 权重是 M²
                l_vel = ((mask * (v - args.flow_alpha * (x1 - x0))) ** 2).mean()

                # 推理路径 K 步 Euler，得到 P_K 再加最终监督
                x = x0
                for k in range(K):
                    tk = torch.full((B,), k / K, device=image.device)
                    x = x + mask * model.velocity(x, tk, cond, s1['priors']) / K
                l_bce = F.binary_cross_entropy_with_logits(x, gt)
                l_dice = _soft_dice(torch.sigmoid(x), gt)
                loss = args.w_vel * l_vel + args.w_bce * l_bce + args.w_dice * l_dice
                (loss / args.grad_accum).backward()
                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 12.0)
                    opt.step()
                    opt.zero_grad()
                if epoch_num % 5 == 0 and i % 20 == 0:
                    writer.add_scalar('stage2/loss', loss.item(), epoch_num * len(trainloader) + i)
                    writer.add_scalar('stage2/loss_vel', l_vel.item(), epoch_num * len(trainloader) + i)
            logging.info("stage2 epoch %d: loss=%.4f vel=%.4f bce=%.4f dice=%.4f"
                         % (epoch_num + 1, loss.item(), l_vel.item(), l_bce.item(), l_dice.item()))

            ep = epoch_num + 1
            if args.eval_test_every > 0 and (ep % args.eval_test_every == 0 or ep == args.stage2_epochs):
                _, summary = _eval(model, testloader, args.root_path, stage=2)
                row = _curve_row(ep, summary)
                curve_rows.append(row)
                _write_csv(curve, curve_rows, ["epoch", "dice", "assd", "hd95", "hd", "abs_delta_area"])
                _log_curve("stage2", row)
                for k in ("dice", "assd", "hd"):
                    writer.add_scalar('stage2_curve/' + k, row[k], ep)
                if ep == args.stage2_epochs:
                    torch.save(model.state_dict(), os.path.join(snapshot_path, 'flow_ep%d.pth' % ep))
            writer.add_scalar('stage2/lr', opt.param_groups[0]['lr'], ep)

        # 终点重推理 + 逐切片指标
        records, summary = _eval(model, testloader, args.root_path, stage=2)
        _write_csv(os.path.join(snapshot_path, 'metrics.csv'), records,
                   ["case", "dice", "assd", "hd95", "hd", "delta_area"])
        logging.info("stage2 final: dice=%.2f%%±%.2f%% assd=%.3f hd=%.2f hd95=%.2f |dA|=%.1f" % (
            summary["dice"]["mean"] * 100, summary["dice"]["std"] * 100,
            summary["assd"]["mean"], summary["hd"]["mean"], summary["hd95"]["mean"],
            summary["abs_delta_area"]["mean"]))
        torch.save(model.state_dict(), os.path.join(snapshot_path, 'ccfr_final.pth'))

    writer.close()
    return "Training Finished!"
