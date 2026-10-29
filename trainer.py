import argparse
import csv
import functools
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from metrics_carotid import original_grid_pair

def _worker_init(worker_id, seed):
    # 必须是模块级函数，Windows spawn 下才能被序列化到子进程
    random.seed(seed + worker_id)


@torch.no_grad()
def _eval_test(model, testloader, root_path):
    """在测试集上逐样本计算四项指标，返回 (records, summary)"""
    from metrics_carotid import compute_case_metrics, original_grid_pair, summarize
    was_training = model.training
    model.eval()
    records = []
    for batch in testloader:
        logits = model(batch['image'].cuda())

        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        pred_o, gt_o, spacing = original_grid_pair(pred, root_path, batch['case'][0])
        record = compute_case_metrics(pred_o, gt_o, spacing=spacing)
        record['case'] = batch['case'][0]
        records.append(record)
    if was_training:
        model.train()
    return records, summarize(records)


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    if args.dataset == 'Carotid':
        from datasets.dataset_carotid import CarotidDataset
        split_dir = os.path.join(args.list_dir, args.subset)
        db_train = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "train.txt"),
                                  split="train", img_size=args.img_size,
                                  use_wavelet=bool(args.use_wavelet))
        test_set = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "test.txt"),
                                  split="test", img_size=args.img_size,
                                  use_wavelet=bool(args.use_wavelet))
        testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
        curve_path = os.path.join(snapshot_path, 'test_curve.csv')
        curve_rows = []
    else:
        testloader = None
        curve_path = None
        db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, persistent_workers=args.num_workers > 0,
                             worker_init_fn=functools.partial(_worker_init, seed=args.seed))
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    grad_accum = max(1, getattr(args, 'grad_accum', 1))
    steps_per_epoch = (len(trainloader) + grad_accum - 1) // grad_accum
    max_iterations = args.max_epochs * steps_per_epoch  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        optimizer.zero_grad()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            (loss / grad_accum).backward()

            # 梯度累积：攒够 grad_accum 个 micro-batch（或到 epoch 末尾）才更新一次
            if (i_batch + 1) % grad_accum != 0 and (i_batch + 1) != len(trainloader):
                continue
            optimizer.step()
            optimizer.zero_grad()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 每 epoch 记录所有融合门控 gamma 的绝对值最大值，用于判断门控是否已经在学
        gamma_now = {n: p.detach().abs().max().item()
                     for n, p in model.named_parameters() if "gamma" in n}
        if gamma_now:
            logging.info("epoch %d gamma absmax: %s" % (
                epoch_num + 1, {k: round(v, 6) for k, v in gamma_now.items()}))
            for k, v in gamma_now.items():
                writer.add_scalar('fusion/' + k, v, epoch_num + 1)

        if testloader is not None and args.eval_test_every > 0 and (
                (epoch_num + 1) % args.eval_test_every == 0 or epoch_num >= max_epoch - 1):
            _, summary = _eval_test(model, testloader, args.root_path)
            curve_rows.append({
                "epoch": epoch_num + 1,
                "dice": summary["dice"]["mean"],
                "assd": summary["assd"]["mean"],
                "hd95": summary["hd95"]["mean"],
                "abs_delta_area": summary["abs_delta_area"]["mean"],
            })
            with open(curve_path, "w", newline="", encoding="utf-8") as f:
                curve_writer = csv.DictWriter(
                    f, fieldnames=["epoch", "dice", "assd", "hd95", "abs_delta_area"])
                curve_writer.writeheader()
                curve_writer.writerows(curve_rows)
            row = curve_rows[-1]
            logging.info("epoch %d test curve: dice=%.2f%% assd=%.3f hd95=%.2f |dA|=%.1f" % (
                epoch_num + 1, row["dice"] * 100, row["assd"], row["hd95"], row["abs_delta_area"]))

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()

    if args.dataset == 'Carotid':
        from torch.utils.data import DataLoader as _DL
        from metrics_carotid import compute_case_metrics, original_grid_pair, summarize
        test_set = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "test.txt"),
                                  split="test", img_size=args.img_size,
                                  use_wavelet=bool(args.use_wavelet))
        testloader = _DL(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
        model.eval()
        records = []
        with torch.no_grad():
            for batch in testloader:
                logits = model(batch['image'].cuda())
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                pred_o, gt_o, spacing = original_grid_pair(pred, args.root_path, batch['case'][0])
                record = compute_case_metrics(pred_o, gt_o, spacing=spacing)
                record['case'] = batch['case'][0]
                records.append(record)
        summary = summarize(records)
        with open(os.path.join(snapshot_path, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["case", "dice", "assd", "hd95", "delta_area"])
            writer_csv.writeheader()
            writer_csv.writerows(records)
        logging.info("test summary: dice=%.2f%%±%.2f%% assd=%.3f±%.3f hd95=%.2f±%.2f "
                     "dA=%.1f±%.1f |dA|=%.1f" % (
                         summary["dice"]["mean"] * 100, summary["dice"]["std"] * 100,
                         summary["assd"]["mean"], summary["assd"]["std"],
                         summary["hd95"]["mean"], summary["hd95"]["std"],
                         summary["delta_area"]["mean"], summary["delta_area"]["std"],
                         summary["abs_delta_area"]["mean"]))
    return "Training Finished!"