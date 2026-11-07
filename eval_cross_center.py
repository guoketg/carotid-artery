# coding=utf-8
"""跨中心零样本评测：拿某中心训练出的权重，直接在另一中心的 test 集上推理。

为什么单独写一个脚本：train.py 的评测写在 trainer 里、和训练循环耦合，
跨中心评测需要「换 test list 不换权重」，且不加载 ImageNet 预训练（权重
会被 checkpoint 全量覆盖，省掉 npz 依赖）。

口径与训练内评测完全一致（§0）：
  - 224×224 输入，单通道原图；预测 argmax 后经 original_grid_pair 回到原
    分辨率网格，spacing 取自 NIfTI 头；
  - 指标 dice / assd / hd(最大表面距离) / hd95 / |ΔArea|，逐切片平均。

用法：
  python eval_cross_center.py --ckpt <快照目录或 .pth> --arch baseline \
      --train_subset SPARC --test_subset Zhongnan --epoch 199
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics_carotid import compute_case_metrics, original_grid_pair, summarize
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg

ROOT_PATH = r'E:\mkl\code\datasets\Carotid2D'
LIST_DIR = './lists/lists_Carotid2D'
OUT_CSV = os.path.join('logs', 'queue', 'cross_center_eval.csv')
CSV_HEADER = ['ckpt', 'arch', 'train_subset', 'test_subset', 'epoch',
              'dice', 'assd', 'hd95', 'hd', 'abs_delta_area', 'n']


def _primary_out(outputs):
    """FDS 返回 dict；XNet 返回双输出（取低频主支）；单输出原样返回。"""
    if isinstance(outputs, dict):
        return outputs['out']
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def _wait_free_mem(need_gb=2.5, timeout_s=1800):
    """训练队列还在跑时，等显存腾出足够余量再建模型，避免把训练 job 挤爆。"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        free, total = torch.cuda.mem_get_info()
        if free >= need_gb * 1024 ** 3:
            print('[eval] free %.1f / %.1f GB' % (free / 1024 ** 3, total / 1024 ** 3))
            return True
        print('[eval] free %.1f GB < %.1f GB, wait 30s' % (free / 1024 ** 3, need_gb))
        time.sleep(30)
    return False


def build_model(args):
    """与 train.py 的构造分支保持一致（未覆盖的 arch 直接报错，免得出错配权重）。"""
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3
    config_vit.transformer.num_layers = 12
    config_vit.patches.grid = (int(args.img_size / 16), int(args.img_size / 16))
    if args.arch == 'baseline':
        return ViT_seg(config_vit, img_size=args.img_size, num_classes=2,
                       edge_in=bool(args.edge_in), edge_op=args.edge_op,
                       wavelet_basis=args.wavelet_basis,
                       ddbe_level=args.ddbe,
                       acam=bool(args.acam), hf_branch=bool(args.hf_branch),
                       hf_lambda=args.hf_lambda)
    if args.arch == 'unet':
        from networks.unet import UNet
        return UNet(in_channels=1, num_classes=2, edge_in=bool(args.edge_in),
                    edge_op=args.edge_op, wavelet_basis=args.wavelet_basis,
                    ddbe_level=args.ddbe,
                    acam=bool(args.acam), asr=bool(args.asr))
    if args.arch == 'wmren':
        from networks.unet_wmren import UNetWMREN
        return UNetWMREN(in_channels=1, num_classes=2)
    raise SystemExit('eval_cross_center.py 未接 arch=%s（当前只覆盖 baseline/unet/wmren）' % args.arch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='快照目录（自动取 epoch_<epoch>.pth）或直接给 .pth 路径')
    parser.add_argument('--arch', type=str, default='baseline',
                        choices=['baseline', 'unet', 'wmren'])
    parser.add_argument('--train_subset', type=str, default='SPARC')
    parser.add_argument('--test_subset', type=str, default='Zhongnan')
    parser.add_argument('--epoch', type=int, default=199)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--edge_in', type=int, default=0)
    parser.add_argument('--edge_op', type=str, default='wavelet')
    parser.add_argument('--wavelet_basis', type=str, default='haar',
                        help='与训练时保持一致（haar/db2/db4/sym4/coif1/bior1.5）')
    parser.add_argument('--ddbe', type=int, default=0)
    parser.add_argument('--acam', type=int, default=0)
    parser.add_argument('--asr', type=int, default=0)
    parser.add_argument('--hf_branch', type=int, default=0)
    parser.add_argument('--hf_lambda', type=float, default=0.3)
    parser.add_argument('--out', type=str, default=OUT_CSV)
    args = parser.parse_args()

    ckpt = args.ckpt
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, 'epoch_%d.pth' % args.epoch)
    if not os.path.exists(ckpt):
        raise SystemExit('checkpoint 不存在: %s' % ckpt)

    if not _wait_free_mem():
        raise SystemExit('显存一直不够，放弃本次评测')

    net = build_model(args).cuda()
    state = torch.load(ckpt, map_location='cpu')
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()

    from datasets.dataset_carotid import CarotidDataset
    test_set = CarotidDataset(root=ROOT_PATH,
                              list_file=os.path.join(LIST_DIR, args.test_subset, 'test.txt'),
                              split='test', img_size=args.img_size, use_wavelet=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print('[eval] %s | %s/%s | %d cases' % (os.path.basename(os.path.dirname(ckpt)),
                                            args.train_subset, args.test_subset, len(test_set)))

    records = []
    with torch.no_grad():
        for batch in loader:
            logits = _primary_out(net(batch['image'].cuda()))
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            pred_o, gt_o, spacing = original_grid_pair(pred, ROOT_PATH, batch['case'][0])
            rec = compute_case_metrics(pred_o, gt_o, spacing=spacing)
            rec['case'] = batch['case'][0]
            records.append(rec)
    s = summarize(records)
    row = [os.path.basename(os.path.dirname(ckpt)), args.arch, args.train_subset,
           args.test_subset, args.epoch,
           round(s['dice']['mean'] * 100, 2), round(s['assd']['mean'], 4),
           round(s['hd95']['mean'], 4), round(s['hd']['mean'], 4),
           round(s['abs_delta_area']['mean'], 4), s['n_cases']]
    print('[eval] dice=%.2f assd=%.4f hd95=%.4f hd=%.4f |dA|=%.4f n=%d'
          % (row[5], row[6], row[7], row[8], row[9], row[10]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)
    with open(args.out, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow(row)


if __name__ == '__main__':
    main()
