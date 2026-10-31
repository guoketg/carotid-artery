import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Carotid', help='experiment_name')
parser.add_argument('--subset', type=str,
                    default='SPARC', choices=['SPARC', 'Zhongnan'], help='carotid subset')
parser.add_argument('--num_workers', type=int, default=1, help='dataloader workers, keep 1 on Windows')
parser.add_argument('--eval_test_every', type=int, default=5,
                    help='evaluate test set every N epochs to draw the epoch-vs-metric curve (0=off)')
parser.add_argument('--list_dir', type=str,
                    default='', help='list dir；留空用数据集默认，传 ./lists/lists_Carotid_dev 即快筛子集')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--grad_accum', type=int, default=1,
                    help='梯度累积步数；等效 batch = batch_size * grad_accum（显存进入换页区时用）')
parser.add_argument('--use_wavelet', type=int, default=0,
                    help='1 = build 3 channels (raw / wavelet denoised / detail energy)')
parser.add_argument('--fusion', type=str, default='none', choices=['none', 'se', 'cross'],
                    help='feature fusion between main and wavelet branch')
parser.add_argument('--arch', type=str, default='baseline',
                    choices=['baseline', 'dual', 'split', 'unet', 'xnet', 'gobletnet'],
                    help='baseline = TransUNet / dual = 双骨干交错 + 四级双向融合 (D11) / '
                         'split = 局部(R50)与全局(ViT-B/16)彻底分离、14² 单点融合 / '
                         'unet|xnet|gobletnet = 纯卷积复现网络（各自独立训练，与 baseline 互不混用）')
parser.add_argument('--levels', type=str, default='1,2,3,4',
                    help='dual 融合接点：1=112²/64, 2=56²/256, 3=28²/512, 4=14²/768')
parser.add_argument('--fuse_op', type=str, default='auto',
                    choices=['auto', 'window', 'global', 'gate', 'concat', 'deform', 'aff', 'frc'],
                    help='auto = 112/56 窗内 + 28/14 全局；deform = 可变形大核；aff = MS-CAM；'
                         'frc = Syn-Net FrCFusion（DCT 频率通道注意力 + 空间支）')
parser.add_argument('--window', type=int, default=8, help='112² 接点的窗口边长')
parser.add_argument('--dk_ratio', type=int, default=2, help='注意力 d_k = C / dk_ratio')
parser.add_argument('--bidir', type=int, default=1, help='1 = 双向融合；0 = 只侧→主（R7 对照）')
parser.add_argument('--cross_scale', type=int, default=0, help='1 = 加 14²→28²→56²→112² 跨尺度回流')
parser.add_argument('--side_pretrained', type=int, default=1, help='1 = 侧路复用 ImageNet21k R50 权重')
parser.add_argument('--tag', type=str, default='', help='输出目录附加后缀，用于区分同类实验')
parser.add_argument('--heads', type=int, default=4, help='attention heads of WGA')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_layers', type=int, default=12,
                    help='ViT 编码器层数；0 = 去掉全局 Transformer 段（纯 R50 局部支拆支实验 L）')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--main_input', type=str, default='raw', choices=['raw', 'll'],
                    help='dual 主路输入：raw = 原图复制 3 通道（现状）；ll = 小波 LL 低频近似图（XNet 式低频/高频双流，高频全在侧路）')
parser.add_argument('--edge_in', type=int, default=0,
                    help='1 = 边缘特征融合·输入级：stem 多吃一张边缘图（baseline 拼到 3 通道后；unet 拼到 1 通道后）')
parser.add_argument('--edge_op', type=str, default='wavelet',
                    choices=['wavelet', 'wavelet_lam', 'energy', 'sobel', 'laplacian'],
                    help='输入级边缘图算子（仅 --arch unet --edge_in 1 可换；baseline 固定 wavelet λ=0）：'
                         'wavelet = Haar 高频 IDWT 重建 / wavelet_lam = 再加 λ·LL 回灌 / '
                         'energy = 细节能量图 / sobel / laplacian')
parser.add_argument('--ddbe', type=int, default=0,
                    help='1~3 = 解码器第 N 级输出（28/56/112/224，level3=112×112/64ch）挂 BDFR-Net 式 DDBE；'
                         '4 = 四级全挂 DDBE×4（BDFR-Net 完整版）；0 = 关（仅 baseline）')
parser.add_argument('--acam', type=int, default=0,
                    help='1 = 每个 decoder 级对 skip 特征做 ACAM 跨级语义对齐（ddbefull = --ddbe 4 --acam 1；仅 baseline）')
parser.add_argument('--hf_branch', type=int, default=0,
                    help='1 = GobletNet 式：轻量 HF 支 + FAM 与 skip 逐级互标定（仅 baseline）')
parser.add_argument('--hf_lambda', type=float, default=0.3,
                    help='HF 图低频回灌权重 λ（GobletNet Eq.5：H=HL+LH+HH+λ·LL，论文最优 0~0.3）')
args = parser.parse_args()

if args.ddbe and args.arch != 'baseline':
    parser.error('--ddbe 只接了 baseline 解码器（dual 的融合在编码器接点，未接 DDBE）')
if not 0 <= args.ddbe <= 4:
    parser.error('--ddbe 取值 0~4')
if args.acam and args.arch != 'baseline':
    parser.error('--acam 只接了 baseline 解码器（与 --ddbe 同挂载面）')
if args.hf_branch and args.arch != 'baseline':
    parser.error('--hf_branch 只接了 baseline 解码器（FAM 挂在 skip 上，dual 未接）')
if not 0 <= args.hf_lambda <= 1:
    parser.error('--hf_lambda 取值 0~1（GobletNet 论文最优区间 0~0.3）')
if not 0 <= args.vit_layers <= 12:
    parser.error('--vit_layers 取值 0~12（预训练只含 12 层权重，0 = 去掉全局 Transformer 段）')
if args.edge_op != 'wavelet' and not (args.arch == 'unet' and args.edge_in):
    parser.error('--edge_op 只在 --arch unet --edge_in 1 时生效（baseline 的 edge_in 固定 wavelet λ=0 口径）')
if args.arch in ('unet', 'xnet', 'gobletnet'):
    # 三个独立复现网络在 forward 内自算小波输入，数据集必须给 1 通道原图
    args.use_wavelet = 0
    if args.ddbe or args.acam or args.hf_branch:
        parser.error('--ddbe/--acam/--hf_branch 只接了 baseline 解码器，'
                     'unet/xnet/gobletnet 是独立复现网络')
    if args.edge_in and args.arch != 'unet':
        parser.error('--edge_in 只在 unet 上接了（xnet/gobletnet 自带频率支）')
    if args.fusion != 'none':
        parser.error('--fusion 只接了 baseline，unet/xnet/gobletnet 是独立复现网络')
    if args.main_input != 'raw':
        parser.error('--main_input 只接了 dual')
if args.arch == 'split':
    # split 自带完整两路编码器，输入只给 1 通道原图
    args.use_wavelet = 0
    if args.edge_in or args.fusion != 'none' or args.ddbe or args.acam or args.hf_branch:
        parser.error('split 未接 --edge_in/--fusion/--ddbe/--acam/--hf_branch')
    if args.main_input != 'raw':
        parser.error('--main_input 只接了 dual')
    if args.levels != '1,2,3,4':
        parser.error('split 固定 14² 单点融合，不接受 --levels')


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Carotid': {
            'root_path': r'E:\mkl\code\datasets\Carotid2D',
            'list_dir': './lists/lists_Carotid2D',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = args.list_dir or dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    if args.vit_layers != 12:
        # 拆支实验 L（--vit_layers 0）必须与 12 层的 A0 分目录
        snapshot_path += '_vitL' + str(args.vit_layers)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    if args.grad_accum > 1:
        snapshot_path += '_ga' + str(args.grad_accum)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    if dataset_name == 'Carotid':
        snapshot_path = snapshot_path + '_' + args.subset
        if args.use_wavelet:
            snapshot_path += '_w3'
        if args.edge_in:
            snapshot_path += '_edge'
        if args.fusion != 'none':
            snapshot_path += '_' + args.fusion
        if args.ddbe:
            # 快照名必须反映真实配置（status.md 3.3：名不副实 = 新旧权重混目录）
            snapshot_path += '_ddbe' + str(args.ddbe)
        if args.acam:
            snapshot_path += '_acam'
        if args.hf_branch:
            # λ 量化到 0.1 粒度进快照名（λ=0.3 → _hfbr3）
            snapshot_path += '_hfbr' + str(int(round(args.hf_lambda * 10)))
        if args.arch in ('unet', 'xnet', 'gobletnet'):
            snapshot_path += '_' + args.arch
            if args.arch == 'unet' and args.edge_in:
                # 边缘算子横扫：五种算子各自独立目录
                snapshot_path += '_eo' + args.edge_op
            if args.arch == 'gobletnet' and abs(args.hf_lambda - 0.3) > 1e-9:
                # λ 非默认 0.3 时进快照名（λ=0.1 → _lam01），避免同目录混权重
                snapshot_path += '_lam' + str(args.hf_lambda).replace('.', '')
            if args.tag:
                snapshot_path += '_' + args.tag
        if args.arch == 'dual':
            # 双骨干自己从原图算子带，数据集只给 1 通道
            args.use_wavelet = 0
            snapshot_path += '_dual_L' + args.levels.replace(',', '') + '_' + args.fuse_op
            if args.cross_scale:
                snapshot_path += '_cs'
            if not args.bidir:
                snapshot_path += '_uni'
            if args.main_input=='ll':
                snapshot_path += '_mainLL'
            if not args.side_pretrained:
                snapshot_path += '_sp0'
            if args.tag:
                snapshot_path += '_' + args.tag
        if args.arch == 'split':
            # 局部/全局分离 + 14² 单点融合；算子名必须进目录（同一 arch 多算子对照）
            snapshot_path += '_split_' + args.fuse_op
            if args.tag:
                snapshot_path += '_' + args.tag
        if 'lists_Carotid_dev' in args.list_dir.replace('\\', '/'):
            # 快筛子集的结果绝不能和全量结果混在一起
            snapshot_path += '_dev'
        if 'lists_Carotid2D' in args.list_dir.replace('\\', '/'):
            snapshot_path+='_c2d'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.transformer.num_layers = args.vit_layers
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    if args.arch == 'dual':
        from networks.dual_encoder_model import DualEncoderTransUNet
        net = DualEncoderTransUNet(config_vit, img_size=args.img_size,
                                   num_classes=config_vit.n_classes,
                                   levels=args.levels, fuse_op=args.fuse_op,
                                   window=args.window, dk_ratio=args.dk_ratio,
                                   bidir=bool(args.bidir),
                                   side_pretrained=bool(args.side_pretrained),
                                   cross_scale=bool(args.cross_scale),
                                   main_input=args.main_input,
                                   edge_in=bool(args.edge_in)).cuda()
    elif args.arch == 'split':
        # 局部 R50 与全局 ViT-B/16 并联，仅 14² 单点融合
        from networks.split_encoder_model import SplitLocalGlobalTransUNet
        net = SplitLocalGlobalTransUNet(config_vit, img_size=args.img_size,
                                        num_classes=config_vit.n_classes,
                                        fuse_op=args.fuse_op, window=args.window,
                                        dk_ratio=args.dk_ratio,
                                        bidir=bool(args.bidir)).cuda()
    elif args.arch == 'unet':
        # 容量对齐基线：与 xnet/gobletnet 同积木的纯卷积 U-Net
        from networks.unet import UNet
        net = UNet(in_channels=1, num_classes=args.num_classes,
                   edge_in=bool(args.edge_in), edge_op=args.edge_op).cuda()
    elif args.arch == 'xnet':
        # 小波低/高频双分支 + X 形融合，返回 (out_lf, out_hf) 双输出
        from networks.xnet import XNet
        net = XNet(in_channels=1, num_classes=args.num_classes).cuda()
    elif args.arch == 'gobletnet':
        # 语义 + HF 双编码器，FAM 四级融合，单输出
        from networks.gobletnet import GobletNet
        net = GobletNet(in_channels=1, num_classes=args.num_classes,
                        hf_lambda=args.hf_lambda).cuda()
    elif args.fusion != 'none':
        from networks.wavelet_model import WaveletTransUNet
        net = WaveletTransUNet(config_vit, img_size=args.img_size,
                               num_classes=config_vit.n_classes,
                               fusion=args.fusion, heads=args.heads).cuda()
    else:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes,
                      edge_in=bool(args.edge_in), ddbe_level=args.ddbe, acam=bool(args.acam),
                      hf_branch=bool(args.hf_branch), hf_lambda=args.hf_lambda).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'Synapse': trainer_synapse, 'Carotid': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)