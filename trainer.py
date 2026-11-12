import argparse
import csv
import functools
import logging
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _primary_out(outputs):
    """统一取主输出：FDS 网络返回 {'out', 'aux'}；XNet 返回 (logits_lf, logits_hf)
    双输出，评估/可视化统一取低频主支；单输出网络原样返回。"""
    if isinstance(outputs, dict):
        return outputs['out']
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


class WeaveSegFiveTermLoss(nn.Module):
    """WeaveSeg（ICCV 2025）Eq.(12) 的五项损失，权重取论文的经验值 1.0/0.5/0.5/0.3/1.0：

        L = 1.0·L_mask + 0.5·L_fg + 0.5·L_bg + 0.3·L_tri + 1.0·L_uncert

    逐项对应关系（论文原文位置见 docs/papers_txt/...WeaveSeg...txt L437–443）：
      - L_mask：主分割 BCE；
      - L_fg / L_bg：只在前景 / 背景像素上取的 BCE（论文里是 Sf、Sb 两个辅助前景背景预测，
        本项目网络只有一个输出，故按区域切分主预测来等价实现）；
      - L_tri：三项分解（内部 / 边界带 / 背景）的软标签 BCE，边界带目标取 0.5；
      - L_uncert：Eq.(11) 的 MSE(Su, U*)，Su 由 1×1 头在 logits 上预测，U* 由 GT 边界生成。

    **两处必须标注的还原缺口**（正文没有给公式，细节在补充材料，本项目只有正文）：
      1. L_tri 的具体形式论文只写了 "trilateral decomposition-inspired loss"，未给公式，
         这里用"内部=1 / 边界带=0.5 / 背景=0"的三值软标签 BCE 作为可复现的替代；
      2. 目标不确定图 U* 的生成只在 Figure 4 的图注里描述（GT → 灰度三值图 → 梯度幅值 →
         腐蚀/膨胀 → 不确定性行为图），无公式，这里用 7×7 均值池化后的 `1-|2b-1|`
         实现：b 是局部前景比例，边界处 b≈0.5 故 U*≈1，前景内部/背景 U*≈0。

    另外注意：论文只用 BCE、不含 Dice；本项目 §0 是 0.5CE+0.5Dice，所以本损失与
    ce_dice 的差异同时包含"换损失族"与"加四项"，作对比时要按这两点解释。
    """

    def __init__(self, lam_mask=1.0, lam_fg=0.5, lam_bg=0.5, lam_tri=0.3,
                 lam_uncert=1.0, band=5, blur=7):
        super(WeaveSegFiveTermLoss, self).__init__()
        self.lam_mask = lam_mask
        self.lam_fg = lam_fg
        self.lam_bg = lam_bg
        self.lam_tri = lam_tri
        self.lam_uncert = lam_uncert
        self.band = band
        self.blur = blur
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # Su 的 1×1 头；挂在损失侧，避免污染模型 state_dict（下游 eval 直接 load 不受影响）。
        # 零初始化：起步 Su≡0.5，梯度不回灌主干，先让头自己学会边界，避免 λ=1 的不确定项
        # 在训练初期用随机头的噪声污染分割梯度。
        self.unc_head = nn.Conv2d(2, 1, kernel_size=1)
        nn.init.zeros_(self.unc_head.weight)
        nn.init.zeros_(self.unc_head.bias)

    def _targets(self, label):
        y = label.float()
        if y.dim() == 3:
            y = y.unsqueeze(1)
        k, p = self.band, self.band // 2
        outer = F.max_pool2d(y, k, 1, p)                      # 膨胀
        inner = -F.max_pool2d(-y, k, 1, p)                    # 腐蚀
        band = (outer - inner).clamp(0.0, 1.0)
        tri = y * (1.0 - band) + 0.5 * band                   # 内部 1 / 边界 0.5 / 背景 0
        kb = self.blur
        local = F.avg_pool2d(y, kb, 1, kb // 2)
        ustar = (1.0 - (2.0 * local - 1.0).abs()).clamp(0.0, 1.0)
        return y, outer, band, tri, ustar

    def forward(self, out, label, maps=None):
        """maps：weaveseg 网络返回的 {'sf','sb','su'} 三个上下文 logits。

        给 maps 时，L_fg / L_bg / L_uncert 作用在网络自己的解耦头上（论文口径）；
        不给时退化为旧口径（复用主预测 + 损失侧 1×1 头），供不带上下文头的
        网络（例如 unet --loss weave5）继续可用。
        """
        y, outer, band, tri, ustar = self._targets(label)
        logit = out[:, 1:2]
        pix = self.bce(logit, y)
        l_mask = pix.mean()
        bgw = 1.0 - outer
        if maps is not None:
            l_fg = (self.bce(maps['sf'], y) * y).sum() / y.sum().clamp(min=1.0)
            l_bg = (self.bce(maps['sb'], 1.0 - y) * bgw).sum() / bgw.sum().clamp(min=1.0)
            su = torch.sigmoid(maps['su'])
        else:
            l_fg = (pix * y).sum() / y.sum().clamp(min=1.0)
            l_bg = (pix * bgw).sum() / bgw.sum().clamp(min=1.0)
            su = torch.sigmoid(self.unc_head(out))
        l_tri = self.bce(logit, tri).mean()
        l_uncert = ((su - ustar) ** 2).mean()
        total = (self.lam_mask * l_mask + self.lam_fg * l_fg + self.lam_bg * l_bg
                 + self.lam_tri * l_tri + self.lam_uncert * l_uncert)
        return total, l_mask


class UncertaintyWeightedCEDice(nn.Module):
    """Kendall & Gal (CVPR'18) 同方差不确定性加权：把 CE / Dice 两项的权重也变成可学习量。

        L = 0.5·e^{-s_ce}·L_CE + 0.5·s_ce + 0.5·e^{-s_dice}·L_Dice + 0.5·s_dice

    其中 s = log σ² 是两个标量参数（初始 0）。初始 0 时 e^{-s}=1，等价于 §0 的
    0.5·CE + 0.5·Dice，因此与 ce_dice 的单变量差异就是"损失权重是否自适应"。
    注意：WMREN 原论文只给了固定的 0.4/0.6，没有不确定项，这是额外的对照实验。
    """

    def __init__(self, ce_loss, dice_loss):
        super(UncertaintyWeightedCEDice, self).__init__()
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.log_var_ce = nn.Parameter(torch.zeros(()))
        self.log_var_dice = nn.Parameter(torch.zeros(()))

    def forward(self, out, label):
        l_ce = self.ce_loss(out, label[:].long())
        l_dice = self.dice_loss(out, label, softmax=True)
        loss = (0.5 * (torch.exp(-self.log_var_ce) * l_ce + self.log_var_ce)
                + 0.5 * (torch.exp(-self.log_var_dice) * l_dice + self.log_var_dice))
        return loss, l_ce


@torch.no_grad()
def _eval_test(model, testloader, root_path):
    """在测试集上逐样本计算四项指标，返回 (records, summary)"""
    from metrics_carotid import compute_case_metrics, original_grid_pair, summarize
    was_training = model.training
    model.eval()
    records = []
    for batch in testloader:
        logits = _primary_out(model(batch['image'].cuda()))
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
                                  use_wavelet=bool(args.use_wavelet),
                                  gamma_aug=bool(getattr(args, 'gamma_aug', 0)))
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
    use_domain_loss = bool(getattr(args, 'domain_loss', 0))
    loss_name = getattr(args, 'loss', 'ce_dice')
    aux_loss = None
    if use_domain_loss:
        from utils import BoundaryWeightedCELoss, area_consistency_loss, far_fp_loss
        ce_bw = BoundaryWeightedCELoss(lam=float(getattr(args, 'dl_lambda', 3.0)),
                                       radius=int(getattr(args, 'dl_radius', 2)))
        dl_area_w = float(getattr(args, 'dl_area_w', 0.05))
        dl_far_w = float(getattr(args, 'dl_far_w', 0.1))
        dl_margin = int(getattr(args, 'dl_margin', 10))

        def seg_loss(out, label):
            """领域适配：边界带加权 CE + Dice + 面积一致性 + 远处假阳性惩罚。"""
            l_ce = ce_bw(out, label.long())
            l_dice = dice_loss(out, label, softmax=True)
            l_area = area_consistency_loss(out, label)
            l_fp = far_fp_loss(out, label, margin=dl_margin)
            return (0.5 * l_ce + 0.5 * l_dice + dl_area_w * l_area
                    + dl_far_w * l_fp), l_ce
    elif loss_name == 'bce_wiou':
        from utils import BCEWIoULoss
        bce_wiou = BCEWIoULoss()

        def seg_loss(out, label):
            """Syn-Net 口径：mean(BCE + weighted IoU)，主输出与三级 FDS 共用。"""
            return bce_wiou(out, label), torch.zeros((), device=label.device)
    elif loss_name == 'focal':
        from utils import FocalLoss
        focal = FocalLoss()

        def seg_loss(out, label):
            """损失基线：0.5·Focal(γ=2,α=0.25) + 0.5·Dice。"""
            return (0.5 * focal(out, label) + 0.5 * dice_loss(out, label, softmax=True),
                    torch.zeros((), device=label.device))
    elif loss_name == 'tversky':
        from utils import TverskyLoss
        tversky = TverskyLoss()

        def seg_loss(out, label):
            """损失基线：Tversky(α=0.3,β=0.7) 单独使用（自身已含区域重合）。"""
            return tversky(out, label), torch.zeros((), device=label.device)
    elif loss_name == 'boundary':
        from utils import BoundaryLoss
        boundary = BoundaryLoss()

        def seg_loss(out, label):
            """损失基线：0.5·CE + 0.5·Dice + 0.5·Boundary（Kervadec 距离图加权）。"""
            l_ce = ce_loss(out, label[:].long())
            return (0.5 * l_ce + 0.5 * dice_loss(out, label, softmax=True)
                    + 0.5 * boundary(out, label)), l_ce
    elif loss_name == 'farfp':
        from utils import far_fp_loss
        far_w = float(getattr(args, 'far_w', 0.1))
        far_margin = int(getattr(args, 'far_margin', 10))

        def seg_loss(out, label):
            """损失改进 A：0.5CE+0.5Dice + far_w·远处假阳性项（单独归因 HD 尾部）。"""
            l_ce = ce_loss(out, label[:].long())
            return (0.5 * l_ce + 0.5 * dice_loss(out, label, softmax=True)
                    + far_w * far_fp_loss(out, label, margin=far_margin)), l_ce
    elif loss_name == 'bndsoft':
        from utils import boundary_soft_ce
        bnd_eps = float(getattr(args, 'bnd_eps', 0.25))
        bnd_radius = int(getattr(args, 'bnd_radius', 1))

        def seg_loss(out, label):
            """损失改进 C：边界带软标签 CE + 硬 Dice（单变量降低歧义带监督强度）。"""
            l_ce = boundary_soft_ce(out, label, eps=bnd_eps, radius=bnd_radius)
            return 0.5 * l_ce + 0.5 * dice_loss(out, label, softmax=True), l_ce
    elif loss_name in ('weave5', 'weave4'):
        # weave4 = 同一套五项里的 L_uncert 权重置 0，用来单变量判断不确定项本身值不值
        aux_loss = WeaveSegFiveTermLoss(
            lam_uncert=(1.0 if loss_name == 'weave5' else 0.0)).cuda()

        def seg_loss(out, label):
            """WeaveSeg Eq.(12)：L_mask + 0.5L_fg + 0.5L_bg + 0.3L_tri [+ L_uncert]。"""
            return aux_loss(out, label)
    elif loss_name == 'bce_iou_dice':
        def seg_loss(out, label):
            """FDFUNet 论文 Eq.(6)：0.5·BCE + 0.5·IoU + Dice（前景概率上的 soft 形式）。"""
            prob = torch.softmax(out, dim=1)[:, 1]
            tgt = label.float()
            if tgt.dim() == 4:
                tgt = tgt[:, 0]
            bce = F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), tgt)
            inter = (prob * tgt).sum()
            union = prob.sum() + tgt.sum() - inter
            iou = 1.0 - (inter + 1.0) / (union + 1.0)
            dice = 1.0 - (2.0 * inter + 1.0) / (prob.sum() + tgt.sum() + 1.0)
            return 0.5 * bce + 0.5 * iou + dice, bce
    elif loss_name == 'ce_dice46':
        def seg_loss(out, label):
            """论文 Eq.(14) 口径：0.4·CE + 0.6·Dice（无类别权重）。"""
            l_ce = ce_loss(out, label[:].long())
            return 0.4 * l_ce + 0.6 * dice_loss(out, label, softmax=True), l_ce
    elif loss_name == 'ce_dice_unc':
        aux_loss = UncertaintyWeightedCEDice(ce_loss, dice_loss).cuda()

        def seg_loss(out, label):
            """Kendall 同方差不确定性加权 CE+Dice（初始等价 0.5/0.5）。"""
            return aux_loss(out, label)
    else:
        def seg_loss(out, label):
            l_ce = ce_loss(out, label[:].long())
            return 0.5 * l_ce + 0.5 * dice_loss(out, label, softmax=True), l_ce
    # 优化器口径：sgd = 项目历次实验（momentum 0.9 + poly 0.9 衰减）；
    # adamw = BDFR-Net 论文口径（AdamW，lr/wd 由命令行给，恒定 lr，论文未给调度）
    optim_name = getattr(args, 'optim', 'sgd')
    if optim_name == 'adamw':
        params = list(model.parameters()) + (list(aux_loss.parameters()) if aux_loss is not None else [])
        optimizer = optim.AdamW(params, lr=base_lr, weight_decay=1e-4)
    elif optim_name == 'adam':
        # FDFUNet 论文口径：Adam，lr 由命令行给（1e-4），无 weight decay
        adam_params = list(model.parameters()) + (list(aux_loss.parameters()) if aux_loss is not None else [])
        optimizer = optim.Adam(adam_params, lr=base_lr, weight_decay=0)
    elif aux_loss is not None:
        # 损失侧的可学习参数（log σ² 或 Su 的 1×1 头）也要进优化器
        optimizer = optim.SGD(list(model.parameters()) + list(aux_loss.parameters()),
                              lr=base_lr, momentum=0.9, weight_decay=0.0001)
    else:
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
            aux = None
            maps = None
            if isinstance(outputs, dict):
                # FDS：主输出 + 三级辅助 logits（各自上采样到 GT 分辨率后再算损失）
                # weaveseg：额外带 sf/sb/su 三个上下文 logits，供 Eq.(12) 的后三项
                aux = outputs.get('aux')
                maps = {k: outputs[k] for k in ('sf', 'sb', 'su') if k in outputs} or None
                outputs = outputs['out']
            if isinstance(outputs, (tuple, list)):
                # XNet 双输出（低频/高频支）：每支各算一次分割损失，再对支数取平均
                loss = 0.0
                loss_ce = 0.0
                for out in outputs:
                    l, c = seg_loss(out, label_batch)
                    loss = loss + l
                    loss_ce = loss_ce + c
                loss = loss / len(outputs)
                loss_ce = loss_ce / len(outputs)
            elif maps is not None and aux_loss is not None:
                # weaveseg：Eq.(12) 的后三项作用在网络返回的 sf/sb/su 头上
                loss, loss_ce = aux_loss(outputs, label_batch, maps)
            else:
                loss, loss_ce = seg_loss(outputs, label_batch)
            if aux:
                hw = label_batch.shape[-2:]
                for aux_logits, w_aux in aux:
                    if aux_logits.shape[-2:] != hw:
                        aux_logits = F.interpolate(aux_logits, size=hw, mode='bilinear',
                                                   align_corners=False)
                    l_aux, _ = seg_loss(aux_logits, label_batch)
                    loss = loss + w_aux * l_aux
            (loss / grad_accum).backward()

            # 梯度累积：攒够 grad_accum 个 micro-batch（或到 epoch 末尾）才更新一次
            if (i_batch + 1) % grad_accum != 0 and (i_batch + 1) != len(trainloader):
                continue
            optimizer.step()
            optimizer.zero_grad()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            if optim_name == 'adamw':
                lr_ = base_lr          # 论文只给了 AdamW lr=1e-4，未给衰减策略 → 恒定 lr
            elif optim_name == 'adam':
                # FDFUNet 论文：cosine 退火，1e-4 → 1e-5，按 epoch 计算
                t = min(1.0, epoch_num / max(1, max_epoch - 1))
                lr_ = 1e-5 + 0.5 * (base_lr - 1e-5) * (1.0 + math.cos(math.pi * t))
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
                outputs = torch.argmax(torch.softmax(_primary_out(outputs), dim=1), dim=1, keepdim=True)
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

        if isinstance(aux_loss, UncertaintyWeightedCEDice):
            # 记录学到的 log σ²（权重 = 0.5·e^{-s}），用于判断自适应权重往哪边偏
            s_ce = aux_loss.log_var_ce.item()
            s_dice = aux_loss.log_var_dice.item()
            logging.info("epoch %d uncertainty: logvar_ce=%.4f logvar_dice=%.4f "
                         "(w_ce=%.4f w_dice=%.4f)" % (
                             epoch_num + 1, s_ce, s_dice,
                             0.5 * np.exp(-s_ce), 0.5 * np.exp(-s_dice)))

        if testloader is not None and args.eval_test_every > 0 and (
                (epoch_num + 1) % args.eval_test_every == 0 or epoch_num >= max_epoch - 1):
            _, summary = _eval_test(model, testloader, args.root_path)
            curve_rows.append({
                "epoch": epoch_num + 1,
                "dice": summary["dice"]["mean"],
                "assd": summary["assd"]["mean"],
                "hd95": summary["hd95"]["mean"],
                "hd": summary["hd"]["mean"],
                "abs_delta_area": summary["abs_delta_area"]["mean"],
            })
            with open(curve_path, "w", newline="", encoding="utf-8") as f:
                curve_writer = csv.DictWriter(
                    f, fieldnames=["epoch", "dice", "assd", "hd95", "hd", "abs_delta_area"])
                curve_writer.writeheader()
                curve_writer.writerows(curve_rows)
            row = curve_rows[-1]
            logging.info("epoch %d test curve: dice=%.2f%% assd=%.3f hd=%.2f hd95=%.2f |dA|=%.1f" % (
                epoch_num + 1, row["dice"] * 100, row["assd"], row["hd"],
                row["hd95"], row["abs_delta_area"]))

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
                logits = _primary_out(model(batch['image'].cuda()))
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                pred_o, gt_o, spacing = original_grid_pair(pred, args.root_path, batch['case'][0])
                record = compute_case_metrics(pred_o, gt_o, spacing=spacing)
                record['case'] = batch['case'][0]
                records.append(record)
        summary = summarize(records)
        with open(os.path.join(snapshot_path, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["case", "dice", "assd", "hd95", "hd", "delta_area"])
            writer_csv.writeheader()
            writer_csv.writerows(records)
        logging.info("test summary: dice=%.2f%%±%.2f%% assd=%.3f±%.3f hd=%.2f±%.2f hd95=%.2f±%.2f "
                     "dA=%.1f±%.1f |dA|=%.1f" % (
                         summary["dice"]["mean"] * 100, summary["dice"]["std"] * 100,
                         summary["assd"]["mean"], summary["assd"]["std"],
                         summary["hd"]["mean"], summary["hd"]["std"],
                         summary["hd95"]["mean"], summary["hd95"]["std"],
                         summary["delta_area"]["mean"], summary["delta_area"]["std"],
                         summary["abs_delta_area"]["mean"]))
    return "Training Finished!"