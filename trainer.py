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


def _edge_target(label, k=3):
    """由原 GT 掩膜经边缘算子得到边缘标签（与 CCFR Stage1 同口径，可复现）：

      1) 掩膜二值化（> 0.5，数据集返回的是 resize 后的 long 标签）；
      2) k×k 形态学梯度 dilate − erode，k=3 → 1 像素内环 + 1 像素外环的边缘带；
      3) clamp 到 [0,1] 作为边缘目标（224² 网格，与边缘解码器输出同网格）；
         训练时按主分割同形式 CE+Dice 用 (.long())，评测时按概率图用 (0/1)。
    """
    m = (label.float().unsqueeze(1) > 0.5).float()
    dil = F.max_pool2d(m, k, 1, k // 2)
    ero = -F.max_pool2d(-m, k, 1, k // 2)
    return (dil - ero).clamp(0.0, 1.0)


def _sdf_target(label, tau=12.0):
    """由 GT 掩膜逐样本算带符号距离场（像素单位，内部为正），裁剪到 ±tau。

    只依赖"GT 是二值掩膜"这一件事：没有面积/形状/尺度等群体统计量，换数据集不用改参数。
    224² 的欧氏距离变换每样本约 1 ms，CPU 上算（batch 4 可忽略）。
    """
    from scipy import ndimage as ndi
    m = (label.detach().cpu().numpy() > 0)
    out = np.empty(m.shape, dtype=np.float32)
    for i in range(m.shape[0]):
        d = ndi.distance_transform_edt(m[i]) - ndi.distance_transform_edt(~m[i])
        out[i] = np.clip(d, -tau, tau)
    return torch.from_numpy(out)


def _edge_scores(prob, gt_edge, tol=2):
    """边缘预测打分（224² 网格）：prob 为 sigmoid 概率，gt_edge 为 0/1 边缘带。

    严格口径给 edge Dice / precision / recall；另加 ±tol 像素容差口径——
    1 px 边缘线的严格重合对副解码器过苛，容差口径回答「线是不是落在正确位置附近」。
    同时记录 pred_edge_frac / gt_edge_frac，用来暴露「全 0 / 全 1」的退化解
    （BCE 均值对稀疏目标很容易被常数图糊弄过去，Dice 与占比一起看）。
    """
    p = (prob > 0.5).float()
    g = (gt_edge > 0.5).float()
    tp = (p * g).sum()
    gd = F.max_pool2d(g[None, None], 2 * tol + 1, 1, tol)[0, 0]
    pd = F.max_pool2d(p[None, None], 2 * tol + 1, 1, tol)[0, 0]
    return dict(
        edge_dice=(2 * tp / (p.sum() + g.sum() + 1e-6)).item(),
        edge_prec=(tp / (p.sum() + 1e-6)).item(),
        edge_rec=(tp / (g.sum() + 1e-6)).item(),
        edge_prec_t2=((p * gd).sum() / (p.sum() + 1e-6)).item(),
        edge_rec_t2=((pd * g).sum() / (g.sum() + 1e-6)).item(),
        pred_edge_frac=p.mean().item(), gt_edge_frac=g.mean().item())


def _save_edge_panels(path, rows):
    """四列拼图肉眼核验边缘分支：原图+GT 轮廓 / GT 边缘带 / 预测边缘概率 / 预测边缘叠加。

    rows: [(case, img, gt, gt_edge, prob, main_dice, edge_dice), ...]
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(11.5, 2.6 * n), squeeze=False)
    for i, (case, img, gt, gt_e, prob, main_dice, edge_dice) in enumerate(rows):
        ax = axes[i, 0]
        ax.imshow(img, cmap='gray')
        ax.contour(gt, [0.5], colors='lime', linewidths=0.8)
        ax.set_title('%s  mainDice=%.3f  edgeDice=%.3f' % (case, main_dice, edge_dice),
                     fontsize=8)
        axes[i, 1].imshow(gt_e, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('GT edge (morph 3x3)', fontsize=8)
        axes[i, 2].imshow(prob, cmap='magma', vmin=0, vmax=1)
        axes[i, 2].set_title('pred edge prob', fontsize=8)
        ax = axes[i, 3]
        ax.imshow(img, cmap='gray')
        ax.imshow(np.ma.masked_where(prob < 0.5, prob), cmap='autumn', alpha=0.75)
        ax.set_title('pred edge > 0.5', fontsize=8)
        for ax in axes[i]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


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
        # 边缘目标来源：只有开边缘分支时数据集才额外返回 "edge"（sobel_orig = 原尺寸 Sobel）
        edge_mode = (getattr(args, 'edge_mode', '') or '') if getattr(args, 'edge_branch', 0) else ''
        sw_mode = getattr(args, 'sw_mode', '')
        weight_file = None
        if sw_mode:
            weight_file = (getattr(args, 'sw_file', '') or
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'analysis',
                                        'sample_weights_%s_%s.csv' % (args.subset, sw_mode)))
            if not os.path.isfile(weight_file):
                raise FileNotFoundError('样本权重表不存在: %s（先跑 scripts/build_sample_weights.py）'
                                        % weight_file)
        db_train = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "train.txt"),
                                  split="train", img_size=args.img_size,
                                  use_wavelet=bool(args.use_wavelet),
                                  gamma_aug=bool(getattr(args, 'gamma_aug', 0)),
                                  weight_file=weight_file,
                                  edge_mode=edge_mode)
        test_set = CarotidDataset(root=args.root_path, list_file=os.path.join(split_dir, "test.txt"),
                                  split="test", img_size=args.img_size,
                                  use_wavelet=bool(args.use_wavelet), edge_mode=edge_mode)
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
        bl_w = float(getattr(args, 'bl_w', 5.0))

        def seg_loss(out, label):
            """0.5·CE + 0.5·Dice + bl_w·Boundary（Kervadec 距离图加权，权重按初始量级标定）。"""
            l_ce = ce_loss(out, label[:].long())
            return (0.5 * l_ce + 0.5 * dice_loss(out, label, softmax=True)
                    + bl_w * boundary(out, label)), l_ce
    elif loss_name == 'farfp':
        from utils import far_fp_loss
        far_w = float(getattr(args, 'far_w', 0.1))
        far_margin = int(getattr(args, 'far_margin', 10))

        def seg_loss(out, label):
            """损失改进 A：0.5CE+0.5Dice + far_w·远处假阳性项（单独归因 HD 尾部）。"""
            l_ce = ce_loss(out, label[:].long())
            return (0.5 * l_ce + 0.5 * dice_loss(out, label, softmax=True)
                    + far_w * far_fp_loss(out, label, margin=far_margin)), l_ce
    elif loss_name == 'cvartail':
        from utils import cvar_distance_tail_loss, area_consistency_loss
        tl_w = float(getattr(args, 'tl_w', 0.1))
        tl_q = float(getattr(args, 'tl_q', 0.1))
        tl_area_w = float(getattr(args, 'tl_area_w', 0.05))

        def seg_loss(out, label):
            """距离尾部 + 面积一致性：0.5CE+0.5Dice + 面积项 + CVaR_q(距离加权不一致)。"""
            l_ce = ce_loss(out, label[:].long())
            return (0.5 * l_ce + 0.5 * dice_loss(out, label, softmax=True)
                    + tl_area_w * area_consistency_loss(out, label)
                    + tl_w * cvar_distance_tail_loss(out, label, q=tl_q)), l_ce
    elif loss_name == 'rhd':
        # #82 区域 Hausdorff 距离损失（官方实现 lisaGUZZI/HD-Loss 的 LAHsymLoss 2D 版）
        from utils import regional_hd_loss
        rhd_term = getattr(args, 'rhd_term', 'sym')
        rhd_w = float(getattr(args, 'rhd_w', 0.5))
        rhd_iters = int(getattr(args, 'rhd_iters', 6))
        rhd_alpha = float(getattr(args, 'rhd_alpha', 150.0))

        def seg_loss(out, label):
            """0.5CE+0.5Dice + rhd_w·区域 HD 项（两侧距离剖面按各自质量归一）。"""
            l_ce = ce_loss(out, label[:].long())
            l_dice = dice_loss(out, label, softmax=True)
            l_rhd = regional_hd_loss(out, label, term=rhd_term, iters=rhd_iters,
                                     alpha=rhd_alpha)
            seg_loss.parts = (l_ce.detach(), l_dice.detach(), l_rhd.detach())
            return (0.5 * l_ce + 0.5 * l_dice + rhd_w * l_rhd), l_ce
        seg_loss.parts = None
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
    elif loss_name == 'sw':
        from utils import SampleWeightedCEDice
        sw_loss = SampleWeightedCEDice(num_classes)
        # 权重按 batch 传入：seg_loss 的签名被十几个分支共用，用可变容器避免全量改签名
        sw_state = {'w': None}

        def seg_loss(out, label):
            """样本级加权 0.5CE+0.5Dice；w 全 1 时与 §0 基线数值一致。"""
            return sw_loss(out, label, sw_state['w'])
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
            if loss_name == 'sw':
                sw_state['w'] = sampled_batch['weight'].cuda()
            outputs = model(image_batch)
            aux = None
            maps = None
            edge_logits = None
            sdf_map = None
            l_edge = None
            l_sdf = None
            if isinstance(outputs, dict):
                # FDS：主输出 + 三级辅助 logits（各自上采样到 GT 分辨率后再算损失）
                # weaveseg：额外带 sf/sb/su 三个上下文 logits，供 Eq.(12) 的后三项
                # edge_branch：额外带 edge（边缘解码器 logits）
                aux = outputs.get('aux')
                maps = {k: outputs[k] for k in ('sf', 'sb', 'su') if k in outputs} or None
                edge_logits = outputs.get('edge')
                sdf_map = outputs.get('sdf')
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
            if edge_logits is not None:
                # 边缘分支：与原 GT 经边缘算子得到的边缘标签做**与主分割同形式、同权重**
                # 的监督（同一 seg_loss，默认 0.5CE+0.5Dice），权重 --edge_branch_w 默认 1.0；
                # 主分割输出与主损失不变。
                if 'edge' in sampled_batch:
                    # --edge_mode sobel_orig：边缘目标由数据集在原始尺寸上算好（已同尺寸放大）
                    edge_t = sampled_batch['edge'].cuda()
                else:
                    edge_t = _edge_target(label_batch,
                                          int(getattr(args, 'edge_kernel', 3))).long()
                if edge_t.dim() == 4 and edge_t.shape[1] == 1:
                    edge_t = edge_t[:, 0]
                l_edge, _ = seg_loss(edge_logits, edge_t)
                loss = loss + float(getattr(args, 'edge_branch_w', 1.0)) * l_edge
            if sdf_map is not None:
                # 连续界面表示：只在 |d| < tau 的窄带内监督距离场（tanh 饱和之外无梯度），
                # 外加 eikonal 项让场保持"梯度模长为 1"的距离场性质。
                # 目标逐样本由 GT 距离变换生成，与数据集统计无关。
                tau = float(getattr(args, 'sdf_tau', 12.0))
                sdf_t = _sdf_target(label_batch, tau).unsqueeze(1).cuda()
                if sdf_map.shape[-2:] != sdf_t.shape[-2:]:
                    sdf_map = F.interpolate(sdf_map, size=sdf_t.shape[-2:],
                                            mode='bilinear', align_corners=False)
                band = (sdf_t.abs() < tau).float()
                cnt = band.sum().clamp(min=1.0)
                l_sdf = ((sdf_map - sdf_t).abs() * band).sum() / cnt / tau
                gy, gx = torch.gradient(sdf_map, dim=(2, 3))
                eik = (((gy ** 2 + gx ** 2).sqrt() - 1.0) ** 2 * band).sum() / cnt
                loss = loss + float(getattr(args, 'sdf_w', 1.0)) * (l_sdf + 0.02 * eik)
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
            if l_edge is not None:
                writer.add_scalar('info/loss_edge', l_edge, iter_num)
            if l_sdf is not None:
                writer.add_scalar('info/loss_sdf', l_sdf, iter_num)
            rhd_parts = getattr(seg_loss, 'parts', None)
            if loss_name == 'rhd' and rhd_parts is not None:
                writer.add_scalar('info/loss_rhd', rhd_parts[2], iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f%s%s%s' % (
                iter_num, loss.item(), loss_ce.item(),
                '' if l_edge is None else ', loss_edge: %f' % l_edge.item(),
                '' if l_sdf is None else ', loss_sdf: %f' % l_sdf.item(),
                '' if (loss_name != 'rhd' or rhd_parts is None) else
                ', rhd_ce %.4f, rhd_dice %.4f, rhd_term %.4f'
                % tuple(t.item() for t in rhd_parts)))

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
                                  use_wavelet=bool(args.use_wavelet), edge_mode=edge_mode)
        testloader = _DL(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
        model.eval()
        records = []
        edge_records = []
        edge_detail = []
        with torch.no_grad():
            for batch in testloader:
                outputs = model(batch['image'].cuda())
                logits = _primary_out(outputs)
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                pred_o, gt_o, spacing = original_grid_pair(pred, args.root_path, batch['case'][0])
                record = compute_case_metrics(pred_o, gt_o, spacing=spacing)
                record['case'] = batch['case'][0]
                records.append(record)
                # 边缘分支：同一批样本顺带算边缘指标并留图，确认副解码器真的学到了边缘
                if isinstance(outputs, dict) and 'edge' in outputs:
                    # 2 类边缘 logits → 第 1 类概率作为边缘概率
                    prob = torch.softmax(outputs['edge'], dim=1)[0, 1].cpu()
                    gt_e = batch['edge'][0].float() if 'edge' in batch else _edge_target(batch['label'],
                                        int(getattr(args, 'edge_kernel', 3)))[0, 0]
                    es = _edge_scores(prob, gt_e)
                    # 对照：把主分割预测过同一边缘算子能得到多少 edge Dice。
                    # 边缘头明显高于它才算真学到，≈ 则只是把主掩膜重新描了一遍边。
                    es["edge_dice_main"] = _edge_scores(
                        _edge_target(torch.from_numpy(pred)[None],
                                     int(getattr(args, 'edge_kernel', 3)))[0, 0],
                        gt_e)["edge_dice"]
                    edge_records.append(dict(case=batch['case'][0], dice=record['dice'], **es))
                    edge_detail.append((batch['case'][0], batch['image'][0, 0].cpu().numpy(),
                                        batch['label'][0].cpu().numpy(), gt_e.numpy(),
                                        prob.numpy(), record['dice'], es['edge_dice']))
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
        if edge_records:
            def _mean(key):
                return float(np.mean([r[key] for r in edge_records]))
            with open(os.path.join(snapshot_path, "edge_metrics.csv"), "w", newline="",
                      encoding="utf-8") as f:
                cols = ["case", "dice", "edge_dice", "edge_prec", "edge_rec",
                        "edge_prec_t2", "edge_rec_t2", "pred_edge_frac", "gt_edge_frac",
                        "edge_dice_main"]
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(edge_records)
            n_ok = sum(1 for r in edge_records if r["edge_dice"] > 0.3)
            logging.info(
                "edge summary (%d cases): edgeDice=%.3f | prec=%.3f rec=%.3f | "
                "prec@2px=%.3f rec@2px=%.3f | pred_frac=%.4f gt_frac=%.4f | "
                "edgeDice(main-derived)=%.3f | edgeDice>0.3: %d/%d"
                % (len(edge_records), _mean("edge_dice"), _mean("edge_prec"),
                   _mean("edge_rec"), _mean("edge_prec_t2"), _mean("edge_rec_t2"),
                   _mean("pred_edge_frac"), _mean("gt_edge_frac"),
                   _mean("edge_dice_main"), n_ok, len(edge_records)))
            if _mean("pred_edge_frac") < 0.001:
                logging.warning("边缘头输出几乎全 0：副解码器没有学到边缘（检查 --edge_w / 学习率）")
            elif _mean("pred_edge_frac") > 0.5:
                logging.warning("边缘头输出几乎全 1：副解码器退化为常数图")
            # 最差 3 例 + 最好 3 例，一张图看清边缘是否贴合
            order = sorted(range(len(edge_detail)), key=lambda i: edge_detail[i][6])
            picks = order[:3] + order[-3:] if len(order) > 6 else order
            vis_dir = os.path.join(snapshot_path, "edge_vis")
            os.makedirs(vis_dir, exist_ok=True)
            rows = [edge_detail[i] for i in picks]
            _save_edge_panels(os.path.join(vis_dir, "edge_branch_cases.png"), rows)
            for i, d in zip(picks, rows):
                case_name = str(d[0]).replace("/", "_").replace("\\", "_")
                _save_edge_panels(os.path.join(vis_dir, "%s.png" % case_name), [d])
            logging.info("edge visualization saved to %s" % vis_dir)
    return "Training Finished!"