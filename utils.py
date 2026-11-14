import math

import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class BoundaryWeightedCELoss(nn.Module):
    """边界带加权交叉熵（颈动脉超声领域适配，只依赖 GT，不依赖图像细节）。

    权重 w = 1 + lam * band，band 由 GT 前景的膨胀-腐蚀差得到并加宽到 ±radius 像素。
    图像整体模糊、管壁不可分辨，误差主要落在 GT 边界附近，这里显式抬升该区域的损失。
    """

    def __init__(self, lam=3.0, radius=2):
        super(BoundaryWeightedCELoss, self).__init__()
        self.lam = lam
        self.radius = radius

    def forward(self, logits, label):
        fg = (label > 0).float().unsqueeze(1)
        dil = F.max_pool2d(fg, 3, 1, 1)
        ero = 1.0 - F.max_pool2d(1.0 - fg, 3, 1, 1)
        band = (dil - ero).clamp(0, 1)
        k = 2 * self.radius + 1
        band = F.max_pool2d(band, k, 1, self.radius).squeeze(1)
        weight = 1.0 + self.lam * band
        ce = F.cross_entropy(logits, label.long(), reduction='none')
        return (ce * weight).mean()


def far_fp_loss(logits, label, margin=10):
    """远离真值的假阳性惩罚：预测前景落在「GT 膨胀 margin 像素」之外的质量占比。

    实测 GT 是半宽约 19 px 的厚块，断裂/漂浮小岛才是 HD95 尾部误差的主要来源，
    这一项专门压这类远离 GT 的假阳性；不需要任何图像细节假设。
    """
    prob = torch.softmax(logits, dim=1)[:, 1:2]
    gt = (label > 0).float().unsqueeze(1)
    reach = gt
    for _ in range(int(margin)):
        reach = F.max_pool2d(reach, 3, 1, 1)
    far = 1.0 - reach
    return (prob * far).sum() / (prob.sum() + 1e-6)


def area_consistency_loss(logits, label):
    """面积一致性：|预测前景面积 − 真值面积| / 真值面积，抑制 DeltaArea 漂移。"""
    prob = torch.softmax(logits, dim=1)[:, 1:2]
    gt = (label > 0).float().unsqueeze(1)
    return (prob.sum() - gt.sum()).abs() / (gt.sum() + 1e-6)


class BCEWIoULoss(nn.Module):
    """Syn-Net（JBHI 2025）的 BCE + weighted IoU。

    论文未给权重定义，这里取 w = 1 + 5·|gt − mean(gt)|（前景抬权、与前景面积无关）。
    输出 (BCE + wIoU) / 2，与 `mean(BCE + wIoU)` 同量纲。
    """

    def __init__(self, alpha=5.0):
        super(BCEWIoULoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits, label):
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        gt = (label > 0).float().unsqueeze(1)
        w = 1.0 + self.alpha * (gt - gt.mean(dim=(2, 3), keepdim=True)).abs()
        bce = F.binary_cross_entropy(prob, gt, weight=w, reduction='mean')
        inter = (prob * gt * w).sum()
        union = (prob * w + gt * w - prob * gt * w).sum()
        return 0.5 * (bce + (1.0 - inter / (union + 1e-6)))


class FocalLoss(nn.Module):
    """Focal Loss（Lin et al. 2017）：γ=2、α=0.25，压易分像素的权重。

    作为损失基线使用，与 Dice 各占 0.5，仅替换 CE 项，其余训练口径与 §0 完全一致。
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, label):
        prob = torch.softmax(logits, dim=1)[:, 1]
        gt = (label > 0).float()
        pt = prob * gt + (1.0 - prob) * (1.0 - gt)
        at = self.alpha * gt + (1.0 - self.alpha) * (1.0 - gt)
        return (-at * (1.0 - pt).pow(self.gamma) * pt.clamp_min(1e-6).log()).mean()


class TverskyLoss(nn.Module):
    """Tversky Loss（Salehi et al. 2017）：α=0.3、β=0.7，偏向召回。

    颈动脉斑块欠分割比过分割更常见，取 β>α；该项自身已含区域重合，不再叠加 Dice。
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, label):
        prob = torch.softmax(logits, dim=1)[:, 1]
        gt = (label > 0).float()
        tp = (prob * gt).sum()
        fp = (prob * (1.0 - gt)).sum()
        fn = ((1.0 - prob) * gt).sum()
        return 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


def _signed_distance(gt):
    """GT 的有符号距离图：内部为负、外部为正，按图像边长归一化到 ~[0,1]。"""
    import numpy as np
    from scipy import ndimage as ndi

    g = gt.astype(bool)
    if not g.any():
        return np.ones_like(g, dtype=np.float32)
    inner = ndi.distance_transform_edt(g)
    outer = ndi.distance_transform_edt(~g)
    phi = np.where(g, -inner, outer).astype(np.float32)
    return phi / float(max(g.shape))


def cvar_distance_tail_loss(logits, label, q=0.1, d0=10.0, cap=3.0):
    """距离尾部损失（CVaR-DSL）：只优化最差 q% 的「距离加权不一致」。

    e(x) = |p(x) − gt(x)| · clamp(D(x)/d0, max=cap)
      D = 到 GT 边界的距离（像素，|有符号距离图|）
    再对每张切片取 e 最大的 q% 的均值（CVaR），最后按 batch 平均。

    动机：HD 是 max 型泛函，而 CE / Dice / 边界加权 / 面积 / 远假阳全是 mean 型聚合，
    对尾部不敏感（阶段 0 离线验证：均值代理与 HD 的 rho=0.69~0.75，
    top10% 代理 rho=0.86~0.90，控制均值后仍有 0.77 的偏相关）。
    q=1 时逼近单侧 HD，q=100 时退化为均值型距离项。
    """
    import numpy as np

    prob = torch.softmax(logits, dim=1)[:, 1]
    gt = (label > 0).float()
    gtn = gt.detach().cpu().numpy().astype(bool)
    dist = []
    for g in gtn:
        if not g.any():
            dist.append(np.zeros_like(g, dtype=np.float32))
            continue
        phi = _signed_distance(g) * float(max(g.shape))  # 还原成像素距离
        dist.append(np.abs(phi).astype(np.float32))
    d = torch.from_numpy(np.stack(dist)).to(prob.device).float()

    e = torch.abs(prob - gt) * (d / d0).clamp(max=cap)
    b, h, w = e.shape
    k = max(1, int(round(h * w * float(q))))
    vals = torch.topk(e.reshape(b, -1), k, dim=1).values
    valid = (gt.reshape(b, -1).sum(dim=1) > 0).float()
    per = vals.mean(dim=1) * valid
    return per.sum() / valid.sum().clamp(min=1.0)


def positive_distance_map_2d(x, iters=2):
    """可微正距离剖面 d+（区域 HD 损失 #82，2D 版）。

    对应官方实现（lisaGUZZI/HD-Loss）的 SoftErosion3D + PositiveDistanceMap：
        d+(p) = -erode(1-p, I)
    每次「腐蚀」在 3x3 邻域对 1-p 连乘（含中心 9 项，对应 3D 的 26/27 连通），
    乘性形式处处可导；外边界补 1（图像外视为背景，不在边界处收缩）。
    单次腐蚀记 im_k = erosion^k(1-x)，返回 Σ_{k=0}^{I-1} im_k，
    二值输入下等于到前景的棋盘距离，并在 I 处截断。

    参数 x 可为 bool 或概率（B,1,H,W），返回同形状 float。
    """
    im = 1.0 - x.float()
    total = torch.zeros_like(im)
    for _ in range(int(iters)):
        total = total + im
        p = F.pad(im, (1, 1, 1, 1), mode='constant', value=1.0)
        win = p.unfold(2, 3, 1).unfold(3, 3, 1)          # (B,C,H,W,3,3)
        im = win.prod(dim=-1).prod(dim=-1)
    return total


def regional_hd_loss(logits, label, term='sym', iters=6, alpha=150.0, eps=1e-6):
    """区域 Hausdorff 距离损失（#82，Guzzi et al., MLMI 2025）的 2D 实现。

    fp = <d+(G)>_{y}  ：预测前景里、离 GT 前景多远（extra 侧）
    fn = <d+(y)>_{G}  ：GT 前景里、离预测前景多远（miss 侧）
    两者都按自身质量归一（论文 norm=0），消除面积偏置。

    term:
      sym = 0.5*(fp+fn)                  论文主用的 LAHsymLoss
      ah  = max(fp, fn)                  论文 LAHLoss（逐样本精确最大值）
      h   = max(smax(fp), smax(fn))      论文 LHLoss，smax 为 alpha 的 LogSumExp
    空前景样本按有效样本数归一后置 0，避免 bs 内个别空切片污染与 NaN。
    """
    prob = torch.softmax(logits, dim=1)[:, 1:2]
    gt = (label > 0).float()
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    with torch.no_grad():
        dm_gt = positive_distance_map_2d(gt, iters)
    dm_hat = positive_distance_map_2d(prob, iters)

    axes = (1, 2, 3)
    fp = (dm_gt * prob).sum(dim=axes) / (prob.sum(dim=axes) + eps)
    fn = (dm_hat * gt).sum(dim=axes) / (gt.sum(dim=axes) + eps)
    if term == 'sym':
        per = 0.5 * (fp + fn)
    elif term == 'ah':
        per = torch.maximum(fp, fn)
    elif term == 'h':
        # 官方 logsum: alpha * log(sum(exp(x/alpha)))，按空间求和后取两侧较大者。
        # 减去 alpha*log(N)：直接使用时该式含常数项 alpha*log(N)（224² 约 1624），
        # 会把 0.5CE+0.5Dice（约 0.6）完全淹没，无法等权相加；减去常数后是标准
        # soft-max（≈ 剖面内最大距离，上界为 iters），与 sym/ah 同量级、可加。
        n_pix = float(dm_gt[0].numel())
        offset = alpha * math.log(n_pix)
        fp_l = alpha * torch.log(torch.sum(torch.exp((dm_gt * prob) / alpha), dim=axes) + eps) - offset
        fn_l = alpha * torch.log(torch.sum(torch.exp((dm_hat * gt) / alpha), dim=axes) + eps) - offset
        per = torch.maximum(fp_l, fn_l)
    else:
        raise ValueError('rhd term 必须是 sym / ah / h，收到 %r' % (term,))

    valid = (gt.sum(dim=axes) > 0).float()
    return (per * valid).sum() / valid.sum().clamp(min=1.0)


def boundary_soft_ce(logits, label, eps=0.25, radius=1):
    """边界带软标签 CE：GT 边界带 ±radius 像素内的目标向 0.5 软化。

    前景像素目标由 (1,0) 变为 (1-eps, eps)，背景像素由 (0,1) 变为 (eps, 1-eps)，
    eps ∈ [0, 0.5]；eps=0.5 时带内目标恒为 (0.5, 0.5)。

    动机（docs/low-score-samples.md §5）：低分误差集中在贴 GT 的 1 px 边界带，
    该带梯度仅为全图均值的 1.9 倍，属标注歧义区。继续用硬标签逼模型贴合等于
    放大标签噪声，故降低带内监督强度，把容量让给可学的内部与远端区域。

    eps=0 时严格退化为普通交叉熵（冒烟脚本用该恒等性做校验）。
    """
    fg = (label > 0).float().unsqueeze(1)
    dil = F.max_pool2d(fg, kernel_size=3, stride=1, padding=1)
    ero = 1.0 - F.max_pool2d(1.0 - fg, kernel_size=3, stride=1, padding=1)
    band = (dil - ero).clamp(0.0, 1.0)
    for _ in range(int(radius) - 1):
        band = F.max_pool2d(band, kernel_size=3, stride=1, padding=1)
    soft = fg * (1.0 - eps) + (1.0 - fg) * eps
    p_fg = torch.where(band > 0, soft, fg)
    target = torch.cat([1.0 - p_fg, p_fg], dim=1)
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


class BoundaryLoss(nn.Module):
    """Boundary Loss（Kervadec et al., MIDL 2019）：用 GT 有符号距离图给预测概率加权。

    只依赖 GT 几何、不假设图像边界锐利，与本项目的数据实测前提一致；
    与 Dice 互补，用于检验「显式边界项」是否等价于本文的边界带加权 CE。
    """

    def forward(self, logits, label):
        import numpy as np

        prob = torch.softmax(logits, dim=1)[:, 1]
        gt = (label > 0).detach().cpu().numpy()
        phi = torch.from_numpy(np.stack([_signed_distance(g) for g in gt]))
        return (prob * phi.to(prob.device).float()).mean()


class SampleWeightedCEDice(nn.Module):
    """样本级加权 0.5·CE + 0.5·Dice（按切片给权重，不是按像素）。

    与基线 [DiceLoss] 的唯一差别是聚合方式：基线在整 batch 上做全局求和，
    样本权重无法作用；这里先算每个样本的 CE / Dice，再按 w 加权平均。
    w 在 batch 内除以自身均值（保持平均权重 1），避免全难样本的 batch
    把有效学习率放大；w 全为 1 时与 0.5·CE + 0.5·DiceLoss(n_classes) 数值一致。
    """

    def __init__(self, n_classes=2):
        super(SampleWeightedCEDice, self).__init__()
        self.n_classes = n_classes

    def forward(self, logits, label, weight=None):
        label = label.long()
        if label.dim() == 4:
            label = label[:, 0]
        ce = F.cross_entropy(logits, label, reduction='none').mean(dim=(1, 2))

        prob = torch.softmax(logits, dim=1)
        tgt = torch.stack([(label == i).float() for i in range(self.n_classes)], dim=1)
        inter = (prob * tgt).sum(dim=(2, 3))
        # 分母口径必须与 DiceLoss 一致：soft 概率平方和 + 目标平方和（不是概率和）
        denom = (prob * prob).sum(dim=(2, 3)) + (tgt * tgt).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)      # (B, C)
        dice = dice.mean(dim=1)                                  # 与 DiceLoss 一致：两类取平均

        if weight is None:
            w = torch.ones_like(ce)
        else:
            w = weight.to(ce.device).float().reshape(-1)
        w = w / w.mean().clamp_min(1e-6)
        loss_ce = (w * ce).sum() / w.sum()
        loss_dice = (w * dice).sum() / w.sum()
        return 0.5 * loss_ce + 0.5 * loss_dice, loss_ce


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list