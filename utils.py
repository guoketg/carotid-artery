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