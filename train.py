import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.wavelet import WAVELET_BASES
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Carotid', help='experiment_name')
parser.add_argument('--subset', type=str,
                    default='SPARC', choices=['SPARC', 'Zhongnan', 'ArPlaqSegm1'],
                    help='carotid subset；ArPlaqSegm1 = 论文公开数据集（scripts/prepare_arplaq.py 转换）')
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
                    choices=['baseline', 'dual', 'split', 'unet', 'xnet', 'gobletnet', 'wmren', 'ccfr',
                             'fdfunet', 'weaveseg'],
                    help='baseline = TransUNet / dual = 双骨干交错 + 四级双向融合 (D11) / '
                         'split = 局部(R50)与全局(ViT-B/16)彻底分离、14² 单点融合 / '
                         'unet|xnet|gobletnet = 纯卷积复现网络（各自独立训练，与 baseline 互不混用）/ '
                         'ccfr = ESWA-D-25-34301 两阶段复现（EA-Net + rectified flow 精修）/ '
                         'fdfunet = Neural Networks 2024 多尺度频率滤波网络复现（UNet* + DRDAC + FEM）')
parser.add_argument('--levels', type=str, default='1,2,3,4',
                    help='dual 融合接点：1=112²/64, 2=56²/256, 3=28²/512, 4=14²/768')
parser.add_argument('--fuse_op', type=str, default='auto',
                    choices=['auto', 'window', 'global', 'gate', 'concat', 'deform', 'aff', 'frc', 'frc_s'],
                    help='auto = 112/56 窗内 + 28/14 全局；deform = 可变形大核；aff = MS-CAM；'
                         'frc = Syn-Net FrCFusion（DCT 频率通道注意力 + 空间支）；'
                         'frc_s = 严格版（单层 Linear 通道权重、空间支无 ReLU、纯残差）')
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
parser.add_argument('--no_pretrain', type=int, default=0,
                    help='1 = 跳过 ImageNet21k 权重加载（仅 baseline）：R50/ViT 全部随机初始化，'
                         '用于把「R50 结构」与「ImageNet 预训练」两个因素拆开（§1.13）')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--main_input', type=str, default='raw', choices=['raw', 'll'],
                    help='dual 主路输入：raw = 原图复制 3 通道（现状）；ll = 小波 LL 低频近似图（XNet 式低频/高频双流，高频全在侧路）')
parser.add_argument('--edge_in', type=int, default=0,
                    help='1 = 边缘特征融合·输入级：stem 多吃一张边缘图（baseline 拼到 3 通道后；unet 拼到 1 通道后）')
parser.add_argument('--edge_op', type=str, default='wavelet',
                    choices=['wavelet', 'wavelet_lam', 'energy', 'sobel', 'laplacian'],
                    help='输入级边缘图算子（--arch unet/baseline 且 --edge_in 1 时可换）：'
                         'wavelet = Haar 高频 IDWT 重建 / wavelet_lam = 再加 λ·LL 回灌 / '
                         'energy = 细节能量图 / sobel / laplacian')
parser.add_argument('--wavelet_basis', type=str, default='haar', choices=list(WAVELET_BASES),
                    help='小波基选择实验（只作用于 --edge_op wavelet / wavelet_lam）：'
                         'haar / db2 / db4 / sym4 / coif1 / bior1.5；分解级数、对数域、'
                         'LL 置零与标准化口径全部不变，只换滤波器组')
parser.add_argument('--ddbe', type=int, default=0,
                    help='1~3 = 解码器第 N 级输出（28/56/112/224，level3=112×112/64ch）挂 BDFR-Net 式 DDBE；'
                         '4 = 四级全挂 DDBE×4（BDFR-Net 完整版）；0 = 关（仅 baseline）')
parser.add_argument('--acam', type=int, default=0,
                    help='1 = 每个 decoder 级对 skip 特征做 ACAM 跨级语义对齐（ddbefull = --ddbe 4 --acam 1；仅 baseline）')
# ---- BDFR-Net（3612 Submission）忠实复现：v2 解码器（Eq.1–10） ----
parser.add_argument('--bdfr_v2', type=int, default=0,
                    help='1 = 换成 BDFR-Net 忠实解码器（DDBE×4 + ACAM 替代 concat；仅 --arch baseline，'
                         '与 --ddbe/--acam 互斥）')
parser.add_argument('--bdfr_ddbe', type=int, default=1,
                    help='--bdfr_v2 1 时是否启用 DDBE 块；论文 Eq.1 里 ECA 是 DDBE 的第一步，'
                         '两者绑定（0 = DDBE 与 ECA 一起去掉，只留 ACAM）')
parser.add_argument('--bdfr_acam', type=int, default=1,
                    help='--bdfr_v2 1 时是否启用 ACAM；0 = ACAM 回退为 concat（其余不变）')
parser.add_argument('--bdfr_wave_op', type=str, default='haar', choices=['haar', 'lap', 'pyr'],
                    help='--bdfr_v2 1 时 DDBE 内 HF/LF 的分解基：haar（论文原版）/ lap（单尺度拉普拉斯）/ '
                         'pyr（多尺度拉普拉斯金字塔）')
parser.add_argument('--ffusion', type=int, default=0,
                    help='1 = 解码器每个 skip 融合点换成 FreqFusion（NeurIPS 2024，'
                         'ALPF+offset+AHPF，Eq.6–13；仅 --arch baseline，不与 --bdfr_v2 同开）')
parser.add_argument('--edge_fusion', type=str, default='none',
                    choices=['none', 'rel', 'relconst', 'mref', 'fem', 'abf', 'abftied'],
                    help='解码器 skip 汇合点的边缘/频域融合：'
                         'none / rel（R-GEF 单支拉普拉斯 + 逐像素可靠性，仅 baseline）/ '
                         'relconst（r≡0.5，参数完全匹配的对照，仅 baseline）/ '
                         'mref（MREF 多支：多尺度拉普拉斯 + Sobel + 斑点噪声图 + 双头门控，仅 baseline）/ '
                         'fem（FDFUNet 的 FEM：MFDF 多尺度频域滤波 + ASCA 注意力，仅 baseline）/ '
                         'abf（自研非对称频带融合：baseline = R50 版；unet = 弱骨干版）/ '
                         'abftied（α_H = −α_L 的参数匹配对照）')
parser.add_argument('--edge_gamma', type=float, default=0.0,
                    help='R-GEF 注入强度 γ 初值。0 = 起点等价基线（实测 200ep 后 γ 仍≈0，模块未启用）；'
                         '0.1 = 修正版，模块从头参与优化')
parser.add_argument('--edge_depth', type=int, default=1,
                    help='--edge_fusion mref 时每个 skip 汇合点上串联的 MREF 块数（推荐 4）；'
                         '单块容量有限，级联 4 块 = 全模型 12 个融合块')
parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adamw', 'adam'],
                    help='sgd = 项目历次口径（momentum 0.9 + poly）；adamw = 论文口径（恒定 lr，wd 1e-4）；'
                         'adam = FDFUNet 论文口径（Adam lr 1e-4 + cosine 退火到 1e-5，按 epoch）')
parser.add_argument('--gamma_aug', type=int, default=0,
                    help='1 = 训练集加 gamma 扰动（论文 augmentation 之一，默认关以保持既有实验口径）')
parser.add_argument('--asr', type=int, default=0,
                    help='1 = unet 解码器前三级（28²/56²/112²）把 up_conv 换成 WeaveSeg ICCV 2025 的 '
                         'ASR 自适应光谱细节精炼（cat + DBB 结构不变）')
parser.add_argument('--weave_asr', type=int, default=1,
                    help='weaveseg：1 = 启用 ASR（解码器各级把 up+cat+DBB 换成 1×1+ASR）；'
                         '0 = 该级回到 U-Net 的 up+cat+DBB')
parser.add_argument('--weave_icw', type=int, default=1,
                    help='weaveseg：1 = 末端接 ICW（T 次迭代的对比编织）；0 = 只用上下文头')
parser.add_argument('--weave_iters', type=int, default=3,
                    help='weaveseg：ICW 迭代次数 T（论文 Table 2 最优为 3）')
parser.add_argument('--weave_k', type=int, default=7,
                    help='weaveseg：ICW 对比注意力的局部窗口边长 K（论文只写 K×K，未给数值）')
parser.add_argument('--weave_asr_levels', type=str, default='0,1,2,3',
                    help='weaveseg：启用 ASR 的解码层级，逗号分隔；0=28²/512 1=56²/256 '
                         '2=112²/128 3=224²/64（显存不足时可去掉 3）')
parser.add_argument('--hf_branch', type=int, default=0,
                    help='1 = GobletNet 式：轻量 HF 支 + FAM 与 skip 逐级互标定（仅 baseline）')
parser.add_argument('--hf_lambda', type=float, default=0.3,
                    help='HF 图低频回灌权重 λ（GobletNet Eq.5：H=HL+LH+HH+λ·LL，论文最优 0~0.3）')
parser.add_argument('--domain_loss', type=int, default=0,
                    help='1 = 颈动脉超声领域适配损失：边界带加权 CE + clDice + 面积一致性 '
                         '（替代原 0.5CE+0.5Dice：0.5*ce_bw + 0.5*dice + 0.3*(1-clDice) + 0.05*area）')
parser.add_argument('--dl_lambda', type=float, default=3.0,
                    help='领域损失：边界带加权系数 λ（权重 1+λ·band），扫参用')
parser.add_argument('--dl_radius', type=int, default=2,
                    help='领域损失：边界带宽 ±radius 像素')
parser.add_argument('--dl_area_w', type=float, default=0.05,
                    help='领域损失：面积一致性项权重（压 DeltaArea）')
parser.add_argument('--dl_far_w', type=float, default=0.1,
                    help='领域损失：远处假阳性项权重（压 HD 尾部）')
parser.add_argument('--dl_margin', type=int, default=10,
                    help='领域损失：远处假阳性的 GT 膨胀 margin（像素）')
parser.add_argument('--fds', type=int, default=0,
                    help='1 = Syn-Net Flux Deep Supervision：解码器 14²/28²/56² 三级辅助监督 '
                         '(0.2/0.3/0.5)，辅助 logits 上采样到 GT 分辨率算损失；仅 dual/split')
parser.add_argument('--loss', type=str, default='ce_dice',
                    choices=['ce_dice', 'ce_dice46', 'ce_dice_unc', 'weave5', 'weave4',
                             'bce_wiou', 'bce_iou_dice', 'focal', 'tversky', 'boundary'],
                    help='ce_dice = 0.5CE+0.5Dice（§0 口径）；ce_dice46 = 论文 Eq.(14) 的 0.4CE+0.6Dice；'
                         'ce_dice_unc = Kendall 同方差不确定性加权 CE+Dice（初值等价 0.5/0.5，额外对照）；'
                         'weave5 = WeaveSeg Eq.(12) 五项损失 1.0Mask+0.5L_fg+0.5L_bg+0.3L_tri+1.0L_uncert；'
                         'weave4 = 同上但 λ_uncert=0（单变量判断不确定项）；'
                         'bce_wiou = Syn-Net 的 mean(BCE+wIoU)；'
                         'bce_iou_dice = FDFUNet 论文 Eq.(6) 的 0.5BCE+0.5IoU+Dice；'
                         'focal = 0.5Focal+0.5Dice；tversky = Tversky(0.3/0.7)；'
                         'boundary = 0.5CE+0.5Dice+0.5Boundary（损失基线，均为 §0 其余口径不变）')
parser.add_argument('--fdfunet_variant', type=str, default='full', choices=['full', 'unetstar'],
                    help='full = 完整 FDFUNet（DRDAC + MFDF + ASCA）；unetstar = 论文同构基线 UNet*')
parser.add_argument('--fdfunet_ablate', type=str, default='',
                    choices=['', 'no_drdac', 'no_mfdf', 'no_asca'],
                    help='FDFUNet 单模块减法消融（只在完整版上关一个模块）：'
                         'no_drdac = 编码器 DRDAC 换普通 3×3 卷积；'
                         'no_mfdf = FEM 只留 ASCA；no_asca = FEM 只留 MFDF')
parser.add_argument('--wmren_ablate', type=str, default='',
                    choices=['', 'unet_enc', 'unet_dec',
                             'no_wave', 'no_mlkm', 'no_rem', 'no_gate'],
                    help='WMREN 模块归因的原位替换：unet_enc = 四个 WRMM 换 U-Net 编码块；'
                         'unet_dec = 三级 REM 换 U-Net 的 cat+DBB 融合；'
                         'no_wave = WRDM 只留残差支；'
                         'no_gate = 去掉 Eq.(4)(5) 的 softmax 软加权；'
                         'no_mlkm = WRMM 去掉多尺度大核；no_rem = REM 退化为 Concat+1×1')
parser.add_argument('--wmren_wave_op', type=str, default='haar', choices=['haar', 'lap', 'pyr'],
                    help='WMREN 的 WRDM 高频分解算子：haar（默认）/ lap（§1.12 拉普拉斯算子替换）/ '
                         'pyr（拉普拉斯金字塔：3 个真实尺度、保留符号，§1.12 的正统多尺度臂）')
parser.add_argument('--wmren_no_wave', type=int, default=0,
                    help='1 = WRDM 去掉左支小波（只留残差支）；可与 --wmren_ablate unet_dec 叠加，'
                         '用于在 K2 宿主上做「小波支整体值多少」控制臂')
parser.add_argument('--wmren_no_gate', type=int, default=0,
                    help='1 = WRDM 去掉 Eq.(4)(5) 的 softmax 软加权（固定 0.5）；同样可与 unet_dec 叠加')
parser.add_argument('--wmren_no_mlkm', type=int, default=0,
                    help='1 = WRMM 去掉 MLKM 多尺度大核单元（WRDM 输出直接 1×1 回通道）；同样可与 unet_dec 叠加')
parser.add_argument('--wmren_enc', type=str, default='wrmm', choices=['wrmm', 'r50', 'r50deep', 'r50wrmm', 'r50seq'],
                    help="WMREN 的编码器：wrmm（默认，stem+4×WRMM，从零训）/ "
                         "r50（R50 conv_root+block1~3 整体替换编码器、WRMM 全不建，"
                         "用于问「换成 R50 编码器值多少」，不是「WMREN 的预训练版」）/ "
                         "r50deep（只把后两级 WRMM 原位换成 R50 block2/block3，"
                         "输入输出逐级同形：56²/256→28²/512、28²/512→14²/1024，"
                         "浅两级 WRMM 与小波支全保留）/ "
                         "r50wrmm（R50 全段不截断，conv_root+block1~3 的 3/4/6 出四级特征，"
                         "再逐级接 WRMM 的 stride=1 同级精炼，形状不变；"
                         "即「强骨干提特征 + 自己的编码器吃这些特征」）/ "
                         "r50seq（R50 与 WRMM 串行：WRMM1 的 112² 输出既作 skip 又喂 block1，"
                         "WRMM2/3/4 的输出同理既作 skip 又喂下一级 block，"
                         "WRMM 特征真正进入编码器主链）")
parser.add_argument('--ddbe_hf_op', type=str, default='haar', choices=['haar', 'lap', 'pyr', 'none', 'plain'],
                    help='DDBE 高频支算子：haar（默认）/ lap（§1.12 拉普拉斯替换）/ '
                         'pyr（拉普拉斯金字塔，3 个真实尺度、保留符号）/ '
                         'none（HF 支置零，作为「频域支整体值多少」对照）/ '
                         'plain（ddbectl 去分解对照臂 2026-11-09：三支全吃同一普通特征，'
                         '参数量与 haar/lap 逐位一致，归因「拼接改写 vs 频率内容」）')
parser.add_argument('--r50_fdf', type=str, default='none', choices=['none', 'drdac', 'drdac_mfdf', 'full'],
                    help='FDFUNet 模块渐进上 R50（2026-11-09 规划，验证 A.7 减法消融'
                         '排除不掉的组合协同）：drdac = 编码器四级 + DRDAC（恒等起步）；'
                         'drdac_mfdf = +编码/解码各四级 FEM（MFDF only）；'
                         'full = FEM 开 ASCA（完整模块组）。全部外加不替换')
parser.add_argument('--fgr', type=int, default=0,
                    help='1 = 解码器四级挂 FGR（拉普拉斯频带 + FcaNet 式 DCT 谱门 + '
                         '1×1 改写，fuse 零初始化恒等起步；「只有频带内容 + 轻门控」，'
                         '与 ddbectl/plain 构成双向归因对照）')
parser.add_argument('--fgr_mode', choices=['whole', 'band'], default='whole',
                    help='--fgr 1 时的改写核档位：whole = 单枚 2C→C 核搅两带（原 FGR）；'
                         'band = 高低频各一枚 C→C 零核并联（FBGR 分带档，R50 上解决'
                         'FGR 与 DDBE 同点串联互覆的算子同型问题）')
# ---- FreqDINO（arXiv 2512.11335）移植批：MFEA / FGBR 四级全挂解码器输出 ----
parser.add_argument('--fd_mfea', type=int, default=0,
                    help='1 = 解码器四级（28²/256→224²/16）全挂 MFEA：两级 Haar 分解 + '
                         '边界/结构通道注意力改写，λ=0.3、α=β=0.5，γ 零初始化恒等起步')
parser.add_argument('--fd_fgbr', type=int, default=0,
                    help='1 = 同四级全挂 FGBR：两级高频逐级降维出 8×min(64,C) 维边界原型，'
                         'Q 来自空间特征、K/V 来自原型（8 头×min(128,C) 维，ω=0.2），'
                         'W_O 零初始化恒等起步')
# ---- CCFR（ESWA-D-25-34301）两阶段训练 ----
parser.add_argument('--ccfr_stage', type=str, default='both', choices=['both', '1', '2'],
                    help='both = 先训 EA-Net 再训 flow；1 = 只训 EA-Net；2 = 只训 flow（须给 --ccfr_stage1_ckpt）')
parser.add_argument('--ccfr_stage1_ckpt', type=str, default='',
                    help='--ccfr_stage 2 时加载的 EA-Net 权重')
parser.add_argument('--stage1_epochs', type=int, default=100, help='EA-Net 训练轮数（论文 100 ep）')
parser.add_argument('--stage1_pick', type=str, default='best',
                    help="stage2 用哪一轮 EA-Net：'best' = 按 val Dice 选最优（论文 best-performing），"
                         "或写整数轮次")
parser.add_argument('--stage1_val_ratio', type=float, default=0.1,
                    help='Stage-1 选模用的 val 比例（按病例从 train 切，0 = 关掉、退回 test 曲线）')
parser.add_argument('--stage1_lr', type=float, default=1e-3)
parser.add_argument('--stage1_wd', type=float, default=0.01)
parser.add_argument('--edge_w', type=float, default=0.4, help='边缘监督权重（论文 0.4）')
parser.add_argument('--edge_kernel', type=int, default=3,
                    help='由 GT 掩膜提边缘标签的形态学核尺寸（论文未给算子，固定 3 = 1 像素环带）')
parser.add_argument('--stage2_epochs', type=int, default=20)
parser.add_argument('--stage2_lr', type=float, default=1e-4)
parser.add_argument('--stage2_wd', type=float, default=0.01)
parser.add_argument('--flow_steps', type=int, default=5, help='推理 Euler 步数 K（论文 K=5）')
parser.add_argument('--flow_alpha', type=float, default=1.0, help='速度目标缩放 α（论文 α=1）')
parser.add_argument('--w_vel', type=float, default=5.0, help='速度损失权重（论文 5:1:5）')
parser.add_argument('--w_bce', type=float, default=1.0)
parser.add_argument('--w_dice', type=float, default=5.0)
parser.add_argument('--ccfr_base', type=int, default=56, help='EA-Net 基础通道（56 → 24.4M，对齐论文 U-Net 量级）')
parser.add_argument('--flow_base', type=int, default=32, help='flow 头基础通道')
parser.add_argument('--ccfr_mfloor', type=float, default=0.1, help='不确定性门 M 下限')
parser.add_argument('--ccfr_medgew', type=float, default=0.5, help='M 中边缘强度权重（其余为 Pc(1-Pc)）')
parser.add_argument('--ccfr_nocond', type=int, default=0, help='1 = 去掉结构条件注入（w/o condition 消融）')
parser.add_argument('--ccfr_nounc', type=int, default=0, help='1 = 去掉不确定性门（w/o uncertainty 消融）')
parser.add_argument('--ccfr_noattn', type=int, default=0,
                    help='1 = Stage 1 去掉每级空间注意力（=U-Net 底座），边缘支与其余超参不动')
parser.add_argument('--ccfr_pretrain', type=int, default=0,
                    help='1 = EA-Net 编码器换成 ImageNet21k 预训练 R50（灰度复制 3 通道输入），'
                         '对齐论文；0 = 从零 1 通道纯卷积 U-Net（统一口径版本）。'
                         '1 时 --ccfr_base 无效（先验通道由 R50 固定为 64/256/512/1024）')
parser.add_argument('--stage1_lr_enc', type=float, default=1e-4,
                    help='--ccfr_pretrain 1 时预训练编码器的 lr（解码器/边缘支仍用 --stage1_lr）')
args = parser.parse_args()

if args.ddbe and args.arch not in ('baseline', 'unet'):
    parser.error('--ddbe 只接了 baseline/unet 解码器（dual 的融合在编码器接点，未接 DDBE）')
if not 0 <= args.ddbe <= 4:
    parser.error('--ddbe 取值 0~4')
if args.acam and args.arch not in ('baseline', 'unet'):
    parser.error('--acam 只接了 baseline/unet 解码器（与 --ddbe 同挂载面）')
if args.r50_fdf != 'none':
    if args.arch != 'baseline':
        parser.error('--r50_fdf 只接了 baseline（R50 编码器 + TransUNet 解码器）')
    if args.ddbe or args.acam or args.bdfr_v2 or args.ffusion or args.edge_fusion != 'none':
        parser.error('--r50_fdf 与 --ddbe/--acam/--bdfr_v2/--ffusion/--edge_fusion 互斥'
                     '（A.7 组合协同单独成臂，不与已有模块纠缠）')
if args.fgr:
    if args.arch not in ('baseline', 'unet'):
        parser.error('--fgr 只接了 baseline/unet（解码器四级输出改写）')
    if args.arch == 'baseline' and (args.bdfr_v2 or args.ffusion):
        parser.error('--fgr 与 --bdfr_v2/--ffusion 互斥（同为解码器改写面；'
                     '--ddbe 允许组合 = FGR+DDBE 双模块臂，同点串联 ddbe→fgr）')
if args.fgr_mode != 'whole' and not args.fgr:
    parser.error('--fgr_mode 只在 --fgr 1 时有效（没有 FGR 就没有档位）')
if args.fd_mfea or args.fd_fgbr:
    # 只接 TransUNet 解码器（--arch baseline）；且与其它解码器改写/融合臂互斥，
    # 保证移植批每个臂只变一个模块
    if args.arch != 'baseline':
        parser.error('--fd_mfea/--fd_fgbr 目前只接了 baseline（R50-TransUNet 解码器）')
    if (args.fgr or args.ffusion or args.bdfr_v2 or args.edge_fusion != 'none'
            or args.r50_fdf != 'none'):
        parser.error('--fd_mfea/--fd_fgbr 与 --fgr/--ffusion/--bdfr_v2/--edge_fusion/'
                     '--r50_fdf 互斥（单模块移植批只变一个变量）')
if args.bdfr_v2 and args.arch != 'baseline':
    parser.error('--bdfr_v2 只接了 TransUNet 解码器（--arch baseline）')
if args.bdfr_v2 and (args.ddbe or args.acam):
    parser.error('--bdfr_v2 自带 DDBE×4 + ACAM，不能与 v1 的 --ddbe/--acam 同时开')
if args.bdfr_wave_op != 'haar' and not args.bdfr_v2:
    parser.error('--bdfr_wave_op 只作用于 --bdfr_v2 1 的 DDBEv2（v1 请用 --ddbe_hf_op）')
if args.ffusion and args.arch != 'baseline':
    parser.error('--ffusion 只接了 TransUNet 解码器（--arch baseline）')
if args.ffusion and args.bdfr_v2:
    parser.error('--ffusion 与 --bdfr_v2 都替换解码器融合点，不能同开')
if args.edge_fusion != 'none' and args.arch not in ('baseline', 'unet'):
    parser.error('--edge_fusion 只接了 baseline（TransUNet 解码器）与 unet 解码器')
if args.arch == 'unet' and args.edge_fusion not in ('none', 'abf', 'abftied'):
    parser.error('--arch unet 只接了自研 ABF（--edge_fusion abf / abftied）')
if args.edge_fusion != 'none' and (args.bdfr_v2 or args.ffusion):
    parser.error('--edge_fusion 与 --bdfr_v2/--ffusion 都改解码器融合点，不能同开')
if args.asr and args.arch != 'unet':
    parser.error('--asr 只接了 unet 解码器（替换 up_conv，baseline/dual/split/xnet/gobletnet 未接）')
if args.hf_branch and args.arch != 'baseline':
    parser.error('--hf_branch 只接了 baseline 解码器（FAM 挂在 skip 上，dual 未接）')
if not 0 <= args.hf_lambda <= 1:
    parser.error('--hf_lambda 取值 0~1（GobletNet 论文最优区间 0~0.3）')
if not 0 <= args.vit_layers <= 12:
    parser.error('--vit_layers 取值 0~12（预训练只含 12 层权重，0 = 去掉全局 Transformer 段）')
if args.no_pretrain and not (args.arch == 'baseline'
                             or (args.arch == 'wmren'
                                 and args.wmren_enc in ('r50', 'r50deep', 'r50wrmm', 'r50seq'))):
    parser.error('--no_pretrain 只支持 --arch baseline（A0 家族）与 --arch wmren 的 r50/r50deep/r50wrmm/r50seq 编码器'
                 '（预训练臂的 scratch 对照）：unet/wmren(wrmm) 本来就是随机初始化，'
                 'dual/split 另有 --side_pretrained 控制侧路')
if args.edge_op != 'wavelet' and not (args.arch in ('unet', 'baseline') and args.edge_in):
    parser.error('--edge_op 只在 --arch unet/baseline --edge_in 1 时生效')
if args.wavelet_basis != 'haar':
    if args.edge_op not in ('wavelet', 'wavelet_lam'):
        parser.error('--wavelet_basis 只作用于 --edge_op wavelet / wavelet_lam')
    if not (args.arch in ('unet', 'baseline') and args.edge_in):
        parser.error('--wavelet_basis 只在 --arch unet/baseline --edge_in 1 时生效')
if args.fds and args.arch not in ('dual', 'split'):
    parser.error('--fds 只接了 dual/split 的解码器（baseline/unet 的解码器未挂三级辅助头）')
if args.loss == 'bce_wiou' and args.arch not in ('baseline', 'dual', 'split'):
    parser.error('--loss bce_wiou 只接了 baseline/dual/split（unet 等复现网络未接）')
if args.wmren_ablate and args.arch != 'wmren':
    parser.error('--wmren_ablate 只接了 --arch wmren')
if args.wmren_wave_op != 'haar':
    if args.arch != 'wmren':
        parser.error('--wmren_wave_op 只接了 --arch wmren')
    if args.wmren_ablate == 'unet_enc':
        parser.error('--wmren_wave_op 对 unet_enc 无效（四个 WRMM 已被替换）')
    if args.wmren_ablate == 'no_wave':
        parser.error('--wmren_wave_op 与 no_wave 互斥')
if args.wmren_enc in ('r50', 'r50deep', 'r50wrmm', 'r50seq'):
    if args.arch != 'wmren':
        parser.error('--wmren_enc 只接了 --arch wmren')
    if args.wmren_ablate == 'unet_enc':
        parser.error('--wmren_enc %s 的浅层是 R50，没有 WRMM 可换' % args.wmren_enc)
    if args.wmren_enc in ('r50', 'r50deep') and args.wmren_ablate in ('no_wave', 'no_gate', 'no_mlkm'):
        parser.error('--wmren_enc %s 不含 WRMM，不能与 WRMM 级消融共用' % args.wmren_enc)
    if args.wmren_enc == 'r50' and args.wmren_wave_op != 'haar':
        parser.error('--wmren_enc r50 没有 WRDM，--wmren_wave_op 无意义')

_wr_flags = [('--wmren_no_wave', args.wmren_no_wave),
             ('--wmren_no_gate', args.wmren_no_gate),
             ('--wmren_no_mlkm', args.wmren_no_mlkm)]
if args.wmren_enc in ('r50', 'r50deep') and any(v for _, v in _wr_flags):
    parser.error('--wmren_enc %s 不含完整 WRMM，不能与 WRMM 级消融共用' % args.wmren_enc)
if any(v for _, v in _wr_flags):
    if args.arch != 'wmren':
        parser.error('--wmren_no_wave/--wmren_no_gate/--wmren_no_mlkm 只接了 --arch wmren')
    if sum(1 for _, v in _wr_flags if v) > 1:
        parser.error('--wmren_no_wave/--wmren_no_gate/--wmren_no_mlkm 三者互斥（一次只做留一）')
    if args.wmren_ablate in ('no_wave', 'no_gate', 'no_mlkm', 'unet_enc'):
        parser.error('--wmren_no_wave/--wmren_no_gate/--wmren_no_mlkm 与 --wmren_ablate %s 冲突'
                     % args.wmren_ablate)
    if args.wmren_no_wave and args.wmren_wave_op != 'haar':
        parser.error('--wmren_no_wave 与 --wmren_wave_op 互斥（小波支已去掉）')
if args.ddbe_hf_op != 'haar':
    if not args.ddbe:
        parser.error('--ddbe_hf_op 需要同时给 --ddbe 1~4')
    if args.arch not in ('baseline', 'unet'):
        parser.error('--ddbe_hf_op 只接了 baseline/unet 的 DDBE 挂载面')
if args.fdfunet_ablate and args.arch != 'fdfunet':
    parser.error('--fdfunet_ablate 只接了 --arch fdfunet')
if args.fdfunet_ablate and args.fdfunet_variant != 'full':
    parser.error('--fdfunet_ablate 只能叠在完整版上（--fdfunet_variant full）')
if args.arch in ('unet', 'xnet', 'gobletnet', 'wmren', 'fdfunet'):
    # 独立复现网络在 forward 内自算小波输入，数据集必须给 1 通道原图
    args.use_wavelet = 0
    if args.hf_branch:
        parser.error('--hf_branch 只接了 baseline 解码器（FAM 挂在 skip 上，unet/xnet/gobletnet/fdfunet 未接）')
    if args.edge_in and args.arch != 'unet':
        parser.error('--edge_in 只在 unet 上接了（xnet/gobletnet/fdfunet 自带频率支）')
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


if args.arch == 'ccfr':
    # CCFR 从零训练、只用 1 通道原图，其余融合/损失开关一律不接
    args.use_wavelet = 0
    if args.edge_in or args.fusion != 'none' or args.ddbe or args.acam or args.asr \
            or args.hf_branch or args.fds or args.domain_loss:
        parser.error('ccfr 是独立复现网络，不接受 --edge_in/--fusion/--ddbe/--acam/--asr/'
                     '--hf_branch/--fds/--domain_loss')
    if args.loss != 'ce_dice':
        parser.error('ccfr 的两阶段损失在 trainer_ccfr 内部定义，不接受 --loss')
    if args.ccfr_nocond and args.ccfr_nounc:
        parser.error('--ccfr_nocond 与 --ccfr_nounc 不同时开（论文消融是分别去掉）')


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
    args.is_pretrain = not bool(args.no_pretrain)
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
            if args.edge_op != 'wavelet':
                # 算子非默认（wavelet λ=0）时进目录名，同一 arch 多算子对照不互相覆盖
                snapshot_path += '_eo' + args.edge_op
            if args.wavelet_basis != 'haar':
                # 小波基非默认时进目录名（_wbdb2 / _wbsym4 ...），基横扫不互相覆盖
                snapshot_path += '_wb' + args.wavelet_basis
        if args.fusion != 'none':
            snapshot_path += '_' + args.fusion
        if args.ddbe:
            # 快照名必须反映真实配置（status.md 3.3：名不副实 = 新旧权重混目录）
            snapshot_path += '_ddbe' + str(args.ddbe)
            if args.ddbe_hf_op != 'haar':
                snapshot_path += '_ddhf' + args.ddbe_hf_op
        if args.acam:
            snapshot_path += '_acam'
        if args.bdfr_v2:
            # BDFR-Net 忠实版（Eq.1–10）：完整 / 去 DDBE+ECA / 去 ACAM 三组必须分目录
            snapshot_path += '_bdfr2'
            if not args.bdfr_ddbe:
                snapshot_path += '_nod'
            if not args.bdfr_acam:
                snapshot_path += '_na'
            if args.bdfr_wave_op != 'haar':
                snapshot_path += '_bw' + args.bdfr_wave_op
        if args.ffusion:
            snapshot_path += '_ff'
        if args.edge_fusion != 'none':
            # R-GEF：rel 为正式模块，relconst 为参数匹配的常数门控对照
            snapshot_path += '_eg' + args.edge_fusion
            if args.edge_fusion == 'mref':
                snapshot_path += str(int(args.edge_depth))
            if args.edge_gamma != 0.0:
                snapshot_path += '_gm' + str(int(round(args.edge_gamma * 100)))
        if args.asr:
            # v1（分组共享核 8 组）为 _asr；v2 逐通道核 + 软偏移，必须分目录
            snapshot_path += '_asr2'
        if args.hf_branch:
            # λ 量化到 0.1 粒度进快照名（λ=0.3 → _hfbr3）
            snapshot_path += '_hfbr' + str(int(round(args.hf_lambda * 10)))
        if args.r50_fdf != 'none':
            snapshot_path += '_r50fdf_' + {'drdac': 'dr', 'drdac_mfdf': 'mf', 'full': 'full'}[args.r50_fdf]
        if args.fgr:
            # whole 沿用历史段名 _fgr；band 是新算子档（FBGR），必须分目录
            snapshot_path += '_fgr' if args.fgr_mode == 'whole' else '_fgrb'
        if args.fd_mfea:
            snapshot_path += '_fdmfea'
        if args.fd_fgbr:
            snapshot_path += '_fdfgbr'
        if args.domain_loss:
            # 损失口径不同必须分目录，否则与 0.5CE+0.5Dice 的权重混在一起
            snapshot_path += '_dl'
            # 扫参时非默认超参必须进目录名，否则多次 sweep 互相覆盖
            dl_tag = 'L%sR%dA%sF%sM%d' % (args.dl_lambda, args.dl_radius,
                                          args.dl_area_w, args.dl_far_w, args.dl_margin)
            if dl_tag != 'L3.0R2A0.05F0.1M10':
                snapshot_path += dl_tag
        if args.loss != 'ce_dice':
            # 任何非 §0 损失都必须与 0.5CE+0.5Dice 的权重分目录（避免写进 A0 目录）
            snapshot_path += '_l' + args.loss.replace('_', '')
        if args.arch in ('unet', 'xnet', 'gobletnet', 'wmren', 'fdfunet'):
            snapshot_path += '_' + args.arch
            if args.arch == 'unet' and args.edge_fusion in ('abf', 'abftied'):
                snapshot_path += '_eg' + args.edge_fusion
                if args.edge_gamma != 0.0:
                    snapshot_path += '_gm' + str(int(round(args.edge_gamma * 100)))
            if args.arch == 'unet' and args.edge_in and args.edge_op == 'wavelet':
                # unet 的边缘算子横扫：默认 wavelet 沿用历史目录名 _eowavelet（非默认算子已在上方加过）
                snapshot_path += '_eo' + args.edge_op
            if args.arch == 'fdfunet' and args.fdfunet_variant != 'full':
                snapshot_path += '_' + args.fdfunet_variant
            if args.arch == 'fdfunet' and args.fdfunet_ablate:
                # 减法消融必须与完整版分目录（_abnodrdac / _abnomfdf / _abnoasca）
                snapshot_path += '_ab' + args.fdfunet_ablate.replace('_', '')
            if args.arch == 'gobletnet' and abs(args.hf_lambda - 0.3) > 1e-9:
                # λ 非默认 0.3 时进快照名（λ=0.1 → _lam01），避免同目录混权重
                snapshot_path += '_lam' + str(args.hf_lambda).replace('.', '')
            if args.arch == 'wmren' and args.wmren_ablate:
                # 模块归因消融必须与完整版分目录
                snapshot_path += '_ab' + args.wmren_ablate.replace('_', '')
            if args.arch == 'wmren' and args.wmren_wave_op != 'haar':
                # 算子替换必须与 Haar 版分目录
                snapshot_path += '_wo' + args.wmren_wave_op
            if args.arch == 'wmren' and args.wmren_no_wave:
                snapshot_path += '_nw'
            if args.arch == 'wmren' and args.wmren_no_gate:
                snapshot_path += '_ng'
            if args.arch == 'wmren' and args.wmren_no_mlkm:
                snapshot_path += '_nm'
            if args.arch == 'wmren' and args.wmren_enc == 'r50':
                snapshot_path += '_encr50'
            if args.arch == 'wmren' and args.wmren_enc == 'r50deep':
                snapshot_path += '_encr50deep'
            if args.arch == 'wmren' and args.wmren_enc == 'r50wrmm':
                snapshot_path += '_encr50wrmm'
            if args.arch == 'wmren' and args.wmren_enc == 'r50seq':
                snapshot_path += '_encr50seq'
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
            if args.fds:
                snapshot_path += '_fds'
            if args.tag:
                snapshot_path += '_' + args.tag
        if args.arch == 'split':
            # 局部/全局分离 + 14² 单点融合；算子名必须进目录（同一 arch 多算子对照）
            snapshot_path += '_split_' + args.fuse_op
            if args.fds:
                snapshot_path += '_fds'
            if args.tag:
                snapshot_path += '_' + args.tag
        if args.arch == 'ccfr':
            # 两阶段复现：_ccfr；只跑 stage2 的消融另立目录，权重不互相覆盖
            snapshot_path += '_ccfr'
            if args.ccfr_pretrain:
                # 预训练 R50 编码器版本，与从零版本的权重绝不能混目录
                snapshot_path += '_pr50'
            if args.ccfr_stage == '2':
                snapshot_path += '_s2'
            if args.ccfr_nocond:
                snapshot_path += '_nocond'
            if args.ccfr_nounc:
                snapshot_path += '_nounc'
            if args.ccfr_noattn:
                snapshot_path += '_noattn'
            if args.flow_base != 32 or (not args.ccfr_pretrain and args.ccfr_base != 56):
                snapshot_path += '_b%df%d' % (args.ccfr_base, args.flow_base)
            if args.tag:
                snapshot_path += '_' + args.tag
        if args.arch == 'weaveseg':
            # 四行消融只差 ASR/ICW 开关，变体名必须进目录，否则四行互相覆盖
            if args.weave_asr and args.weave_icw:
                variant = 'full'
            elif args.weave_asr:
                variant = 'noicw'
            elif args.weave_icw:
                variant = 'noasr'
            else:
                variant = 'base'
            snapshot_path += '_weave_' + variant
            if args.tag:
                snapshot_path += '_' + args.tag

        if 'lists_Carotid_dev' in args.list_dir.replace('\\', '/'):
            # 快筛子集的结果绝不能和全量结果混在一起
            snapshot_path += '_dev'
        if 'lists_Carotid2D_union' in args.list_dir.replace('\\', '/'):
            # 多中心联合训练（SPARC 350 + Zhongnan 448），必须与单中心 A0 分目录
            snapshot_path += '_mcUnion'
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
                                   edge_in=bool(args.edge_in),
                                   fds=bool(args.fds)).cuda()
    elif args.arch == 'split':
        # 局部 R50 与全局 ViT-B/16 并联，仅 14² 单点融合
        from networks.split_encoder_model import SplitLocalGlobalTransUNet
        net = SplitLocalGlobalTransUNet(config_vit, img_size=args.img_size,
                                        num_classes=config_vit.n_classes,
                                        fuse_op=args.fuse_op, window=args.window,
                                        dk_ratio=args.dk_ratio,
                                        bidir=bool(args.bidir),
                                        fds=bool(args.fds)).cuda()
    elif args.arch == 'unet':
        # 容量对齐基线：与 xnet/gobletnet 同积木的纯卷积 U-Net
        from networks.unet import UNet
        net = UNet(in_channels=1, num_classes=args.num_classes,
                   edge_in=bool(args.edge_in), edge_op=args.edge_op,
                   wavelet_basis=args.wavelet_basis,
                   ddbe_level=args.ddbe, acam=bool(args.acam),
                   asr=bool(args.asr), ddbe_hf_op=args.ddbe_hf_op,
                   fgr=bool(args.fgr), fgr_mode=args.fgr_mode)
        if args.edge_fusion in ('abf', 'abftied'):
            net.attach_edge_fusion(kind=args.edge_fusion, init_gamma=args.edge_gamma)
        net = net.cuda()
    elif args.arch == 'weaveseg':
        # WeaveSeg（ICCV 2025）：U-Net 底座 + ASR + ICW，返回 {'out','sf','sb','su'}
        from networks.weaveseg import WeaveSeg
        levels = tuple(int(v) for v in args.weave_asr_levels.split(',') if v.strip() != '')
        net = WeaveSeg(in_channels=1, num_classes=args.num_classes,
                       asr=bool(args.weave_asr), icw=bool(args.weave_icw),
                       asr_levels=levels, iters=args.weave_iters,
                       k=args.weave_k).cuda()
    elif args.arch == 'wmren':
        # WMREN 骨架替换：4×WRMM（WRDM + MLKM）编码 + 3×REM 解码（U-Net 底座 60.3M → 58.1M）
        from networks.unet_wmren import UNetWMREN
        net = UNetWMREN(in_channels=1, num_classes=args.num_classes,
                        ablate=args.wmren_ablate or None,
                        wave_op=args.wmren_wave_op,
                        no_wave=bool(args.wmren_no_wave),
                        no_gate=bool(args.wmren_no_gate),
                        no_mlkm=bool(args.wmren_no_mlkm),
                        enc=args.wmren_enc).cuda()
    elif args.arch == 'xnet':
        # 小波低/高频双分支 + X 形融合，返回 (out_lf, out_hf) 双输出
        from networks.xnet import XNet
        net = XNet(in_channels=1, num_classes=args.num_classes).cuda()
    elif args.arch == 'gobletnet':
        # 语义 + HF 双编码器，FAM 四级融合，单输出
        from networks.gobletnet import GobletNet
        net = GobletNet(in_channels=1, num_classes=args.num_classes,
                        hf_lambda=args.hf_lambda).cuda()
    elif args.arch == 'ccfr':
        # 两阶段复现：EA-Net（边缘感知 U-Net）+ rectified flow 精修头
        from networks.ccfr import CCFR
        net = CCFR(num_classes=args.num_classes, base=args.ccfr_base,
                   flow_base=args.flow_base, steps=args.flow_steps,
                   cond=not bool(args.ccfr_nocond), unc=not bool(args.ccfr_nounc),
                   alpha=args.flow_alpha, m_floor=args.ccfr_mfloor,
                   m_edge_w=args.ccfr_medgew, img_size=args.img_size,
                   pretrain=bool(args.ccfr_pretrain),
                   noattn=bool(args.ccfr_noattn)).cuda()
        if args.ccfr_pretrain:
            # 只载 R50 部分；解码器/边缘支/flow 头仍从零（论文也是从零）
            n_units = net.ea.load_pretrained(np.load(config_vit.pretrained_path))
            print("[ccfr] EA-Net encoder <- %s (%d bottleneck units)"
                  % (config_vit.pretrained_path, n_units))
    elif args.arch == 'fdfunet':
        # Neural Networks 2024 多尺度频率滤波网络：UNet* 5 级 + DRDAC + 每级 FEM(MFDF+ASCA)
        from networks.fdfunet import FDFUNet
        full = args.fdfunet_variant == 'full'
        net = FDFUNet(in_channels=1, num_classes=args.num_classes,
                      img_size=args.img_size,
                      use_drdac=full and args.fdfunet_ablate != 'no_drdac',
                      use_mfdf=full and args.fdfunet_ablate != 'no_mfdf',
                      use_asca=full and args.fdfunet_ablate != 'no_asca').cuda()
    elif args.fusion != 'none':
        from networks.wavelet_model import WaveletTransUNet
        net = WaveletTransUNet(config_vit, img_size=args.img_size,
                               num_classes=config_vit.n_classes,
                               fusion=args.fusion, heads=args.heads).cuda()
    else:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes,
                      edge_in=bool(args.edge_in), edge_op=args.edge_op,
                      wavelet_basis=args.wavelet_basis,
                      ddbe_level=args.ddbe, acam=bool(args.acam),
                      ddbe_hf_op=args.ddbe_hf_op,
                      hf_branch=bool(args.hf_branch), hf_lambda=args.hf_lambda,
                      bdfr_v2=bool(args.bdfr_v2), bdfr_ddbe=bool(args.bdfr_ddbe),
                      bdfr_acam=bool(args.bdfr_acam),
                      bdfr_wave_op=args.bdfr_wave_op,
                      ffusion=bool(args.ffusion),
                      edge_fusion=args.edge_fusion != 'none',
                      edge_gate='const' if args.edge_fusion == 'relconst' else 'spatial',
                      edge_kind=('mref' if args.edge_fusion == 'mref' else
                                'fem' if args.edge_fusion == 'fem' else
                                args.edge_fusion if args.edge_fusion in ('abf', 'abftied') else 'rel'),
                      edge_depth=args.edge_depth,
                      edge_gamma=args.edge_gamma,
                      r50_fdf=args.r50_fdf, fgr=bool(args.fgr),
                      fgr_mode=args.fgr_mode,
                      fd_mfea=bool(args.fd_mfea), fd_fgbr=bool(args.fd_fgbr)).cuda()
    if args.arch != 'ccfr' and not args.no_pretrain:
        if args.arch == 'wmren' and args.wmren_enc in ('r50', 'r50deep', 'r50wrmm', 'r50seq'):
            # wmren 的 load_from 是空操作（WRMM 无可迁移形状）；R50 编码器臂单独吃 R50 那部分权重
            net.load_r50(weights=np.load(config_vit.pretrained_path))
        else:
            net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'Synapse': trainer_synapse, 'Carotid': trainer_synapse,}
    if args.arch == 'ccfr':
        from trainer_ccfr import trainer_ccfr
        trainer_ccfr(args, net, snapshot_path)
    else:
        trainer[dataset_name](args, net, snapshot_path)