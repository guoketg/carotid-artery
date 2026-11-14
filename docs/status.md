# 项目状态（精简版）

> 本文件只回答三件事：**现在在跑什么 / 已经得到了什么结论 / 下一步等什么**。
> **结果数字统一记在 [results.md](results.md)**（纯结果台账，本文件不放大表）；计划与判定规则见 [experiments.md](experiments.md)。
> 2026-10-23 ~ 10-27 的完整过程记录已归档到 [archive/status-history-2026-10.md](archive/status-history-2026-10.md)。
> 术语解释见 [study/edge-fusion-explained.md](study/edge-fusion-explained.md)；融合算子逐个讲解见 [study/fusion-operators-explained.md](study/fusion-operators-explained.md)。

最后更新：2026-11-14 20:30（**新增 `queue_c2d_cvartail_r50_3seed_zn` 5 job：R50 cvartail 的 SPARC 3 seed 补 2 run + 中南 3 run，已挂载等 GPU，见 §2 首条**。此前：**报数口径定稿：5-seed 集成降级为内部诊断，不作论文方法/读数；训练侧改推区域 HD 损失 #82，规划见 §4 第 6 条与 [experiments.md](experiments.md) §2.1，未启动训练**。此前：**FGREnc 补跑成功 88.62 / HD 0.93，位次翻盘**：enc ≈ dec（差 0.37，不可分辨）> skip 87.76 > EFF 87.48 > 基线 86.55，见 [results.md](results.md) §7(5)；结论从「越靠近输出越好」改为「改写必须落在主链上，内容比位置重要」。GPU 现已空闲；`queue_c2d_edge_branch` 两臂都 exit 1，训练与测试摘要都正常打印、崩在 `edge_vis` 逐例 PNG 存盘（目录未建），EB-U 读数 86.65 无效）。此前：2026-11-13 20:05（新增**聚类版样本加权**队列 `queue_c2d_cluster_weight` 4 job，等 GPU，见 §2 首条）。此前：2026-11-13（新增**边缘监督分支**队列 `queue_c2d_edge_branch` 2 job，已 WMI 挂载、等 GPU 与其它队列；见 §2 末条）。此前：2026-11-13（新增 FGR 位置研究 + EFF 编码器融合队列 `queue_fgr_pos_eff` 3 job，已在跑；见 §2）。此前：2026-11-12 13:40（新增损失改进筛选队列 `queue_c2d_loss_unet` 2 job 挂载（等 GPU，见 §2 末条）；**在跑：queue_nopre_ddbe_fgr 2 job**（DDBE / FGR 的 R50 随机初始化归因，2026-11-12 12:23 启动，见 §2 末条）；接力链 queue_freqdino 已 11-11 11:06 收口 exit 0×3，结果待回填。此前状态：**queue_fgr_multiseed_zn 3 job**，约 23:10 收口；follow-up 链已挂两级：ZUB 补 Zhongnan unet 基线 → **queue_fbgr_full 三臂接力**（FBGR 分带档已实现、冒烟全绿，见 §2 与 [module-suite-design.md](module-suite-design.md)）；Syn-Net 经用户决策放弃，见 §4 item 2）。`queue_fgr_ddbe_combo` 3 job 15:39 全部收口 exit 0×3：R50 上 DDBE+FGR 组合 90.632 判负（同点双改写互覆）；弱骨干上 FGR +1.70 / 组合 +1.84 双门槛全过，**FGR 为头号候选**。上一批 `queue_r50fdf_fgr` 5 job 已于 00:57 收口，结果在 §5(12)–(16)）

---

## 1. 环境（只列会变的东西）

| 项 | 值 |
|---|---|
| 代码根 | `E:\mkl\code\python_code\carotid-artery` |
| 数据根 | `E:\mkl\code\datasets\Carotid2D`（`PlaqueDataNii_SPARC_S` 510 图 / `PlaqueDataNii_Zhongnan_S` 636 图） |
| 权重根 | `E:\mkl\code\python_code\model`（`vit_checkpoint\imagenet21k\R50+ViT-B_16.npz`、`TU_Carotid224` 快照） |
| Python | `venv\Scripts\python.exe`（3.11.5，numpy 2.4.6 / scipy 1.17.1） |
| GPU | RTX 3060 12GB |
| 划分 | 官方划分 `lists/lists_Carotid2D/{SPARC,Zhongnan}/{train,test}.txt`，病例级无泄漏 |

---

## 2. 队列状态（`queue_c2d_cvartail_r50` 在跑：1 job；`queue_c2d_cvartail_r50_3seed_zn` 已挂载：5 job，等 GPU）

**2026-11-14 21:20 `queue_c2d_distloss_unet.ps1` 已挂载（PID 26160，等 GPU 与其它队列，实测已在等待）**：U-Net 弱骨干上的**三条「降距离指标」损失臂横评**，各 1 job 单变量，全部 `--arch unet`、SPARC seed 1234 bs4 ga2 ep200（约 100 min/job，合计约 5 h）。① **RHDSYM** = `--loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6`，区域 HD 平均项（Guzzi et al., MLMI 2025 主用式，miss/extra 两侧按各自质量归一）；② **RHDH** = 同前但 `--rhd_term h`，soft-max 压最坏一点（机制上等同 Karimi & Salcudean TMI 2020「直接优化 HD」一族）；③ **BND** = `--loss boundary --bl_w 5.0`，Kervadec 边界图加权（MIDL 2019）。**第 3 条不是 cbDice**：cbDice（MICCAI 2024）需可微骨架 + 欧氏距离变换 + 骨架半径加权，本机 venv 无 skimage/cuCIM、公式细节未核，故用同族的边界 Dice 写法替代，已在脚本注释写死。配对对照 U-Net 基线 86.55 / ASSD 0.225 / HD 1.14；判定主判 HD（ΔHD ≤ −0.1 mm）、护栏 Dice（ΔDice ≥ −0.3）。**代码改动**：`utils.py` 的 `regional_hd_loss` 中 `term='h'` 减掉常数 `alpha*log(N)`（224² 约 1624，原式会淹没 0.5CE+0.5Dice 的 ~0.6，无法等权相加），减后是标准 soft-max、上界 = iters；`train.py` 新增 `--bl_w`（默认 5.0，按初始量级标定：离线 L_B≈0.07 vs 0.5CE+0.5Dice≈0.7），`trainer.py` 的 boundary 分支改用该权重，快照名加 `_B<w>`（仅非默认值）。冒烟 [scripts/smoke_distloss_unet.py](../scripts/smoke_distloss_unet.py) 全绿：U-Net 前向 (2,2,224,224)、三条损失对参数梯度有限非零（grad_sum 6.9e4–7.8e4）、rhd-h 3.35 落在 [0, iters]、boundary×5 = 0.49 与 Dice 同量级、空前景 rhd=0 且无 NaN。状态 CSV `logs/queue/queue_c2d_distloss_unet_status.csv`。

> **2026-11-14 21:50 实测进度**：队列已自动开跑（首 job RHDSYM 于 21:45:00 启动），日志 `logs/queue/unet_rhd_sym_20261114_214500.log` 分项正常打印（iter 234：ce 0.293 / dice 0.121 / rhd 0.717，与 `0.5CE+0.5Dice+0.5·rhd` 逐项对得上 → **真实数据 ep1 接线检查通过**）。实测 **0.98 s/iter**（256→348 iters / 90 s），每 epoch 88 iter、总 8800 iter → 单 job 约 **145 min**（比历史 U-Net 的 90 min 慢，因区域 HD 项的 3×3 连乘腐蚀在 224² 上做 6 次）；三臂串行 ETA ≈ **次日 05:00–05:30**。**注意：`queue_c2d_cvartail_r50_3seed_zn` 已不在运行**——状态表只有 s2024 一行（21:09:59 结束），之后脚本进程消失，剩余 4 个 job（SPARC s3407、中南 s1234/s2024/s3407）未跑；现 GPU 被本队列占用，**不要并行重挂**。

**2026-11-14 20:30 `queue_c2d_cvartail_r50_3seed_zn.ps1` 已挂载（等前一条队列收口 + GPU 空闲）**：按用户决策，**不等 seed 1234 判定，直接补 cvartail 的 3 seed 与中南**（偏离 experiments.md §2「先单 seed 过线再补」的常规顺序，记录在此备查）。5 job 串行，损失参数与迁移臂逐字相同（`--loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`，`--arch baseline --vit_layers 0`，bs4 ga2 ep200）：① SPARC seed 2024；② SPARC seed 3407（与已有 1234 合成三 seed，报均值±sd）；③ 中南 seed 1234；④ 中南 seed 2024；⑤ 中南 seed 3407。约 42 min/job，合计约 3.5 h。快照前缀 `TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_[s<seed>_]<subset>_lcvartail_W0.1Q0.1A0.05_c2d`。状态 CSV `logs/queue/queue_c2d_cvartail_r50_3seed_zn_status.csv`。**判定缺口**：中南只有 seed 1234 有配对基线 Z0（`..._224_Zhongnan_c2d`），s2024 / s3407 目前只能看方向，要报 Δ 的均值±sd 需再补 2 个中南基线 run（未入队）。

**2026-11-14 19:52 `queue_c2d_cvartail_r50.ps1` 已启动（在跑）**：cvartail 从 U-Net 迁移到 R50，命令与 U-Net 那条 +0.82 Dice / HD −0.18 的臂逐字同参（`--loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`，seed 1234 / bs4 ga2 / 200 ep / `--arch baseline --vit_layers 0`），约 42 min，快照 `TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lcvartail_W0.1Q0.1A0.05_c2d`。跑完与 A0 90.4903 / 0.14473 / HD 0.7062 配对判定（ΔHD ≤ −0.1 且 ΔDice ≥ −0.3）。

**2026-11-14 18:48 `queue_c2d_loss_transfer` 已收口（2 job exit 0）**：① R50 + 组合域损失（`--domain_loss 1 --dl_lambda 3.0 --dl_radius 2 --dl_area_w 0.05 --dl_far_w 0.1`）90.43 / 0.142 / HD 0.66 → ΔDice −0.06、ΔHD −0.046（判负）；② U-Net + FGREnc + 同一域损失 89.07 / 0.166 / HD 0.85 → 相对 UNet+FGREnc 88.62 / HD 0.93 为 +0.45 / −0.08（仍不过线）。**同批之前的 `queue_c2d_loss_unet` 里，cvartail（`--tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`）在 U-Net 上拿到 87.37 / 0.200 / HD 0.96，+0.82 / HD −0.18，是唯一双门槛全过的损失臂**。明细 [results.md](results.md) §10；据此定的下一批见 §4 第 6 条。

**`queue_fgr_multiseed_zn.ps1`（2026-11-10 19:41 启动，已收口）**：① UFs2024 = unet+fgr SPARC seed 2024；② UFs3407 = unet+fgr SPARC seed 3407；③ ZUF = unet+fgr Zhongnan seed 1234。目的：把 §7(4) 的 unet+fgr +1.70（单 seed）钉成多 seed 证据（与 s1234 合并为均值±标准差一行判定，门槛不变），并做 Zhongnan 方向性验证。状态 CSV：`logs/queue/queue_fgr_multiseed_zn_status.csv`。

**接力链已挂两级（全自动，无需值守）**：① `queue_zn_unet_base.ps1`（PID 16328）等本队列 3 行状态落盘 + GPU 空闲后跑 Zhongnan unet 基线（seed 1234，快照 `..._224_Zhongnan_unet_c2d`），补 ZUF 的成对 Δ（状态：`queue_zn_unet_base_status.csv`）；② **`queue_fbgr_full.ps1`（2026-11-10 21:09 WMI 启动，PID 39212）**等 ZUB 状态落盘 + GPU 空闲后跑 FBGR 三臂（SPARC seed 1234 bs4 ga2 ep200，全走 `--fgr_mode band` 分带档）：FBGR-UNet（`..._SPARC_fgrb_unet_c2d`）、FBGR-R50（`..._vitL0_..._SPARC_fgrb_c2d`，对照 DDBE-lap 90.996 看互覆是否消除）、FULL-U = HFE+FBGR 同开（`..._SPARC_edge_fgrb_unet_eowavelet_c2d`，补消融阶梯最后一行）。第三臂原定 HFE-UNet，但同口径结果已存在（87.42 / +0.87，归档 A.4 C 组）经用户决策改跑 full 档。设计、诊断与判定规则见 [module-suite-design.md](module-suite-design.md)；冒烟 [scripts/smoke_fbgr_band.py](../scripts/smoke_fbgr_band.py) 全绿（恒等起步逐位 0、UNet 参数 +960、梯度链路、旗标接线）。状态 CSV：`logs/queue/queue_fbgr_full_status.csv`。

**接力链第三级：`queue_freqdino.ps1`（2026-11-10 挂载，等 FBGR 三行落盘 + GPU 空闲）**：FreqDINO（arXiv 2512.11335，超声分割）移植批三臂，全部 `--arch baseline --vit_layers 0`（R50-TransUNet）单 seed 1234、bs4 ga2 ep200，模块**四级全挂解码器输出**（28²/256 → 56²/128 → 112²/64 → 224²/16，与 DDBE 四级全挂同口径；用户决策：单点只挂最深层的接法在以往批次里普遍拿不到增益）：① FD-MFEA（`--fd_mfea 1`，两级 Haar + 边界/结构通道注意力，λ=0.3、α=β=0.5，γ 零初始化恒等起步，+0.030M 参数）；② FD-FGBR（`--fd_fgbr 1`，边界原型作 K/V，深两级 8×64 原型 + 8 头×128 维照抄论文，浅两级按通道收缩为 8×min(64,C) / min(128,C)，ω=0.2，W_O 零初始化恒等起步，+1.603M 参数）；③ FD-BOTH（论文原序 MFEA→FGBR 串联）。判定门槛不变（ΔDice ≥ +0.5 且 HD 同向），口径**不引用论文的 +2.21/+0.96**——其消融基线是「DINOv3 冻结 + 轻 adapter」82.35 Dice 的弱头，不是 U-Net。冒烟 [scripts/smoke_freqdino.py](../scripts/smoke_freqdino.py)：四级逐级初始化输出与输入逐位相同、闸门梯度非零、四种开关组合端到端前向通过（105.276 / 105.306 / 106.879 / 106.908 M）。状态 CSV：`logs/queue/queue_freqdino_status.csv`。

**2026-11-12 `queue_nopre_ddbe_fgr.ps1`（在跑，2 job，12:23 启动）**：DDBE / FGR 的**骨干归因批**——把 R50 换成随机初始化（`--no_pretrain 1`），只变这一个因素，回答「§5(3) DDBE +0.435 / §5(16) FGR +0.305 的增益来自 ResNet 结构本身，还是来自 ImageNet 预训练特征」。两臂均 `--arch baseline --vit_layers 0`、SPARC seed 1234 bs4 ga2 ep200、模块配置与预训练臂逐字一致：① NP-DDBE（`--ddbe 4`，Haar / ACAM 关，快照 `..._SPARC_ddbe4_c2d`）；② NP-FGR（`--fgr 1` whole 档，快照 `..._SPARC_fgr_c2d`）。**NP-CTL 不重跑**：随机初始化基线（`..._SPARC_c2d`，2026-11-07，89.59 / ASSD 0.16520 / HD 0.79 / HD95 0.51）与预训练臂同协议同 seed，指标代码此后未改；DDBE/FGR 模板在基础权重初始化之后才创建（RNG 对齐纪律），同 seed 下与 NP-CTL 逐位对齐，仍是成对比较。冒烟：两臂各跑 1 epoch 通过（快照命名确认无 `_pretrain` 段，冒烟目录已清）。归因读法：Δ_nopre ≈ Δ_pre → 增益属结构；Δ_nopre ≪ Δ_pre → 增益依赖预训练；Δ_nopre ≤ 0 → 模块只对预训练特征有效。状态 CSV：`logs/queue/queue_nopre_ddbe_fgr_status.csv`，脚本 [scripts/queue_nopre_ddbe_fgr.ps1](../scripts/queue_nopre_ddbe_fgr.ps1)。

**2026-11-12 13:16 `queue_c2d_head_hr.ps1`（挂载，等 `queue_nopre_ddbe_fgr` 两行落盘 + GPU 空闲后起跑）**：单臂 HR-SPARC = `--head_hr 1`（224² 全分辨率证据支路：I / 5×5 均值 / 21×21 均值 / 9×9 局部标准差 4 通道 → 两层 3×3 → logits 残差，末层零初始化），其余与基线 A0 逐字一致（SPARC seed 1234 bs4 ga2 ep200 `--arch baseline --vit_layers 0`），快照 `TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d_hr`。动机与诊断见 [low-score-samples.md](low-score-samples.md) §6：该级是网络里唯一「无 skip、也不接触原图」的接点，低分样本的 1 px 边界误差正缺这一级证据。冒烟（CPU）：参数 5282、基础权重与基线逐位一致、初始 logits 差 0.0。判定：ΔDice ≥ +0.5 且 ASSD / HD 同向不变差；未过线按该文档 §5 收口。状态 CSV：`logs/queue/queue_c2d_head_hr_status.csv`，脚本 [scripts/queue_c2d_head_hr.ps1](../scripts/queue_c2d_head_hr.ps1)。**13:42 核查：进程列表已无该脚本（status 只有表头），当前并未在等；要跑请在本批队列结束后手动起，避免抢卡。**

**2026-11-12 13:42 `queue_c2d_backbone_3seed.ps1`（挂载，等 GPU 空闲且无其它 `queue_*.ps1` 在等，实际排在 `queue_nopre_ddbe_fgr` → `queue_c2d_loss_unet` 之后）**：骨干归因的 **3-seed 补跑**（用户指定 seed 42 / 317 / 1029，与课题组其它工作对齐以便横向对比；该集合已写入 experiments.md §0 与 CLAUDE.md §15），把 §3.1 的四条单 seed 归因钉成均值±sd。四臂 × 3 seed = 12 job，全部 SPARC bs4 ga2 ep200、单变量只换骨干：① U-Net from scratch（`--arch unet`，快照 `..._s{seed}_SPARC_unet_c2d`）；② R50 结构 + 随机初始化（`--arch baseline --vit_layers 0 --no_pretrain 1`，`..._vitL0_..._s{seed}_SPARC_c2d`）；③ R50 + ImageNet21k 预训练（`--arch baseline --vit_layers 0`，`TU_pretrain_..._vitL0_..._s{seed}_SPARC_c2d`）；④ TransUNet 预训练（`--arch baseline`，`TU_pretrain_..._s{seed}_SPARC_c2d`）。读法按 seed 配对：Δ_pretrain = ③−②，Δ_R50struct = ②−①。seed 1234 的现成读数见 [results.md](results.md) §3.1/§4，不进本批集合；已完成的 job 由 `epoch_199.pth` 判定后直接复用（「有现成的话就直接用」），未完成的旧目录归档 `_bak_*` 后重跑。状态 CSV：`logs/queue/queue_c2d_backbone_3seed_status.csv`，脚本 [scripts/queue_c2d_backbone_3seed.ps1](../scripts/queue_c2d_backbone_3seed.ps1)。

**2026-11-12 13:40 `queue_c2d_loss_unet.ps1`（挂载，等 `queue_nopre_ddbe_fgr` + `queue_c2d_head_hr` 全部落盘且 GPU 空闲后起跑）**：损失改进**筛选批**，2 job 单变量，跑在弱骨干 U-Net（基线 86.55）上，目的是低成本的信号筛选，胜者再单独搬回 R50 复验（弱骨干结论不外推，见 §5 教训）。两臂均 `--arch unet`、SPARC seed 1234 bs4 ga2 ep200、不叠加 `--domain_loss`：① LFA `--loss farfp`（0.5CE+0.5Dice+0.1·远处假阳性，对准 HD 尾部——D1 领域损失里该项只有 w=0.1 且被边界 CE 稀释，从未单独成臂，快照 `..._SPARC_lfarfp_unet_c2d`）；② LBC `--loss bndsoft`（边界带软标签 CE，eps=0.25 / radius=1，+ 硬 Dice，对低可分辨带**降监督**而非加权，快照 `..._SPARC_lbndsoft_unet_c2d`）。依据：[low-score-samples.md](low-score-samples.md) §4–§5 的 CNR~Dice 相关 0.142(p=0.074) / −0.024(p=0.741) 方向不一致，故不做「低 CNR 加权」；误差是 1 px 边界带 + 大斑块单点越界，分别对应 LBC / LFA。冒烟 [scripts/smoke_loss_unet.py](../scripts/smoke_loss_unet.py) 全绿（eps=0 退化为 CE 逐位一致、边界带几何、eps=0.5 全带在 logits=0 处等于 ln2、远 FP 项 gt=pred→0 / 远处孤立块→1.0、两条损失梯度有限）。判定不变（ΔDice ≥ +0.5 且 HD 同向）。状态 CSV：`logs/queue/queue_c2d_loss_unet_status.csv`，脚本 [scripts/queue_c2d_loss_unet.ps1](../scripts/queue_c2d_loss_unet.ps1)。

**2026-11-13 20:05 `queue_c2d_cluster_weight.ps1`（已收口，4 job 串行）**：把 §8 的连续分位加权换成**真正的聚类**（老师原方案）。全训练集上对二维特征 `[rel_echo, cnr]` 做 k-means（k=2，标准化参数与簇中心只在 train 拟合，test 只 assign），低回声簇 1.5 / 高回声簇 0.5（3:1、均值 1）。权重表由 [build_cluster_weights.py](../scripts/build_cluster_weights.py) 生成，两种粒度：`cluster_slice`（逐切片归簇）与 `cluster_patient`（多切片患者按特征中位数归簇、整例共用权重）。**训练侧与 §8 完全一致**（`--loss sw` 的样本级 0.5CE+0.5Dice、batch 内归一化），因此是单变量对比：只换权重表来源。离线分组（A0 逐切片指标）：slice 粒度两库方向一致——SPARC 低簇 243 切片(69%) 90.36/0.144/0.734 vs 高簇 107 切片(31%) 90.90/0.146/0.617；ZN 低簇 350(78%) 91.89/0.125/0.589 vs 高簇 98(22%) 92.36/0.117/0.493。**patient 粒度在 SPARC 上方向不一致**（低簇 Dice 90.44 更差、但 HD 0.666 反而更好，高簇 90.58/0.774），故主判为 slice 粒度，patient 臂作对照。状态 CSV：`logs/queue/queue_c2d_cluster_weight_status.csv`。

**2026-11-13 19:17 `queue_c2d_sample_weight.ps1` 已收口（4 job 全 exit 0，27 / 27 / 32.9 / 32.8 min）**：老师提出的**样本级损失加权**两臂 × 两子集，`--loss sw`（样本级 0.5CE+0.5Dice，权重在 batch 内归一化；w 全 1 时与 A0 逐位同值，已用 `Temp/check_sw_loss.py` 校验 diff 0.0）。权重表由 [build_sample_weights.py](../scripts/build_sample_weights.py) 生成，参考分布只在 train 上拟合：指标先换算成 train 内的**难度分位** u（echo：rel_echo 越小越难；cnr：越小越难），再取 `w = exp(γ(u−0.5))`、γ=ln3，即最难 : 最易 = 3:1 且连续过渡，最后归一化到 train 均值 1（batch 内还会再归一化，所以绝对尺度无意义，唯一自由度是这个 3:1 比值）。① `--sw_mode echo` = rel_echo（前景均值 − 外 6px 环均值）；② `--sw_mode cnr` = |前景−外环| / 合并标准差。**结果：两臂整体判负**（SPARC ΔDice +0.076 / +0.080，Zhongnan +0.004 / +0.107；HD 降 0.008~0.045 mm，均不足 0.1 mm 有效线），明细见 [results.md](results.md) §8。回声臂与离线先验一致（两组基线 Dice 差仅 0.008/0.003 且两库反向），实质为零扰动；对比度臂只在 SPARC 低对比度子组 HD −0.077，Zhongnan 同子组反向。状态 CSV：`logs/queue/queue_c2d_sample_weight_status.csv`。

**2026-11-10 00:57 `queue_r50fdf_fgr` 收口**：5 job 全部 exit 0（CTL 56.3min / A1 42.4min / A2 54.6min / A3 59.2min / FGR 44.1min），结果已回填 [results.md](results.md) §5(12)–(16)。要点：ddbeCTL +0.060 ≈ 噪声（DDBE 增益归因频率内容，机制问题落地）；A1/A2/A3 FDFUNet 渐进 +0.03/−0.37/−0.19 判负（MFDF 是负贡献主源）；FGR +0.305、ASSD/HD 同向改善，未过线但为多 seed 复核候选。HD 自本批起统一 medpy 口径。

历史队列（全部收口，无失败 job；唯一异常是融合算子 AFF 首跑在 ep5 因 GroupNorm 崩溃，修复后于批次 0+1 重跑成功）：

| 队列 | job 数 | 内容 | 状态 CSV |
|---|---|---|---|
| `queue_r50fdf_fgr.ps1`（A.7 组合协同 + FGR） | 5 | ddbectl 去分解对照、DRDAC/DRDAC+MFDF/完整 FDFUNet 渐进上 R50、FGR 频门改写 | `logs/queue/queue_r50fdf_fgr_status.csv` |
| `queue_fgr_ddbe_combo.ps1`（FGR+DDBE 组合 + 弱骨干臂） | 3 | R50 解码器 DDBE×4+FGR 同点串联（90.632 判负）、UNet 仅 FGR（88.25，+1.70 双门槛全过）、UNet FGR+DDBE（88.39，复现线新高） | `logs/queue/queue_fgr_ddbe_combo_status.csv` |
| `queue_c2d.ps1`（主队列） | 6 | 单骨干基线 seed2024/3407、双骨干跨支 QKV 注意力 seed2024/3407、输入级边缘图（基线+1 通道）、边缘图+双骨干 | `logs/queue/queue_c2d_status.csv` |
| `queue_c2d_fuse.ps1`（融合算子横评） | 6 | AFF（崩溃 1 次）、gate 门控、deform 可变形大核、DDBE 单级挂载、基线 seed1234/3407 | `logs/queue/queue_c2d_fuse_status.csv` |
| `queue_c2d_repro.ps1`（批次 0+1） | 4 | DDBE 四级全挂、DDBE×4+ACAM（BDFR 完整版）、AFF 重跑、主路吃 LL 低频图（XNet 便宜变体） | `logs/queue/queue_c2d_repro_status.csv` |
| `eval_batch_ep199.ps1`（评测） | 16 | 终点 epoch 200 重算，16 个全部 exit 0（含补记快照 edgedeform） | `logs/queue/eval_batch_ep199_status.csv` |
| `queue_repro_nets.ps1`（文献复现） | 3 | 纯卷积 U-Net / XNet / GobletNet（官方划分 SPARC、seed1234、200ep），全部 exit 0 | `logs/queue/queue_repro_nets_status.csv` |

**2026-11-13 `queue_fgr_pos_eff.ps1`（已收口，4 臂全部有读数）**：FGR **挂载位置研究** + 编码器融合模块 EFF。三臂全在弱骨干 U-Net、SPARC seed 1234 bs4 ga2 ep200，参数增量都是 +2,788,160，唯一变量是挂载面，对照已有点 unet+fgr = 88.2484（§7(4)，不重跑）。结果（含 11-14 补跑）：**FGREnc 88.62 / FGR dec 88.2484 / skip 87.76 / EFF 87.48 / 基线 86.55**——enc 与 dec 差 0.37 在判定线内、只能记并列；skip 比两条主链低 0.5~0.9；EFF 与同位置同参数的 FGREnc 差 1.14，说明差距在内容（跨尺度 ctx）不在位置。**FGREnc 失败**：ep5 评测时 `AttributeError: 'CarotidDataset' object has no attribute 'weights'`——`datasets/dataset_carotid.py` 13:30:53 被改成带样本权重表，而该 job 12:48 已启动，评测新建 worker 重新导入新版类、主进程传入旧实例所致，**与模块无关**；教训：队列在跑时不要改会被 worker 重新导入的文件。已给脚本加 `-Only <名>` 过滤参数，11-14 01:25→03:16 补跑完成（exit 0，110.7 min）。EFF-v2（`--eff_mode band` 分带双核 +960 参数、`--eff_lfdiff 1` 用 `LL(ctx)−LL(x)`、无新增参数）已实现，冒烟覆盖 IDENTITY / 参数量 / 双核梯度链路 / 旗标校验，全绿但未入队。设计与公式见 [module-suite-design.md](module-suite-design.md) §9；冒烟 [scripts/smoke_fgr_pos_eff.py](../scripts/smoke_fgr_pos_eff.py)。状态 CSV：`logs/queue/queue_fgr_pos_eff_status.csv`。

**2026-11-14 `R50-FGREnc` 单臂（在跑，11-14 起，PID 31312 的 WMI 子进程，日志 `logs/queue/r50_fgre_20261114.log`）**：把位置研究从弱骨干搬到 **R50-TransUNet**（`--arch baseline --vit_layers 0`，SPARC seed 1234 bs4 ga2 ep200），指令为用户的「先试一下，不管容量」。接线：`ResNetV2.forward` 新增 `fgr_enc` four-stage inline 改写（模块由 `VisionTransformer.__init__` 在基础权重建完后挂到 `hybrid_model`，形状 root 112²/64 → block1 55²/256（padding hack 之前）→ block2 28²/512 → block3 14²/1024，改写后的 x 继续下采样并作 skip），`train.py` 放开 `--fgr_pos enc` 对 baseline 的限制（`skip` 仍只接 unet）、并把 `fgr_pos` 透传给 `ViT_seg`。**容量混杂必须记**：编码器四级 +11,048,384 参数（解码器位置为 +691,568，差 16 倍），本臂只能回答「R50 编码器改写有没有信号」，不能与 R50 dec 90.795（§7(4)）作纯位置比较。冒烟 [scripts/smoke_fgr_pos_eff.py](../scripts/smoke_fgr_pos_eff.py) 新增 R50 段：同 seed 恒等逐位 0、参数增量精确 11,048,384；端到端 2 epoch 冒烟通过（训练与 ep1 评测均正常打出，快照目录已清）。快照 `..._vitL0_epo200_bs4_ga2_224_SPARC_fgre_c2d`。判定不变：|ΔDice| ≥ 0.5 且 HD 降 ≥ 0.1 mm，对照 R50 预训练基线 90.4903 / HD 0.7062。

**2026-11-14 `queue_c2d_loss_transfer.ps1`（已 WMI 挂载，等 GPU 空闲后起跑，2 job）**：把 UNet 弱骨干上唯一过线的组合损失（D2 = 0.5·ce_bw(λ=3,r=2) + 0.5·Dice + 0.05·area + 0.1·far_fp(margin=10)，87.01 / HD 0.99，Dice +0.46 未过、HD −0.15 过线）做**跨骨干 / 跨模块迁移**，损失口径逐字不动、全默认超参：① **LR50** `--arch baseline --vit_layers 0 --domain_loss 1`，对照 R50 vitL0 基线 90.4903 / 0.14473 / HD 0.7062（§5），参考同一损失在 12 层 A0 上 D1 90.73（对 3-seed 均值 +0.41、对同 seed +0.24），快照 `..._vitL0_epo200_bs4_ga2_224_SPARC_dl_c2d`；② **UFG** `--arch unet --fgr 1 --fgr_pos enc --domain_loss 1`，对照 UNet+FGREnc 88.62（§7(4)）与 UNet 基线 86.55 / 0.2250 / 1.14，快照 `..._SPARC_fgre_dl_unet_c2d`。两臂均 SPARC seed 1234 bs4 ga2 ep200，单变量。**UFG 容量混杂必须记**：FGREnc 比 UNet 多 +2,788,160 参数，该臂只回答「FGR 改写 + 组合监督能否叠加」，不能把增益单独归因给损失；若与 88.62 持平或下降，说明两者打的是同一批误差（边界带 + 远处小岛），存在功能重叠。判定不变：|ΔDice| ≥ 0.5 且 HD 同向。状态 CSV：`logs/queue/queue_c2d_loss_transfer_status.csv`，脚本 [scripts/queue_c2d_loss_transfer.ps1](../scripts/queue_c2d_loss_transfer.ps1)（原来的 `queue_c2d_loss_r50.ps1` 已并入本队列后删除，避免两个等待器互相卡死）。注意：11-14 的 `queue_c2d_edge_tu` 启动器已不在进程表（status 只有表头 + 一行等待日志）。

**2026-11-13 `queue_c2d_edge_branch.ps1`（已收口，2 臂 exit 1，均崩在边缘拼图存盘）**：用户要求的**边缘监督分支**批——原标签经边缘算子得到边缘 GT → 编码器共享 → 解码器两个（主分割 + 边缘分支），在 UNet 与 R50 预训练骨干上分别验证，单 seed。实现：边缘解码器与主解码器**逐层对等**（UNet 新增完整 `UNetDecoder`，+22.6M；R50-TransUNet 新增第二个 `DecoderCup` + 2 类分割头，+7.4M），**不是轻量头**；UNet 与 VisionTransformer 新增 `--edge_branch`，模块**最后创建**并沿用同初始化，保证同 seed 基础权重逐位对齐；训练侧 `trainer._edge_target` 用 k×k 形态学梯度（dilate−erode，`--edge_kernel` 默认 3 → 1 px 内环 + 1 px 外环，与 CCFR Stage1 同口径），边缘项与主分割**同形式同权重**（同一个 `seg_loss` = 0.5CE+0.5Dice，`--edge_branch_w` 默认 **1.0**），主输出/评估口径不变。用户先验：边缘分支在 TransUNet 骨干上应有约 **+1.0 Dice**，明显没到优先当实现/口径错误排查。两臂均 SPARC seed 1234 bs4 ga2 ep200：① EB-U `--arch unet --edge_branch 1`（快照 `..._SPARC_unet_c2d_eb`）；② EB-R50 `--arch baseline --vit_layers 0 --edge_branch 1`（`..._vitL0_..._SPARC_c2d_eb`）。成对基线不重跑：UNet 86.55 / 0.2250 / HD 1.14（§5）与 R50 预训练 90.4903 / 0.14473 / HD 0.7062（§4）。**容量混杂需记**：边缘臂比基座多一个完整解码器，若增益只出现在边缘臂，后续需补「同容量但目标退化为常数/随机」的对照臂才能归因到边缘监督本身。冒烟 [scripts/smoke_edge_branch.py](../scripts/smoke_edge_branch.py) 全绿（输出形状 2 类边缘 logits、同 seed 基础权重对齐、参数增量、边缘/主头梯度链路、边缘带几何、边缘指标边界行为与拼图落盘）。**边缘是否真的学到**：训练结束自动写 `edge_metrics.csv`（每例 edge Dice / precision / recall / ±2 px 容差口径 / 预测边缘占比 / **主掩膜派生边缘 Dice 对照**）与 `edge_vis/edge_branch_cases.png`（边缘 Dice 最差 3 例 + 最好 3 例的四列拼图：原图+GT 轮廓 / GT 边缘带 / 预测边缘概率 / 预测边缘叠加），并在 log.txt 打 `edge summary`；判定读法：边缘头 edgeDice 明显高于「主掩膜过同一边缘算子」的对照才算真学到，≈ 则只是把主掩膜重新描边；预测边缘占比 <0.1% 或 >50% 会显式告警（常数图判负）。**实际结果（11-13 20:58 → 11-14 04:03 收口）：两臂 exit 1，但训练与 `test summary` / `edge summary` 都已正常打出** —— EB-U 86.65 / 0.216 / HD 1.08（对 UNet 86.55 仅 **+0.10**），EB-R50 90.41 / 0.144 / HD 0.69（对 R50 90.4903 **−0.08**），两臂都在噪声内：**边缘监督在无预训练与预训练骨干上都没吃到增益**，与「TransUNet 上 +1.0」的用户先验相反；边缘头 edgeDice 0.321 vs 主掩膜派生 0.279，略高于「重新描边」但不是真学到。崩点在最后存图：[trainer.py](../trainer.py#L654-L655) 用 `d[0]`（含 `/` 的病例 id，如 `PlaqueDataNii_SPARC_S/877824…`）直接拼单图文件名，`edge_vis/<子集>/<病例>/` 子目录不存在 → `FileNotFoundError`。**指标与权重都已落盘，只丢逐例拼图**；修法是把 `/`、`\` 替换成 `_` 再拼名（或对 `dirname` 做 `makedirs`）。状态 CSV：`logs/queue/queue_c2d_edge_branch_status.csv`，脚本 [scripts/queue_c2d_edge_branch.ps1](../scripts/queue_c2d_edge_branch.ps1)。

**队列启动方式必须用 WMI**（否则训练会被终端回收，踩过坑）：

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& ''...\queue_xxx.ps1'' *>> ''...\launcher.log''"'}
```

注意：WMI 不经 shell 解析，**`-File` 形式后不能带 `*>> log` 这类重定向参数**——重定向符与日志路径会被当成位置参数绑进脚本 `param`（2026-11-09 踩坑：脚本空转退出、日志路径被 Join-Path 建成目录）；要捕获队列脚本自身输出必须用上面的 `-Command "& 'script' *>> 'log'"` 包装形式。

---

## 3. 已经得到的结论

### 3.1 基线与横评（口径：官方划分 + bs4/ga2 + 200 epoch，判定点 **终点 epoch 200**，SPARC test n=160）

**全部结果数字（3-seed 基线/双骨干、单 seed 变体横评、复现三网络）统一见 [results.md](results.md) §1–§4**，此处只留读法：

- A0 三 seed 均值 **90.32%**、dual auto 三 seed 均值 **90.32%**（完全持平）→ 双骨干判负。
- 单 seed 横评最佳点：edgedual **90.74%** → ddbefull 90.65% → deform / edgedeform 90.60%——均为单 seed 方向证据，无一达到幅度门槛（|ΔDice| ≥ 0.5% 或 HD 降 ≥ 0.1 mm）。
- 补记快照 `*_SPARC_edge_dual_L1234_deform_c2d`（边缘图 + 双骨干 deform）已纳入本轮 epoch_199 评测，90.60% / 0.1410 / 0.61（见 results.md §2）。

### 3.2 边缘融合（输入级）：信息增量确认，正式训练未过门槛

代码位置：[networks/wavelet.py](../networks/wavelet.py) `build_hf_image()`；开关 `train.py --edge_in 1`。

做法：stem 吃 4 通道 = 原图 3 通道 + 1 张小波高频重建边缘图。

- **先行诊断**（小网络门槛实验）：SPARC +1.81% / Zhongnan +1.83%，两个子集一致 → 边缘通道确实携带原图没有的信息。
- **正式训练**（§1.8）：单骨干 edge ΔDice −0.01%（无差异）；双骨干 edgedual 90.74% 全场最佳但 ΔDice +0.30% → **线索保留，不作结论**。备选：高频能量图（改一行）。

术语与指标含义看不懂 → [study/edge-fusion-explained.md](study/edge-fusion-explained.md)。

### 3.3 踩过的坑（避免重犯）

→ 全部集中记录在 **[pitfalls.md](pitfalls.md)**（进程管理 / Windows 环境 / 网络实现 / 实验管理 / 数据口径 五类），新坑持续追加到该文件。

### 3.4 文献复现三网络（2026-10-30 队列跑完，exit 0×3）

论文底座复现线：XNet（ICCV 2023 小波低/高频双分支 + X 形融合，双输出各自监督）、GobletNet（TMI 2024 双编码器 + FAM 逐级融合）、以及与两者同积木同容量的纯卷积 U-Net 基线。三个网络各自独立训练（`--arch unet|xnet|gobletnet`），与 baseline/dual 线互不混用；官方划分 SPARC、seed1234、200 epoch、等效 batch 8、全部 from scratch（无预训练可载——自定义 block 架构与任何公开 checkpoint 不匹配，且两篇论文官方协议即从头训）。

ep200 指标（Dice / ASSD / HD、参数量、训练耗时）**完整表见 [results.md](results.md) §4**，读法：

1. **XNet vs 同底座 U-Net 基线：方向为正**（87.06% vs 86.55%，+0.5%）——低频/高频双分支 + X 形融合在这个数据上没有兑现论文声称的收益，方向证据保留、不作结论。
2. **GobletNet 更差**（86.28%，比 U-Net 低 0.3%，边界两项也更差），133M 参数但双编码器 + FAM 在小数据上更难训（ep5-10 曲线也落后）。
3. **纯卷积复现线整体比 TransUNet 基线（90.32%）低约 4 个点**：这是不同骨干家族的容量/归纳偏置差异，**不是**实验结论的直接比较对象——复现线内部（U-Net ↔ XNet ↔ GobletNet）同积木同口径互比才有效。
4. 口径注意：本线读数（含 2026-10-31 补算的 HD）取 `epoch_199.pth` 重推理，dice/assd/hd 与 `test_curve.csv` ep200 行一致（终点 epoch 200，与全项目口径一致）；跨线比较只看方向不看小数位。

快照目录（权重根 `E:\mkl\code\python_code\model\TU_Carotid224\`）：`..._epo200_bs4_ga2_224_SPARC_unet_c2d`、`..._epo200_bs2_ga4_224_SPARC_xnet_c2d`、`..._epo200_bs4_ga2_224_SPARC_gobletnet_c2d`。队列脚本 [scripts/queue_repro_nets.ps1](../scripts/queue_repro_nets.ps1)，状态 `logs/queue/queue_repro_nets_status.csv`。网络实现：[networks/unet.py](../networks/unet.py)、[networks/xnet.py](../networks/xnet.py)、[networks/gobletnet.py](../networks/gobletnet.py)。

### 3.5 距离指标（ASSD / HD）诊断（2026-11-14，仅评测不训练；多 seed 集成只作诊断）

完整数字见 [results.md](results.md) §9，读法：

- HD 尾部是**整段局部错位**（每例 >5 px 缺失像素约 96 个），方向近似对称（漏检侧主导 108/160）；只压远处假阳性覆盖不到 2/3。
- 评测后处理（双线性回原网格 / 概率高斯 σ=1 / 3×3 中值）与连通域清理全部无效（ASSD/HD 变化 ≤0.001 mm；158/160 例 n_cc=1）。
- seed 噪声大：单模型 HD 0.614–0.719，逐例 seed 内 sd 0.145 mm，高 HD 组 0.318 mm。**5 seed 概率集成（不训练）把 HD 0.658→0.595、ASSD 0.147→0.137、Dice +0.55**，与单 seed 结构臂最好点持平或更优 → 现有结构增益落在 seed 噪声内。**集成本身只作内部诊断，不作论文方法与报数口径**（2026-11-14 用户决策）；论文报单 seed 配对，过线后 3 seed（42/317/1029）复核报均值±sd。
- 与 best-of-seed 下界（HD 0.490）仍差约 0.10 mm，剩余是系统性局部误差；方向只剩「原分辨率证据 / 漏检侧尾部损失 / 距离图回归」，纯评测侧已无空间。

---

## 4. 下一步等什么

1. **批次 2 已实现未入队**（GobletNet 式：单主干 + 轻 HF 支 + FAM 逐层）：冒烟 5 项全过（恒等起步 max|Δlogits|=0、+1.206M 参数），入队命令 `--hf_branch 1 --hf_lambda 0.3`（快照 `*_SPARC_hfbr3_c2d`），批次 0+1 结论已落，可入队。实现细节见 [experiments.md](experiments.md) §1.9 执行记录②。
2. **批次 3（Syn-Net）已放弃**（2026-11-10 用户决策）：原型臂 split+frc（FrCFusion，局部/全局分离双骨干）88.34，比 A0 低 2.2 点，负增益证实（experiments.md §2 待跑表）；`--fuse_op dct` 不再实现。改道：FGR 复核（在跑）→ 过线则设计 FGR-v2，不过线再议其他论文方法。批次 4（WMREN WRDM）视结论排期。
3. 全部结论目前只在 SPARC 单子集成立；尚无变体达到幅度门槛（|ΔDice| ≥ 0.5% 或 HD 降 ≥ 0.1 mm）——edgedual / ddbefull / deform / edgedeform 四个方向为正的点，择优去 Zhongnan 复现。
4. 病人图像仍在 git 历史里，彻底清除需 `git filter-repo` 重写历史（会改全部 commit hash，**待用户批准**）。
5. **距离指标（ASSD/HD）的诊断与报数口径（2026-11-14 修订：训练侧改推区域 Hausdorff 距离损失）**。证据见 [results.md](results.md) §9.3：量化地板仅 0.003/0.032（**提高分辨率或学习式上采样无空间**）；HD 与斑块面积 ρ=0.64（HD 由大斑块最坏一处决定）；4 条损失臂落差全在 ΔHD ≤0.13、ΔASSD ≤0.016。**难例审计已扩到 40 例（覆盖全部 30 例 5-seed persistent、占全部 HD 的 47.0%）**：GT 全部单连通、无孔洞，GT 轮廓梯度与其余病例相当（0.392 vs 0.389，ρ=−0.075）→ 无标注质量问题的证据；主体是「强边上贴错位置」的跨 seed 稳定口径偏移（39/40，HD 中位 1.03、范围 0.4–2.2，Dice 仍 84–92%；最坏点处图像梯度中位 0.48、17/39 ≥0.6、大斑块中位 0.73，即该处本有强边），**仅 1/40** 是孤立假阳性块（331495，329 px，HD 5.46，随机崩例）。逐例清单 [hard_cases.csv](../Temp/hard_cases.csv)、40 张叠加图汇总页 [review.html](../Temp/hard_cases/review.html)。据此顺序：① **报数口径定稿**——论文一律报单 seed 配对结果（基线 90.4903 / 0.14473 / HD 0.7062），过门槛的臂再按 3 seed（42/317/1029）复核报均值±sd；**5-seed 概率集成只作内部诊断（难例排序、seed 噪声估计），不作论文方法、不作论文读数**；**翻转 TTA 已实测无效（0.9049→0.9097，且属评测期技巧、不进论文），代码已移除、禁用**；② 把难例清单交标注方复核「亮带贴内缘/外缘」口径；③ 训练侧主推**区域 Hausdorff 距离损失**（venue-survey #82，见第 6 条）。文献对照见 [venue-survey.md](venue-survey.md) §14。
6. **下一批：距离类损失两臂（① 在跑，② 代码就绪等 GPU）**。① **cvartail→R50**：`--loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`，与 U-Net 上那条 **+0.82 Dice / HD −0.18** 的臂逐字同参（不调权重），R50 vitL0 / seed 1234 约 42 min，配对 A0 90.4903 / 0.14473 / HD 0.7062——这是目前唯一在本项目数据上双门槛全过的损失臂，先验证它能否迁移（见 [results.md](results.md) §10）；**2026-11-14 19:52 已启动**。② **区域 HD 损失 #82**（文献 MLMI 2025）：主臂 `--loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6`，直接对 miss/extra 两侧距离剖面回传，不是已失败的「像素重加权」写法，同样单 seed 配对，另排论文默认 `--rhd_iters 2` 变体。**代码已落地**（[utils.py](../utils.py) 的 `positive_distance_map_2d` / `regional_hd_loss`，[train.py](../train.py) 注册 `--loss rhd` 与 `--rhd_term/--rhd_w/--rhd_iters/--rhd_alpha`，[trainer.py](../trainer.py) 接线并分项打印），**5 项离线冒烟全绿**（[scripts/smoke_rhd_loss.py](../scripts/smoke_rhd_loss.py)：硬软腐蚀逐像素一致、梯度 vs 差分 3.06e-4、λ=0 diff 0.0、尺度 3.7×、空前景 0）；**尚未跑真实数据 ep1 接线检查、未入队**，GPU 被 ① 占用。判定 ΔHD ≤ −0.1 mm 且 ΔDice ≥ −0.3，过线后 3 seed 复核。细节见 [experiments.md](experiments.md) §2 与 §2.1。

---

## 5. 已知风险

- 病人隐私：工作区与 git 历史仍有病人图像/标识文件（已从当前索引移除，历史未清）。
- 队列不自动续跑：机器重启或断电后需按 `logs/queue/queue_c2d_repro_status.csv` 手工 `-Skip` 重启。
- 旧口径数字：10-25 之前是「旧划分 + 像素口径」；2026-10-30 起判定口径从 ep150/ep149 换为**终点 epoch 200**（A0 90.16%→90.32%），更早文档里的 ep149/ep150 数字全部作废，以 [results.md](results.md) 为准。

---

## 6. 状态归档（2026-11-07 自 CLAUDE.md 「当前状态」移入）

> CLAUDE.md 只保留规则，以下进度记录迁到此节，按时间倒序追加。

- 环境、预训练权重（R50+ViT-B_16）、依赖、Windows `pjoin` 修复：已完成。
- 单通道原图基线与三通道早融合、三通道＋小波引导注意力的消融已完成；**三通道＋小波引导注意力相对单通道原图基线无增益**（差异在 seed 噪声带内），旧轻量侧支路线已否。
- **双骨干交错四级融合**已实现并通过探针验证。融合分支此前因「输出投影零初始化 + γ 零初始化」双重零相乘永久冻结，2026-10-25 已修；最深 14² 接点「侧路无下游消费者」的结构性死支路也已改为串联双向（先更新侧路、主路再用 `qkv_b(B2)` 重取 K/V），探针 `max|Δ|=0`、`gamma_b` 第 1 步起有非零梯度。修复后跑完的首个活融合配置（`--fuse_op window`，前三个接点并行双向、最深接点当时仍是 A→B 单向）Dice 0.9051，**相对单通道基线 0.9092 无增益、四项指标全差**，但存在 `bs8` vs `bs4+ga2` 口径不一致与单 seed 两个混杂。
- **2026-10-24 夜队列（只跑全量 SPARC 单子集）**：`scripts/night_queue_dual.ps1 -Skip r2_sparc,r3_sparc,r7_sparc,*zhongnan` → `scripts/night_queue_dual2.ps1 -Skip *zhongnan`，实际顺序为四点全开·窗内/全局自适应融合 → 可变形大核 → AFF 式幅度调制 → 自适应融合＋跨尺度回流。
- **实测耗时**：全量 SPARC 0.70 s/iter（107 iter/epoch ≈ 75 s）→ **2.5 h/job**；瓶颈是 dataloader（`num_workers=1`），GPU 仅 63% util。旧文档“34.6 s/epoch、72 min”是错的。
- **dev 快筛子集（已撤销）**：`lists_Carotid_dev` + `scripts/night_queue_dev.ps1` + `scripts/summarize_dev.py` 保留备用，中间产物已删。
- 训练超参变更：**batch 4 + `--grad_accum 2`**（等效 8）；batch 8 触发 WDDM 换页，单步 2.6×，禁用。
- **2026-10-26 频域文献复盘**：本地 8 篇 PDF 精读完毕，XNet（ICCV2023）与 GobletNet（TMI2025）结论不矛盾——XNet 的 raw 基线是单编码器（增量混容量），GobletNet 参数量对齐后只剩「高频有用、低频有害」。据此确定：**变换种类不是主要矛盾，注入位置才是**；不再新增更重的融合算子。详见 `technical-design.md`「频率注入位置」。
- **2026-10-26 老师指示（改排期）**：① 新数据集口径——`Carotid2D`（`_S` 目录）＋官方划分（SPARC：训练 350 张/约 101 例、测试 160 张/约 43 例；Zhongnan 448/187 张），旧划分弃用，**全部历史数字只在旧口径下有效**；② 下一结构实验先做**边缘特征融合**，插队到「stem 加 HF 通道」与「特征级 DWT 解耦」之前。
- **2026-10-26 新口径落地**：① Carotid2D 接入完成——root/list_dir 已切、划分四文件就位、快照带 `_c2d` 后缀，冒烟 n=160 与官方一致；② **指标物理单位化**——预测经 `original_grid_pair` 重采样回原分辨率网格后用 NIfTI 头 spacing 算 medpy 距离，**assd/hd 单位 mm、DeltaArea 单位 mm²**，历史数字均为像素口径不可直接比；③ XNet 式低频主路已实现（`--main_input ll`）。
- **2026-10-28 三队列收口（SPARC，官方划分，ep150 判定；旧口径）**：① 双骨干 QKV 跨支注意力三 seed 判负——3-seed 均值 0.9012 vs 单骨干基线 0.9016；② 融合算子五档横评齐：deform 0.9061 唯一三项全优但未过门槛，gate 与 auto 同噪声带，aff 垫底 0.9017；③ BDFR-Net 复现：单级挂载判负，四级全挂 0.9033、+ACAM 完整版 0.9053，方向为正但均未过门槛；④ 主路换 LL 低频图与原图无差异（0.9043 vs 0.9040）；⑤ 输入级边缘图单用无效、配双骨干 0.9072 为全场最佳点但未过门槛。
- **2026-11-07 R50 新协议（`vit_layers=0`，R50+DecoderCup）全部收口**：基线 A 90.4903 / ASSD 0.14473 / HD95 0.41634（seed 1234）。FreqFusion（NeurIPS 2024）90.5729（+0.083，不显著）；DDBE×4 去 ACAM 90.9251 / HD 0.6292（+0.435，ΔHD −0.077，未过线）；DDBE×4+ACAM 关+拉普拉斯 90.9960 / HD 0.6581（+0.506，ΔHD −0.048，单 seed 刚过 Dice 线）；自研 R-GEF 边缘融合 90.4039（−0.086，权重复查 `gamma` 终值 −0.0044/+0.0013/−0.0005，模块实际未启用）。
- **2026-11-07 WRMM 线**：基线 K2（WRMM 随机编码器 + `unet_dec`）87.7742 / HD95 0.6060；去 MLKM −1.006（MLKM 值 +1.01）；去门控 +0.452（门控有害）；去小波支 +0.336。R50 全段 + WRMM 精炼 + `unet_dec` 89.7390 / HD95 0.5051（vs K2 +1.965，vs R50 基线 −0.751）。
- **2026-11-07 在跑队列** `queue_c2d_r50_seedcheck.ps1`（5 job，串行）：R-GEF γ 初值 0→0.1 修正版；基线 A 的 seed 2024/3407；DDBE×4+ACAM+lap 的 seed 2024/3407（给 F3 的 +0.506 做 3-seed 确认）。状态 CSV：`logs/queue/queue_c2d_r50_seedcheck_status.csv`。
- **2026-11-08 seedcheck 收口**（5 job 全 exit 0）：R-GEF γ₀=0.1 → 90.6230（+0.133，不显著）；基线 s2024 90.3567 / s3407 90.4244；DDBE+lap 复跑 s2024 90.9342 / s3407 91.0241（注意这两条是 **ACAM 开**，与 seed 1234 的 ACAM 关不同臂，已按定稿口径移出结果引用）。实测耗时：R-GEF 33 min、R50 基线 28 min、DDBE×4 48 min。
- **2026-11-08 定稿口径**：ACAM 全关，不再进表；后续所有 DDBE 实验只跑 ACAM 关版本。多 seed 默认禁止（CLAUDE.md 第 15 条），需明确同意才排。
- **2026-11-08 extend 队列收口**（`queue_c2d_wmren_r50_extend.ps1`，7 job 全 exit 0，10:59→21:48，合计 10.8 h）：R50U 56 min / RW_nm 95.2 / RW_ng 102.1 / RW_nw 83.2 / RW_lap 111.2 / K2_cmb 98 / RW_cmb 103.1。结果见 results.md §6 表 6-A/6-B/6-C——**WRMM 的四条结论在强骨干（预训练 R50）上全部归零**（最大 |Δ| 0.342，无一过 0.5 门槛）。
- **2026-11-08 当前在跑**：`queue_c2d_r50seq_goblet0.ps1` 的 S1（`r50seq` 串行 WRMM 编码器，22:20 起）。relconst 已收口：90.4366 / HD95 0.3991，相对基线 −0.054、相对空间门控版 −0.186，30.1 min。其后为 GobletNet λ=0 → `queue_c2d_mref.ps1`（MREF depth=4）。三者均为前台 powershell 串行等待，互斥不会并行抢卡。
- **耗时基准（实测）**：R50+DecoderCup 类 28~48 min/job；R-GEF 32~33 min；`r50wrmm`（72M）+`unet_dec` 83~111 min；GobletNet（133M）149 min。databloader 仍是瓶颈（`num_workers=1`）。
- **2026-11-08 结果回填**：`docs/results.md` 新增 §5（R50 新协议：基线 3-seed、FreqFusion、DDBE×4 两档、R-GEF 两版）与 §6（K2 / r50 直连 / r50wrmm 三张骨干分开的 WRMM 留一表 + 组合臂）。
- **2026-11-09 三 job 收口**：`queue_c2d_r50seq_goblet0.ps1` 的 S1/S2 与 `queue_c2d_mref.ps1` 全部 exit 0，GPU 空闲。
  1. **r50seq（WRMM 串进 R50 主链，71.41M）89.6193 / 0.15645 / HD95 0.4449**（95.6 min）：相对 r50wrmm 并联 −0.120、相对 r50 直连 −0.012 → 串行同样零增益（results.md §6(3)）。
  2. **GobletNet λ=0（87.1472 / 0.2092 / HD95 0.6919，144.9 min）**：相对原版 λ=0.3（86.277 / 0.2287）**+0.870**，说明原配置的 0.3·LL 回灌是负资产；HD 两项口径不同（旧快照无 hd95 列），不可直接比（results.md §4(4)）。
  3. **MREF×4 级联（32.38M，12 块，γ₀=0.1）90.3750 / 0.14701 / HD95 0.43244**：ΔDice −0.115 → 容量放大 30 倍仍无增益，排除「模块容量不足」（results.md §5(8)）。
- **待用户决策**：MREF 在强骨干上未过 0.5 门槛，是否按原计划搬到弱骨干（`unet_dec`，需给 `UNetFusion` 加可选 egf 参数并放开 `--edge_fusion` 仅 baseline 的限制）。
- **2026-11-09 A.7 组合协同 + FGR 队列启动**（用户批准的设计，`queue_r50fdf_fgr.ps1` 5 job 串行，20:40 起）：
  1. CTL `--ddbe 4 --ddbe_hf_op plain`：DDBE 去分解对照——三支全吃普通特征、参数量与 haar 版逐位一致，归因「拼接改写 vs 频率内容」（results.md §7(3) 机制问题）；
  2. A1 `--r50_fdf drdac`：FDFUNet DRDAC 挂 R50 编码器四级——残差钩子 + fuse 零初始化恒等起步（**TF32 下不能让 x 过 eye 恒等 conv**：cuDNN 会把输入激活舍入到 10-bit 尾数，1.0×TF32(x)≠x，四级复累 → logits 差 O(1)；残差 + 零初始化 0×任何值=0 与 DDBE 同型、逐位精确，坑已记 pitfalls.md）；
  3. A2 `--r50_fdf drdac_mfdf`：+ 编码器四级/解码器四级 FEM（MFDF only，论文口径 inline 不包 γ；编码器 block1 挂载点 55²/256——TransUNet 的 padding hack 在挂载点之后，FEM 按 (256,55,55) 注册；非恒等属预期）；
  4. A3 `--r50_fdf full`：FEM 开 ASCA（FDFUNet 完整三模块组）；
  5. FGR `--fgr 1`：自研频门改写（拉普拉斯频带 + FcaNet 式 DCT 谱门 + 1×1 改写，门只作用频带支保证 fuse 零初始化精确恒等起步），挂解码器四级。
  统一 §5 R50 协议（200ep / SGD 0.01 / bs4+ga2 / seed 1234），判定 |ΔDice| ≥ 0.5 或 HD95 降 ≥ 0.1，对照 A0=90.4903 / ASSD 0.14473 / HD95 0.41634。快照：`*_SPARC_{ddbe4_ddhfplain,r50fdf_dr,r50fdf_mf,r50fdf_full,fgr}_c2d`。实现：train.py `--r50_fdf/--fgr` 与互斥校验、[networks/fgr.py](../networks/fgr.py)（新）、[networks/vit_seg_modeling.py](../networks/vit_seg_modeling.py) 挂载块、[networks/vit_seg_modeling_resnet_skip.py](../networks/vit_seg_modeling_resnet_skip.py) 钩子、冒烟 [scripts/smoke_r50_fdf.py](../scripts/smoke_r50_fdf.py)（全绿：恒等/参数量/预训练加载）。
  **收口（2026-11-10 00:57，exit 0×5）**：CTL 90.5505（+0.060，plain ≈ 噪声 → DDBE 增益归因频率内容）；A1 90.5246（+0.034）；A2 90.1181（−0.372）；A3 90.2999（−0.190）——FDFUNet 渐进整体判负；FGR **90.7955 / 0.13912 / HD 0.69732**（+0.305，三项同向，未过线、列为多 seed 复核候选）。详见 results.md §5(12)–(16)。

