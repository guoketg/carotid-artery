# 项目状态（精简版）

> 本文件只回答三件事：**现在在跑什么 / 已经得到了什么结论 / 下一步等什么**。
> **结果数字统一记在 [results.md](results.md)**（纯结果台账，本文件不放大表）；计划与判定规则见 [experiments.md](experiments.md)。
> 2026-10-23 ~ 10-27 的完整过程记录已归档到 [archive/status-history-2026-10.md](archive/status-history-2026-10.md)。
> 术语解释见 [study/edge-fusion-explained.md](study/edge-fusion-explained.md)；融合算子逐个讲解见 [study/fusion-operators-explained.md](study/fusion-operators-explained.md)。

最后更新：2026-11-12 13:40（新增损失改进筛选队列 `queue_c2d_loss_unet` 2 job 挂载（等 GPU，见 §2 末条）；**在跑：queue_nopre_ddbe_fgr 2 job**（DDBE / FGR 的 R50 随机初始化归因，2026-11-12 12:23 启动，见 §2 末条）；接力链 queue_freqdino 已 11-11 11:06 收口 exit 0×3，结果待回填。此前状态：**queue_fgr_multiseed_zn 3 job**，约 23:10 收口；follow-up 链已挂两级：ZUB 补 Zhongnan unet 基线 → **queue_fbgr_full 三臂接力**（FBGR 分带档已实现、冒烟全绿，见 §2 与 [module-suite-design.md](module-suite-design.md)）；Syn-Net 经用户决策放弃，见 §4 item 2）。`queue_fgr_ddbe_combo` 3 job 15:39 全部收口 exit 0×3：R50 上 DDBE+FGR 组合 90.632 判负（同点双改写互覆）；弱骨干上 FGR +1.70 / 组合 +1.84 双门槛全过，**FGR 为头号候选**。上一批 `queue_r50fdf_fgr` 5 job 已于 00:57 收口，结果在 §5(12)–(16)）

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

## 2. 队列状态（在跑：queue_fgr_multiseed_zn，3 job 串行）

**在跑 `queue_fgr_multiseed_zn.ps1`（2026-11-10 19:41 启动，预计约 3.5h）**：① UFs2024 = unet+fgr SPARC seed 2024；② UFs3407 = unet+fgr SPARC seed 3407；③ ZUF = unet+fgr Zhongnan seed 1234。目的：把 §7(4) 的 unet+fgr +1.70（单 seed）钉成多 seed 证据（与 s1234 合并为均值±标准差一行判定，门槛不变），并做 Zhongnan 方向性验证。状态 CSV：`logs/queue/queue_fgr_multiseed_zn_status.csv`。

**接力链已挂两级（全自动，无需值守）**：① `queue_zn_unet_base.ps1`（PID 16328）等本队列 3 行状态落盘 + GPU 空闲后跑 Zhongnan unet 基线（seed 1234，快照 `..._224_Zhongnan_unet_c2d`），补 ZUF 的成对 Δ（状态：`queue_zn_unet_base_status.csv`）；② **`queue_fbgr_full.ps1`（2026-11-10 21:09 WMI 启动，PID 39212）**等 ZUB 状态落盘 + GPU 空闲后跑 FBGR 三臂（SPARC seed 1234 bs4 ga2 ep200，全走 `--fgr_mode band` 分带档）：FBGR-UNet（`..._SPARC_fgrb_unet_c2d`）、FBGR-R50（`..._vitL0_..._SPARC_fgrb_c2d`，对照 DDBE-lap 90.996 看互覆是否消除）、FULL-U = HFE+FBGR 同开（`..._SPARC_edge_fgrb_unet_eowavelet_c2d`，补消融阶梯最后一行）。第三臂原定 HFE-UNet，但同口径结果已存在（87.42 / +0.87，归档 A.4 C 组）经用户决策改跑 full 档。设计、诊断与判定规则见 [module-suite-design.md](module-suite-design.md)；冒烟 [scripts/smoke_fbgr_band.py](../scripts/smoke_fbgr_band.py) 全绿（恒等起步逐位 0、UNet 参数 +960、梯度链路、旗标接线）。状态 CSV：`logs/queue/queue_fbgr_full_status.csv`。

**接力链第三级：`queue_freqdino.ps1`（2026-11-10 挂载，等 FBGR 三行落盘 + GPU 空闲）**：FreqDINO（arXiv 2512.11335，超声分割）移植批三臂，全部 `--arch baseline --vit_layers 0`（R50-TransUNet）单 seed 1234、bs4 ga2 ep200，模块**四级全挂解码器输出**（28²/256 → 56²/128 → 112²/64 → 224²/16，与 DDBE 四级全挂同口径；用户决策：单点只挂最深层的接法在以往批次里普遍拿不到增益）：① FD-MFEA（`--fd_mfea 1`，两级 Haar + 边界/结构通道注意力，λ=0.3、α=β=0.5，γ 零初始化恒等起步，+0.030M 参数）；② FD-FGBR（`--fd_fgbr 1`，边界原型作 K/V，深两级 8×64 原型 + 8 头×128 维照抄论文，浅两级按通道收缩为 8×min(64,C) / min(128,C)，ω=0.2，W_O 零初始化恒等起步，+1.603M 参数）；③ FD-BOTH（论文原序 MFEA→FGBR 串联）。判定门槛不变（ΔDice ≥ +0.5 且 HD 同向），口径**不引用论文的 +2.21/+0.96**——其消融基线是「DINOv3 冻结 + 轻 adapter」82.35 Dice 的弱头，不是 U-Net。冒烟 [scripts/smoke_freqdino.py](../scripts/smoke_freqdino.py)：四级逐级初始化输出与输入逐位相同、闸门梯度非零、四种开关组合端到端前向通过（105.276 / 105.306 / 106.879 / 106.908 M）。状态 CSV：`logs/queue/queue_freqdino_status.csv`。

**2026-11-12 `queue_nopre_ddbe_fgr.ps1`（在跑，2 job，12:23 启动）**：DDBE / FGR 的**骨干归因批**——把 R50 换成随机初始化（`--no_pretrain 1`），只变这一个因素，回答「§5(3) DDBE +0.435 / §5(16) FGR +0.305 的增益来自 ResNet 结构本身，还是来自 ImageNet 预训练特征」。两臂均 `--arch baseline --vit_layers 0`、SPARC seed 1234 bs4 ga2 ep200、模块配置与预训练臂逐字一致：① NP-DDBE（`--ddbe 4`，Haar / ACAM 关，快照 `..._SPARC_ddbe4_c2d`）；② NP-FGR（`--fgr 1` whole 档，快照 `..._SPARC_fgr_c2d`）。**NP-CTL 不重跑**：随机初始化基线（`..._SPARC_c2d`，2026-11-07，89.59 / ASSD 0.16520 / HD 0.79 / HD95 0.51）与预训练臂同协议同 seed，指标代码此后未改；DDBE/FGR 模板在基础权重初始化之后才创建（RNG 对齐纪律），同 seed 下与 NP-CTL 逐位对齐，仍是成对比较。冒烟：两臂各跑 1 epoch 通过（快照命名确认无 `_pretrain` 段，冒烟目录已清）。归因读法：Δ_nopre ≈ Δ_pre → 增益属结构；Δ_nopre ≪ Δ_pre → 增益依赖预训练；Δ_nopre ≤ 0 → 模块只对预训练特征有效。状态 CSV：`logs/queue/queue_nopre_ddbe_fgr_status.csv`，脚本 [scripts/queue_nopre_ddbe_fgr.ps1](../scripts/queue_nopre_ddbe_fgr.ps1)。

**2026-11-12 13:16 `queue_c2d_head_hr.ps1`（挂载，等 `queue_nopre_ddbe_fgr` 两行落盘 + GPU 空闲后起跑）**：单臂 HR-SPARC = `--head_hr 1`（224² 全分辨率证据支路：I / 5×5 均值 / 21×21 均值 / 9×9 局部标准差 4 通道 → 两层 3×3 → logits 残差，末层零初始化），其余与基线 A0 逐字一致（SPARC seed 1234 bs4 ga2 ep200 `--arch baseline --vit_layers 0`），快照 `TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d_hr`。动机与诊断见 [low-score-samples.md](low-score-samples.md) §6：该级是网络里唯一「无 skip、也不接触原图」的接点，低分样本的 1 px 边界误差正缺这一级证据。冒烟（CPU）：参数 5282、基础权重与基线逐位一致、初始 logits 差 0.0。判定：ΔDice ≥ +0.5 且 ASSD / HD 同向不变差；未过线按该文档 §5 收口。状态 CSV：`logs/queue/queue_c2d_head_hr_status.csv`，脚本 [scripts/queue_c2d_head_hr.ps1](../scripts/queue_c2d_head_hr.ps1)。**13:42 核查：进程列表已无该脚本（status 只有表头），当前并未在等；要跑请在本批队列结束后手动起，避免抢卡。**

**2026-11-12 13:42 `queue_c2d_backbone_3seed.ps1`（挂载，等 GPU 空闲且无其它 `queue_*.ps1` 在等，实际排在 `queue_nopre_ddbe_fgr` → `queue_c2d_loss_unet` 之后）**：骨干归因的 **3-seed 补跑**（用户指定 seed 42 / 317 / 1029，与课题组其它工作对齐以便横向对比；该集合已写入 experiments.md §0 与 CLAUDE.md §15），把 §3.1 的四条单 seed 归因钉成均值±sd。四臂 × 3 seed = 12 job，全部 SPARC bs4 ga2 ep200、单变量只换骨干：① U-Net from scratch（`--arch unet`，快照 `..._s{seed}_SPARC_unet_c2d`）；② R50 结构 + 随机初始化（`--arch baseline --vit_layers 0 --no_pretrain 1`，`..._vitL0_..._s{seed}_SPARC_c2d`）；③ R50 + ImageNet21k 预训练（`--arch baseline --vit_layers 0`，`TU_pretrain_..._vitL0_..._s{seed}_SPARC_c2d`）；④ TransUNet 预训练（`--arch baseline`，`TU_pretrain_..._s{seed}_SPARC_c2d`）。读法按 seed 配对：Δ_pretrain = ③−②，Δ_R50struct = ②−①。seed 1234 的现成读数见 [results.md](results.md) §3.1/§4，不进本批集合；已完成的 job 由 `epoch_199.pth` 判定后直接复用（「有现成的话就直接用」），未完成的旧目录归档 `_bak_*` 后重跑。状态 CSV：`logs/queue/queue_c2d_backbone_3seed_status.csv`，脚本 [scripts/queue_c2d_backbone_3seed.ps1](../scripts/queue_c2d_backbone_3seed.ps1)。

**2026-11-12 13:40 `queue_c2d_loss_unet.ps1`（挂载，等 `queue_nopre_ddbe_fgr` + `queue_c2d_head_hr` 全部落盘且 GPU 空闲后起跑）**：损失改进**筛选批**，2 job 单变量，跑在弱骨干 U-Net（基线 86.55）上，目的是低成本的信号筛选，胜者再单独搬回 R50 复验（弱骨干结论不外推，见 §5 教训）。两臂均 `--arch unet`、SPARC seed 1234 bs4 ga2 ep200、不叠加 `--domain_loss`：① LFA `--loss farfp`（0.5CE+0.5Dice+0.1·远处假阳性，对准 HD 尾部——D1 领域损失里该项只有 w=0.1 且被边界 CE 稀释，从未单独成臂，快照 `..._SPARC_lfarfp_unet_c2d`）；② LBC `--loss bndsoft`（边界带软标签 CE，eps=0.25 / radius=1，+ 硬 Dice，对低可分辨带**降监督**而非加权，快照 `..._SPARC_lbndsoft_unet_c2d`）。依据：[low-score-samples.md](low-score-samples.md) §4–§5 的 CNR~Dice 相关 0.142(p=0.074) / −0.024(p=0.741) 方向不一致，故不做「低 CNR 加权」；误差是 1 px 边界带 + 大斑块单点越界，分别对应 LBC / LFA。冒烟 [scripts/smoke_loss_unet.py](../scripts/smoke_loss_unet.py) 全绿（eps=0 退化为 CE 逐位一致、边界带几何、eps=0.5 全带在 logits=0 处等于 ln2、远 FP 项 gt=pred→0 / 远处孤立块→1.0、两条损失梯度有限）。判定不变（ΔDice ≥ +0.5 且 HD 同向）。状态 CSV：`logs/queue/queue_c2d_loss_unet_status.csv`，脚本 [scripts/queue_c2d_loss_unet.ps1](../scripts/queue_c2d_loss_unet.ps1)。

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

---

## 4. 下一步等什么

1. **批次 2 已实现未入队**（GobletNet 式：单主干 + 轻 HF 支 + FAM 逐层）：冒烟 5 项全过（恒等起步 max|Δlogits|=0、+1.206M 参数），入队命令 `--hf_branch 1 --hf_lambda 0.3`（快照 `*_SPARC_hfbr3_c2d`），批次 0+1 结论已落，可入队。实现细节见 [experiments.md](experiments.md) §1.9 执行记录②。
2. **批次 3（Syn-Net）已放弃**（2026-11-10 用户决策）：原型臂 split+frc（FrCFusion，局部/全局分离双骨干）88.34，比 A0 低 2.2 点，负增益证实（experiments.md §2 待跑表）；`--fuse_op dct` 不再实现。改道：FGR 复核（在跑）→ 过线则设计 FGR-v2，不过线再议其他论文方法。批次 4（WMREN WRDM）视结论排期。
3. 全部结论目前只在 SPARC 单子集成立；尚无变体达到幅度门槛（|ΔDice| ≥ 0.5% 或 HD 降 ≥ 0.1 mm）——edgedual / ddbefull / deform / edgedeform 四个方向为正的点，择优去 Zhongnan 复现。
4. 病人图像仍在 git 历史里，彻底清除需 `git filter-repo` 重写历史（会改全部 commit hash，**待用户批准**）。

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

