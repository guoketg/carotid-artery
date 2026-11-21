# 实验方案与结论

> 本文件只放三样东西：**当前口径、在跑 / 待跑、仍在用的结论**。
> 已关闭 / 已判负 / 已作废的线与臂全部移到 [experiments-archive.md](experiments-archive.md)，这里不再复述。
> 完整指标横评见 [results.md](results.md)，结构细节见 [technical-design.md](technical-design.md)。

## 0. 统一口径

| 项 | 值 |
|---|---|
| 数据 | SPARC / Zhongnan 分开训练、分开测试，不混合；开发集固定 SPARC。**现役划分（2026-11-21 起）= 同组研究生划分文件**：SPARC 280 / val 70 / test 160，Zhongnan 357 / 88 / 191，落库 `lists/lists_official/`（`--list_dir ./lists/lists_official`）。**历史臂（2026-11-20 及之前）用旧划分**：SPARC 350 / 160、Zhongnan 448 / 187（`lists/lists_Carotid2D/`），本文其它位置出现的「官方划分」「n=160」「448/187」一律指旧口径，不可与新实验同表比较（详见 [dataset.md](dataset.md) §4） |
| 默认骨干 | **ResNet50 = `--arch baseline --vit_layers 0`**（ImageNet21k 预训练 R50 四级编码器 + DecoderCup，基线 90.4903 / 0.14473 / HD 0.7062）；**U-Net = `--arch unet --unet_style plain --unet_base 64`**（原版 Ronneberger U-Net，from scratch，28.95 M；2026-11-21 起，旧残差版 60.30 M 及其 86.5455 / 0.22500 / HD 1.13618 一行**作废**，新基线以 `logs/queue/queue_punet_bs8_*` 落盘为准）。两骨干 × SPARC/Zhongnan = 四张表，一表一骨干、Δ 不跨表继承 |
| 输入 | 224×224；基线单通道灰度 repeat 3 通道；变体按需加通道（边缘图 / 方向子带） |
| 训练 | **现役（2026-11-21 起 = 对齐锚）**：`--batch_size 8 --grad_accum 1`（真批 8、不做累积，见 §0.1.1）；`--max_epochs 100` + cosine（A0 配方）；`--eval_test_every 0` + best-val 选模。**历史（2026-11-20 及之前）**：`--batch_size 4 --grad_accum 2`（等效 bs8）、`--max_epochs 200`（poly 绑死）、`--eval_test_every 5` |
| 优化 | SGD，lr 0.01，momentum 0.9，wd 1e-4；loss = 0.5·CE + 0.5·Dice |
| 预训练 | R50 编码器默认加载 ImageNet21k 预训练权重（`--arch baseline --vit_layers 0`；对照臂 `--arch wmren --wmren_enc r50 --wmren_ablate unet_dec` 同源权重）。**完整 TransUNet（`--vit_layers>=1`）已封存**，只作历史复现。`--no_pretrain 1` 只作归因对照（V-N1 基线 89.59；DDBE/FGR 的随机初始化归因批见 status.md §2） |
| 读数 | 终点 ep200：`test_curve.csv` 末行，或 `epoch_199.pth` 重推理 |
| 指标 | Dice / ASSD / HD / DeltaArea；ASSD、HD 单位 mm，只用 HD 不用 HD95 |
| 判定 | 主看 ASSD / HD，两项方向一致才认边界改善；\|ΔDice\| ≥ 0.5% 或 HD 降 ≥ 0.1 mm 才算有效 |
| 统计 | 整个测试集直接平均（旧划分 SPARC 160 张 / Zhongnan 187 张；现役划分 160 / 191 张），不做显著性检验；单 seed 噪声带 ±0.3 |
| 多 seed 集合 | **42 / 317 / 1029**（用户指定，与课题组其它工作对齐，便于横向对比；3-seed 补跑一律用这组）；单 seed 默认仍为 1234，历史批次 2024 / 3407 保留原口径 |

### 0.1 现役配方 A0（2026-11-21 起，所有新实验一律用这套）

一条命令点亮：`--recipe a0 --list_dir ./lists/lists_official`，显式传入的命令行项优先于预设。

| 项 | 值 | 来源 |
|---|---|---|
| 数据链 | 先 `scipy.ndimage.zoom(order=1)` 双线性 resize 到 224，再做 1%–99% 分位 min-max；mask 最近邻 + 阈值 | 研究生 A0 包 `dataset.py` |
| 增强 | 水平翻转 0.5、垂直 0.2、±10° 仿射 0.4、gamma 0.5、轻对比度 0.25（不再加 CLAHE） | 同上 |
| 优化器 | AdamW，lr 1e-4，**weight_decay 0.01** | 研究生 `train.py` 默认 |
| 调度 | cosine 退火到 1e-6，绑 `max_epochs` | 同上 |
| 梯度裁剪 | max_norm 1.0 | 同上 |
| 损失 | BCEWithLogits + soft Dice，各权重 1.0 相加（`--loss bce_dice_sum`） | 同上 |
| epoch / batch | 100 ep；**bs 8 + grad_accum 1（真批 8、不做累积）**，2026-11-21 14:12 定案（`Temp\probe_mem.py`：base 臂 bs8 reserved 5830 MiB，dec_WRMM 臂 bs8 实测 8886 MiB 可跑；bs16 在 12 GB 卡上装不下） | 差异项，已登记（他的原包是 bs16） |
| 选模 | 每轮算 val 逐样本 Dice 均值，最高一轮存 `best_model.pth`，测试前载回（`--select_val 1`） | 研究生 best-val 口径 |
| 划分 | `lists/lists_official/`：SPARC 280/70/160，Zhongnan 357/88/191 | §0 数据行 |
| seed | 42（单 seed 只作筛选，过线后扩 42/317/1029） | §0 多 seed 集合 |

与 A0 复现包（`logs/repro_a0/*/run_config.json`：plain unet、bs **16**、base_channel **32**、100 ep、lr 1e-4、wd 0.01、min_lr 1e-6、clip 1.0、224）的差异只剩两处：**batch 8（他 16）**、**默认通道宽 base64 / 28.95 M（他 base32 / 7.24 M）**。`--unet_base 32` 能建出与他逐位同构的 7,239,778 参数版本（他 7,239,745，差 33 来自 2 类头）。「U-Net 行数字一致」的严格核对要么走 `logs/repro_a0/`（他的原包权重，SPARC 0.8803 / Zhongnan 0.9025），要么走 base32 臂；默认 base64 臂只在本仓库内做同协议成对对照。

### 0.1.1 对齐锚：bs 8 · 无梯度累积 · 原版 U-Net（2026-11-21 用户定案，唯一对齐基准）

- **三项锁定**：`--batch_size 8 --grad_accum 1`（真批 8、一次 forward/backward 就 step，不做累积，BN 看到的就是 8 —— `trainer.py` 的累积分支在 `grad_accum=1` 时恒为假）；宿主 = **原版 U-Net**（`--arch unet --unet_style plain --unet_base 64`，Ronneberger 2015 拓扑：MaxPool 下采样 + 纯双卷积 + 双线性上采样 + 1×1 降维，无残差、无预训练、28,946,626 参数）；配方 = §0.1 的 `--recipe a0`。
- **对齐对象与逐项核对**：研究生 A0 包 vs 本仓库 —— lr 1e-4 / wd 0.01 / min_lr 1e-6 / clip 1.0 / 100 ep / 224 / AdamW+cosine / bce_dice_sum / best-val 选模**全部一致**；差异只有 batch（8 vs 16）与通道宽（64 vs 32）。`queue_a0_wrmm.ps1` 与 `queue_a0_matrix.ps1` 的 12 条 U-Net 臂已逐条核对为 `--batch_size 8 --grad_accum 1 --unet_style plain --unet_base 64`。
- **措辞约束**：bs8 + base64 的读数只与「同为本仓库 bs8 + base64」的臂成对比较；与他的表（bs16 + base32）比只能作方向参照，**不得写「与研究生/论文数字一致」**；要真对齐他的数值，另起 `--unet_base 32 --batch_size 8` 的诊断臂并在表头写明 batch 差。
- **作废（不得再入表、不得再当对齐参照）**：旧残差 U-Net（`--unet_style res`，60.30 M）的全部读数（86.5455 / 0.22500 / HD 1.13618、SPARC 89.68、三 seed 87.83）；`bs4 + ga4`（等效 16）那批 U-Net 读数（含 base 臂 89.20）；`bs8 + ga2`（等效 16）批次；以及 **2026-11-21 12:20 之前**的所有 U-Net 读数。
- **同时作废的口径来源**：`scripts/queue_align_unet.ps1`（11-20 写的「对齐」队列）对他配置的解读（wd 0 / 恒定 lr / clip 10 / 0.5+0.5 bce_dice）是拿到 A0 包之前的猜测，与 `run_config.json` 不符，**不得再挂载**。


### 0.2 A0 口径下的现役消融队列（2026-11-21 重跑批）

划分与配方换代后，旧台账数字全部作废，按 `--recipe a0 --list_dir ./lists/lists_official`、seed 42 单 seed 重跑。**基线**由 `scripts/queue_a0_wrmm.ps1` 的 arm 1/4/7/10 提供（U-Net 与 R50+DecoderCup 各 × SPARC/Zhongnan）；**消融**见 `scripts/queue_a0_matrix.ps1`（p=29，16 臂 = 四格 × {DDBE+lap, RHD, DDBE+lap+RHD, FGR(解码器)}）：

| 臂 | 命令行差异（其余逐字相同） | 判定（跑前登记，单 seed 只作筛选） |
|---|---|---|
| DDBE+lap | U-Net `--ddbe 4 --ddbe_hf_op lap`；R50 `--bdfr_v2 1 --bdfr_wave_op lap` | ΔDice ≥ +0.5 且 ΔHD ≤ −0.1 mm（对同格 A0 基线） |
| RHD | `--loss rhd --rhd_base bce_dice_sum --rhd_term sym --rhd_w 0.5`，U-Net `--rhd_iters 6` / R50 `2` | 同上 |
| DDBE+lap+RHD | 上面两者叠加 | 对结构单臂与损失单臂各判一次可加性 |
| FGR（解码器四级） | `--fgr 1` | 同上 |

RHD 的基础项走 `--rhd_base bce_dice_sum`，与 A0 基线只差 `rhd_w·L_rhd`；历史 `ce_dice` 基础仍是默认，旧臂不受影响。**未纳入**：FGR+RHD（旧口径零读数，不属重跑项）、DDBE+FGR 同点串联（已知互覆，只在需要归因时单跑）。

## 1. 在跑（2026-11-07）

| 编号 | 内容 | 状态 | 回填位置 |
|---|---|---|---|
| M1 | K2 + `--wmren_no_mlkm 1`（55.35 M）：MLKM 值多少 | leaveout 队列在跑 | §3.2 |
| M2 | K2 + `--wmren_no_gate 1`（54.37 M）：WRDM 软加权 vs 固定 0.5 | 同上 | §3.2 |
| F1 | R50 全段 → 每级 WRMM `stride=1` 精炼 → `unet_dec`（72.17 M） | r50pre 队列已就绪，等前序退出，未开跑 | §3.3 |
| FGREnc+cvar | 弱骨干 U-Net 同时挂 **FGR（编码器四级频带门控改写，`--fgr_pos enc`，改写后继续下采样并作 skip）** 与 **cvartail 损失（CVaR 距离尾部 + 面积一致项）**：`--arch unet --fgr 1 --fgr_pos enc --loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`，seed **42 / 317 / 1029**（§0 三 seed 集合）。两者作用面不重叠（特征改写 vs 优化目标），无 DDBE+FGR 那种同点互覆风险。判定：3 seed 合并一行报均值 ± sd，主判 ΔDice ≥ +0.5 或 ΔHD ≤ −0.1 mm，参照 UNet 基线 86.5455 / HD 1.13618 与 FGREnc 单挂 88.62 / HD 0.93；另记与 cvartail 单挂 87.37 / HD 0.96 的差额看两项是否可加。**成对 Δ 缺口**：FGREnc 单挂的三 seed 尚未按 42/317/1029 跑（历史 2024/3407 是解码器位置，不属本 seed 集合），要写成对 Δ 需补 3 个对照 run（`-WithControl` 开关已就绪，默认未开） | `queue_c2d_fgr_cvartail_3seed.ps1` 已挂载（2026-11-15 09:00 重挂，PID 7428，优先级 30，排在 `queue_c2d_rhd` 之后、`queue_c2d_distloss_unet` 之前） | results.md §7 |
| **叠加批（结构 + 损失，2026-11-15 09:20 挂载）** | 选臂规则：同宿主同 seed 各自单变量最强 + 机制面不重叠。① `U-FGREnc-RHD` = U-Net + `--fgr 1 --fgr_pos enc` + `--loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6`，对 FGREnc 88.62 / HD 0.93 与 RHD 88.21 / HD 0.88 双参照；② `R50-BDFRlap-RHD` = R50 vitL0 + `--bdfr_v2 1 --bdfr_acam 0 --bdfr_wave_op lap` + 同一 RHD，对 BDFR2-na-bwlap 90.9957 / HD 0.65807 与 A0 90.4903 / 0.7062 双参照。主判 ΔDice ≥ +0.5 且 ΔHD ≤ −0.1 mm，过线后 3 seed（42/317/1029）；不做结构+结构（R50 注入式融合全负、U-Net 上 DDBE 单挂 −0.21） | `queue_c2d_stack.ps1` 已挂载（优先级 5，排当日全部队列之后，ETA 次日 02:00） | results.md §10 |

> 2026-11-16 16:30 状态：臂 1（U-FGREnc-RHD）于 11-16 14:49:41 起跑（前序队列全部收口后自动交棒），臂 2（R50-BDFRlap-RHD）随后自动开始，全部 ETA ≈ 19:35。读数回填 [results.md](results.md) §12。

## 2. 待跑 / 未启动

| 项 | 说明 |
|---|---|
| **cvartail λ=0.2 在 U-Net 上的补齐臂（用户 2026-11-18 直接指派，已挂载等卡）** | `scripts/queue_c2d_cvar_unet_w02.ps1`（优先级 6，单臂，2026-11-18 20:30 挂载）：宿主 `--arch unet`（from scratch，60.3M），SPARC / seed 1234 / bs4+ga2 / ep200 / `eval_test_every 5`，与已过线的 λ=0.1 臂**只差 `--tl_w`**（0.1 → 0.2），`tl_q=0.1`、`tl_area_w=0.05` 逐字不变，快照 `..._224_SPARC_lcvartail_W0.2Q0.1A0.05_unet_c2d`。**动机**：R50 侧的份额探针反解出「按梯度份额对齐需 λ≈0.18~0.2」，但那只是 R50 的口径；U-Net 上只跑过默认 λ=0.1（+0.82 / HD −0.18），因此本臂回答「λ 再抬一档是否还有增量」，同时给出与 λ0.1 的同宿主直接可比点（四个指标同向才算有用）。**〔2026-11-18 口径更正〕「抬到对齐点」这个说法不成立**：U-Net 的 A0 份额 34.9% 本身就是锚点，其 λ=0.1 已等价于 R50 的 λ≈0.18，故 0.2 是 **2× 上沿剂量**而非对齐点；把「份额对齐」按三种口径搬到 U-Net 分别反解出 0.10（A0 模长份额）/ 0.15（整段积分剂量 λ·∫e）/ 0.20（上沿），本臂取上沿。预注册预测（末端 e→0.070~0.074、ΔHD 只能 −0.02~−0.03 故 −0.1 门大概率不可达、ΔDice +0.7~+0.9 与 λ0.1 不可分辨）与推导见 [results.md](results.md) §10.3(d)。判定：对 U-Net 基线 86.5455 / 0.22500 / 1.13618 要求 ΔDice ≥ +0.5 且 ΔHD ≤ −0.1 mm；与 λ0.1 臂比不设门槛、只看方向。**单 seed，按第 21 条只作筛选，不作主张**。挂在 `queue_c2d_cluster_weight_v2`（p=7）之后自动开跑，预计 23:30 起、约 100–120 min。 |
| **cvartail 剂量消融（R50-TransUNet 线，已收口判负，2026-11-17 起封存）** | `queue_c2d_cvar_lambda_r50.ps1`（优先级 48，5 臂单 seed，2026-11-15 10:45 挂载、2026-11-16 11:30 收口）：**宿主是 `--arch baseline --vit_layers 0`（R50 + DecoderCup，不含 ViT）= 2026-11-17 起的默认 ResNet50 对外口径**，读数照常入表；SPARC / seed 1234 / bs4 ga2 / ep200，只动 `tl_w` = 0.05 / **0.2（探针反解的对齐点）** / 0.4 / 1.0，加一条选择器臂 `--tl_q 0.02 --tl_w 0.2`；`tl_area_w` 固定 0.05。判定 ΔHD ≤ −0.1 mm 且 ΔDice ≥ −0.3（配对 A0 90.4903 / 0.14473 / 0.7062）。依据与零训练诊断见 [results.md](results.md) §10.3：收益只落在 A0 HD 的 Q4、同 λ 下 R50 梯度份额只有 U-Net 的一半、q=0.1 的 top-k 仅 20% 是真误差。**结果（§10.3(c)）**：λ 加到 1.0 仍只有 −0.011 HD 且 Dice 单调下滑到 89.71 →「剂量不是瓶颈」预注册判据成立，闸门在误差类型。该臂**未在纯 ResNet50 上重做**，纯 R50 只有默认剂量一臂（ΔDice −0.10 / ΔHD +0.004，判负），见 [results.md](results.md) §13。**2026-11-17 16:44 已补挂纯 ResNet50 单臂 λ=0.2**（`queue_c2d_resnet_loss.ps1 -Only RES-cvar-w0.2`，与 `RES-cvar` 只差 `tl_w`，其余宿主/协议/seed/`tl_q`/`tl_area_w` 逐字相同；排在现有 10 臂之后自动开跑，约 67 min） |
| W-L3 | K2 + `--wmren_wave_op lap --wmren_no_gate 1`，条件臂，未排期 |
| WMREN 损失口径 | `--loss ce_dice46`（原文 0.4/0.6）/ `weave4` / `ce_dice_unc`，代码已接，未排 |
| WMREN 补 seed | K2 / W2 的 seed 2024、3407 未跑 |
| ASR v2 | 已重写（逐通道自适应核、1×1→PixelShuffle、tanh 软约束，60.10 M），`queue_c2d_asr2.ps1` 就绪，未启动 |
| **RHD 的 h / ah 两档在默认 ResNet50 宿主上（2026-11-19 已挂载在跑）** | 宿主 `--arch baseline --vit_layers 0`（R50 + DecoderCup，不含 ViT），配对 A0 90.4903 / 0.14473 / HD 0.7062；两臂只差 term：①「压最坏一点」`--rhd_term h --rhd_w 0.5 --rhd_iters 6`（alpha-LogSumExp 后取最大，α=150），②「两侧取最大」`--rhd_term ah --rhd_w 0.5 --rhd_iters 6`。SPARC 官方划分 / seed 1234 / bs4+ga2 / ep200。判定 ΔHD ≤ −0.1 mm 且 ΔDice ≥ −0.3，单 seed 配对；同宿主参照是 sym I6 +0.15 / −0.036、sym I2 +0.33 / −0.056（均判负）。补跑理由：term 覆盖面，h / ah 此前只在弱骨干与另一套解码器上有读数，Δ 不得跨表继承。脚本 `queue_c2d_rhd_r50_term.ps1`（p=16），状态 `logs/queue/queue_c2d_rhd_r50_term_status.csv` | results.md §10.2a |
| 损失基线 | A0 × Focal / Tversky / Boundary / wIoU，`queue_c2d_loss_baselines.ps1` Phase 1 就绪，未启动 |
| 领域损失消融 | 去边界带 / 去面积项 / 去远假阳项，同队列 Phase 2 就绪，未启动 |
| 领域损失补 seed | 主配置 seed 2024 / 3407，同队列 Phase 3 就绪，未启动 |
| 远假阳项归因（UNet） | 对照 D2（87.01 / 0.202 / 0.99）与基线（86.55 / 0.225 / 1.14）：AB-noFar（`--dl_far_w 0`）判"0.15 的 HD 降幅是否来自该项"，DL-far03（`--dl_far_w 0.3`）看剂量；连同 LFA 已入 `queue_c2d_loss_unet.ps1`，启动器在 `queue_c2d_loss_unet_status.csv` |
| HFBranch + FAM | 代码已实现、冒烟通过，未入队 |
| 跨中心反向 | 待补 Zhongnan → SPARC（需先训 Zhongnan 权重）与 dual / split 分支 |
| split + frc | 88.34（A0 −2.2 点），HD 未重算，未收口 |
| 输入分辨率 320 vs 224 | 未做，且不作为论文卖点；若补须所有对比方法同口径 |
| 边缘监督分支 | 共享编码器 + 两个对等解码器（主分割 + 边缘解码器）；边缘 GT = 原标签 3×3 形态学梯度（dilate−erode），边缘项与主分割同一 `seg_loss`、同权重 1.0。UNet（+22.6M）与 R50 预训练（+7.4M）各一臂，SPARC seed 1234 bs4 ga2 ep200；已挂 `queue_c2d_edge_branch.ps1`，状态见 status.md §2。用户先验 TransUNet 骨干 +1.0 Dice；存在解码器容量混杂 |
| **cvartail 迁移 R50（先跑，零新代码）** | `--loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`，与 U-Net 上那条 +0.82 / HD −0.18 的臂**逐字同参**（不调权重）。R50 vitL0 / SPARC / seed 1234 / bs4 ga2 / ep200（约 42 min），配对 A0 90.4903 / 0.14473 / HD 0.7062。这是唯一已在本项目数据上过双门槛的损失臂，迁移是复现而非调参 |
| **区域 HD 损失 · 主臂（#82）** | `--loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6`：0.5CE + 0.5Dice + λ·`L_AH_sym`（λ=0.5 使 CE:HD = 1:1，对应论文 `weight_ce = weight_lah = 1`）。R50 vitL0 / SPARC / seed 1234 / bs4 ga2 / ep200（约 42 min）。配对对照 A0 90.4903 / 0.14473 / HD 0.7062。**2026-11-15 08:54 已启动**（与下方 I2 臂串行，`queue_c2d_rhd.ps1`，优先级最高）|
| **区域 HD 损失 · 论文默认 I=2** | 同上，仅 `--rhd_iters 2`（论文 `iters=2`）；单变量验证「剖面截断在 2 px 是否够」，我们 HD 尾部中位 5.4 px |
| **区域 HD 损失 · U-Net 复现（已挂载）** | `--arch unet --loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6`（约 100 min）。配对对照 §4(1) 86.55 / 0.225 / 1.14；回答「距离损失在弱骨干上的效应能否复现」（同类 `cvartail` 已给 87.37 / 0.200 / 0.96，+0.82 / HD −0.18）。快照 `..._lrhd_TsymW0.5I6_unet_c2d` |
| **区域 HD 损失 · 最大项（已挂载）** | `--arch unet --loss rhd --rhd_term h --rhd_w 0.5 --rhd_iters 6`：soft-max 压最坏一点，机制上等同 Karimi & Salcudean（IEEE TMI 2020）的「直接优化 HD」一族，与上一行的均值项构成单变量对照（同一实现只换 term）。注意：官方 logsum 含常数 `alpha*log(N)`（224² 约 1624），已在该项内减掉，减后上界 = iters，可与 0.5CE+0.5Dice 等权相加。快照 `..._lrhd_ThW0.5I6_unet_c2d` |
| **边界图加权损失（已挂载）** | `--arch unet --loss boundary --bl_w 5.0`：Kervadec et al.（MIDL 2019）Boundary Loss，用 GT 有符号距离图给预测概率加权，监督落在边界位置而非区域面积。论文用 α 由 0 递增的调度，本项目无调度、改用固定权重，默认 5.0 按初始量级标定（离线 L_B≈0.07 vs 0.5CE+0.5Dice≈0.7），**不扫参**。**该臂替代 cbDice**（MICCAI 2024）：cbDice 需可微骨架 + 欧氏距离变换 + 骨架半径加权，本机 venv 无 skimage/cuCIM 且公式细节未核；cbDice 对距离指标起作用的是「边界 Dice」部分，本臂即该族的无中心线版本。快照 `..._lboundary_unet_c2d` |
| **损失全线迁移到 R50 + 普通 U-Net 解码器（2026-11-16 起，在跑）** | `scripts/queue_c2d_resnet_loss.ps1`（优先级 8，12 臂 + 补挂 cvartail λ0.2 共 13 臂，串行，2026-11-16 16:31 起跑 → **2026-11-18 06:46 全部收口**；13 臂无一过门槛，读数见 results.md 表 13-A）。宿主 `--arch wmren --wmren_ablate unet_dec --wmren_enc r50`（R50 四级 ImageNet21k 预训练直连普通 UNet 解码器，31.00M，无 DecoderCup）——**这是对照臂**，默认 ResNet50 口径是 §5 的 R50 + DecoderCup（`--arch baseline --vit_layers 0`，90.4903 / 0.14473 / HD 0.7062）；本表基线 **89.6309 / 0.16380 / HD 0.82617**（[results.md](results.md) §6 表 6-A「r50 直连」），Δ 不得跨表搬到 DecoderCup 那条线上。只动损失、其余逐字不动：cvartail / RHD-sym I6 / RHD-sym I2 / RHD-h / RHD-ah / boundary / farfp / bndsoft / domain_loss / cons / sw-echo / sw-cnr。判定 ΔDice ≥ +0.5 或 ΔHD ≤ −0.1。**动因**：此前所有「R50」都是 R50-TransUNet，损失在该线上 19 条 run 全灭；纯 ResNet 线损失覆盖为 0。跑前登记：RHD-sym 预期 +0.6~+1.2 / −0.10~−0.20（按骨干强度插值，本线 HD 0.826 夹在 U-Net 1.136 与 TransUNet 0.706 之间）；若这条也只有 +0.2 级，则结论升级为「损失只在无预训练骨干上有效」**（已触发：13 臂最强只有 cons +0.095 / rhd-ah −0.042，见 results.md 表 13-A）** |
| **编码器分带重投影 BRP · R50 位置消融（用户 2026-11-18 直接指派）** | 设计见 [module-suite-design.md](module-suite-design.md) §11。把**解码器上唯一有 R50 正证据的形态**（分带 → 拼接 → 一枚 1×1 重投影 → 零初始化残差，DDBE 四级 +0.435 / 单尺度拉普拉斯 +0.506，plain 对照仅 +0.060）搬到编码器，4C²+28C/级。五臂设计单 seed 1234 / SPARC n=160 / bs4 ga2 / ep200：① 浅层两级 112²/64+55²/256（+287,488）；② 深层两级 28²/512+14²/1024（+5,285,888）；③ 四级全挂（+5,573,376，与 FGR 编码器位置 +11.05M 同宿主可直比）；④ 浅层 + 跨级门（浅级空间门 → 深级 HF/LF，只传门不传内容，+2 标量）；⑤ 浅层 + 随机频带对照（同参数量，归因容量 vs 内容）。**实际只先跑 ①~③ 三臂**（消融主体）；④ 是我方设计主张、⑤ 只在结果为正时才有意义，两者**降为条件臂**（浅层过线再按 `-Only` 追加）。**主读浅 vs 深**：浅正深负 → 位置问题；两端同负 → 形态问题（编码器主链改写关闭）；两端同正 → 由对照臂判容量。四级臂对 FGR 编码器位置 88.85 / 0.1793 / HD 0.8234 比，明显更高才算「位置被形态否掉」。判定沿用 §0：ΔDice ≥ +0.5 或 ΔHD ≤ −0.1 mm（对 A0 90.4903 / 0.14473 / 0.7062），单 seed 过线才按 42/317/1029 复核（须用户同意）。冒烟 [scripts/smoke_band_reproject.py](../scripts/smoke_band_reproject.py) 六项全绿 | `queue_c2d_brp_r50.ps1` 5 臂串行，优先级 26，2026-11-18 07:45 挂载，等当前 U-Net+FGREnc+RHD 臂收口后自动开跑 | results.md §7 |
| 熵门控边界监督（#83，备选） | `--loss bndsoft` 的边界带 CE 乘 `(1−H/ln2)`（H 为预测二类熵，stop-grad），只在高可信边界带保留监督。单臂、仅在第 82 号主臂判负后再排 |
| **损失设计 v2（换角度，见 §2.2；已收口）** | **不再加距离/max 族**（已有 cvartail / RHD-sym / RHD-h / RHD-ah 四个成员）。三个新角度：几何场回归（`--sdf aux`）、斑点扰动一致性（`--cons_w`）、多尺度深监督（`--ds`）。**2026-11-15 已实现 + 冒烟全绿 + 排入队列**：弱骨干 U-Net 五臂（三个单臂 → 几何场+一致性 → 三个全叠）→ 强骨干 R50 四臂（三个单臂 + 全叠），**全部单 seed 1234**。脚本 `queue_c2d_lossv2_unet.ps1`（优先级 45）/ `queue_c2d_lossv2_r50.ps1`（优先级 40），排在 RHD-I6 之后、3-seed 队列之前。**结果（[results.md](results.md) §11）**：U-Net 侧斑点一致性 **+1.39 / −0.182 过线**、几何场+一致性 +1.19 / −0.170、几何场 +0.36、深监督 +0.33，**三项全叠 NaN 作废**；R50 侧四臂 +0.02~+0.22 / −0.027~+0.006 全不过 → 损失侧改造只在弱骨干兑现 |
| **标签侧边缘算子四臂（用户 2026-11-18 直接指派，先 R50 后 U-Net）** | 在已跑过的边缘监督分支（共享编码器 + 两个对等解码器）上只改边缘目标的两个自由度，结构/权重（`--edge_branch_w 1.0`）/seed 1234/bs4+ga2/ep200 逐字不动。① `EB-SOBEL-*`：`--edge_mode sobel_orig` = 原始小 crop 上对二值标签算 Sobel（>0→1），再 nearest 放大到 224；② `EB-MORPH7-*`：`--edge_mode morph224 --edge_kernel 7` = 224 上固定 3 px 内 + 3 px 外。**带宽实测**（3 例，GT 面积 19.4%）：k=3 单边 1 px / 2.44%、k=5 2 px / 4.69%、k=7 3 px / 6.90%（=k=3 的 2.83 倍）、k=9 4 px / 9.11%、k=11 5 px / 11.32%；sobel_orig 2.8%~13.5%（随原始 crop 边长浮动）。若 k=7 过线，归因只能是「带宽 / label 软化」；k=7 与 k=3 的读数差可直接当带宽剂量。对照 morph224 k=3 臂 U-Net 86.65 / R50 90.41（均噪声内）；容量混杂照旧（多一个解码器，UNet +22.6M / R50 +7.4M） | `queue_c2d_edge_label_ops.ps1` 优先级 12，2026-11-18 23:38 挂载（等待聚类加权批收口后自动接力） | status.md |
| **标签侧算子等价性检验（2026-11-18 23:52，第二段队列同日 23:57 取消）** | 用户要求"除 Canny 外还有哪些边缘算子，都试一下"。逐位等价性检验（21 个算子 × 3 例，`Temp/edge_viz/check_operator_equivalence.py`）：Prewitt / Sobel3×3 / Scharr / Robinson / Frei-Chen / Kirsch8 / Laplacian8 与 morph k=3 的 Jaccard **0.998~1.000**（1220~1222 px 同一批点，换算子 ≡ 复跑）；Sobel5×5 ≡ morph k=5（同为 2352 px）。**真正点集不同且带宽可用的四个**：Roberts 1.19%、Laplacian4 2.16%、Canny σ=1 1.45%、多尺度 log-Gabor 能量 2.11%。排除留痕：LoG σ=1/2 → 24%/28%、DoG → 16.5%（正负样本 ≈1:3，标签设计错误），顶帽 7×7 → 0.15%（近乎全空），结构张量相干性 2.81%（连续图不适合当二值标签）。实现保留：`datasets/label_edge_ops.py`（四算子，均作用 224 掩膜），`train.py --edge_mode` choices 扩到 `morph224/sobel_orig/roberts/lap4/canny/pc`，冒烟 `Temp/edge_viz/check_new_modes.py` 六模式形状与占比全合理。**用户 2026-11-18 23:57 决定不排这四个**（带宽 1.19%~2.16% 全部 ≤ k=3 的 2.44%，监督信息量与已跑骨架臂等量，没有判读价值）；队列脚本 `queue_c2d_edge_label_more.ps1` 与优先级表项已删除，代码与 choices 保留以便随时单臂补跑 | 无（不排队） | 无 |

### 2.1 区域 HD 损失（#82）实现与判定

论文做法（Guzzi et al., MLMI 2025，DOI 10.1007/978-3-032-09513-8_7，官方仓库 `lisaGUZZI/HD-Loss` 的 `lah.py` / `lh.py`）：

- **可微正距离图**：`d+(p) = −erode(1−p, I)`，每次腐蚀在 3×3×3（3D）/ 3×3（2D）邻域做**乘积**（梯度连续），I 次后得到的剩余概率即距离剖面；`D+(G)` 同理作用于 GT one-hot。
- **平均项（主用）**：`L_AH_sym = 0.5·[ Σ(D+(G)∘y)/|Y| + Σ(d+(y)∘G)/|G| ]`，分别对应 miss 侧与 extra 侧的距离能量，`norm=0`（各按自身质量归一，消除面积偏置）。
- **最大值项（备选）**：`L_H = smax(smax(D+(G)∘y), smax(d+(y)∘G))`，`smax` 为 `α=150` 的 LogSumExp。
- **组合**：论文 CE 与 HD 项等权（`weight_ce = weight_lah = 1`）；本实现为 0.5CE+0.5Dice，故 λ=0.5 即等权。论文 `iters=2`、`connec=26`（3D）；本实现 2D 取 8 连通。

实现落点：损失函数加在 [utils.py](../utils.py)，`--loss` 注册在 [train.py](../train.py)，分项接线在 [trainer.py](../trainer.py)；新增旗标 `--rhd_term {sym,ah,h}`、`--rhd_w`（默认 0.5）、`--rhd_iters`（默认 6）、`--rhd_alpha`（默认 150）。

入队前冒烟（[scripts/smoke_rhd_loss.py](../scripts/smoke_rhd_loss.py)，全绿才起训练）：

1. **精确性**：二值 GT 上 `d+` 与 `scipy.ndimage.distance_transform_cdt(metric='chessboard')` 在 `I` 层内逐像素相等（软腐蚀 = 硬腐蚀）。
2. **可微性**：概率输入下对 logits 的 autograd 梯度与有限差分相对误差 < 1e-3；`d+` 随 I 单调不减。
3. **恒等性**：`--rhd_w 0` 与原 loss 逐位相同（diff 0.0）。
4. **尺度探针**：ep1 起日志分项打印 `ce / dice / rhd`；λ·`L_AH_sym` 与 0.5CE+0.5Dice 落在同一量级（相差 >5× 则按 train 集读数一次性重定 λ 并记入本文件，不在 test 上调）。
5. **空前景守卫**：GT 无前景的合成 batch 返回 0 且无 NaN（斑块小，bs4 下个别 step 会整批无前景）。

**实现状态（2026-11-14）**：代码已落地，5 项冒烟全绿——`d+` 与 `distance_transform_cdt`（chessboard）在 I=2/4/8 逐像素相等；梯度 vs 中心差分最大相对误差 3.06e-4；`--rhd_w 0` 与原 loss diff 0.0；32² 合成斑块上 ce 0.739 / dice 0.415 / rhd 2.143（0.5CE+0.5Dice = 0.577，**3.7×**，未超 5× 上限）；空前景返回 0。224² 真实尺寸单批（bs4）CPU 前向+反传 0.13 s，rhd 2.718（同批 CE+Dice 约 1.3，约 2×），**开销可忽略**。

**2026-11-14 补充（`term='h'` 的常数偏移修正）**：官方 `L_H = smax(smax(D+∘y), smax(d+∘y))`，其中 `smax(x)=alpha*log Σ exp(x/alpha)` **自带常数项 `alpha*log(N)`**（224² 时 `N≈5.0e4`，常数约 1624），原样与 0.5CE+0.5Dice（约 0.6）相加会把后者完全淹没（实测 1625.96 vs 0.58）。现在该项内减掉 `alpha*log(N)`，得到标准 soft-max：数值上 ≈ 剖面内最大距离，上界 = `iters`，与 `sym`（2.43）/ `ah`（4.93）同量级。**这是对论文式的必要归一化，不是改机制**；主臂 `sym` 的数值不受影响（`rhd_w=0` 恒等性仍为 diff 0.0）。

**U-Net 三臂接线冒烟（2026-11-14）**：新增 [scripts/smoke_distloss_unet.py](../scripts/smoke_distloss_unet.py)，在真实网络上前向+反传一次，不加载数据。结果：U-Net 输出 `(2,2,224,224)`；三臂总损失有限、对参数梯度非零有限（grad_sum 6.9e4–7.8e4）；`rhd-h` = 3.354 落在 `[0, iters]`；`boundary×5` = 0.492 与 Dice 同量级；空前景 `rhd`=0、无 NaN。**三臂已入队 `queue_c2d_distloss_unet.ps1`，排在 cvartail 系列之后，见 status.md §2**；真实数据 ep1 接线检查由该队列首个 job 的日志确认。

判定（本次起统一口径）：单 seed 1234 配对，**主判 HD**（ΔHD ≤ −0.1 mm），**护栏 Dice**（ΔDice ≥ −0.3，防止用 Dice 换 HD）；过线后按 3 seed（42/317/1029）复核报均值±sd，再跑 Zhongnan；5-seed 集成不参与判定，只作事后的难例诊断。

执行顺序与多重比较纪律：① **cvartail→R50**（零新代码，同参复现，先跑）；② 实现 #82 并跑完 5 项冒烟；③ RHD 主臂（I=6）→ 论文默认（I=2）；④ 主臂过线则直接进 3 seed，不过线再决定是否跑 U-Net 复现臂。三条臂在跑前已在本文登记判据，**不按结果挑臂重报**；同一 seed 多条臂只作为「机制筛选」，对外结论只取过线且经 3 seed 复核的那条。

### 2.2 损失设计 v2：换角度（2026-11-15 设计 / 同日实现并排期，**全部单 seed 1234**）

> 触发：用户要求「多叠几个损失，但角度要换，不要再加 Hausdorff/max 那一族」。下面是盘点 + 三个新角度 + 叠加方案 + 归因设计。
>
> **2026-11-15 09:40 更新：三项已全部实现并排期**（用户指定顺序：先在 U-Net 上验证三个损失 → 逐级叠加 1→2→3 → 再迁到 R50 验证，全部单 seed）。
> 落地：`networks/sdf.py` 的 SDFHead 同时接 U-Net（`networks/unet.py`）与 R50（`networks/vit_seg_modeling.py`）；
> U-Net/R50 两侧各加深监督辅助头；一致性项在 `trainer.py` 训练循环里实现（含 `--cons_off` 控制臂）；
> `train.py` 新增 `--ds` / `--cons_w` / `--cons_sigma` / `--cons_gain` / `--cons_warmup` / `--cons_off`，并放开 `--sdf` 的 unet 限制。
> 冒烟 `scripts/smoke_lossv2.py` 八项全绿（结果见 §2.2⑤）。

**① 已占用角度盘点（④ 族不再扩张）**

| 族 | 监督的是什么 | 现有成员 | 项目读数 |
|---|---|---|---|
| ① 区域重叠（mean） | 面积/重合 | CE、soft Dice、面积一致性（cvartail 子项） | 基线 A0 90.4903 / U-Net 86.55 |
| ② 像素重加权（mean 的加权版） | 哪些像素更重要 | Kervadec Boundary、far_fp、DL(ce_bw)、sw/cluster | U-Net ΔDice ≤ +0.46、ΔHD ≤ 0.13；R50 与集群两臂归零（§8、§9.3(d)） |
| ③ 歧义带降监督 | 标注歧义带少给梯度 | bndsoft（LBC）、#83 熵门控（备选） | +0.16 / −0.03 |
| ④ **距离剖面聚合（max/tail）** | **误差 × 距离的聚合算子** | cvartail(CVaR-q)、RHD-sym(mean)、RHD-h(soft-max)、RHD-ah(exact max) | U-Net +0.82 / +1.66；R50 待 I6 |
| ⑤ 固定距离图加权 | 边界邻域加权 | 与 ② 同族（Kervadec / Boundary Weighted CE） | 见 ② |

**④ 已有四个成员，共享同一个信息源（GT 距离图），只换聚合算子**：sym→h 是换聚合，h→ah 是换聚合，CVaR 是换分位。再加第五个（MHD / Zipf / LogSumExp 变体）预期增量趋零，故本批明确排除该族。

**② 三个新角度（互相正交，均不依赖「距离剖面聚合」）**

**N1 几何场回归（SDF / level-set）——零新代码，实现已在库里但从未跑过**

- 监督对象：每个像素的**有符号**距离值 `φ(x) ∈ (−τ, τ)`，`τ = 12 px`（约 1.6 mm，覆盖本数据 HD 全域）；损失 = 窄带 L1（按 τ 归一）+ eikonal `0.02·(‖∇φ̂‖−1)²`。
- 与 ④ 的区别：④ 是 `Σ(误差 × 无符号距离)` 的**聚合**；N1 是 `Σ|φ̂ − φ|` 的**场回归**，携带内外方向与连续位置，梯度把整张场拉成距离函数。
- 诊断依据：39/40 稳定难例是「强边上贴错位置」，且**面积差 ±10% 与 HD 朝向一致**（§9.3(e)）、HD/ASSD = 5.35。这是**有符号位移**；④ 两侧各按质量归一后，无法区分「整体外扩 1 px」与「局部漏 1 px」。
- 落点：`networks/sdf.py`（`SDFHead`，输出 `tanh(body)·τ`）+ `trainer._sdf_target`（scipy 距离变换，逐样本、无群体统计）。旗标 `--sdf aux|out --sdf_w 1.0 --sdf_tau 12`，**已同时接 `--arch baseline`（224²/16，+13.985 k）与 `--arch unet`（224²/64，+27.809 k）**。
- 臂：`aux`（保留分类头，只加窄带辅助项）本轮跑；`out`（输出即场，掩膜 = `φ>0`，logits = ±0.5·φ）冒烟已核对 argmax 等价性，但概率量程饱和到 [1.7e−4, 0.99998]（见 §2.2⑤），只作机制对照、本轮不排。
- 权重：L1 已按 τ 归一，`sdf_w=1.0` 与 0.5CE+0.5Dice 同量级；ep1 日志分项校验，落在 0.2×–5× 之外则一次性重定（不在 test 上调）。
- 判定：主判 `ΔHD ≤ −0.1 mm`，护栏 `ΔDice ≥ −0.3`（与 RHD 同口径，因为它也是冲距离指标去的）。

**N2 不变性监督（斑点扰动一致性）——已实现（`trainer.py` 训练循环 + `trainer` 前置开关），零新参数**

- 监督对象：**同一图像两种斑点实现下的预测一致性**，`L_cons = 1 − Dice(p(x), p(x'))`；`x' = x·(1+n)·g`，`n~N(0, σ²)`、σ = 0.08（乘性斑点），`g ~ U(1−0.1, 1+0.1)` 为每样本标量增益（**实现上不能用幂变换**：输入已做零均值单位标准化、含负值，`x^γ` 无定义，故增益改用乘性标量，物理上同样对应增益/TGC 旋钮）。两视各自照旧算 0.5CE+0.5Dice，一致性只作用于前景概率。
- 与已有各族区别：①②③④ 都在改**逐样本偏差**，N2 改的是**跨扰动方差**，完全不接触 GT 距离图。
- 诊断依据：§9.3(c) 的尾部含**随机崩例**（sd 1.6–2.5，3/40 例但决定上尾）；5-seed 集成把 HD 0.658→0.595 证明「可被平均掉的方差」真实存在 → 一致性正则就是在训练期做**可微自平均**。
- 必带控制臂：**同视角两次前向（不加扰动）**，此时 `L_cons ≡ 0`，用于排除「双前向/BN 统计」带来的假增益。
- 权重：预热——`s = min(1, epoch/20)` 同时乘在第二视角主损失与一致性项上；第 0 轮 `s=0` 时**不做**第二视角前向，与不挂该损失逐位一致，之后线性升到 `--cons_w 0.2`；不动 CE:Dice 比例。
- 成本：+1 前向/步 ≈ 1.6×（U-Net 92→145 min；R50 42→70 min）。

**N3 多尺度深监督（训练方案，推理零开销）——已实现，U-Net 与 R50 标准 DecoderCup 各自挂辅助头**

- 监督对象：112² / 56² 的中间解码输出各算一次 CE+Dice（权重 0.3 / 0.15），推理时丢弃。
- 与其他族的区别：不改损失形式，改**监督加在哪一层**；对 HD 的作用是让粗尺度先把「整段局部错位」对齐，而不是等最后一级去修。
- 已有反证：Syn-Net 的 `--fds`（dual 解码器 14²/28²/56² 三级，0.2/0.3/0.5）给 +0.40，噪声带内、未迁主方法（archive A.4 F1）→ **预期收益小**；但零推理成本、零容量混杂，作为「叠加上限」的探针很便宜。
- 判定同上。

**N4（备选，不排）频带一致性**：`L_spec = Σ_s ‖HP_s(p) − HP_s(gt)‖₁`（拉普拉斯 2 级）。角度是「频带能量」而非位置；风险是把概率推向过置信（GT 是二值，频谱能量集中在边界），超声斑点下容易学噪声。仅在 N1–N3 全灭时再考虑。

**③ 叠加方案（2026-11-15 按用户指定改为「逐级叠加 1→2→3」）**

- 本轮叠加**只在这三项之间**做，不含 RHD-sym（距离剖面族的可加性另案判断，避免一次改两类作用面）。
- 组合式：`L = 0.5·CE + 0.5·Dice + 1.0·L_SDF + 0.2·L_cons + (0.3·L_DS112 + 0.15·L_DS56)`，`L_cons` 前 20 轮从 0 线性升到满权。
- 臂序（每臂单 seed 1234，同宿主同基线配对）：① 几何场单臂 → ② 一致性单臂 → ③ 深监督单臂 → ④ 几何场+一致性 → ⑤ 三个全叠。
- 读数方式：单臂与基线比给**绝对量**；**相邻两臂的差**给边际（④−① 是一致性在几何场之上的增量，⑤−④ 是深监督在两者之上的增量）。
- 判定（跑前登记，不按结果挑臂重报）：主判 ΔHD ≤ −0.1 mm，护栏 ΔDice ≥ −0.3（不许用 Dice 换 HD），ASSD 需与 HD 同向；过线后如要 3 seed（42/317/1029）须先经用户同意（CLAUDE.md §15）。
- 一致性臂若过线，**必须**补 `--cons_off 1` 控制臂（同图双前向、一致性恒 0）才能把增益归给一致性本身；两个队列都留 `-WithControl` 开关，默认未开。

**④ 排期与预算（2026-11-15 已挂载，全部单 seed）**

| 阶段 | 臂 | 宿主 | 单价 | 合计 |
|---|---|---|---|---|
| A 弱骨干筛选 | 几何场 / 扰动一致性 / 深监督（3 臂） | U-Net | 92 / 145 / 92 min | ≈ 5.5 h |
| B 逐级叠加 | 几何场+一致性 / 三项全叠（2 臂） | U-Net | 145 / 145 min | ≈ 4.8 h |
| C 强骨干验证 | 三个单臂 + 全叠（4 臂） | R50 vitL0 | 45–70 min | ≈ 4 h |

- 队列：`queue_c2d_lossv2_unet.ps1`（优先级 45）→ 自动交棒 `queue_c2d_lossv2_r50.ps1`（40）；两者与现有队列全序互斥，任何时刻只有一个 `train.py` 占卡。
- 合计约 14 h（不含控制臂；控制臂两段各加 145 / 70 min）。
- 参照读数（同宿主同 seed）：U-Net 基线 86.5455 / 0.3138 / HD 1.13618；R50 vitL0 基线 90.4903 / 0.14473 / HD 0.7062。

**⑤ 入队前冒烟（`scripts/smoke_lossv2.py`，CPU 跑、不占卡；2026-11-15 实测八项全绿）**

| 检查 | 结果 |
|---|---|
| 恒等起步 | `--ds` / `--sdf aux` 只多挂键（4 / 8 个），共有权重与不挂版本逐位相同；`--sdf_w 0` 时总损失 4.402690 与基线**逐位相同** |
| 参数量 | U-Net：深监督两头 +0.772 k、几何场头 +27.809 k（全程 60.297 M）；R50：+13.985 k / +0.388 k（20.236 M）；一致性零新参数 |
| 输出契约 | 深监督 aux 形状 U-Net 112²/56²（权重 0.3/0.15）、R50 56²/112²（0.15/0.3）；几何场 `aux` 只加监督、主输出不变 |
| 几何场目标 | 方形 GT：内部中心 +12.0、外部 −12.0、窄带占比 0.156；窄带 L1+eik 0.638（主项 4.40）；对主干梯度 6.8e4（辅助项确实训主干） |
| 深监督 | 两级损失 0.53 / 0.67，aux 头梯度 2.95e1、主干 6.6e4 |
| 一致性 | 同图两次前向 2.3e−6（float32 累加误差，非 0）；扰动后概率最大差 0.178、梯度 6.6e4 |
| 空前景守卫 | GT 全空时几何场项目标取 −tau（**修掉了 scipy 对全前景输入返回 1 的坑，原先会给出假的 −1 目标**），该项贡献 0.0，无 NaN |
| `--sdf out` 量程 | argmax 与 `sdf>0` 逐像素一致，但前景概率量程 [1.7e−4, 0.99998]（窄带外 tanh 饱和）→ **本轮只跑 `aux`**，`out` 留作机制对照 |

## 3. 仍在用的结论（每条一行，明细见归档）

### 3.1 基线

- **A0 = 90.32 / 0.146 / 0.64**（3 seed：90.32 / 90.23 / 90.40）。
- **ViT 全局段零贡献**：去掉 12 层后 90.49（+0.17，噪声带内）；只留 ViT 去掉 R50 skip 为 89.19。
- **预训练只值 +0.90**：V-N1（R50 随机初始化）89.59 vs A 90.49；R50 结构本身相对 U-Net 贡献 +3.04。

### 3.2 WMREN 弱骨干

- **K2（WRMM + U-Net 解码）= 87.77 / 0.1923 / HD 0.979**，是本家族最优弱骨干；W2 完整版 87.14。
- 归因：WRMM **+1.22**、REM **−0.63**（负）、交互项 −0.98（强非可加）。
- 门控实测 w_wavelet 0.0032 / 0.0019，小波支实际只经 F1 concat 进入。
- **Haar 支是负资产**（W-L2 去掉后 88.11，+0.34）；**换拉普拉斯 88.75 / HD 0.817（+0.98，过门槛）**；`pyr` 88.38 与 lap 并列。
- DDBE 强骨干对三种基全部不敏感：`lap` 90.32（+0.01）、去掉 90.26（−0.05）、`pyr` 90.29（−0.02）→ 换基收益是骨干强弱相关的。

### 3.3 R50 强骨干单臂（F1）

- **F1 接线**：R50 全段（`block_units=(3,4,6)`，不截断）出 112²/64、56²/256、28²/512、14²/1024 四级特征 → **每级接一个 `WRMM(stride=1)` 只重写该级特征（不出图、不改分辨率）** → 精炼后的四级按 WMREN 方式进 `unet_dec`（浅三级 skip，14² 瓶颈），解码器不动。72.17 M，R50 用 ImageNet21k 预训练。
- 读数：`F1 vs K2`（同一 `unet_dec`、WRMM 编码器从零训，87.77 / 0.1923 / HD 0.979）= 换强骨干值多少；`F1 vs A0`（90.32）只作跨家族参照，不判门槛。
- **E1 已删**（R50 直连 `unet_dec`，31.00 M）：容量只有 U-Net 一半，不是公平的「ResNet + U-Net」对照。`r50deep` 与随机初始化对照同样排除。
- 局限：`stride=1` 同级精炼是我们改的 WRMM 变体（原版 WRDM 自带 stride-2），不能称原文结构；WRMM 本身仍随机初始化。

### 3.4 其他仍在用的结论

- **领域适配损失**：D1（A0）90.73 / D2（U-Net）87.01，唯一在两个骨干上同向为正的改动，但均未达门槛。
- **FDFUNet 模块不可归因**：去掉 drdac / mfdf / asca 只掉 0.02~0.17，HD 反而变好。
- **A0 + 梯度边缘图零增益**（90.45 / 90.46）；梯度优势只在弱骨干（U-Net）成立。
- **多中心联合 91.06（+0.74）靠的是训练集 350 → 798 张**，不能当方法结论。
- **CCFR flow 精修在 SPARC / Zhongnan ≈ 0**（−0.02 ~ −0.13）；A1 的 +1.58 说明收益取决于 Stage-1 残余。
- **小波基维持 Haar**；`bior1.5` 只用于与 XNet / Frontiers 2025 对齐。
- **WeaveSeg 不复现**（3/4 臂不收敛），不再引用；其 ASR 上采样首版判为实现不等价。
- 双边界平行性、后方声影先验均已测并判负。

## 4. 运行命令

```powershell
# 病例级划分（每个子集独立 train/test）
.\venv\Scripts\python.exe datasets\dataset_carotid.py

# 在跑 / 在等的队列（先 -DryRun 看计划）
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_leaveout.ps1   # M1/M2
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_r50pre.ps1      # F1
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lap_swap.ps1 -Only "B-L3"

# 就绪未启动
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_asr2.ps1
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_loss_baselines.ps1

# 单条训练（统一口径）
.\venv\Scripts\python.exe train.py --subset SPARC --seed 1234 `
  --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5

# 批量评测（ep200 / epoch_199.pth，整测试集切片直接平均）
powershell -ExecutionPolicy Bypass -File scripts\eval_batch_ep199.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\eval_batch_ep199.ps1
```

## 5. 日志与产物

- 快照目录：`../model/TU_Carotid224/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_<变体>_c2d/`
  - `log.txt`、`test_curve.csv`（每 5 epoch，末行即 ep200）、`metrics.csv`（逐切片）、`epoch_199.pth`
- 队列日志与状态：`logs/queue/<job>_<ts>.log`、`logs/queue/queue_*_status.csv`
- 禁止复用基线目录名，避免权重覆盖。
