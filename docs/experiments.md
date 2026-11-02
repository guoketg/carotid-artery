# 实验方案与结论

> 只保留**当前口径、已完成结论、待办**。排队时间、pid、故障修复等过程流水见 git 历史与 `logs/`。
> 完整指标表见 [results.md](results.md)；融合结构细节见 [technical-design.md](technical-design.md) 与 `docs/study/`。

## 0. 统一口径（当前有效）

| 项 | 值 |
|---|---|
| 数据 | SPARC / Zhongnan 分开训练、分开测试，不混合；开发集固定 SPARC |
| 输入 | 224×224；基线单通道灰度 repeat 3 通道；变体按需加通道（边缘图 / 方向子带） |
| 训练 | `--batch_size 4 --grad_accum 2`（等效 bs8）；`--max_epochs 200`（poly 衰减总长绑死，禁止中途停）；`--eval_test_every 5` |
| 优化 | SGD，lr 0.01，momentum 0.9，wd 1e-4；loss = 0.5·CE + 0.5·Dice |
| 预训练 | R50+ViT-B_16（ImageNet21k） |
| 读数 | **终点 ep200**：`test_curve.csv` 末行，或 `epoch_199.pth`（权重名 0 基）重推理 |
| 指标 | **Dice / ASSD / HD / DeltaArea**；ASSD、HD 单位 mm，DeltaArea mm²。距离指标**只用 HD（最大表面距离）**，不用 95 分位版本 |
| 判定 | 主看 ASSD / HD；两项方向一致才认边界改善 |
| 统计 | **一次只报整个测试集的一个数**：SPARC 测试集全部 160 张切片直接平均；**不做病例聚合、不做 Wilcoxon/显著性检验** |
| 门槛 | **\|ΔDice\| ≥ 0.5% 或 HD 降 ≥ 0.1 mm**；seed 噪声带 ±0.3% |
| 报告 | 每个变体同时给 Params / FLOPs / 单张时延 |

## 1. 已完成实验

### 1.1 基线（A0，ep200）

| seed | Dice | ASSD | HD |
|---|---|---|---|
| 1234 | 90.32 | 0.146 | 0.64 |
| 2024 | 90.23 | 0.149 | 0.66 |
| 3407 | 90.40 | 0.144 | 0.61 |
| **均值** | **90.32** | 0.146 | 0.64 |

单骨干 TransUNet：R50 局部 + ViT 全局串联，解码器拼 R50 skip。

### 1.2 预处理与早期融合（2026-10-23，已关闭）

A1 三通道早融合 / A2 伪小波支路 / A3 14×14 小波引导注意力。三者 Dice、ASSD 与 A0 的差异全部落在 seed 噪声带内；唯一方向明确的是 A0→A1 的 DeltaArea（预测面积更小，负向）。**结论：单路特征 + 一次重加权无效，该线停止。**

### 1.3 双骨干与融合算子（2026-10-24 ~ 10-28）

结构：主路 = A0 吃原图；侧路 = 独立 R50 吃 LH/HL/HH 三方向子带；112²/56²/28²/14² 四点双向跨支注意力，零初始化 γ 残差；解码器只接主路。

| 配置 | Dice | ASSD | HD | 判定 |
|---|---|---|---|---|
| dual auto s1234 / s2024 / s3407 | 90.46 / 90.37 / 90.12 | 0.146 / 0.146 / 0.153 | 0.65 / 0.67 / 0.70 | 3-seed 均值 90.32 = A0；s3407 掉到噪声带之外 → **判负** |
| dual gate | 90.24 | 0.146 | 0.67 | 未过门槛 |
| dual aff | 90.35 | 0.144 | 0.64 | 未过门槛 |
| dual deform | 90.60 | 0.142 | 0.61 | 方向为正但未过门槛，不作结论 |
| edge（主路 + 边缘图第 4 通道） | 90.32 | 0.146 | 0.61 | 零增益 |
| edgedual | **90.74** | 0.140 | 0.63 | 单 seed 方向证据，未过门槛 |
| edgedeform | 90.60 | 0.141 | 0.61 | 单 seed 方向证据 |
| mainLL（主路吃 LL 低频图） | 90.39 | 0.148 | 0.65 | 无增益 |

**结论：融合算子（gate / aff / deform / frc）均未过 §0 门槛；dual 3-seed 均值与 A0 持平、单 seed 掉出噪声带 → 双骨干线停止追加投入。**

### 1.4 DDBE / ACAM（BDFR-Net）

| 配置 | Dice | ASSD | HD | 判定 |
|---|---|---|---|---|
| ddbe3（单级门槛） | 90.13 | 0.151 | 0.72 | 未过门槛，反而变差（HD 升 0.08） |
| ddbe4（解码器四级全挂） | 90.31 | 0.144 | 0.64 | 无增益 |
| ddbefull（四级 DDBE + ACAM） | 90.65 | 0.142 | 0.61 | 单 seed 方向证据，不作结论 |

### 1.5 文献复现三网络（seed1234，from scratch，ep200）

| 网络 | 参数量 | Dice | ASSD | HD |
|---|---|---|---|---|
| U-Net | 60.3 M | 86.55 | 0.225 | 1.14 |
| XNet | 326.2 M | 87.06 | 0.212 | 1.11 |
| GobletNet | 133.3 M | 86.28 | 0.229 | 1.22 |

三者都比 A0 低约 3.5~4 点（骨干家族差异），**只在复现线内部互比**。GobletNet 垫底：其设计假设（无散斑、边界=锐利高频）不匹配颈动脉超声，高频支补进来的是斑点噪声，参数 2 倍仍倒退。

### 1.6 拆支归因 + 边缘算子横扫 + FrCFusion（2026-10-31 完成）

口径同 §0；8 个 job 串行，00:06→10:05，全部 exit 0（约 10 h）。队列脚本 [queue_c2d_split.ps1](../scripts/queue_c2d_split.ps1)。F1 / S1 两行于 2026-11-01 16:12→18:24 补跑（[queue_c2d_syn_fds.ps1](../scripts/queue_c2d_syn_fds.ps1)）。

| 编号 | 名称 | 配置 | Dice | HD mm | 对照 ΔDice |
|---|---|---|---|---|---|
| A | `cnn_only` | `--vit_layers 0`（R50 保留，ViT 清零） | 90.49 | 0.71 | A0 90.32：+0.17 |
| B | `vit_only` | `--vit_name ViT-B_16 --n_skip 0` | 89.19 | 0.78 | A0：**−1.13** |
| D | `dual_frc` | `--arch dual --fuse_op frc`（四级） | 90.40 | 0.67 | dual auto（s1234）90.46：−0.06 |
| F1 | `dual_frc_fds` | `--arch dual --fuse_op frc --fds 1`：四级 frc 不变，14²/28²/56² 加深监督（0.2/0.3/0.5），主损失仍 0.5CE+0.5Dice | 90.80 | 0.60 | D 90.40：+0.40；A0 90.32：+0.48 |
| S1 | `syn_split_fds` | `--arch split --fuse_op frc_s --fds 1 --loss bce_wiou`：R50 局部支 + ViT-B/16 全局支，两支均吃原图；FrCFusion 严格版；FDS；损失 mean(BCE+wIoU) | 88.40 | 0.84 | split_frc 88.34：+0.06；A0 90.32：−1.92 |

**判定**：

1. **A ≈ A0（+0.17，在 ±0.3 噪声带内）→ ViT 全局段在本任务零贡献**，A0 的分数几乎全部来自 R50 局部支。Syn-Net / frc / 双骨干线全部停止投入。
2. **B 明显更差（−1.13%）**：全局支能单独跑，但 `n_skip=0` 的解码器缺 skip，边界锐度明显下降。
3. **D frc 与 auto 等价（−0.06）**：FrCFusion 的 DCT 频率通道注意力 + 空间支没有超过窗内注意力，多花 2.43 M 参数无收益。问题在「换信息源」这条路，不在融合算子形式。
4. **F1（FDS 深监督）是这条线上唯一正向的一块**：D→F1 为 +0.40（Dice 90.40→90.80），HD 0.67→0.60（−0.07 mm）、ASSD 0.147→0.139（−8%）、|ΔArea| 2.15→2.06。方向一致但幅度仍在 ±0.3 噪声带附近，且 F1 相对 A0 也只有 +0.48，**不足以支撑把 FDS 迁到主方法**。
5. **S1 与 split_frc 持平（+0.06）**：FDS + BCE+wIoU 在「R50 局部支 + ViT 全局支、单点 14² 融合」的骨架上完全不生效（88.34→88.40），HD 0.84 也与 split 线同量级。说明该论文的两块改动只对它有 DCT 频率支的融合结构有效，换回本项目骨干后收益消失。

F1 / S1 是把该论文剩下的两块（FDS 深监督、BCE+weighted IoU）补到 D 行上，看换成本项目骨干与颈动脉数据后是否真涨点。结构边界先写死、防止回填时外推：全局支是单尺度 ViT-B/16（14²/196 token），FrCFusion 只能落 14²，「三级融合」退化为 FDS 的三级监督；DCT 频点沿用 FcaNet top-16，weighted IoU 权重取 `w = 1 + 5·|gt − mean(gt)|`；两支均吃原图，112²/56²/28² 不存在局部-全局互补；主输出另加权重 1.0；保留 ImageNet21k 预训练（论文为 from scratch）。

#### C 组：U-Net 边缘算子横扫

U-Net 无边缘基线 86.55%。网络结构、参数量（60.298 M）、超参完全一致，只换输入第 2 通道的算子。

| 算子 | 定义 | Dice | HD mm | ASSD mm | ΔArea mm²（均值±标准差） | ΔDice | 判定 |
|---|---|---|---|---|---|---|---|
| 无边缘（基线） | 单通道原图 | 86.55 | 1.14 | 0.225 | +0.7±7.1 | — | 参照 |
| `wavelet` | 对数域 1 级 Haar，LL 置零后 IDWT（λ=0） | 87.42 | 1.16 | 0.205 | −0.7±4.0 | +0.87 | 过 Dice 门槛，HD 升 0.02，边界未改善 |
| `wavelet_lam` | 同上，掺回 λ·LL（λ=0.3） | 86.78 | **1.09** | 0.214 | +0.6±5.6 | +0.23 | 落在噪声带内 |
| `energy` | 细节能量 √(LH²+HL²+HH²)，半分辨率插值 | 85.62 | 1.26 | 0.240 | −0.3±5.4 | −0.93 | 唯一负增益，弃用 |
| `sobel` | Sobel 梯度幅值 | 87.52 | **1.06** | 0.199 | −0.4±4.9 | +0.97 | 过 Dice 门槛，HD 降 0.08，方向一致 |
| `laplacian` | 4 邻域 Laplacian 幅值 | **87.90** | 1.25 | **0.195** | −0.0±3.8 | **+1.35** | Dice 最好但 HD 升 0.11，边界变差 |

**结论**：

- 算子确实有影响：Dice 跨度 2.28%，是 seed 噪声带（0.6% 全幅）的 3.8 倍。
- **梯度类（laplacian / sobel）全面优于小波类**：保留方向与原分辨率相位；`energy` 丢方向 + 半分辨率上采样，唯一负增益，弃用。
- `λ·LL` 回灌在 from-scratch U-Net 上无效（+0.23%）。
- **所有算子都补不上架构差距**：最好 87.90 仍比 A0 低 2.42 点，比 U-Net 基线只涨 1.35%。边缘图是有用的输入自由度，但补不上骨干家族差距。
- 后续若在主方法上换梯度算子，需补 A0+edge(laplacian) 对照；本次**未做**，留作待补项。

#### 代码改动（冒烟全过）

| 文件 | 改动 |
|---|---|
| [networks/wavelet.py](../networks/wavelet.py) | `EDGE_OPS` 注册表、`_sobel_mag` / `_laplacian_mag` / `_detail_energy`、统一入口 `build_edge_map`；`build_hf_image` 加 `lf_weight` |
| [networks/unet.py](../networks/unet.py) | `UNet` 加 `edge_in / edge_op`：stem 1→2 通道，forward 内拼边缘图 |
| [networks/dual_encoder_model.py](../networks/dual_encoder_model.py) | 新增 `BidirFrC`（FrCFusion 移植）、`make_fusion` 的 `frc` 分支；`_MSCAM.glb` 去 GroupNorm（修 eval batch=1 崩溃） |
| [networks/ddbe.py](../networks/ddbe.py) | BDFR-Net 式 DDBE（HF / LF / CTX 三支 + 零初始化 + CBAM 残差） |
| [networks/hf_branch.py](../networks/hf_branch.py) | GobletNet 式 HFBranch + FAM（已实现，未入队） |
| [networks/vit_seg_modeling.py](../networks/vit_seg_modeling.py) | `DecoderCup` 加 ddbe 挂载点；`VisionTransformer` 加 `ddbe_level / hf_branch / hf_lambda` |
| [train.py](../train.py) | 加 `--vit_layers`、`--edge_op`、`--fuse_op frc`、`--ddbe`、`--hf_branch / --hf_lambda`；快照名追加对应后缀 |
| [scripts/eval_carotid.py](../scripts/eval_carotid.py) | 同步上述参数；输出 HD |

### 1.7 2026-11-01 队列：解码器整体替换 + 领域适配损失（完成）

**本轮设计依据（两条独立结论，别混为一谈）**：

1. **噪声与伪影确实存在**——散斑（乘性、空间相关，实测残差 lag-1 自相关 0.78）、混响、搏动/探头运动伪影、设备与操作者域差异。不能再假设图像干净。
2. **边界不锐利**——前景/背景 CNR 中位数 0.95、GT 边界带梯度仅全图平均 1.90 倍、5×5 残差高频能量只占方差 0.3%。所以「边界 = 锐利高频」的频域先验在本数据上不成立。

痛点来源、证据级别（已读 / 仅摘要）与逐项实测数字见 [literature.md](literature.md)「二维颈动脉超声的成像痛点」；硬约束见 [CLAUDE.md](../CLAUDE.md) 第 3 节第 12 条。本轮由此得出：**设计只能依赖 GT 几何与低对比度下的形状/面积先验**，不依赖任何「边缘/高频显著」的图像假设。

**GT 几何实测（决定损失设计，先测后设计）**：SPARC 训练集前 30 张，前景面积中位数 4540 px，等效半宽 2A/P 中位数 **18.2 px**（11.3~33.2），**全部 > 10 px → 实心厚块**，不是细管状结构。

据此**否决 clDice**：官方软骨架（方向腐蚀 + 开运算差）在实心区域上恒为空集，实测 `skel_gt.sum = 0`，损失恒等于 1、梯度≈0（已冒烟验证）。这正是「套用文献先验」的典型失败，与 CLAUDE.md 第 3 节第 12 条一致。

**采用 `--domain_loss 1`（只依赖 GT 与预测自身的几何，不假设任何图像层面的边界/噪声特性）**：

| 项 | 权重 | 作用 | 冒烟表现（厚块 GT） |
|---|---|---|---|
| 边界带加权 CE（GT 膨胀−腐蚀带 ±2 px，λ=3） | 0.5 | 误差集中在 GT 边界，抬升该带损失 | 完美预测 0；全背景 4.44 |
| Dice | 0.5 | 区域重合（原有用） | 完美 0 |
| 面积一致性 \|Σp−Σgt\|/Σgt | 0.05 | 直接压 DeltaArea | 半面积 0.50，等面积平移 0 |
| 远处假阳性（GT 膨胀 10 px 之外的预测前沿质量占比） | 0.1 | 压 HD 尾部：漂浮小岛/断裂 | 等面积平移 40 px → 0.375，完美 0 |

**队列**（[queue_c2d_domain_loss.ps1](../scripts/queue_c2d_domain_loss.ps1)，等前面两个队列与所有 `train.py` 退出后接力；约 2.7 h/job）：

| 编号 | 配置 | 对照 |
|---|---|---|
| U1 | `--arch unet --ddbe 4` | U-Net 86.55 |
| U2 | `--arch unet --ddbe 4 --acam 1` | U1；A0 上 BDFR 完整版 90.65 |
| W1 | `--arch wmren`（**已作废**：旧版缺 MLKM、WRDM 右支仅单层 3×3 s2、REM 只两级带 CBAM） | U-Net 86.55 |
| W2 | `--arch wmren` 完整版（4×WRMM = WRDM 双支 + MLKM，3×REM = CBAM+CREM+SAFM，58.1 M） | W1；U-Net 86.55 |
| D1 | `--arch baseline --domain_loss 1` | A0 90.32 / HD 0.64 |
| D2 | `--arch unet --domain_loss 1` | U-Net 86.55 |
| V1 | `--arch unet --asr 1`（WeaveSeg ICCV 2025 ASR 替换 28²/56²/112² 的 up_conv，57.4 M） | U-Net 86.55 |

**结果（2026-11-01 全部 exit 0；下表为 `test_curve.csv` ep200 训练内读数，整测试集切片直接平均）**：

| 编号 | 配置 | Dice% | ASSD mm | HD mm | \|ΔArea\| mm² | 对照 ΔDice |
|---|---|---|---|---|---|---|
| — | A0 基线 | 90.32 | 0.146 | 0.64 | 2.11 | — |
| D1 | A0 + 领域损失 | **90.73** | **0.141** | 0.65 | 2.16 | +0.41 |
| — | U-Net 基线 | 86.55 | 0.225 | 1.14 | 3.36 | — |
| D2 | U-Net + 领域损失 | **87.01** | **0.202** | **0.99** | 3.02 | +0.46 |
| U1 | U-Net + 四级 DDBE | 86.34 | 0.217 | 1.09 | 3.34 | −0.21 |
| U2 | U1 + ACAM（= BDFR 完整版） | 86.20 | 0.224 | 1.08 | 3.50 | −0.35 |
| W1 | WMREN 骨架整体替换（**不完整版，作废**：缺 MLKM、右支无残差块、REM 非完整三件套） | 86.77 | 0.220 | 1.07 | 3.34 | +0.22 |
| W2 | WMREN 完整版（4×WRMM + 3×REM，58.1 M） | **87.14** | **0.201** | **0.93** | **2.90** | W1：+0.37；U-Net：+0.59 |
| X1 | U-Net + 完整 BDFR + 领域损失 | 86.81 | 0.216 | 1.10 | 3.22 | U2：+0.61；D2：−0.20 |
| X2 | A0 + 完整 BDFR + 领域损失 | 90.67 | 0.142 | 0.64 | 2.06 | ddbefull：+0.02；D1：−0.06 |
| V1 | U-Net + ASR 上采样替换（WeaveSeg ICCV 2025） | 81.30 | 0.335 | 1.63 | 5.16 | U-Net：**−5.25** |

判定：

1. **WMREN 完整版是本轮唯一同时过 §0 两项门槛的结构替换**：W2 87.14 比 U-Net 86.55 高 +0.59（≥0.5%），HD 1.14→0.93（−0.21 mm），ASSD 0.225→0.201、\|ΔArea\| 3.36→2.90 同向；相对自身的不完整版 W1 为 +0.37 / HD 1.07→0.93。补齐 MLKM、WRDM 双支与三级 REM 确实带来了差距。**仅 seed 1234 单次，需复跑 seed 2024 才能定论**；且它仍在纯卷积骨干家族内，与 A0 的 90.32 差 3.2 点，不能跨族比较。
2. **BDFR 完整版（DDBE×4 + ACAM）在纯卷积 U-Net 上不复制它在 A0 的 +0.33**：U1 −0.21、U2 −0.35，X1（再叠领域损失）相对 D2 仍为 −0.20；HD 依次 1.09 / 1.08 / 1.10，全部差于 D2 的 0.99。ACAM 归因 U2 − U1 = −0.14，本任务上是负贡献。
3. **领域适配损失是唯一两骨座同向为正的改动**：D1 +0.41、D2 +0.46，D2 的 ASSD 降 10%、HD 由 1.14 降到 0.99（−0.15 mm）。但均未达 §0 门槛（\|ΔDice\| ≥ 0.5% 或 HD 降 ≥ 0.1 mm）。
4. **结构改造与损失改造不叠加**：X2 90.67 vs ddbefull 90.65（+0.02）、vs D1 90.73（−0.06），A0 上两块互不增强；X1 的 +0.61 相对 U2 基本由领域损失单独贡献（D2 单独 +0.46），结构项净贡献落在噪声带内。
5. **ASR（V1 81.30，−5.25）判为实现不等价，暂不作论文结论**：曲线平滑收敛到 81.3 而非发散，排除训练不稳定，指向结构语义偏差——当前实现有三处已知降低自由度（每像素 K² 权重组内共享、核在 H×W 预测后再 PixelShuffle、偏移 clamp ±4），且参数量比 U-Net 少 2.93 M（57.4 M）。要判定 ASR 本身有效与否，必须按官方代码重核后再跑一次。

**接力（2026-11-01 12:20 起，seed 1234，已跑完、结果见上表 X1/X2 行）**：完整 BDFR + 领域损失的两组组合实验，检验“结构改造”与“损失改造”能否叠加：

| 编号 | 配置 | 快照后缀 | 对照 |
|---|---|---|---|
| X1 | `--arch unet --ddbe 4 --acam 1 --domain_loss 1` | `_ddbe4_acam_dl_unet_c2d` | U2 86.20；D2 87.01 |
| X2 | `--arch baseline --ddbe 4 --acam 1 --domain_loss 1` | `_ddbe4_acam_dl_c2d` | A0 上 ddbefull 90.65；D1 90.73 |

## 2. 待办 / 未排期

| 项 | 状态 |
|---|---|
| GobletNet 式 HFBranch + FAM 移植（批次 2） | 代码已实现、冒烟通过，**未入队** |
| WMREN WRDM 双分支 DWT（批次 4） | 已实现（`networks/unet_wmren.py`，4×WRMM + 3×REM），W2 已跑；**待复跑 seed 2024 确认 +0.59** |
| WeaveSeg ASR 实现重核（V1 81.30 判为实现不等价） | **v2 已重写并排队**（逐通道自适应核 C·K²、1×1→PixelShuffle 拆开、偏移改 tanh 软约束；CPU + 整网冒烟通过，60.10 M）；`scripts/queue_c2d_asr2.ps1` 等第二轮 sweep 退出后接力跑 `_asr2_unet_c2d` |
| A0 + edge(laplacian/sobel) 对照 | **已入队**（[queue_c2d_edge_union.ps1](../scripts/queue_c2d_edge_union.ps1) E1/E2）：`ViT_seg` 已透传 `--edge_op`（`build_edge_map` 统一入口，wavelet 路径与旧 `build_hf_image` 逐位一致，A0 复现不受影响；CPU 整网前向通过）。依据：A0+edge(wavelet)=90.32 零增益，但梯度算子只在 U-Net 上测过（sobel 87.52 / laplacian 87.90 vs wavelet 87.42） |
| 多中心联合训练（SPARC 350 + Zhongnan 448 = 798） | **已入队**（同队列 E3）：`lists/lists_Carotid2D_union`（train 798 行、test 仍为 SPARC test 160，ep200 可与 A0 90.32 直接比）；快照 `_mcUnion_c2d`。预期 +0.5~1.5（训练集翻倍），训完用 `eval_cross_center.py` 补 Zhongnan test |
| Zhongnan 跨子集复现（dual / edge 线） | 未跑；换信息源线已判负，优先级最低 |
| split 架构 + frc（`split_frc_frcstrict`，2026-10-31 11:39 跑完） | 结果 88.34%，低于 A0 约 2.2 点；HD 未重算；**尚未收口入档** |
| 损失基线对比（A0 骨干 × Focal / Tversky / Boundary / wIoU） | 代码已接（`--loss` 取 focal / tversky / boundary / bce_wiou，CPU 冒烟通过、梯度有限）；[queue_c2d_loss_baselines.ps1](../scripts/queue_c2d_loss_baselines.ps1) Phase 1，4 个 job，排在 ASR2 之后 |
| 领域损失消融（去边界带 λ=0 / 去面积项 / 去远假阳项） | 同队列 Phase 2，3 个 job；`$dlBestArgs` 待 sweep 结果出来后改这一行 |
| 领域损失主配置补 seed 2024 / 3407 | 同队列 Phase 3，2 个 job；跨 seed 稳定性（§1.1 噪声带 ±0.3%） |
| 跨中心零样本评测 | [eval_cross_center.py](../eval_cross_center.py) 已出 5 条，写入 `logs/queue/cross_center_eval.csv`。**跨中心全线不掉点**（A0 90.32→90.75、U-Net +1.90、W2 +2.15），D1 在 Zhongnan 91.18（相对 A0 +0.43）。待补：反向 Zhongnan→SPARC（需先训 Zhongnan 权重）、dual/split 分支 |
| 双边界平行性先验（平行曲线 / 管壁连续性正则） | **已测，判负**（[stat_priors.py](../scripts/stat_priors.py)，SPARC train 350 张、有效 290）：沿主轴分箱后厚度 CV 中位 **0.333**（IQR 0.272~0.380），两侧边界切向夹角中位 **20.8°**（IQR 16.6~25.1）→ 两条边界不平行，平行曲线正则的前提不成立。附注：相邻箱厚度变化仅 0.039，说明是**沿长轴平滑渐变**而非平行 |
| 后方声影 / 衰减先验（钙化斑块深侧暗区） | **已测，判负**（有效 202/350）：斑块带/同深度侧方参考带比值，下方中位 1.10、上方 0.89，below−above 中位 +0.50 且 **36% 为负** → 无一致方向性。参考区几何存在组织类型混杂，此项只能作弱结论 |
| 输入分辨率敏感性（320 vs 224） | 未做，**且不作为论文卖点**（分辨率是超参数）；若补，须所有对比方法同口径 |

## 3. 运行命令

```powershell
# 病例级划分（每个子集独立 train/test）
.\venv\Scripts\python.exe datasets\dataset_carotid.py

# 训练队列（串行，一次一个 job；先 -DryRun 看计划）
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_split.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_split.ps1

# 单条训练（统一口径）
.\venv\Scripts\python.exe train.py --subset SPARC --seed 1234 `
  --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5

# 领域适配损失（边界带加权 CE + Dice + 面积一致性 + 远处假阳性；快照名追加 _dl）
.\venv\Scripts\python.exe train.py --subset SPARC --seed 1234 --arch baseline --domain_loss 1 `
  --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5

# 批量评测（ep200 / epoch_199.pth，整测试集切片直接平均）
powershell -ExecutionPolicy Bypass -File scripts\eval_batch_ep199.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\eval_batch_ep199.ps1
```

## 4. 日志与产物

- 快照目录：`../model/TU_Carotid224/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_<变体>_c2d/`
  - `log.txt`：完整训练日志
  - `test_curve.csv`：每 5 epoch 的测试集指标（末行即 ep200 读数）
  - `metrics.csv`：逐切片四项指标
  - `epoch_199.pth`：终点权重（另有每 50 epoch 的中途快照）
- 队列日志与状态：`logs/queue/<job>_<ts>.log`、`logs/queue/queue_*_status.csv`
- 禁止复用基线目录名，避免权重覆盖。
