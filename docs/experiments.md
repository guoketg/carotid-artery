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

### 1.8 CCFR 复现（ESWA-D-25-34301，2026-11-03 起跑）

论文两阶段：EA-Net（U-Net + 每级空间注意力 + 多尺度边缘监督，Dice+BCE+0.4·Edge）→ logit 空间 rectified flow 精修（速度场 U-Net，结构先验跨注意 + 时间 FiLM + 不确定性门 M，K=5 步 Euler，损失 5:1:5）。论文数据 Carotid25 不公开（Ar-PlaqSegm1 公开，见 Z1/A1），**C1–C5 在本项目 SPARC 350/160 上复现**，方法细节与实现偏差见 [study/ccfr-reproduction.md](study/ccfr-reproduction.md)。

口径：EA-Net 用 Adam 1e-3 / wd 0.01 / 200 ep（§0 轮数），取 ep200 权重冻结；flow 头 AdamW 1e-4 / wd 0.01 / 20 ep，bs4+ga2（≈论文 bs8），K=5。**注意：不是 §0 的 SGD 0.01 + CE/Dice 口径，结果不能与 A0 直接判门槛，只作方法复现读数。**

| 编号 | 配置 | 快照后缀 | Stage1 / Stage2 Dice（SPARC test 160） |
|---|---|---|---|
| C1 | CCFR full，seed 1234 | `_ccfr_c2d` | 81.53 / 80.49 |
| C2 | w/o condition（复用 C1 的 `stage1_ep200.pth`） | `_ccfr_s2_nocond_c2d` | 81.53 / 82.11 |
| C3 | w/o uncertainty guidance（同上） | `_ccfr_s2_nounc_c2d` | 81.53 / 72.14 |
| C4 | CCFR full，seed 2024 | `_s2024_ccfr_c2d` | 82.44 / 82.37 |
| C5 | full + ImageNet21k 预训练 R50 编码器（`--ccfr_pretrain 1`，论文保真版） | `_ccfr_pr50_c2d` | 90.85 / 90.83 |

结论：C5 − C1 = +9.3（预训练骨干有效）；flow 精修在 SPARC 上 −0.02 ~ −1.04，全部落在 C1/C4 的 seed 噪声带（±1.9）内 → **flow 贡献 ≈ 0**。C3 的 −9.39 是本版 `--ccfr_nounc` 关法把 Stage 2 弄坏，不能当论文 w/o uncertainty 的复现值。

C5 存在的原因：论文原文写 EA-Net 输入为灰度复制的 3 通道，"This setting adapts the input requirement of pretrained ResNet backbones"，即论文编码器带 ImageNet 预训练；C1–C4 为与 §0 口径一致而从零、1 通道。本项目自身的差距读数：预训练 R50+ViT 的 A0 = 90.32，纯卷积从零 U-Net = 86.55 / WMREN = 87.14，预训练这一项值 1~3 点 Dice。C5 vs C1 只能读作"预训练 + 3 通道输入"整块的贡献（EA-Net 架构与参数量都不同：22.45 M vs 33.82 M）。

**Z1 / A1（第七队列，同中心 in-domain）**：把 C5 配置原样搬到另外两个数据集各自的训练/测试划分上，用来检验"flow ≈ 0"是不是 SPARC 特有。

| 编号 | 数据（train/test） | 配置 | 快照后缀 | Stage1 → Stage2 Dice |
|---|---|---|---|---|
| Z1 | Zhongnan 448 / 187（348 / 148 例） | 同 C5 | `_Zhongnan_ccfr_pr50_c2d` | 92.14 → 92.07（−0.07） |
| A1 | Ar-PlaqSegm1 316 / 24（过滤空掩膜后） | 同 C5，`--stage1_pick 200` | `_ArPlaqSegm1_ccfr_pr50_c2d` | 69.43 → 69.31（−0.12） |
| A1-s | 同上 | 论文口径修正（§9：边缘标签显式化、`--stage1_pick best`、`u=Pc(1−Pc)`、`Lv` 用 M²、flow 输出零初始化、Stage1 100 ep） | `_ArPlaqSegm1_ccfr_pr50_c2d`（2026-11-04 重跑，旧权重已归档为 `_bak_*`） | 68.65 → **70.23（+1.58）** |
| C6 | SPARC 350 / 104 | 同 A1-s（严格口径） | `_SPARC_ccfr_pr50_strict_c2d` | 90.17 → 90.08（−0.09） |
| Z2 | Zhongnan 448 / 187 | 同 A1-s（严格口径） | `_Zhongnan_ccfr_pr50_strict_c2d` | 91.93 → 91.80（−0.13） |

结论：严格口径下 A1 的增益（+1.58）与论文同量级，而 SPARC / Zhongnan 仍是 −0.1 左右。两者并存说明 flow 的收益取决于 Stage-1 还剩多少**系统性**边界残差：SPARC/Zhongnan 的 Stage-1 已到 90.17 / 91.93，边界误差只剩噪声量级，flow 无处可修；A1 的 Stage-1 只有 68.65、HD 111.36 像素，残差大且方向一致，才学得动。

A1 上一轮为负的主因是 `--stage1_pick 200` 取了末轮：A1 的 Stage-1 test 曲线最高 72.66@ep75、ep200 只有 69.43，Stage 2 从被训坏的 Stage 1 出发先扣 2.9 点。修正后同一份 Stage-1 权重上 Stage 2 涨 1.58，ASSD 22.02→19.33、HD 111.36→101.55 同向改善，与论文 +1.24 同量级。**基线 68.65 是按 val（按病例 10%，32 张）选出的 ep65，不是 test 最优轮**；test 最优轮只作参考、不参与选模。

A1 的口径限制：公开包 541 对里有 201 对空掩膜（论文写的 493/48 是未过滤的全量），过滤后 test 仅 24 张；斑块占图中位 1.5%（SPARC 19.8%/15.5%），224² 上约 600 像素；spacing 未知记 1.0，HD 单位是像素。转换脚本 `scripts/prepare_arplaq.py`。

论文参照（Carotid25 / Ar-PlaqSegm1 Dice）：CCFR 87.06 / 76.03；w/o flow refinement 85.11 / 74.79；w/o condition 86.39 / 75.65；w/o uncertainty 86.62 / 75.81。

### 1.9 WMREN 归因 + 损失口径对照（2026-11-04 起跑，seed 1234）

**编号对照（本节及后文沿用的四个名字，避免再出现无定义的代号）**：

| 写法 | 结构 |
|---|---|
| **W1** | 早期不完整版 WRMM，70.58 M，已废弃，只作历史记录 |
| **W2** | 完整 WMREN（4×WRMM + 3×REM），58.06 M |
| **K1** | WRMM 整体换回 U-Net 编码块、REM 保留（`--wmren_ablate unet_enc`），60.08 M |
| **K2** | REM 整体换回 U-Net 的 cat+DBB、WRMM 保留（`--wmren_ablate unet_dec`），58.01 M |

**先把两篇论文分开，本节所有 job 的结构都是 IJCAI-2025 WMREN，不是 ICCV-2025 WeaveSeg**：

| | WMREN（IJCAI 2025） | WeaveSeg（ICCV 2025） |
|---|---|---|
| 本项目角色 | **主复现对象**（4×WRMM + 3×REM = W1/W2） | 非复现对象，只借两块 |
| 借用内容 | — | ASR 上采样替换（V1，−5.25 判负）；Eq.(12) 五项损失（`--loss weave5`） |
| 自身损失 | Eq.(14) `0.4·CE + 0.6·Dice`，无不确定项 | Eq.(12) `1.0 Lmask + 0.5 Lfg + 0.5 Lbg + 0.3 Ltri + 1.0 Luncert` |
| 对应 job | **未排**（`--loss ce_dice46` 已实现） | **未排**（`--loss weave5 / weave4` 已实现） |

W2 的 +0.59 是"整套结构 + 一整套超参"的合账，本节只拆 **(a) 哪个模块在起作用**。(b) 损失口径这一支已实现但**本轮不跑**：WMREN 本身没有不确定项，`Luncert` 只能从 WeaveSeg 借，收益不能计入 WMREN 的模块增益；要跑时按下面"代码开关"表加 job 即可。

**先说清三件事实**：

1. **WMREN 自己的 Eq.(14) 只有 `0.4·CE + 0.6·Dice`**，正文没有不确定性项。这只是 WMREN 一文的情况，不能推广成"这类论文都没有不确定损失"。
2. 项目里已有的 **WeaveSeg（ICCV 2025）就明确写了五项损失**，[L437–443](../docs/papers_txt/Li_WeaveSeg_Iterative_Contrast-weaving_and_Spectral_Feature-refining_for_Nuclei_Instance_Segmentation_ICCV_2025_paper.txt)：

   `L = 1.0·Lmask + 0.5·Lfg + 0.5·Lbg + 0.3·Ltri + 1.0·Luncert`，权重"empirically determined via validation tuning"。
   其中 `Luncert` 是 Eq.(11) 的 `MSE(Su, U*)`。所以 `--loss weave5` 是**有出处、按权重照搬**的实现；此前把它说成"论文里没有"是我的错，我引的是 WMREN 那一篇。
3. 但 WeaveSeg 正文**没给** `Ltri` 的公式，`U*` 的生成也只在 Figure 4 图注里描述（GT → 三值灰度图 → 梯度幅值 → 腐蚀/膨胀 → 不确定性行为图），细节在补充材料，本项目只有正文 txt。这两处实现是**文档化的替代**，代码里逐条标注了，不能称为"论文原式复现"。

**代码开关**（均已冒烟：CPU 前向 + 1 epoch 端到端）

| 开关 | 作用 | 参数量 |
|---|---|---|
| `--loss ce_dice46` | 论文 Eq.(14)：0.4 CE + 0.6 Dice | — |
| `--loss ce_dice_unc` | Kendall & Gal 同方差加权：`0.5e^{-s_ce}L_ce + 0.5s_ce + 0.5e^{-s_dice}L_dice + 0.5s_dice`，s 初始 0（等价 0.5/0.5），两个标量进优化器 | — |
| `--loss weave5` | WeaveSeg Eq.(12) 五项：`1.0L_mask + 0.5L_fg + 0.5L_bg + 0.3L_tri + 1.0L_uncert`；`Su` 用损失侧 1×1 零初始化头（不进 state_dict），`U*` 由 GT 边界生成 | +3 |
| `--loss weave4` | 同上但 `λ_uncert=0`，单变量判断不确定项本身 | +3 |
| `--wmren_ablate unet_enc` | **四个 WRMM 整体换成 U-Net 编码块**（down_conv + DBB，最深一级用 BasicBlock） | 60.08 M |
| `--wmren_ablate unet_dec` | **三级 REM 整体换成 U-Net 的 cat + DBB 融合**（其后的 DBB 置空，避免重复卷积） | 58.01 M |
| `--wmren_ablate no_wave` | WRDM 只留残差右支（去 Haar 左支、F1、门控） | 36.31 M |
| `--wmren_ablate no_gate` | 两支保留，Eq.(4)(5) softmax 换成固定 0.5/0.5 | 54.42 M |
| `--wmren_ablate no_mlkm` | WRMM 只保留下采样 | 55.40 M |
| `--wmren_ablate no_rem` | REM 退化为 Concat + 1×1 + BN | 54.91 M |

（完整版 58.06 M；U-Net 60.32 M。`unet_enc/unet_dec` 是原位替换，其余结构逐位不动。）

**WRMM 内部补测（2026-11-07 入队，`scripts/queue_c2d_wmren_leaveout.ps1`）**：`no_mlkm` / `no_gate` 原先只能通过单值 `--wmren_ablate` 设置，无法与 K2 宿主（`unet_dec`）叠加，故新增独立布尔开关 `--wmren_no_mlkm`（与 `--wmren_no_wave` / `--wmren_no_gate` 同构，三者互斥）。宿主固定为 K2，因为 §1.9 已把 REM 判负，只有在 K2 上问「WRMM 内部哪块在起作用」才不被解码器污染。

| 编号 | 配置 | 快照后缀 | 参数量 | 作用 |
|---|---|---|---|---|
| M1 | `--wmren_ablate unet_dec --wmren_no_mlkm 1` | `_abunetdec_nm` | 55.35 M | MLKM 多尺度大核单元整体值多少 |
| M2 | `--wmren_ablate unet_dec --wmren_no_gate 1` | `_abunetdec_ng` | 54.37 M | WRDM Eq.(4)(5) softmax 软加权 vs 固定 0.5 |

对照 K2 = 87.77 / 0.1923 / HD 0.979；对照 W-L2（K2 + `no_wave`）= 88.11。

**零训练成本的先行证据：门控在已训好的 W2 上已把小波支关掉**

在 `epoch_199.pth` 上对 SPARC test 160 张做前向探针，统计 Eq.(4) softmax 里分给小波支的平均权重：

| 层级 | mean(w_wavelet) | max(w_wavelet) | ‖F_hwd‖/‖F_res‖ |
|---|---|---|---|
| WRDM1（112²） | 0.0032 | 0.374 | 0.600 |
| WRDM2（56²） | 0.0019 | 0.522 | 0.609 |
| WRDM3（28²） | 0.0008 | 0.066 | 0.596 |
| WRDM4（14²） | 0.0009 | 0.922 | 0.555 |

平均权重 0.08%–0.32%，即 F2 支路里小波几乎被完全抑制（个别像素可到 0.92，但占比极低）。推论：**W2 的小波信息实际只经由 F1 的 concat 进入网络，Eq.(4)(5) 的可学习软选择没有学出"选小波"的行为**。这是 `no_gate` 预计 ≈ 0 差异的先行依据，也意味着 WMREN 相对 U-Net 的增益不太可能来自"小波门控"，更可能来自 (i) MLKM 的大感受野、(ii) REM 的边界/区域调制、(iii) 参数与解码拓扑差异。

**队列**：[queue_c2d_wmren_attrib.ps1](../scripts/queue_c2d_wmren_attrib.ps1)，**只排 2 个 job**，§0 口径（200 ep、SGD 0.01、bs4×ga2、224²、seed 1234）：

| 编号 | 配置 | 快照后缀 | 预计 |
|---|---|---|---|
| K1 | WMREN − WRMM（四个 WRMM 换 U-Net 编码块） | `_wmren_abunetenc_c2d` | ~1.8 h |
| K2 | WMREN − REM（三级 REM 换 U-Net 融合块） | `_wmren_abunetdec_c2d` | ~1.9 h |

**四阶阶梯**（K1/K2 已于 2026-11-04 跑完，判定点 ep200）：

| 阶梯 | Dice | ASSD | HD | \|ΔArea\| | 来源 |
|---|---|---|---|---|---|
| 标准 U-Net | 86.55 | 0.225 | 1.14 | 3.36 | U-Net 基线（§1.1） |
| K1：WRMM → U-Net 编码块（REM 保留） | 86.90 | 0.2206 | 1.146 | 2.833 | `_wmren_abunetenc_c2d` |
| K2：REM → U-Net 融合（WRMM 保留） | **87.77** | **0.1923** | **0.979** | **2.606** | `_wmren_abunetdec_c2d` |
| W2：WMREN 完整 | 87.14 | 0.2009 | 0.930 | 2.902 | `_wmren_c2d` |

**判定（§0：\|ΔDice\| ≥ 0.5 或 HD 降 ≥ 0.1 mm）**

- **WRMM 是增益载体**：K2 − U-Net = **+1.22**，Dice 与 HD（1.14→0.979，−0.16 mm）同向，过门槛。
- **REM 是负贡献**：W2 − K2 = **−0.63**，反向过门槛；K1 − U-Net = +0.35 未过门槛，说明 REM 单独存在时近乎中性，**只有和 WRMM 同用时才把 K2 的 +1.22 拖回 +0.59**。
- **强非可加性**：两条单替换的边际和 = +1.57，完整版只有 +0.59，交互项 −0.98 → 不能用"逐模块相加"解释 W2。
- **推论**：本家族里最该拿去做算子替换的宿主是 **K2（87.77）**，不是 W2（87.14）。W2 上的频率算子对照会被负贡献的 REM 吃掉 0.63 点分辨率（见 §1.12）。
- **局限**：仍为 seed 1234 单次；K2 相对 W2 的 +0.63 已过门槛，但换论文主表前需补 seed 2024/3407。

### 1.10 FDFUNet 减法消融（Neural Networks 2024，2026-11-05 07:58 入队）

已完成对照（同协议：SPARC 官方划分、seed 1234、224²、bs4+ga2、200 ep、Adam 1e-4 cosine、0.5BCE+0.5IoU+Dice，判定点 ep200）：

| 模型 | 参数量 | Dice | ASSD | HD |
|---|---|---|---|---|
| UNet\*（论文同构基线） | 12.58 M | 88.97 | 0.1759 | 0.9547 |
| FDFUNet 完整 | 28.69 M | 89.98 | 0.1616 | 0.7910 |

本次全部为**减法**：只在完整版上关掉一个模块，其余超参与上面的完整版逐字一致，不做加法定向重建。

| job | 变更 | 参数量 | 快照 |
|---|---|---|---|
| `no_drdac` | 编码器 DRDAC → 普通 3×3 conv+BN+MaxPool+ReLU（即 UNet\* 下采样） | 25.18 M | `_fdfunet_abnodrdac_fdf_c2d` |
| `no_mfdf` | 每级 FEM 去掉 MFDF（含 FDF 频域支路），只留 ASCA | 24.82 M | `_fdfunet_abnomfdf_fdf_c2d` |
| `no_asca` | 每级 FEM 去掉 ASCA，只留 MFDF | 19.96 M | `_fdfunet_abnoasca_fdf_c2d` |

- 入口：`--arch fdfunet --fdfunet_ablate {no_drdac,no_mfdf,no_asca}`，与 `--fdfunet_variant unetstar` 互斥（train.py 参数校验）。
- CPU 冒烟：四档前向均为 `(1,2,224,224)`，参数量 28.69 / 25.18 / 24.82 / 19.96 M，与开关一一对应（no_asca 掉 8.73 M，主要是 ASCA 的 `Linear(2C,C)` + `Conv1d(C,C,3)`）。
- 队列：[queue_fdfunet_ablation.ps1](../scripts/queue_fdfunet_ablation.ps1)，3 job 串行，等当前 CCFR 作业退出后自动接力；状态写 `logs/queue/queue_fdfunet_ablation_status.csv`。
- 结果回填本节，报「完整 − 消融」的 ΔDice/ΔASSD/ΔHD，并按 §0 门槛（\|ΔDice\| ≥ 0.5% 或 HD 降 ≥ 0.1 mm）判定。

### 1.11 小波基选择（2026-11-05 入队，seed 1234）

**问题**：`wavelet` 算子此前固定 Haar。基的选择是否影响 Dice / 边界？此前的对外表述是「不做基对比」，本次按需求补上。

**载体**：C 组宿主 `--arch unet --edge_in 1 --edge_op wavelet`（U-Net 60.298 M + stem 第二通道），**只换滤波器组**。Haar 行直接复用 §1.6 的 `..._SPARC_edge_unet_eowavelet_c2d`（Dice 87.42），不重跑，保证同口径。

**不变量**（全部锁死，否则差异分不清来源）：log1p 域、1 级分解、LL 置零、IDWT 重建、逐样本 z-score、U-Net 结构与参数量、§0 训练口径（SPARC / seed 1234 / bs4+ga2 / 200 ep / 每 5 ep 评估 / 取 ep150）。

| 基 | 支撑长度 | 性质 | Dice | HD mm | 状态 |
|---|---|---|---|---|---|
| `haar` (db1) | 2 | 正交、无相位失真、格点对齐 | 87.42 | 1.16 | 已有（§1.6） |
| `db2` | 4 | 正交、最常用短支撑 | 待填 | 待填 | 已入队 |
| `db4` | 8 | 正交、长支撑（GobletNet 官方用基） | 待填 | 待填 | 已入队 |
| `sym4` | 8 | 近似对称正交 | 待填 | 待填 | 已入队 |
| `coif1` | 6 | 近似对称、低阶消失矩 | 待填 | 待填 | 已入队 |
| `bior1.5` | 10/2 | 双正交（XNet 与 Frontiers 2025 用基） | 待填 | 待填 | 已入队 |

**实现与校验**

- [networks/wavelet.py](../networks/wavelet.py)：`BASIS_FILTERS` 硬编码 PyWavelets 标准系数表（运行时不依赖 pywt），新增 `dwt2 / idwt2` 可分离卷积实现（分析 `dec_*`，合成 `rec_*` 反序），`build_hf_image / build_edge_map` 增加 `basis` 参数；`basis="haar"` 仍走原 `haar_dwt / haar_idwt`，历史结果逐位不变。
- PR 校验（`Temp/probe_wavelet_basis.py`，224²，float32）：6 个基重建 `max|Δ|` 均在 4.8e-07 ~ 9.5e-07，级联可逆。
- 边界处理：延拓长度取整个滤波器长度 L（不是 L/2），否则 db4/sym4 最外圈几行几列无法精确重建。
- 冒烟：6 个基的 `build_edge_map` 输出均为 `(B,1,224,224)`、有限值；UNet 前向均为 `(1,2,64,64)`。
- 入口：`--wavelet_basis {haar,db2,db4,sym4,coif1,bior1.5}`（仅 `--edge_op wavelet/wavelet_lam` + `--edge_in 1` 生效），快照名追加 `_wb<basis>`。
- 队列：[queue_c2d_wavelet_basis.ps1](../scripts/queue_c2d_wavelet_basis.ps1)，5 job 串行，干跑已确认快照名。

**先验注意**（回填结论时要带上）

- 基之间存在相位差（`db*` 非对称），所以 HF 图在空间上有平移；随机图上与 Haar 的相关性 db2 0.92 / db4 0.79 / sym4 0.92 / coif1 0.88 / bior1.5 0.98。
- Haar 对**恰好落在 2 像素块边界**的阶跃无响应（格点对齐盲区），长支撑基没有该问题；真实超声图像一般不会整幅对齐，但这是基差异的真实来源之一。
- `bior1.5` 的分解高通只有 2 个非零抽头（本质是 Haar 高通），预期与 Haar 接近，主要差异来自低通形状；它的价值在于与文献（XNet / Frontiers 2025）对齐，而不是提供新的高频响应。

### 1.12 模块内 Haar → 拉普拉斯替换（2026-11-06 入队）

**问题**：把 IJCAI WMREN 的 WRMM（K2 宿主，§1.9）和 3612 Submission 的 DDBE 高频支（BDFR 复现）里的 Haar 分解换成项目在输入级验证过的拉普拉斯算子，是否还能保住/放大增益。

**算子口径**：直接复用 §1.6 边缘算子测试的 `_laplacian_mag` + `_standardize`（4 邻域 `[[0,1,0],[1,-4,1],[0,1,0]]` → `abs` → 逐样本 z-score），不引入新的签名族。为对齐 DWT 的四子带/半分辨率接口：`abs` + 标准化后做 2×2 均值池化，复制到 3 个细节槽；LL 仍用 Haar 低通。depthwise 到 C 通道；**零参数**。

W-L1 的口径有一个结构性妥协：`abs` 丢符号、且三个细节槽是同一幅幅值图的三份复制（有效频带只有 2）。因此增加 **W-L4 = 拉普拉斯金字塔（DoG）**：`G1=LL(x), G2=LL(G1), G3=LL(G2)`，`L_k = G_{k−1} − up(G_k)`，三个细节槽对应**三个真实尺度**并保留符号，同样零参数、接口逐位同形（`laplacian_pyramid_channels`）。`Δ(W-L4 − W-L1)` 分离出「回符号 + 多尺度」这一项本身的贡献；`Δ(W-L4 − W-L0)` 才是「最正统的拉普拉斯分解 vs Haar」的净效应。

| job | 宿主 | 变更 | 快照 | 回答 |
|---|---|---|---|---|
| W-L0 | K2（WRMM + U-Net 解码） | Haar（已有 87.77） | `_wmren_abunetdec_c2d` | 参照 |
| W-L1 | K2 | `--wmren_wave_op lap` | `_wmren_abunetdec_wolap` | 算子主效应 |
| W-L2 | K2 | `--wmren_no_wave 1`（可与 `unet_dec` 叠加） | `_wmren_abunetdec_nw` | 频域支整体值多少 |
| W-L3 | K2 | `lap` + `--wmren_no_gate 1` | `_wmren_abunetdec_wolap_ng` | 条件臂：算子被真正使用时 |
| W-L4 | K2 | `--wmren_wave_op pyr` | `_wmren_abunetdec_wopyr` | 正统拉普拉斯金字塔臂（对照 W-L1） |
| B-L0 | A0 + DDBE×4 | Haar（已有 90.31） | `_ddbe4_c2d` | 参照 |
| B-L1 | A0 + DDBE×4 | `--ddbe_hf_op lap` | `_ddbe4_ddhflap` | 算子主效应 |
| B-L2 | A0 + DDBE×4 | `--ddbe_hf_op none` | `_ddbe4_ddhfnone` | 频域支整体值多少 |
| B-L3 | A0 + DDBE×4 | `--ddbe_hf_op pyr` | `_ddbe4_ddhfpyr` | 金字塔口径主效应（B-L1 的公平版） |

**结果（2026-11-07；W-L1/W-L2/W-L4/B-L1/B-L2 全部 exit 0，B-L3 排队中；ep200 `test_curve.csv`）**：

| job | Dice% | ASSD | HD95 | HD | \|ΔArea\| | 相对 L0 |
|---|---|---|---|---|---|
| W-L0（K2, Haar） | 87.77 | 0.1923 | 0.6060 | 0.979 | 2.606 | 参照 |
| W-L1（lap） | **88.75** | 0.1710 | 0.5097 | 0.817 | 2.482 | **Δ_op = +0.98** |
| W-L2（no_wave） | 88.11 | 0.1895 | 0.6191 | 0.929 | 2.692 | **Δ_br = +0.34** |
| W-L4（pyr） | 88.38 | 0.1724 | 0.4980 | 0.804 | 2.566 | Δ = +0.61 |
| B-L0（A0+DDBE4, Haar） | 90.31 | 0.1442 | 0.3999 | 0.64\* | 2.097 | 参照 |
| B-L1（ddhflap） | 90.32 | 0.1447 | 0.3916 | 0.6397 | 2.059 | Δ_op = +0.01 |
| B-L2（ddhfnone） | 90.26 | 0.1450 | 0.3902 | 0.6224 | 2.159 | Δ_br = −0.05 |

\* B-L0 的 csv 无 `hd` 列，取自 §1.4 文档值。

**结论**（WMREN 宿主）：

1. **Δ_br = +0.34 > 0 → Haar 小波支本身就是负资产**，去掉它分数反而更高（且 HD 0.979→0.929 同步改善）。所以 §1.9 门控探针（w_wavelet≈0.003）的读数不是「已学会忽略」，而是「这支撑死重」。
2. 在这个前提下 `Δ_op = +0.98` 才有意义：换成拉普拉斯后 Dice 88.75、HD 0.817、ASSD 0.171，**四项指标全部改善，过 §0 两项门槛**。拆开看 ≈ 去掉 Haar 支的 +0.34 + 换成拉普拉斯的 +0.64。
3. **W-L4 与 W-L1 并列**（Dice 88.38 vs 88.75，差 −0.37 在噪声带内；pyr 的 HD 0.804 / HD95 0.498 略优于 lap 的 0.817 / 0.510）。即「回符号 + 三真实尺度」没有带来可分辨收益，**瓶颈不是 `abs` 整流，而是 Haar 基本身**——这正是「两支都为正方可归因为换基」的情形。
4. **DDBE 宿主上一切都在噪声内**（Δ_op +0.01、Δ_br −0.05，HD 0.622~0.640）：该宿主对 HF 支完全不敏感，与 §1.4「ddbe4 零增益」一致。**换基收益是宿主相关的**，不能写成通用结论。

**判定**：`Δ_op = L1 − L0`，`Δ_br = L2 − L0`。|Δ| < 0.3 记「不可分辨」；0.3~0.5 方向证据；≥0.5 且 HD 不退化才过 §0 门槛并补 seed 2024。**只有 `Δ_br < 0` 时 `Δ_op > 0` 才能归因为「拉普拉斯更好」**；若 `Δ_br ≈ 0`，结论是这条频域支整体是死重，算子怎么换都一样。DDBE 侧再读 `Δ_pyr = B-L3 − B-L0`：若 `B-L1 < B-L0 < B-L3`，说明拖累来自 `abs` + 三槽冗余而非拉普拉斯基；若 B-L3 也为负，才是「该宿主不适合换基」。

**预注册预期**：WMREN 的 `|Δ_op| < 0.3`（§1.9 的 F2 门控探针权重 0.003，该支路已基本关闭）；DDBE `≈ 0`（宿主自身 0 增益）。若 DDBE 出现 ≥0.5，是唯一值得追的信号。

**预注册命中情况**：WMREN 侧 `|Δ_op| < 0.3` **被推翻**（实际 +0.98）。失效原因不是「门控已把该支关掉所以换不换都行」，而是「Haar 支本身是负资产」（Δ_br = +0.34）——门控权重小正是因为它有害，不是因为它无害。DDBE 侧 `≈ 0` 命中。

**已知偏差**（回填时必须写）：算子作用在中间特征而非输入图；W-L1 的 `abs` 丢符号（与已判负的 energy 图同病）+ 三槽冗余，由 W-L4 的金字塔口径覆盖（DDBE 侧同理由 B-L3 覆盖）；`_standardize` 在 C 通道上是逐样本全局统计。若 W-L1 为负而 W-L4 为正，说明瓶颈在「整流 + 单尺度」而非「拉普拉斯基」；两支都为负才可判定该宿主不适合换基。

### 1.13 骨干归因：ViT 贡献与 ImageNet 预训练贡献（2026-11-06 入队）

**动机**：需要回答「TransUNet 的分数是不是主要来自 ResNet 特征强、ViT 基本没效果」。本地已有 A vs A0 支持前半句，但 U-Net 86.55 与 A 90.49 的差距**同时混合了两个变量**（换成 R50 结构 + 加载 ImageNet21k 权重），无法单独归因到 ResNet 结构。

**`--vit_layers 0` 关掉了什么**（逐行可查）：

| 环节 | 位置 | 行为 |
|---|---|---|
| 参数改写 | [train.py](../train.py#L499) `config_vit.transformer.num_layers = args.vit_layers` | 层数由 12 改为 0 |
| 建层 | [vit_seg_modeling.py](../networks/vit_seg_modeling.py#L239-L243) `for _ in range(num_layers): self.layer.append(...)` | 0 次循环 → 空 `ModuleList` |
| 前向 | 同文件 L245-L251 `for layer_block in self.layer: hidden_states, weights = layer_block(...)` | 空表不迭代，`hidden_states` 就是 patch embedding + 位置编码，直接过 `encoder_norm` 进解码器 |

所以 R50 卷积主干、`skip3` 三条 skip、解码器、损失、优化器、超参**全部不动**，唯一变量是「14×14 token 上的 12 层全局自注意力」。注意 patch embedding 与位置编码的参数仍在（参数量不等于严格删除 ViT），但全局信息交互为 0——这就是「把 ViT 关掉」的准确含义。

**已有数据**（同一套预处理、seed 1234、ep200、0.5CE+0.5Dice）：

| 编号 | 骨干 | ImageNet21k | Dice | ASSD | HD mm |
|---|---|---|---|---|---|
| U-Net（§1.5，from scratch） | 纯卷积 | 否 | 86.55 | 0.225 | 1.14 |
| A（§1.6） | R50 only（`--vit_layers 0`） | 是 | **90.49** | — | 0.71 |
| A0（§1.1，3 seed 均值） | R50 + ViT-B/16 | 是 | 90.32（90.23~90.40） | 0.146 | **0.64** |
| B（§1.6） | ViT-B/16 only（`--n_skip 0`） | 是 | 89.19 | — | 0.78 |

**缺口与新增臂**：

| 编号 | 配置 | 快照 | 作用 |
|---|---|---|---|
| V-N1 | `--arch baseline --vit_layers 0 --no_pretrain 1` | `TU_R50-ViT-B_16_skip3_vitL0_..._c2d`（无 `_pretrain`） | R50 随机初始化，其余与 A 逐位一致 → 分离「R50 结构」与「预训练」 |

实现：`train.py` 新增 `--no_pretrain`（跳过 L600 的 `net.load_from(...)`，并把 `is_pretrain` 置 False 使快照目录不带 `_pretrain`）；仅允许 `--arch baseline`。

**判定**：`Δ_pretrain = A − V-N1` 为预训练贡献，`Δ_R50struct = V-N1 − U-Net` 为结构贡献。

| 读数 | 结论 |
|---|---|
| V-N1 ≥ 89 | R50 结构本身强，预训练是次要项 → 导师判断成立 |
| V-N1 ≈ 87（U-Net 档） | +3.9 几乎全部来自 ImageNet 预训练 → 只能表述为「预训练强」 |
| V-N1 居中 88~89 | 两者各占一半，按 `Δ_pretrain` / `Δ_R50struct` 分配 |

**预注册预期**：V-N1 落在 87~89。350 张训练图 + 200 epoch 下随机初始化的 R50 收敛不充分，因此 V-N1 是「ResNet 结构 + 本项目训练预算」的下界，不能当作 ResNet 结构的上限。

**结果（2026-11-07 完成，exit 0，27.7 min，[日志](../logs/queue/vitL0_nopre_20261107_093245.log)；350 张 / 8800 iter / ep200 均已核）**：

| 臂 | 骨干 | ImageNet21k | Dice | ASSD | HD95 | HD | \|ΔArea\| |
|---|---|---|---|---|---|---|---|
| A | R50 only | 是 | 90.49 | 0.145 | 0.416 | 0.71 | 1.83 |
| V-N1 | R50 only | 否 | **89.59** | 0.165 | 0.513 | 0.79 | 2.36 |
| **Δ_pretrain（A − V-N1）** | | | **+0.90** | −0.021 | −0.097 | −0.08 | −0.53 |
| U-Net | 纯卷积 from scratch | 否 | 86.55 | 0.225 | 0.771 | 1.14 | 3.36 |
| **Δ_R50struct（V-N1 − U-Net）** | | | **+3.04** | −0.060 | −0.258 | −0.35 | −1.00 |

**结论**：A 相对 U-Net 的 +3.94 = **R50 结构 +3.04（77%）+ ImageNet21k 预训练 +0.90（23%）**，四项指标方向全部一致（预训练让 ASSD −0.021、HD95 −0.097、\|ΔArea\| −0.53）。所以「TransUNet 的分数主要来自 ResNet 特征」成立，但成因是**结构**而非**预训练** —— 预训练只值 0.9 点，虽然越过 ±0.3 噪声带，但远低于「没训过会很差、应有 3~4 点」的先验估计。原因在于本任务 from-scratch 的下限本就很高（U-Net 86.55、颈动脉前景是等效半宽 18 px 的实心厚块），预训练能补的空间被压缩。

**边界**：Δ_R50struct 严格说是「R50 + DecoderCup 解码器」相对「U-Net 编码-解码」的差，含解码器家族差异，不能整体记为 ResNet 结构的贡献；只有 Δ_pretrain 是完全同构的干净对比（同一份代码、同一随机种子，唯一变量是 `load_from`）。

**汇报口径**（可直接引用）：「同骨干对照下，去掉 ViT 的 12 层全局注意力后 Dice 90.49，比保留 ViT 的 TransUNet 90.32 高 0.17，差异在 ±0.3 的 seed 噪声带内；只留 ViT 去掉 R50 skip 则降到 89.19、HD 由 0.64 升到 0.78。因此 ViT 在本任务上贡献可忽略，TransUNet 的分数来自 R50 局部支。把 R50 换成随机初始化（其余逐位不变）后 Dice 89.59、HD 0.79，即 **ImageNet 预训练只贡献 +0.90 / HD −0.08，而 R50 结构本身相对 U-Net 贡献 +3.04**。」

### 1.14 R50 全段 + 两种接法（2026-11-07 入队）

**问题**：老师与我们都默认「**先一个强骨干，再加一个自己的编码器**」。本轮把 R50 完整加载 ImageNet21k（不截断任何 stage）当强骨干，测两件事：① 自己的 WRMM 编码器接在 R50 特征上还有没有增量（`F1 − E1`）；② 相对纯 WMREN 编码器（87.77）换强骨干值多少（`F1/E1 − 参照`）。预训练本身的值已由 §1.13 的 V-N1 单独测过（`Δ_pretrain = +0.90`），本轮不再重复设随机对照。

**两条编码器路线（定性先说清）**：

- **整段替换**（`--wmren_enc r50`）：R50 的 `conv_root + block1~3`（ImageNet21k，block_units=(3,4,6)，**完整不截断**）整体当编码器，输出 112²/64、56²/256、28²/512、14²/1024，四个 WRMM **全不建**，解码器固定 `--wmren_ablate unet_dec`。它就是「R50 编码器 + U-Net 解码器」，**不是 WMREN 的预训练版**。
- **全段 + 自有编码器**（`--wmren_enc r50wrmm`）：R50 同样全段不截断（3/4/6），四级特征**逐级各接一个 WRMM 做同级精炼**（`stride=1`，输出尺度与通道都不变：`1×1 升 64→128 → WRMM1`、`WRMM2`、`WRMM3`、`WRMM4(hidden=512)`），再进同一个 `unet_dec`。即「强骨干提特征 → 自己的编码器吃这些特征」。

> 已排除两件事：① `--wmren_enc r50deep`（R50 只嫁接深两级 block2/block3），开关保留但不排期；② 所有随机初始化对照（`--no_pretrain 1`）——文献里的 ResNet 一律加载预训练，从零训不是本任务的对照条件。

两条路线都用 `unet_dec` 解码器，因为 §1.9 已判 REM 负贡献（87.14 vs 87.77，−0.63），留在臂上会把差数拖进编码器对照。

**对齐方式**：R50 全段输出 112²/64、56²/256、28²/512、14²/1024，与 WRMM1~4 逐级同形，只需一个 1×1 把 112² 的 64 通道升到 128；旧版的 `r50front` 只把 R50 截到 `conv_root + block1`（且 block1 只有 1 个残差单元），等于只用 1/8 的 R50，已废弃。

| 编号 | 配置 | 快照后缀 | 参数量 |
|---|---|---|---|
| F1 | `--wmren_enc r50wrmm --wmren_ablate unet_dec` | `_abunetdec_encr50wrmm_pretrain` | 72.17 M |
| E1 | `--wmren_enc r50 --wmren_ablate unet_dec` | `_abunetdec_encr50_pretrain` | 31.00 M |

队列顺序 `F1 → E1`，F 在前（老师指定的主线）。

**读数口径**（参照 = WRMM 编码器 + U-Net 解码器，87.77 / 0.1923 / HD 0.979）：

- `F1 − E1` = 「在 R50 提的特征上再加自己的 WRMM 编码器」值多少；两者都吃预训练、都用 U-Net 解码器，唯一差别是 WRMM 四级精炼。
- `F1 − 参照` = 换成「强骨干 + WRMM 精炼」相对纯 WMREN 编码器值多少（结构+预训练混在一起）。
- `E1 − 参照` = 「整段换成 R50 预训练编码器」值多少。
- 本轮**不再报** `Δ_pretrain`：随机对照已按要求排除，两个臂都在预训练条件下比较。

**实现**：`WRDM` / `WRMM` 新增 `stride` 参数。`stride=1` 时右支改 stride-1 卷积；小波支一级分解天然落在 H/2，用 bilinear 插值回原尺寸后再融合、门控、MLKM，因此四个输出与输入逐级同形。`r50wrmm` 建 `ResNetV2(block_units=(3,4,6), in_channels=3)`，`wrmm1..4 = WRMM(128,128) / WRMM(256,256) / WRMM(512,512) / WRMM(1024,1024,hidden=512)`，全部 `stride=1`。两支共用 `load_r50`（键与 [ccfr.py](../networks/ccfr.py#L226) 一致）。

**冒烟（CPU）**：R50 全段 + 四级 WRMM 精炼 **72,169,090**，整段替换 31,001,346；`block1` 3 个残差单元、`block3` 6 个，`block1.unit3.conv2` 加载 ImageNet21k 前后不同；前向均为 `(2,2,224,224)`；四个 WRMM 精炼后形状 `112²/128 → 112²/128`、`56²/256 → 56²/256`、`28²/512 → 28²/512`、`14²/1024 → 14²/1024`；`--max_epochs 0` 下两臂命令行均可跑通。队列：[queue_c2d_wmren_r50pre.ps1](../scripts/queue_c2d_wmren_r50pre.ps1)，先等 leaveout 队列（M1/M2）退出再接力，避免两个队列同时抢显存。

**判定 / 预期**：§0 门槛（`|ΔDice| ≥ 0.5` 或 HD 降 ≥ 0.1 mm）。`F1` 若显著超过 87.77，才谈得上「强骨干前置 + 自有编码器有效」；`F1 − E1` 为正才说明 WRMM 在 R50 特征上仍有增量。

**局限（审稿人一定会问）**：① 精炼模式是**我们改的 WRMM**（原版 WRDM 自带 stride-2 下采样），论文原模块没有这个用法，汇报时必须写成「WRMM 的同级精炼变体」，不能说是原文结构；② 两个臂都没有「WMREN 全身预训练」——R50 吃预训练，WRMM 仍是随机初始化。



## 2. 待办 / 未排期

| 项 | 状态 |
|---|---|
| 骨干归因 V-N1（R50 随机初始化，分离 ViT / 预训练贡献） | **已完成**（2026-11-07，89.59 / HD 0.79）：Δ_pretrain = +0.90、Δ_R50struct = +3.04，结论见 §1.13 |
| 预训练宿主：R50 全段（3/4/6 不截断）× 两种接法两臂（F1 R50 四级特征 + WRMM 精炼 / E1 R50 直连 U-Net 解码器），均加载预训练、无随机对照 | **已入队**（[queue_c2d_wmren_r50pre.ps1](../scripts/queue_c2d_wmren_r50pre.ps1)，顺序 F1→E1，先等 leaveout 队列 M1/M2 退出再接力；设计见 §1.14。`r50deep` 与所有 `--no_pretrain` 对照按用户要求排除） |
| WMREN 归因：去 MLKM（M1）/ 固定等权门控（M2） | **已入队**（leaveout 队列在跑；组件归因见 §1.9） |
| FDFUNet 减法消融（DRDAC / MFDF / ASCA，3 job） | **已入队**（[queue_fdfunet_ablation.ps1](../scripts/queue_fdfunet_ablation.ps1)，2026-11-05 07:58 WMI 脱离启动，等 CCFR 作业退出后接力；记录见 §1.10） |
| GobletNet 式 HFBranch + FAM 移植（批次 2） | 代码已实现、冒烟通过，**未入队** |
| WMREN WRDM 双分支 DWT（批次 4） | 已实现（`networks/unet_wmren.py`，4×WRMM + 3×REM），W2 已跑；**归因 K1/K2 已跑完并回填 §1.9**（K1 86.90 / K2 87.77 → WRMM +1.22、REM −0.63）；**Laplacian 替换已入队 §1.12**（`--wmren_wave_op lap|pyr` / `--wmren_no_wave 1` / `--ddbe_hf_op lap|pyr|none`，队列 [queue_c2d_lap_swap.ps1](../scripts/queue_c2d_lap_swap.ps1)，W-L1~W-L4 + B-L1/B-L2/B-L3）；`--loss ce_dice46 / weave5 / weave4 / ce_dice_unc` 已实现但**未排**；**seed 2024 复跑仍未排** |
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

# CCFR 复现（两阶段在同一进程内串联：EA-Net 200ep → 冻结训 flow 20ep）
powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr.ps1
# CCFR 同中心 in-domain（Z1 Zhongnan + A1 Ar-PlaqSegm1，A1 需先转换数据）
.\venv\Scripts\python.exe scripts\prepare_arplaq.py
powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr_indomain.ps1
# 只跑 stage2（消融）：需手动指向 stage1 权重
.\venv\Scripts\python.exe train.py --arch ccfr --ccfr_stage 2 `
  --ccfr_stage1_ckpt ..\model\TU_Carotid224\TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_ccfr_c2d\stage1_ep200.pth `
  --ccfr_nocond 1 --batch_size 4 --grad_accum 2

# C5：预训练 R50 编码器版（会自动等上面 4 个 job 全部落表再开跑）
powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr_pretrain.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr_pretrain.ps1
# 对应快照 ..._SPARC_ccfr_pr50_c2d；stage2-only 时要同时带 --ccfr_pretrain 1 才能匹配结构

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
