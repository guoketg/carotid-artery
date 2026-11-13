# 实验方案与结论

> 本文件只放三样东西：**当前口径、在跑 / 待跑、仍在用的结论**。
> 已关闭 / 已判负 / 已作废的线与臂全部移到 [experiments-archive.md](experiments-archive.md)，这里不再复述。
> 完整指标横评见 [results.md](results.md)，结构细节见 [technical-design.md](technical-design.md)。

## 0. 统一口径

| 项 | 值 |
|---|---|
| 数据 | SPARC / Zhongnan 分开训练、分开测试，不混合；开发集固定 SPARC |
| 输入 | 224×224；基线单通道灰度 repeat 3 通道；变体按需加通道（边缘图 / 方向子带） |
| 训练 | `--batch_size 4 --grad_accum 2`（等效 bs8）；`--max_epochs 200`（poly 衰减总长绑死）；`--eval_test_every 5` |
| 优化 | SGD，lr 0.01，momentum 0.9，wd 1e-4；loss = 0.5·CE + 0.5·Dice |
| 预训练 | R50 + ViT-B_16（ImageNet21k）；默认全部加载预训练。`--no_pretrain 1` 只作归因对照（V-N1 基线 89.59；DDBE/FGR 的随机初始化归因批见 status.md §2） |
| 读数 | 终点 ep200：`test_curve.csv` 末行，或 `epoch_199.pth` 重推理 |
| 指标 | Dice / ASSD / HD / DeltaArea；ASSD、HD 单位 mm，只用 HD 不用 HD95 |
| 判定 | 主看 ASSD / HD，两项方向一致才认边界改善；\|ΔDice\| ≥ 0.5% 或 HD 降 ≥ 0.1 mm 才算有效 |
| 统计 | 整个测试集 160 张切片直接平均，不做病例聚合、不做显著性检验；单 seed 噪声带 ±0.3 |
| 多 seed 集合 | **42 / 317 / 1029**（用户指定，与课题组其它工作对齐，便于横向对比；3-seed 补跑一律用这组）；单 seed 默认仍为 1234，历史批次 2024 / 3407 保留原口径 |

## 1. 在跑（2026-11-07）

| 编号 | 内容 | 状态 | 回填位置 |
|---|---|---|---|
| M1 | K2 + `--wmren_no_mlkm 1`（55.35 M）：MLKM 值多少 | leaveout 队列在跑 | §3.2 |
| M2 | K2 + `--wmren_no_gate 1`（54.37 M）：WRDM 软加权 vs 固定 0.5 | 同上 | §3.2 |
| F1 | R50 全段 → 每级 WRMM `stride=1` 精炼 → `unet_dec`（72.17 M） | r50pre 队列已就绪，等前序退出，未开跑 | §3.3 |

## 2. 待跑 / 未启动

| 项 | 说明 |
|---|---|
| W-L3 | K2 + `--wmren_wave_op lap --wmren_no_gate 1`，条件臂，未排期 |
| WMREN 损失口径 | `--loss ce_dice46`（原文 0.4/0.6）/ `weave4` / `ce_dice_unc`，代码已接，未排 |
| WMREN 补 seed | K2 / W2 的 seed 2024、3407 未跑 |
| ASR v2 | 已重写（逐通道自适应核、1×1→PixelShuffle、tanh 软约束，60.10 M），`queue_c2d_asr2.ps1` 就绪，未启动 |
| 损失基线 | A0 × Focal / Tversky / Boundary / wIoU，`queue_c2d_loss_baselines.ps1` Phase 1 就绪，未启动 |
| 领域损失消融 | 去边界带 / 去面积项 / 去远假阳项，同队列 Phase 2 就绪，未启动 |
| 领域损失补 seed | 主配置 seed 2024 / 3407，同队列 Phase 3 就绪，未启动 |
| 远假阳项归因（UNet） | 对照 D2（87.01 / 0.202 / 0.99）与基线（86.55 / 0.225 / 1.14）：AB-noFar（`--dl_far_w 0`）判"0.15 的 HD 降幅是否来自该项"，DL-far03（`--dl_far_w 0.3`）看剂量；连同 LFA 已入 `queue_c2d_loss_unet.ps1`，启动器在 `queue_c2d_loss_unet_status.csv` |
| HFBranch + FAM | 代码已实现、冒烟通过，未入队 |
| 跨中心反向 | 待补 Zhongnan → SPARC（需先训 Zhongnan 权重）与 dual / split 分支 |
| split + frc | 88.34（A0 −2.2 点），HD 未重算，未收口 |
| 输入分辨率 320 vs 224 | 未做，且不作为论文卖点；若补须所有对比方法同口径 |
| 边缘监督分支 | 共享编码器 + 两个对等解码器（主分割 + 边缘解码器）；边缘 GT = 原标签 3×3 形态学梯度（dilate−erode），边缘项与主分割同一 `seg_loss`、同权重 1.0。UNet（+22.6M）与 R50 预训练（+7.4M）各一臂，SPARC seed 1234 bs4 ga2 ep200；已挂 `queue_c2d_edge_branch.ps1`，状态见 status.md §2。用户先验 TransUNet 骨干 +1.0 Dice；存在解码器容量混杂 |

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
