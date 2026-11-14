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
| **cvartail 迁移 R50（先跑，零新代码）** | `--loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`，与 U-Net 上那条 +0.82 / HD −0.18 的臂**逐字同参**（不调权重）。R50 vitL0 / SPARC / seed 1234 / bs4 ga2 / ep200（约 42 min），配对 A0 90.4903 / 0.14473 / HD 0.7062。这是唯一已在本项目数据上过双门槛的损失臂，迁移是复现而非调参 |
| **区域 HD 损失 · 主臂（#82）** | `--loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6`：0.5CE + 0.5Dice + λ·`L_AH_sym`（λ=0.5 使 CE:HD = 1:1，对应论文 `weight_ce = weight_lah = 1`）。R50 vitL0 / SPARC / seed 1234 / bs4 ga2 / ep200（约 42 min）。配对对照 A0 90.4903 / 0.14473 / HD 0.7062 |
| **区域 HD 损失 · 论文默认 I=2** | 同上，仅 `--rhd_iters 2`（论文 `iters=2`）；单变量验证「剖面截断在 2 px 是否够」，我们 HD 尾部中位 5.4 px |
| **区域 HD 损失 · U-Net 复现（已挂载）** | `--arch unet --loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6`（约 100 min）。配对对照 §4(1) 86.55 / 0.225 / 1.14；回答「距离损失在弱骨干上的效应能否复现」（同类 `cvartail` 已给 87.37 / 0.200 / 0.96，+0.82 / HD −0.18）。快照 `..._lrhd_TsymW0.5I6_unet_c2d` |
| **区域 HD 损失 · 最大项（已挂载）** | `--arch unet --loss rhd --rhd_term h --rhd_w 0.5 --rhd_iters 6`：soft-max 压最坏一点，机制上等同 Karimi & Salcudean（IEEE TMI 2020）的「直接优化 HD」一族，与上一行的均值项构成单变量对照（同一实现只换 term）。注意：官方 logsum 含常数 `alpha*log(N)`（224² 约 1624），已在该项内减掉，减后上界 = iters，可与 0.5CE+0.5Dice 等权相加。快照 `..._lrhd_ThW0.5I6_unet_c2d` |
| **边界图加权损失（已挂载）** | `--arch unet --loss boundary --bl_w 5.0`：Kervadec et al.（MIDL 2019）Boundary Loss，用 GT 有符号距离图给预测概率加权，监督落在边界位置而非区域面积。论文用 α 由 0 递增的调度，本项目无调度、改用固定权重，默认 5.0 按初始量级标定（离线 L_B≈0.07 vs 0.5CE+0.5Dice≈0.7），**不扫参**。**该臂替代 cbDice**（MICCAI 2024）：cbDice 需可微骨架 + 欧氏距离变换 + 骨架半径加权，本机 venv 无 skimage/cuCIM 且公式细节未核；cbDice 对距离指标起作用的是「边界 Dice」部分，本臂即该族的无中心线版本。快照 `..._lboundary_unet_c2d` |
| 熵门控边界监督（#83，备选） | `--loss bndsoft` 的边界带 CE 乘 `(1−H/ln2)`（H 为预测二类熵，stop-grad），只在高可信边界带保留监督。单臂、仅在第 82 号主臂判负后再排 |

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
