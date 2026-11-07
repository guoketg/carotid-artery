# FDFUNet 复现说明与完整性审查

论文：*Medical image segmentation network based on multi-scale frequency domain filtering*（Neural Networks 2024），
后文简称 FDFUNet。代码：[networks/fdfunet.py](../networks/fdfunet.py)；机器审查脚本：`scripts/audit_fdfunet.py`。

## 0. 结论先行

- 结构级复现完成：Eq.(1)–(6)、Fig.5/6/7 的 26 项结构要素全部可由 `scripts/audit_fdfunet.py` 自动校验，当前 **26/26 PASS**。
- 与论文唯一的硬指标交叉验证：FLOPs 比值 FDFUNet/UNet\* = **1.53**（论文 19.83/12.84 = 1.54），误差 0.7%。
- 未做：5 个公共数据集的绝对值核对（本地无数据）、3 次随机划分与 p 值、噪声鲁棒性、跨数据集、Table 7 五档消融（开关已实现，未排队）。清单见 §6。

## 1. 复现范围

| 层级 | 内容 | 本次 |
|---|---|---|
| L1 结构 | DRDAC / MFDF / ASCA / FEM / 主干拓扑 + 单元校验 | 做 |
| L2 对比 | SPARC 上 UNet\* vs FDFUNet，同协议同 seed | 做 |
| L3 数字 | CVC-ClinicDB 88.49 IoU 等论文绝对值 | 不做（无数据） |

## 2. 主干拓扑（Fig.7 的量化解读）

Fig.7 只能数出「5 个 Down、5 个 Up、4 处 skip、4 个解码 FEM」，但没写分辨率。用通道自洽性反推：

| 位置 | 分辨率 | 通道 | 说明 |
|---|---|---|---|
| Down1 | 224 → 112 | 64 | DRDAC 在 224 做，池化后挂 FEM |
| Down2 | 112 → 56 | 128 | 同上 |
| Down3 | 56 → 28 | 256 | |
| Down4 | 28 → 14 | 512 | |
| Down5 | 14 → 7 | 1024 | 瓶颈 |
| Up1–Up4 | 7→14→28→56→112 | 512/256/128/64 | 每个 Up 后 `+skip`、再挂 FEM |
| Up5 | 112 → 224 | 64 | 只上采样，无 skip、无 FEM |
| Final | 224 | 2 | 1×1 conv |

只有「每个 Down 末尾都池化」这一读法能让 4 处 skip 的通道正好等于对应 Up 的输出通道（14:512、28:256、56:128、112:64）。
若改成标准 UNet 的「先卷积再池化、skip 取池化前特征」，通道会错位，且 Fig.7 的 5 个 Up 无法安置。

## 3. 三个模块的落地细节

### 3.1 DRDAC（Eq.1 / Fig.5(e)）

`X1 = Conv3×3(X)`；`X2 = X1 + PWConv(X1 + ASSConv(X1))`；`Y = Conv1×1(Concat(X, X2))`。
注意两点容易写错：**内层 `X1+X1` 的残差出现两次**（Fig.5(e) 的两个 ⊕），外层「残差」是 **concat+1×1** 而非常规相加。
ASSConv = depthwise 空洞卷积（groups=C, rate=3），PWConv = 1×1 无 bias。

### 3.2 FEM / MFDF（Eq.2–4 / Fig.6(a)）

- 融合位置：每个 Down 后、前 4 个 Up 后（顺序 Up → +skip → FEM），共 9 个 FEM，通道与所在级一致，不改变分辨率。
- 轴向：先在**通道维**做逐像素 LayerNorm，再沿通道**对半切**（空间支 X1 / 频率支 X2）；空间支做完 DASConv 后**再对半切**，前 1/4 继续空间卷积、后 1/4 转频域；频率支同理反向。
- 算子：`concat[DasConv(X11), FDF(X12), DasConv(X21), FDF(X22)] + X`（顺序按 Eq.2，不可打乱）。
- FDF：`IFFT(W ⊙ FFT(X))`，W 为 `(C, H, W/2+1)` 复数参数；**初始化 randn×0.02**（GFNet 惯例，论文未给）。
- 参数增量：MFDF 的复数权重合计约 5.95M（224 输入），是 FDFUNet 相对 UNet\* 参数增量的主要来源之一。

### 3.3 ASCA（Eq.5 / Fig.6(b)）

- 轴向池化：H 轴 `AvgPool(1×W) → MaxPool(H×1)`；W 轴 `MaxPool(H×1) → AvgPool(1×W)`，得到两个 (C,1,1) 代表值。
- 门控：`2C → MLP → C`，接 `Conv1d(k=3, pad=1)`，Sigmoid 后按通道乘回；取值严格在 (0,1)。

## 4. 参数量与 FLOPs（224², 单张）

| 模型 | 参数量 | GFLOPs |
|---|---|---|
| UNet\*（无 DRDAC/MFDF/ASCA） | 12.58 M | 10.24 |
| FDFUNet（完整） | 28.69 M | 15.71 |

FLOPs 比值 1.53 vs 论文 19.83/12.84 = 1.54。论文 GFLOPs 的绝对口径无法复刻（其 UNet 55.01 @512 与标准计量差约 4–8 倍），
故只把**比值**当保真度指标，不看绝对值。

## 5. 歧义决策表（论文未写明之处）

| # | 歧义 | 决策 | 依据 / 风险 |
|---|---|---|---|
| A1 | 主干是 4 次还是 5 次池化、skip 取池化前还是后 | 5 次池化，skip 取池化+FEM 后特征 | Fig.7 的 5 Up/4 skip 计数 + 通道自洽；风险：与「标准 UNet」读法差近 2× 参数 |
| A2 | Fig.7 写 "Skip concatenate" 但符号是 ⊕ | 直接相加 | 正文明确 "directly summed"，图注应为笔误 |
| A3 | skip 加在 FEM 前还是后 | 加在 Up 之后、FEM 之前 | Fig.7 的 ⊕ 落在 Up→FEM 的连线上 |
| A4 | DRDAC 内部是否有 BN/ReLU | 无（整块线性），BN/ReLU 只在块外 | Fig.5(e) 未画；风险：可能是作图省略，若增益异常需补对照 |
| A5 | FDF 复数权重初始化 | randn×0.02 | GFNet 惯例；若初始化为 1 则 FDF 初始即恒等，等于关掉频域支 |
| A6 | ASCA 两轴池化顺序 | H:Avg→Max，W:Max→Avg（Table 14 末行） | 正文/Fig.6(b) 画的是两轴 Avg→Max，冲突；代码留 `variant='symmetric'` |
| A7 | ASCA 的 MLP 与 1-D 卷积结构 | `Linear(2C→C)+ReLU`，`Conv1d(k=3,pad=1)` | 论文只写 "MLP" 与 "simple one-dimensional convolution" |
| A8 | 上采样方式与 Conv 位置 | 双线性，且在低分辨率先做 Conv3×3（Conv→BN→Upsample→ReLU） | Fig.7 图例顺序 |
| A9 | 归一化/增强/调度 | 沿用项目口径（1%–99% 截断、零均值单位方差），Adam 1e-4，cosine→1e-5 按 epoch | 论文只写 resize 与 Adam+cosine |

## 6. 完整性审查

### 6.1 已自动校验（`python scripts/audit_fdfunet.py`，26/26 PASS）

覆盖：DRDAC 与式(1) 数值等价、ASSConv 的 groups/dilation、FDF 权重形状与初始化、MFDF 四路通道与梯度、ASCA 池化调用顺序与门控范围、FEM 数量 5+4、主干通道表、skip 为加法、UNet\* 基线剥离、输出形状、全参数非零梯度、FLOPs 比值、train/trainer 接线。

### 6.2 未复现项（明确声明，不算"完成"）

1. 5 个公共数据集（LiTS/ISIC/CVC-ClinicDB/Kvasir/GlaS）本地无数据 → 论文绝对值与 Table 2/3/4 未核对。
2. 论文的 3 次随机划分、均值±std、显著性与 p 值（Table 12）。
3. 噪声鲁棒性（Table 13）、跨数据集泛化（Table 8）、定性图（Fig.8/9）。
4. Table 7 的模块消融：代码已支持（`use_drdac/use_mfdf/use_asca`），2026-11-05 已按减法口径入队（见 §6.4）。
5. GFLOPs 绝对值：口径不明，仅核对比值。

### 6.3 训练结果（2026-11-05 00:39 收口）

两个 job 已由 `scripts/queue_fdfunet.ps1` 跑完（exit 0），判定点 ep200：

| 模型 | Dice | ASSD | HD |
|---|---|---|---|
| UNet\* | 88.97 | 0.1759 | 0.9547 |
| FDFUNet | 89.98 | 0.1616 | 0.7910 |

- [x] 相对增量方向与论文一致（+1.01 Dice；论文公开数据集 +1.9~2.5，量级更小）
- [x] |ΔDice| = 1.01% ≥ 噪声带 0.5%
- [ ] 单 seed，跨 seed 复跑未排

### 6.4 减法消融（2026-11-05 07:58 入队）

`--fdfunet_ablate {no_drdac,no_mfdf,no_asca}`，全部在完整版上只关一个模块，其余超参与 §6.3 逐字一致。
队列 `scripts/queue_fdfunet_ablation.ps1`，3 job 串行；进度看 `logs/queue/queue_fdfunet_ablation_status.csv`。

| job | 变更 | 参数量 | 快照后缀 |
|---|---|---|---|
| `no_drdac` | DRDAC → 普通 3×3 conv 下采样 | 25.18 M | `_abnodrdac` |
| `no_mfdf` | FEM 去掉 MFDF，只留 ASCA | 24.82 M | `_abnomfdf` |
| `no_asca` | FEM 去掉 ASCA，只留 MFDF | 19.96 M | `_abnoasca` |

- [ ] 三档 ep200 的 Dice/ASSD/HD
- [ ] 「完整 − 消融」的 Δ，按 |ΔDice| ≥ 0.5% 或 HD 降 ≥ 0.1 mm 判定

## 7. 运行命令

用队列脚本串行跑两个 job（第二个会等第一个结束，避免抢显存）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\queue_fdfunet.ps1 -DryRun   # 先看命令
powershell -ExecutionPolicy Bypass -File scripts\queue_fdfunet.ps1           # 实际排队
```

等效的两条命令（`--fdfunet_variant` 是唯一差异）：

```powershell
# 同构基线 UNet*（12.58M / 10.24 GFLOPs）
python train.py --dataset Carotid --subset SPARC --seed 1234 --img_size 224 ^
  --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5 ^
  --optim adam --base_lr 1e-4 --loss bce_iou_dice --num_workers 1 ^
  --arch fdfunet --fdfunet_variant unetstar --tag fdf

# 完整 FDFUNet（28.69M / 15.71 GFLOPs）
python train.py --dataset Carotid --subset SPARC --seed 1234 --img_size 224 ^
  --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5 ^
  --optim adam --base_lr 1e-4 --loss bce_iou_dice --num_workers 1 ^
  --arch fdfunet --tag fdf
```

输出目录（`--loss` 与 `--base_lr` 都会进快照名）：

- `TU_Carotid224/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_lr0.0001_224_SPARC_lbceioudice_fdfunet_unetstar_fdf_c2d`
- `TU_Carotid224/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_lr0.0001_224_SPARC_lbceioudice_fdfunet_fdf_c2d`

### 7.1 为什么不能拿已有的 `--arch unet` 当基线

项目里 [networks/unet.py](../networks/unet.py) 是 **ResUNet 式** 的 60.30M 网络（concat skip + DoubleBasicBlock + stride-2 卷积下采样），
论文的 UNet\* 是 **12.58M** 的 plain UNet（skip 直接相加 + MaxPool 下采样）。两者差 4.8 倍参数，
且已有结果是 SGD+poly+0.5CE+0.5Dice 口径，与论文的 Adam+cosine+0.5BCE+0.5IoU+Dice 不同。
直接对比会把「模块增益 + 结构差异 + 优化器差异 + 损失差异」混在一起，故本次另跑同构基线。
