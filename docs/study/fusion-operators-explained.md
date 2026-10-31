# 融合算子逐个讲解：gate / concat / deform / aff / auto / DDBE 是怎么回事

> 2026-10-28 应你要求新建。方法盘点（谁出自哪篇论文、试没试过）见 [fusion-methods.md](fusion-methods.md)，
> 本文只做一件事：**把本项目里每个融合算子的公式、代码位置、实测结果讲透**。
> 所有数字来自 2026-10-28 跑完的 fuse 队列（SPARC，ep150，aggregate 口径）。

## 0. 一页版

这些算子全是**双骨干模型里"两条支路怎么交换信息"的插件**（`--fuse_op` 档位），公式统一为残差形式
`A' = A + γ·Op(A, B)`，γ 零初始化保证第 0 步与基线逐元素相同。

| 算子 | 一行公式 | 轴向 | ep150 Dice / ASSD / HD95 | 结论 |
|---|---|---|---|---|
| gate | `A' = A + γ·A·σ(conv([mean(A),max(A)]))` | 只调幅度 | 90.35% / 0.1434 / 0.4034 | 与 auto 拉不开 |
| auto（QKV） | `A' = A + γ·Proj(Attn(Q=A,K=V=B))` | 跨位置重组 | 90.40% / 0.1476 / 0.4064 | 旧冠军，被 deform 超越 |
| deform | `A' = A + γ·DCN(dst, off(src))` | 几何自适应 | **90.61% / 0.1412 / 0.3788** | **本轮最好**，深讲见 [deform-fusion-explained.md](deform-fusion-explained.md) |
| aff | `A' = A + γ·A·MS-CAM(A+B)` | 只调幅度 | 90.17% / 0.1469 / 0.4011 | 重跑成功，五档垫底 |
| DDBE（单级挂载） | 三分支 DWT 融合 + 残差 | 通道轴 | 90.08% / 0.1513 / 0.4404（vs 基线 89.95%/0.1498/0.3997） | 判负；四级全挂 90.33%、+ACAM 完整版 90.53% 方向为正 |

**高频图的拼接问题**：小波分解出 LH/HL/HH 三个高频子带后，文献里有三种处理，**拼接成 3 通道只是其中一种**
（详见 §1.3）。BDFR-Net 的 HF 支和本项目侧路都选拼接；老师指定的输入级边缘图选的是"重建回单通道"。

## 1. 输入是怎么处理的

### 1.1 两条支路各吃什么

- 主路：原图复制成 3 通道（可选第 4 通道 = 小波高频重建图，`--edge_in 1`）；
- 侧路：[build_detail_channels](../../networks/dual_encoder_model.py#L35-L49) ——
  对数域 1 级 Haar DWT 后取 **LH/HL/HH 三个方向子带 → 拼成 3 通道** → 上采样到 224 → 逐样本 z-score。
  侧路刻意不含低频，与主路信息不重叠。

### 1.2 小波 DWT 一句话

[haar_dwt](../../networks/wavelet.py#L24-L28)：用 4 个固定 2×2 卷积核
（[wavelet.py L9](../../networks/wavelet.py#L9)）stride 2 卷一次，
把每张图拆成 4 张半尺寸的子带：

| 子带 | 卷积核 | 含义 |
|---|---|---|
| LL | `[[1,1],[1,1]]/2` | 低频近似（缩小+模糊的原图） |
| LH | `[[1,1],[-1,-1]]/2` | 水平方向的细节（竖直边缘） |
| HL | `[[1,-1],[1,-1]]/2` | 垂直方向的细节（水平边缘） |
| HH | `[[1,-1],[-1,1]]/2` | 对角线方向的细节 |

### 1.3 高频图一定拼接吗？——不是，三种做法

| 做法 | 运算 | 优点 | 缺点 | 本项目里的例子 |
|---|---|---|---|---|
| **① 拼接成 3 通道** | `cat([LH,HL,HH])` | 保留全部方向与符号信息，后续卷积自己学怎么组合 | 通道变多 | 侧路输入、DDBE 的 HF 支（[ddbe.py L123](../../networks/ddbe.py#L123)） |
| ② 能量图合成 1 通道 | `sqrt(LH²+HL²+HH²)` | 单通道、省算力 | **丢方向与符号**，只剩"哪里有高频" | 旧的高频能量图（[build_three_channel C3](../../networks/wavelet.py#L99)） |
| ③ 重建回单通道 | LL 置零 → IDWT | 与原图同分辨率同分布，方向信息以干涉形式保留 | 单通道，方向被混在一起 | 老师定的输入级边缘图 [build_hf_image](../../networks/wavelet.py#L58-L82) |

文献里的普遍情况：**要让网络"分方向处理"就选拼接**（BDFR-Net HF 支、XNet/GobletNet 族都是）；
**只是把高频当附加输入通道**就选重建或能量图。BDFR-Net 自己也是拼接派——它的 HF 支
`cat(LH,HL,HH) → 1×1 压回`正是做法①。

## 2. 模型与损失

### 2.1 融合算子逐个讲（全部在 [dual_encoder_model.py](../../networks/dual_encoder_model.py)）

融合位置：双骨干四个接点 112²/64、56²/256、28²/512、14²/768（`--fuse_op` 选哪个算子，接点不变）。
所有算子双向对称（A 影响 B、B 影响 A），γ_a、γ_b 各一个标量零初始化——**一条链上只有一个零**，
保证第 0 步与单通道基线逐元素相同。

**① gate 逐像素门控**（[BidirGate L170](../../networks/dual_encoder_model.py#L170-L195)）

```
w_b = σ(Conv7×7([mean(B), max(B)]))          # 把 B 压成一张空间权重图
A'  = A + γ_a · (A ⊙ w_b)                    # 用对方的权重图缩放自己
```

只调幅度、不重组内容：A 的每个位置值不变，只是被 B 的"哪里重要"乘了个 0~1 的系数。参数量几乎为零。
conv 权重零初始化 → 初始 w=σ(0)=0.5，非零，门控能学（这是修死锁后的写法）。

**② concat 拼接回投影**（[BidirConcat L198](../../networks/dual_encoder_model.py#L198-L213)）

```
A' = A + γ_a · Conv1×1(cat([A, B]))          # 拼接后 1×1 线性混合再压回
```

对照档：区分"跨支交互"与"只是多一路特征"。TinyCNN 门槛实验里它只 +0.005（无效）。

**③ auto / window / global 跨支 QKV 注意力**（[BidirCrossAttention L87](../../networks/dual_encoder_model.py#L87-L167)）

```
A' = A + γ_a · Proj(Attn(Q=A, K=B, V=B))     # A 的每个位置去 B 的所有位置取内容
```

唯一**跨空间位置重组内容**的算子。auto = 112²/56² 用 8×8 窗内注意力（全图 token 太多会爆显存），
28²/14² 用全局注意力。已测出的坑：侧路喂全零涨幅反而更高 → **涨幅不来自 HF 内容**（fusion-methods.md §3）。
最深接点用 chain 串联双向，否则 B 无下游消费者、gamma_b 恒 0 悬空。

**④ deform 可变形大核**（[BidirDeform L216](../../networks/dual_encoder_model.py#L216-L260)）

```
off = Conv3×3(red(A))                        # 偏移由"自己"的降维特征预测（不是对方）
A'  = A + γ_a · Proj(DCN(red(B)处内容, off))    # 拿偏移到对方特征上采样，采样点跟着边界走
```

先 1×1 降到 32~128 维再做可变形卷积（省参数）。偏移卷积零初始化 → 初始等价于固定网格卷积
（输出非零），只有 γ=0 保证恒等起步。参数量 3.69 M，四个接点里最大。
术语消歧：torchvision `deform_conv2d` 未传 mask，是 v1 口径（只学偏移、无逐点幅度调制），
不是 DCNv2。为什么本轮最好＋逐行代码：[deform-fusion-explained.md](deform-fusion-explained.md)。

**⑤ aff 两级注意力融合**（[BidirAFF L287](../../networks/dual_encoder_model.py#L287-L312)，核心是 [MS-CAM L263](../../networks/dual_encoder_model.py#L263-L284)）

```
m = MS-CAM(A + B)                            # 先相加整合，再算逐通道空间权重
A' = A + γ_a · (A ⊙ m);  B' = B + γ_b · (B ⊙ (1-m))
```

MS-CAM = 局部支路（保空间分布）+ 全局支路（GAP 抓通道显著性）相加过 sigmoid。
与 gate 同属"只调幅度"家族，但权重是**两支联合**算出来的且互补（m 与 1-m）。参数量 0.47 M。
首跑在 ep5 崩溃（GroupNorm 接在 GAP 后的 1×1 特征上，eval batch=1 报
`Expected more than 1 value per channel`）；修复 = 全局支路改 SE 式 MLP 不带空间归一化，
重跑成功。结果 90.17% / 0.1469 / 0.4011，五档垫底——"只调幅度"且权重全局化，比逐像素 gate 更差。

**⑥ DDBE 三分支**（[ddbe.py](../../networks/ddbe.py)，单级挂载，`--ddbe 3`）

```
f = ECA(x)                                   # 先 1D 卷积通道注意力压斑点响应
HF  = CA(TransConv3×3(1×1(cat(LH,HL,HH))))   # DWT 高频支：拼接子带、精修、上采样
LF  = CA(Bilinear(5×5(LL)))                  # 低频支：近似子带、上采样
CTX = 7×7(f) ⊙ σ(1×1(7×7(f)))                # 上下文支：大核 + 自门控
x'  = x + CBAM(1×1_zero(cat(HF,LF,CTX)))     # 通道拼接 → 1×1（唯一零点）→ CBAM → 残差
```

与其他算子的本质区别：**它不做双支路交换，而是单支路内部把特征按频段拆开再融合**，
融合轴向是纯通道轴（同位置拼接），全程无 QKV。挂载点 = 解码器第 N 级输出
（level3 = 112²/64ch，参数 0.440 M）。零初始化只放最后的 1×1，初始整支恒等。
批次 0+1 已把挂载方式跑全：单级（`--ddbe 3`）90.08% 判负；**四级全挂**（`--ddbe 4`）
90.33% / 0.1439 / 0.3936；**四级全挂 + ACAM skip 对齐**（`--ddbe 4 --acam 1`，BDFR 完整版）
90.53% / 0.1441 / 0.3935——方向为正，ACAM 净贡献 +0.20% Dice；Wilcoxon 已补齐（见 §3）：
四级全挂 p=0.462/0.288/0.876、完整版 p=0.334/0.419/0.731，均未过门槛。

### 2.2 损失与训练口径

损失 = 2 通道 logits 的交叉熵 + Dice（trainer.py 默认，无权重、无深监督）。
优化器/调度/batch = 官方 TransUNet 配置 + bs4 + ga2 + 200 epoch，SPARC 官方划分，
mm 物理单位，每 5 epoch 评估，取 ep150。同一 seed（1234）下所有档位基础权重逐位相同，只差融合算子本身。

## 3. 指标与结果

- **Dice**：预测与真值的重叠度，天花板效应下 0.005 以内的差值多为噪声；
- **ASSD / HD95**（mm）：边界距离，主指标；HD95 是 95 分位豪斯多夫距离，对离群假阳性敏感——
  斑块分割最怕"远处多切一块"，所以 HD95 比 Dice 更能反映临床风险。

| 配置（SPARC, ep150） | Dice ↑ | ASSD ↓ | HD95 ↓ | 相对 auto 的差 |
|---|---|---|---|---|
| 基线 A0（seed1234 重建） | 89.95% | 0.1498 | 0.3997 | — |
| dual **auto** | 90.40% | 0.1476 | 0.4064 | 基准 |
| dual **gate** | 90.35% | 0.1434 | 0.4034 | HD95 −0.003 → 无差异 |
| dual **deform** | **90.61%** | **0.1412** | **0.3788** | **HD95 −0.028** |
| dual aff（重跑） | 90.17% | 0.1469 | 0.4011 | Dice −0.23%，五档垫底 |
| baseline + DDBE(level3) | 90.08% | 0.1513 | 0.4404 | vs 基线：HD95 **+0.041 变差** |
| baseline + DDBE 四级全挂 | 90.33% | 0.1439 | 0.3936 | vs 基线：Dice +0.37% |
| baseline + DDBE×4 + ACAM | 90.53% | 0.1441 | 0.3935 | vs 基线：Dice +0.57% |

怎么读：

1. **deform 是本轮唯一全面变好的档**（三项都优于 auto），但幅度 HD95 −0.028 mm 远小于事先写死的
   0.5 mm 判定线——方向对，量级不够；
2. gate 与 auto 差值全在噪声带，aff 垫底 → **"调幅度 vs 重组内容"这两种轴向在本任务上不敏感**，
   且权重越全局化越差（逐像素 gate > 全局 MS-CAM）；
3. DDBE 单级挂载 Dice 只 +0.12% 且 HD95 变差 → **判负**；但四级全挂与 +ACAM 完整版方向为正
   （ΔDice +0.37% / +0.57%，HD95 −0.006 mm），证明 BDFR 论文"每级都挂 + 每级 ACAM"的挂载方式
   才是对的，单级挂载没有意义；
4. **病例级 Wilcoxon（43 例配对，vs 基线）已全部补齐**：deform ΔDice p=0.113、ΔASSD p=0.154、
   ΔHD95 p=0.276；gate 三项 p≥0.33；aff p=0.454/0.663/0.375；DDBE 四级 p=0.462/0.288/0.876、
   DDBE+ACAM p=0.334/0.419/0.731；mainLL p=0.447/0.742/0.426；双骨干 auto 在 seed3407 上显著
   **变差**（Dice p=0.0036、ASSD p=0.0097）——横评排序成立，显著性全部不成立。

## 4. 代码实现在哪里

- 五种融合算子 + 工厂：[networks/dual_encoder_model.py L84-L330](../../networks/dual_encoder_model.py#L84-L330)
- DDBE：[networks/ddbe.py](../../networks/ddbe.py)；挂载点 [vit_seg_modeling.py DecoderCup](../../networks/vit_seg_modeling.py#L399)
- 小波工具（DWT/IDWT/边缘图）：[networks/wavelet.py](../../networks/wavelet.py)
- 开关：`train.py --fuse_op {auto,window,global,gate,concat,deform,aff}`、`--ddbe 0~4`

## 5. 结论边界与其他

**能说**：同 seed 同口径下 deform > auto > gate > aff（SPARC 单 seed，aff 垫底）；deform 方向三项全正
但 Wilcoxon p=0.113/0.154/0.276 均不显著；DDBE 单级判负（HD95 +0.041 mm），四级全挂 / +ACAM 完整版
方向为正但 p=0.33~0.88 也不显著；mainLL 与主路原图无差异（p=0.447/0.742/0.426）；双骨干 3-seed 交叉
无增益（seed3407 显著变差，p=0.0036）。
**不能说**：① deform 显著更好（未过「HD95 降 ≥0.5 mm 且 p<0.05」门槛）；② 任何 Zhongnan 上的结论
（本轮全是 SPARC）；③ DDBE 四级 / +ACAM 或 mainLL 有增益（方向为正、均不显著）。

已知局限：融合算子横评为单 seed（1234）；43 例检验功效不足，+0.006 量级的 Dice 差判不了显著性。

复现命令：

```powershell
# 任一档位（以 deform 为例）
.\venv\Scripts\python.exe train.py --subset SPARC --seed 1234 --arch dual --levels 1,2,3,4 --fuse_op deform --dataset Carotid --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5
# DDBE 单级
.\venv\Scripts\python.exe train.py --subset SPARC --seed 1234 --ddbe 3 --dataset Carotid --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5
```

与其他方案的关系：本表回答的是"通道轴内部哪种融合算子最好"，与"要不要引入高频信息"（edge 输入级，
见 [edge-fusion-explained.md](edge-fusion-explained.md)）正交——目前证据是两者都只带来小增益，
真正的下一步是把 deform 或 edge+dual 的胜出组合搬到 Zhongnan 复现。
