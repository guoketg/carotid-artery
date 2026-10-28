# GobletNet（TMI 2025）讲解：主路吃原图、小波只补高频——参数对齐后证明「低频多余」

论文：GobletNet: Wavelet-Based High-Frequency Fusion Network for Semantic Segmentation of Electron Microscopy Images，IEEE TMI 44:1058–1069（DOI 10.1109/TMI.2024.3474028），XNet 同团队续作。
本项目落地代码：[networks/hf_branch.py](file:///e:/mkl/code/python_code/carotid-artery/networks/hf_branch.py)；概念背景：[wavelet-input-routes](wavelet-input-routes.md)、[fusion-operators-explained](fusion-operators-explained.md)；正式精读条目：[literature.md](../literature.md) GobletNet 节；速查表：[paper-reading.md](../paper-reading.md) §5.2。

## 0. 一页版

一句话：**在小波双支方案上做减法——低频轮廓交给吃原图的主干自己学，小波只负责提取一张高频图喂给一条轻量支路，再用门控注意力 FAM 逐层把边缘信息「调音」进主干；所有消融都在参数量对齐的前提下做，结论是：只有高频有用，再灌低频反而变差。**

最重要的消融顺序（正文，每步只多一个组件）：
HF 图当额外输入通道 → 涨 ② 加独立 HF 编码支 → 再涨 ③ 加 LF 分量 → **变差** ④ 二级小波 → 无收益 ⑤ 换小波基 → 基本不敏感 ⑥ FAM 挂全部编码层 → 最好。

对本项目的意义：8 篇精读里最重的一篇（literature.md 结论 1/6），因为**参数量对齐**使它推翻了「低频是一路额外信息」的假设——主分支已吃原图，再喂 LL 就是冗余+噪声。本项目批次 2 已按此移植（`--hf_branch 1 --hf_lambda 0.3`），冒烟 5 项全过，**项目侧结果待验证**（等批次 1 结论后入队）。

## 1. 输入是怎么处理的

论文输入是灰度 EM（电子显微镜）图。两条输入路，全是**图像级**构造：

1. **原图 I** → 语义编码器（raw 主干）。
2. **高频图 H（Eq.5）**：1 级小波 DWT 把 I 拆成 4 个子带——LL（低频近似≈模糊版原图）、LH/HL/HH（水平/垂直/对角细节），然后
   `H = LH + HL + HH + λ·LL`
   λ>0 = 往纯高频里掺回少量低频抑制噪声（Table VI：λ∈[0,0.3] 最优，本项目取 0.3）。H 与原图同分辨率，单独喂 HF 编码支。

**定义消歧**（论文叫法 ↔ 本项目代码里的对应物，不得混称）：
- 论文「HF 图」= 本项目 `build_hf_image` 输出：对数域 1 级 Haar DWT → LL 置零（λ=0）或回灌 λ·LL → IDWT 重建回原分辨率 → expm1 回原域 → 逐样本 z-score（[wavelet.py L58-86](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py#L58-L86)）。
- 论文消融①「额外输入通道」= 本项目 `--edge_in`（stem 吃 4 通道，输入级注入）。
- 论文消融②「独立 HF 编码分支」= 本项目批次 2 `--hf_branch`（特征级注入）。
- 论文的 λ·LL 与 XNet 的「LF 编码器」**不是一回事**：前者是高频图里掺的一味抑噪佐料，后者是独立低频支路。

## 2. 模型与损失

### 2.1 结构

| 部件 | 运算 | 作用 |
|---|---|---|
| 语义编码器 | 常规编码器，吃原图 | 全局语义（≈U-Net 主干） |
| HF detail encoder | 轻量卷积支，吃高频图 H | 只提边缘/纹理细节 |
| FAM × 每个编码层 | 见 2.2 | 把 HF 细节逐层「调音」进语义特征 |
| 融合解码器 | 每级 `cat(x_up, x1_fa, x2_fa)` | 两路 FA 后特征一起解码 |

### 2.2 FAM（Fusion Attention Module，最常被追问的部分）

- **融合位置**：编码器每一级，两支同级特征上（消融⑥：全挂最好）。
- **融合轴向**：空间同位置的**通道门控重标定**，不是 QKV 跨位置注意力。算子一行写死：
  `fused = ReLU(BN(Conv1x1(cat[sem, det])))`；`a1 = Sigmoid(Conv1x1(fused))`，`a2 = Sigmoid(Conv1x1(fused))`；官方输出为 `relu(sem·a1)`、`relu(det·a2)` 直接替换。
- **通道适配**：官方两支通道相同（都是 lNc）；本项目 sem/det 通道不等（64/256/512 vs 32/64/128），改为 fusion 压到 sem_ch，att1 出 sem_ch、att2 出 det_ch（[hf_branch.py L65-91](file:///e:/mkl/code/python_code/carotid-artery/networks/hf_branch.py#L65-L91)）。
- **初始化与残差**：官方「直接替换」在本项目改成 `x + γ·relu(x·a)`，**γ 零初始化**——初始时整条 HF 路对主干逐位恒等，同 seed 基础权重与 baseline 完全对齐（冒烟② max\|Δlogits\|=0）。注意「两个零因子相乘」陷阱：inject（1×1 det→sem）也零初始化，但 det 链在 γ 与 inject **两道零门之后**，step 0 只有 γ/inject 有梯度，3 步 SGD 后全链解锁（LoRA B=0 同款一步延迟，冒烟③已验证，属预期行为）。
- **时序编排**：FA 后的 det 特征**继续进下一层下采样**（贴官方「FA 后特征继续进下一层」，[vit_seg_modeling.py L455-469](file:///e:/mkl/code/python_code/carotid-artery/networks/vit_seg_modeling.py#L455-L469)）。
- **参数增量**：本项目 +1.206M（HF 支 +0.29M、FAM +0.84M、inject 微量），相对 105.276M 基线 ≈1.1%，值——这是文献里稳定出增益的第 2 档门控（paper-reading §5.4），比我们 dual 四接点交叉注意力 3.70M 便宜一个量级。
- **对照档位**：FAM 属门控加权档；论文**没做** gate/concat/attn/deform 横评，所以不能引它证明「FAM 优于其他融合算子」，只能证明「有 FAM 比没有强」。

### 2.3 损失与训练配置

标准分割交叉熵，无辅助损失、无深监督——好处是所有涨跌都能归因结构。训练：4×RTX3090，SGD momentum 0.9，batch 16，200 epoch。**可信度关键：消融全部在参数量对齐下做**，对照多出来的参数与实验组相同，涨幅赖不到容量头上。

## 3. 指标与结果

- 数据：7 个公开 EM 数据集（EPFL、CREMI、SNEMI3D、UroCell、MitoEM、Nanowire、BetaSeg），指标 Jaccard/Dice，宣称大幅超过当时 SOTA（逐表数字见原文；本文档不抄录未在本项目验证过的具体值）。
- 正文消融怎么读：

| # | 结论 | 判读 |
|---|---|---|
| ① | HF 图作额外输入通道即提升 | 输入级注入有效，代价仅 +1 通道 |
| ② | 加独立 HF 编码支再提升 | 特征级 > 输入级 |
| ③ | **引入 LF 分量反而变差** | 主干已吃原图，LL 是冗余+噪声（核心结论） |
| ④ | 二级小波无显著提升 | 分解级数不是矛盾 |
| ⑤ | 小波基多数不敏感（UroCell 例外：最好/最差差 8.89%）| 换基不值得花时间 |
| ⑥ | FAM 挂所有编码层最好 | 融合位置：全挂 |

- ①②③④⑥构成控制变量递进链，是证据；⑤是提醒。
- **与 XNet 对照**：XNet +2.74 JI 混着容量红利（raw-only 基线是单编码器，XNet 全网 326M/83G）；GobletNet 对齐后「只有高频有用」。**不能引用 XNet 的 +2.74 支持双骨干方案**（literature.md 结论 1）。

## 4. 代码实现在哪里（本项目批次 2 移植）

| 内容 | 位置 |
|---|---|
| HFBranch（轻量 HF 支 32@112²/64@56²/128@28²） | [hf_branch.py L37-62](file:///e:/mkl/code/python_code/carotid-artery/networks/hf_branch.py#L37-L62) |
| FAM（通道适配 + γ 门控残差） | [hf_branch.py L65-91](file:///e:/mkl/code/python_code/carotid-artery/networks/hf_branch.py#L65-L91) |
| 高频图 H（Eq.5，λ 回灌） | [wavelet.py L58-86](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py#L58-L86) |
| 挂载（RNG 对齐纪律）与 forward 浅→深编排 | [vit_seg_modeling.py L422-469](file:///e:/mkl/code/python_code/carotid-artery/networks/vit_seg_modeling.py#L422-L469) |
| 参数开关（仅 baseline；快照后缀 `_hfbr3`） | [train.py L76-91](file:///e:/mkl/code/python_code/carotid-artery/train.py#L76-L91)、[scripts/eval_carotid.py](file:///e:/mkl/code/python_code/carotid-artery/scripts/eval_carotid.py) |
| 冒烟记录（5 项全过，105.276M→106.482M） | [experiments.md](../experiments.md) §1.9 执行记录② |

## 5. 结论边界与其他

### 能说 / 不能说

| 能说 | 不能说 |
|---|---|
| 参数量对齐后，「加 LF」在 EM 上是负面的 | LF 在任何任务上都负面（BDFR-Net 特征级 LL 支有正贡献，域不同） |
| HF 图+轻量支+FAM 在 EM 7 数据集有效 | 直接迁移到超声有效（论文无超声、不报 HD95/ASSD） |
| 特征级注入 > 输入级注入（①→②） | FAM 优于其他融合算子（论文未做算子横评） |
| 融合挂在全部编码层最好（EM 上） | 在我们 4 级 skip 框架上「全挂」也最好（分辨率/容量不同，待本项目验证） |

### 已知局限

- 仅 EM 图像，无超声；不报边界距离指标（HD95/ASSD），而颈动脉的残差恰恰集中在边界。
- UroCell 上小波基敏感（差 8.89%）说明结论对数据域有依赖。
- 本项目批次 2（hfbr3）尚未训练评估，以上迁移判断全部**待验证**。

### 复现命令（PowerShell，本项目批次 2）

```powershell
E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe train.py `
  --dataset Carotid --subset SPARC --seed 1234 `
  --batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5 `
  --hf_branch 1 --hf_lambda 0.3
# 快照：TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_hfbr3_c2d
```

### 与本项目其他方案的对比

| 方案 | 注入位置 | 融合算子 | 额外参数 | 状态 |
|---|---|---|---|---|
| `--edge_in` | 输入级（stem 4 通道） | 无（靠主干自己学） | +27 | 已有结论：输入级 |
| 批次 2 `--hf_branch`（本篇） | 特征级（3 级 skip） | FAM 门控（第 2 档） | +1.206M | 待入队 |
| dual 双骨干交叉注意力 | 特征级（4 接点） | 跨支 QKV 交叉注意力（第 6 档） | +3.70M | 已跑：无增益 |
| 批次 1 DDBE/ACAM | 解码器级 | DDBE 频域解耦 | +9.5M | 在训 |

同图派生的双输入增益天然偏小（MS-Net 实测 +0.003~+0.008 Dice），批次 2 判定沿用 §1.9 规则：\|ΔDice\| ≥ 0.005 或 HD95 下降 ≥ 0.5 mm 且 p<0.05，否则记负面结果停线。
