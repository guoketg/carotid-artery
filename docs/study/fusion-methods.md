# 融合方法总览：现在有哪些、哪些试过、老师指定先试哪个

> 2026-10-27 应你要求整理。**本文档只做方法盘点，不改任何代码。**
> 老师指定的实验室内部论文（`pdf\3612 Submission.pdf` = BDFR-Net）放在最前，
> 其余按"融合轴向"分类。队列已同步改为只跑 SPARC（中南 4 个 job 已撤）。

## 0. 一页版

| 问题 | 答案 |
|---|---|
| 老师让先试的 | **BDFR-Net（3612）**：特征做 DWT 分解 → 高频/低频/空间上下文三分支各自处理 → **concat + 1×1 + CBAM 融合**。融合走通道轴，不用 QKV 跨位置注意力 |
| 我们已实现且测过的 | 输入级 4 通道拼接（**HF 可归因增益 +0.015~0.018**）；特征级 concat（+0.005 无效）；QKV 交叉注意力（涨幅与 HF 内容无关，侧路喂全零反而最高） |
| 已实现但还没在正式模型上测的 | gate / deform / aff 三种融合算子（双骨干队列只排了 auto=QKV） |
| 完全没试过的 | BDFR 的 DDBE 三分支、ACAM skip 对齐、FcaNet 式 DCT 通道注意力、损失级频率一致性 |
| 你的判断（"低频高频是通道的事"）与文献的关系 | BDFR-Net、WaveFusion、XNet、GobletNet 全部把频率当**子带（通道）**处理，没有任何一篇拿 QKV 跨位置注意力去融合低频和高频 |

## 1. 老师指定：BDFR-Net（`pdf\3612 Submission.pdf`，未发表）

胎儿超声分割（胎头 FH + 耻骨联合 PS），PVTv2-B2 编码器 + 四级级联解码器。
每个解码级依次放三个模块：**ECA → DDBE → ACAM**。损失 = Dice + CE 加权和。

### 1.1 DDBE：双域解耦 + 通道轴融合（本文核心，也是"先试"的对象）

输入特征 F（C×H×W），先 ECA 抑制斑点噪声响应，然后**三条并行支路**：

| 支路 | 运算 | 输出 | 作用 |
|---|---|---|---|
| **高频支路** | 2D-DWT 取 LH/HL/HH 三个方向子带 → **concat 成 3C** → 1×1 压回 C → 3×3 精修 → 转置卷积上采样 → 通道注意力 CE | F_hf | 方向性边缘响应（消融里增益最大的部件） |
| **低频支路** | DWT 取 LL 近似子带 → **5×5 卷积** → 上采样 → CE | F_lf | 大尺度结构骨架，稳住轮廓完整性 |
| **空间上下文支路** | 7×7 卷积 + 通道自门控 `F⊙σ(MLP(GAP(·)))` | F_ctx | 长程区域依赖（两个分离结构间的关系） |

**融合方式（关键）**：

```
F_DDBE = M_sp( M_ch( Conv1x1( concat(F_hf, F_lf, F_ctx) ) ) )
```

即：同位置三特征图**通道拼接** → 1×1 回投影 → CBAM 的通道注意力 + 空间注意力两级加权。
**全程没有跨空间位置的 QKV 注意力**——频率信息从头到尾按子带（通道）处理，
注意力只用来给通道/区域"打分加权"，不用来"重组内容"。这正是你直觉的文献版本。

### 1.2 ACAM：skip 连接的加性融合（第二个可借的模块）

编码器特征 g 与解码器特征 x 直接 concat 会有语义错位。ACAM 对每条支路：
1×1 投影 → softmax 得空间权重 → 加权求和压成上下文向量 → bottleneck 得**逐通道标量权重 W**，
最后 `F = W_g⊙g + W_x⊙x`（**各自定标后相加**）。

### 1.3 消融数据（IUGC 2024，3 次运行均值，std < 0.5%）

| 配置 | mean Dice | mean HD |
|---|---|---|
| 基线（PVT-B2） | 84.74 | 14.55 |
| + 高频支路 | 85.53（**+0.79，单件最大**） | 13.25 |
| + 低频支路 | 85.95 | 12.65 |
| + 上下文支路 | 86.37 | 12.33 |
| 仅 + ACAM | 85.21 | 13.91 |
| 全部（=BDFR-B2） | **86.83** | **11.98** |

对低对比度边界（PS）增益最大：Dice +3.01 超第二名——和我们要解决的颈动脉斑块边界问题同构。

## 2. 融合方法分类总表（按轴向）

状态标记：✅已实现并测过 · ⏳已实现未测 · ❌未实现 · 📖=出处论文

| # | 方法 | 轴向 | 一行原理 | 我们的状态 |
|---|---|---|---|---|
| 1 | **输入级通道拼接**（stem 4 通道 = 原图+HF 图） | 通道 | 多一个通道进第一个卷积 | ✅ **SPARC/中南独立复现 +0.0181/+0.0183**；正式模型 edge 队列在跑 |
| 2 | **特征 concat + 1×1 回投影**（`BidirConcat`） | 通道 | 同位置两支特征合并再压缩 | ✅ TinyCNN 尺度 +0.005（无效）；正式模型未测 |
| 3 | **逐像素门控**（`BidirGate`） | 通道/幅度 | 用对方的空间统计图缩放自己，只调幅度不重组内容 | ⏳ 已实现，未测 |
| 4 | **QKV 跨支交叉注意力**（`BidirCrossAttention`，window/global） | 空间 | 主路每个位置去"查询"侧路所有位置的内容 | ✅ 已测：**涨幅 +0.24 与 HF 内容无关**（侧路全零更高）；正式模型 d1234_sparc 已跑完待分析 |
| 5 | **可变形大核跨支聚合**（`BidirDeform`，D-LKA 式） | 几何 | 采样点跟着边界走而非固定网格 | ⏳ 已实现，未测 |
| 6 | **AFF 两级注意力融合**（MS-CAM + LAFF） | 通道 | 多尺度通道注意力给两支定标后相加 | ⏳ 已实现，未测 |
| 7 | **BDFR DDBE 三分支**（§1.1） | 通道 | DWT 分方向子带，HF/LF/CTX 各配归纳偏置，concat+CBAM | ❌ 未实现，**老师指定先试** 📖3612 |
| 8 | **BDFR ACAM**（§1.2） | 通道/skip | 跨层特征各自定标后加性融合 | ❌ 未实现 📖3612 |
| 9 | **小波域分工块**（WaveFusion：LL→注意力、LH/HL/HH→卷积，IDWT 重建） | 通道 | 频内分工，不做跨频融合 | ❌ 未实现 📖TCSVT2025 |
| 10 | **XNet / GobletNet / Syn-Net 类小波双频网络** | 通道 | 与 DDBE 同族：分带→分处理→融合重建 | ❌ 未实现 📖pdf 目录 |
| 11 | **FcaNet 式 DCT 通道注意力** | 通道 | 把 GAP 换成多频率 DCT 基，通道权重带频域信息 | ❌ 未实现 📖pdf 目录 |
| 12 | **WeaveSeg 频谱特征精炼** | 通道 | 迭代对比编织 + 频谱精炼（核分割） | ❌ 未实现 📖pdf 目录 |
| 13 | **损失级频率一致性正则** | — | 约束预测边界与 HF 图响应一致 | ❌ 未实现（technical-design 备选） |

## 3. 已有实验事实（只列跑过的）

| 实验 | 结果 | 出处 |
|---|---|---|
| TinyCNN 门槛：raw vs raw+HF | +0.0181（SPARC）/ +0.0183（中南），两轮独立复现 | probe_edge_signal.py |
| 融合轴：通道 vs 注意力 | raw_hf +0.015；feat_concat +0.005；feat_attn +0.24 但 feat_attn_zero +0.27 → **注意力涨幅不来自 HF** | probe_fusion_axis.py |
| 双骨干正式模型 | d1234_sparc 已跑完（102 min），**待与基线 seed1234 配对分析** | logs/queue/ |
| BDFR 消融（论文自己的数据） | 高频支路是单件最大增益 +0.79 Dice | 3612 Submission.pdf Table 2 |

## 4. 建议的尝试顺序（等你确认后才动代码）

1. **DDBE 式三分支**（老师指定优先）：可先挂在我们 TransUNet 解码器的某一级做门槛验证，
   再决定全四级；融合算子照抄 concat+1×1+CBAM，不引入 QKV。
2. **ACAM 替换 skip 直连**：改动小（每级一个轻量模块），与 DDBE 正交，可叠加。
3. 已实现未测的 **aff / gate / deform**：作为 DDBE 的对照组，回答"通道轴内部谁最好"。
4. 队列现状（已按你指令改好）：主队列剩 d2024、d3407（双骨干×SPARC），
   edge 队列 e1234、de1234（SPARC），全部约 2×4 个 job、明天下午跑完。

## 5. 代码与文档位置

- 老师论文：[pdf\3612 Submission.pdf](file:///e:/mkl/code/python_code/carotid-artery/pdf/3612%20Submission.pdf)
- 已实现的 4 种融合算子：[networks/dual_encoder_model.py](file:///e:/mkl/code/python_code/carotid-artery/networks/dual_encoder_model.py)（BidirGate/BidirConcat/BidirCrossAttention/BidirDeform/AFF）
- 输入级 edge 融合：[networks/wavelet.py](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py) `build_hf_image`
- 融合轴对照实验：[scripts/probe_fusion_axis.py](file:///e:/mkl/code/python_code/carotid-artery/scripts/probe_fusion_axis.py)（讲解见 [tinycnn-explained.md](tinycnn-explained.md) §6）
- 方案取舍背景：[docs/technical-design.md](file:///e:/mkl/code/python_code/carotid-artery/docs/technical-design.md)「频率注入位置」一节
