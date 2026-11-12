# 低分样本：特性、可视化与针对性模块设计

> 回答三个问题：低分样本长什么样（图 + 指标）、它们错在哪（几何诊断）、据此能设计什么模块且值不值得做。
> 口径与评测链路一致（原分辨率 + 原始 spacing，mm），不引入 TTA。

## 0. 结论先说

1. **低分样本有稳定共性，但不是「低回声」**：低 Dice 的切片更**薄**（短轴 1.12 vs 1.26 mm）、更**长**（长宽比 5.68 vs 4.37，SPARC）、**局部对比度更低**（CNR 0.21 vs 0.42，SPARC）。相对回声强度与 Dice 无关（rho≈0.03 / −0.05，且两库方向相反）。
2. **误差形态是「贴着 GT 边界的一像素带」**：每张切片都有漏检带，带深中位数 0.13 mm（SPARC）/ 0.095 mm（Zhongnan），恰好约等于 1 个原始像素；预测没有孤立连通域（n_cc 中位数 = 1）。
3. **不是整体膨胀/腐蚀**：全局固定偏移扫描的最优 k 就是 0（SPARC 0.9049 / Zhongnan 0.9200）；逐切片最优偏移的收益只有 +0.87 / +0.75 Dice（低分组 +1.71 / +1.64）。方向在切片间分化（SPARC 最优 k：0 → 48 张、+1 → 45 张、−1 → 35 张），所以任何「全局加粗/收缩」类修正在原理上就拿不到分。
4. **高 HD 与低 Dice 不是同一条线**：高 HD 的切片是**更大更厚**的斑块（25.8 vs 16.4 mm²；短轴 1.42 vs 1.11 mm，两库一致，rho≈0.45–0.53），是局部单点越界把最大距离拉高；低 Dice 是薄长结构的边界整体贴不准。
5. 由此设计的模块**不是**再加一层边缘/频率增强（本项目已实测输入级边缘图零增益、边界加权 CE 仅 +0.41 未过线），而是补上网络里**唯一一处完全没有原图证据的接点**：解码器 224² 级（无 skip、最细图像证据只有 112²）。方案与精确改法见 §6，可行性判断见 §7。

## 1. 口径与复现

| 项 | 值 |
|---|---|
| 骨干 / 解码器 | ResNet-50（ImageNet21k 预训练，`vit_layers=0`）+ DecoderCup（TransUNet） |
| 训练协议 | SPARC / Zhongnan 各自独立训练，seed 1234，bs 4 × 梯度累积 2，200 epoch，判定点 `epoch_199.pth` |
| 测试集 | SPARC 160 张切片（病例级划分），Zhongnan 187 张 |
| 指标口径 | 224 预测用 `ndi.zoom(order=0)` 回原网格，GT 读原图；Dice / ASSD / HD 全部按原始 spacing（mm）计算，**全文只用 HD，不用 HD95** |
| 低分定义 | **库内** Dice ≤ p25 记为低 Dice；HD ≥ p75 记为高 HD（与 CLAUDE.md §5 的距离指标口径一致） |
| 样本数 | SPARC：低 Dice 40 / 高 HD 40 / 两者同时 18（Dice p25 = 0.8881，HD p75 = 0.828 mm）；Zhongnan：47 / 47 / 19（p25 = 0.9035，HD p75 = 0.686 mm） |

复现可视化与表格（只读脚本，不改任何权重）：

```powershell
.\venv\Scripts\python.exe scripts\visualize_low_score_cases.py
```

产出：`docs/assets/low-score/SPARC/`（12 张，每样本一张）、`docs/assets/low-score/Zhongnan/`（12 张）、`docs/assets/low-score/shape_scatter.png`、`logs/analysis/low_score_tables.md`。
上游诊断数据来自 `scripts/analyze_error_slices.py`（`logs/analysis/error_slices_*.csv`）、`scripts/analyze_pred_geometry.py`、`scripts/oracle_offset_ceiling.py`、`scripts/analyze_error_side.py`。

## 2. 基线（本表所有 Δ 的参照）

| 臂 | 数据集（n） | Dice | ASSD (mm) | HD (mm) | ΔArea (mm²) |
|---|---|---|---|---|---|
| R50+DecoderCup 基线（seed 1234，epoch 199） | SPARC（160） | 90.49 | 0.145 | 0.706 | −0.045 |
| R50+DecoderCup 基线（seed 1234，epoch 199） | Zhongnan（187） | 92.00 | 0.123 | 0.565 | −0.281 |

## 3. 逐样本可视化

每张样本单独一个文件，一格一件事：左 = 全图（绿 GT 轮廓 / 红预测轮廓）；中 = 斑块局部放大（同色标）；右 = 局部放大误差图（红 FP / 蓝 FN）。标题里带该样本的 Dice / ASSD / HD / 面积 / 长宽比 / 短轴 / CNR / 预测-真值面积比。
取法：**低 Dice（≤ p25）∩ 高 HD（≥ p75）**，按 Dice 升序取前 12 张。

**SPARC（12 张）**

| # | 样本 | Dice | ASSD | HD | 图 |
|---|---|---|---|---|---|
| 01 | 501099_06-20-20180002 | 0.758 | 0.273 | 1.43 | [打开](assets/low-score/SPARC/01_501099_06-20-20180002.png) |
| 02 | 170040_2018-18-010002 | 0.767 | 0.710 | 5.46 | [打开](assets/low-score/SPARC/02_170040_2018-18-010002.png) |
| 03 | 863569_11-26-20180006(1) | 0.792 | 0.491 | 4.81 | [打开](assets/low-score/SPARC/03_863569_11-26-20180006%281%29.png) |
| 04 | 380180_08-01-20190000 | 0.826 | 0.192 | 1.07 | [打开](assets/low-score/SPARC/04_380180_08-01-20190000.png) |
| 05 | 631645_09-10-20180003(1) | 0.826 | 0.271 | 0.91 | [打开](assets/low-score/SPARC/05_631645_09-10-20180003%281%29.png) |
| 06 | 863569_11-26-20180005(2) | 0.835 | 0.241 | 1.73 | [打开](assets/low-score/SPARC/06_863569_11-26-20180005%282%29.png) |
| 07 | 631645_09-10-20180002 | 0.837 | 0.251 | 0.84 | [打开](assets/low-score/SPARC/07_631645_09-10-20180002.png) |
| 08 | 617803_01-18-20180001(2) | 0.839 | 0.251 | 1.82 | [打开](assets/low-score/SPARC/08_617803_01-18-20180001%282%29.png) |
| 09 | 424309_01-06-20150007 | 0.845 | 0.231 | 0.90 | [打开](assets/low-score/SPARC/09_424309_01-06-20150007.png) |
| 10 | 170040_2018-18-010004(2) | 0.853 | 0.258 | 1.49 | [打开](assets/low-score/SPARC/10_170040_2018-18-010004%282%29.png) |
| 11 | 567577_05-26-20140006 | 0.872 | 0.251 | 1.17 | [打开](assets/low-score/SPARC/11_567577_05-26-20140006.png) |
| 12 | 206261_04-08-20190010 | 0.873 | 0.180 | 0.84 | [打开](assets/low-score/SPARC/12_206261_04-08-20190010.png) |

**Zhongnan（12 张）**

| # | 样本 | Dice | ASSD | HD | 图 |
|---|---|---|---|---|---|
| 01 | 20077_3 | 0.769 | 0.526 | 1.39 | [打开](assets/low-score/Zhongnan/01_20077_3.png) |
| 02 | 20314_2 | 0.786 | 0.317 | 1.14 | [打开](assets/low-score/Zhongnan/02_20314_2.png) |
| 03 | 0899_4 | 0.812 | 0.280 | 1.01 | [打开](assets/low-score/Zhongnan/03_0899_4.png) |
| 04 | 0211_1 | 0.845 | 0.211 | 0.99 | [打开](assets/low-score/Zhongnan/04_0211_1.png) |
| 05 | 20589_6 | 0.863 | 0.197 | 0.70 | [打开](assets/low-score/Zhongnan/05_20589_6.png) |
| 06 | 1066_2 | 0.877 | 0.261 | 1.15 | [打开](assets/low-score/Zhongnan/06_1066_2.png) |
| 07 | 1146_2 | 0.881 | 0.204 | 0.70 | [打开](assets/low-score/Zhongnan/07_1146_2.png) |
| 08 | 20690_2 | 0.881 | 0.164 | 0.80 | [打开](assets/low-score/Zhongnan/08_20690_2.png) |
| 09 | 1130_2 | 0.884 | 0.217 | 0.90 | [打开](assets/low-score/Zhongnan/09_1130_2.png) |
| 10 | 1360_5 | 0.889 | 0.172 | 4.22 | [打开](assets/low-score/Zhongnan/10_1360_5.png) |
| 11 | 1149_1 | 0.889 | 0.179 | 0.81 | [打开](assets/low-score/Zhongnan/11_1149_1.png) |
| 12 | 0769_2 | 0.889 | 0.186 | 0.81 | [打开](assets/low-score/Zhongnan/12_0769_2.png) |

肉眼可核对的三点：① 低分切片基本是**细长条**或**极薄的一侧壁**；② 误差几乎全是沿轮廓的**一条细带**（不是整块多检/漏检）；③ 两库的多检/漏检倾向相反（SPARC 偏多检、Zhongnan 偏漏检），所以不能靠单向阈值修。

**逐切片 Dice 与三个特征的散点**

![Dice 与形态/对比度](assets/low-score/shape_scatter.png)

## 4. 指标表

### 4.1 SPARC：最差 12 张（低 Dice ∩ 高 HD）

| case | Dice | ASSD | HD | 面积 mm² | 长宽比 | 短轴 mm | CNR | rel_echo | pred/GT 面积比 |
|---|---|---|---|---|---|---|---|---|---|
| 501099_06-20-20180002 | 0.758 | 0.273 | 1.43 | 16.4 | 6.56 | 1.26 | 0.04 | −0.04 | 1.50 |
| 170040_2018-18-010002 | 0.767 | 0.710 | 5.46 | 71.3 | 9.12 | 1.72 | 0.02 | 0.02 | 1.01 |
| 863569_11-26-20180006(1) | 0.792 | 0.491 | 4.81 | 48.4 | 12.26 | 1.38 | 0.20 | 0.21 | 1.15 |
| 380180_08-01-20190000 | 0.826 | 0.192 | 1.07 | 9.8 | 6.70 | 0.70 | 0.02 | −0.02 | 1.21 |
| 631645_09-10-20180003(1) | 0.826 | 0.271 | 0.91 | 20.5 | 6.64 | 1.04 | 0.57 | 0.50 | 1.17 |
| 863569_11-26-20180005(2) | 0.835 | 0.241 | 1.73 | 25.7 | 7.10 | 1.25 | 0.41 | −0.37 | 1.02 |
| 631645_09-10-20180002 | 0.837 | 0.251 | 0.84 | 12.0 | 2.15 | 1.37 | 1.21 | 1.02 | 0.82 |
| 617803_01-18-20180001(2) | 0.839 | 0.251 | 1.82 | 25.5 | 7.71 | 1.12 | 1.03 | 0.86 | 1.34 |
| 424309_01-06-20150007 | 0.845 | 0.231 | 0.90 | 14.6 | 4.57 | 1.07 | 0.03 | 0.04 | 1.25 |
| 170040_2018-18-010004(2) | 0.853 | 0.258 | 1.49 | 25.9 | 1.89 | 2.31 | 1.11 | 0.84 | 0.84 |
| 567577_05-26-20140006 | 0.872 | 0.251 | 1.17 | 23.7 | 1.90 | 2.13 | 0.97 | 0.84 | 0.78 |
| 206261_04-08-20190010 | 0.873 | 0.180 | 0.84 | 10.7 | 3.61 | 1.06 | 2.15 | 1.53 | 1.23 |

### 4.2 Zhongnan：最差 12 张（低 Dice ∩ 高 HD）

| case | Dice | ASSD | HD | 面积 mm² | 长宽比 | 短轴 mm | CNR | rel_echo | pred/GT 面积比 |
|---|---|---|---|---|---|---|---|---|---|
| 20077_3 | 0.769 | 0.526 | 1.39 | 34.3 | 2.11 | 2.42 | 0.50 | 0.53 | 0.64 |
| 20314_2 | 0.786 | 0.317 | 1.14 | 28.7 | 9.42 | 1.14 | 0.28 | −0.24 | 1.14 |
| 0899_4 | 0.812 | 0.280 | 1.01 | 28.4 | 6.54 | 1.20 | 0.51 | 0.51 | 0.72 |
| 0211_1 | 0.845 | 0.211 | 0.99 | 23.5 | 7.54 | 1.06 | 0.12 | 0.15 | 1.10 |
| 20589_6 | 0.863 | 0.197 | 0.70 | 13.3 | 3.39 | 1.16 | 2.04 | 1.46 | 0.98 |
| 1066_2 | 0.877 | 0.261 | 1.15 | 47.9 | 5.21 | 1.82 | 0.13 | 0.16 | 0.93 |
| 1146_2 | 0.881 | 0.204 | 0.70 | 19.2 | 3.85 | 1.27 | 0.39 | −0.30 | 1.08 |
| 20690_2 | 0.881 | 0.164 | 0.80 | 13.7 | 3.88 | 1.07 | 1.82 | 1.34 | 0.85 |
| 1130_2 | 0.884 | 0.217 | 0.90 | 22.0 | 2.80 | 1.61 | 0.41 | 0.42 | 0.85 |
| 1360_5 | 0.889 | 0.172 | 4.22 | 33.4 | 7.18 | 1.51 | 0.05 | 0.07 | 1.10 |
| 1149_1 | 0.889 | 0.179 | 0.81 | 13.6 | 1.96 | 1.57 | 1.01 | 0.90 | 0.92 |
| 0769_2 | 0.889 | 0.186 | 0.81 | 21.8 | 3.75 | 1.41 | 0.48 | 0.50 | 0.97 |

### 4.3 分组中位数

| SPARC（n=160） | 低 Dice（40） | 其余（120） | 高 HD（40） | 其余（120） |
|---|---|---|---|---|
| 长宽比 | 5.683 | 4.373 | 4.604 | 4.629 |
| 短轴 mm | 1.124 | 1.263 | 1.418 | 1.114 |
| CNR | 0.211 | 0.419 | 0.396 | 0.378 |
| rel_echo | 0.050 | 0.172 | 0.312 | 0.065 |
| 面积 mm² | 20.155 | 17.080 | 25.823 | 16.439 |
| Dice | 0.871 | 0.920 | 0.894 | 0.917 |
| HD mm | 0.787 | 0.520 | 1.025 | 0.495 |

| Zhongnan（n=187） | 低 Dice（47） | 其余（140） | 高 HD（47） | 其余（140） |
|---|---|---|---|---|
| 长宽比 | 3.878 | 3.861 | 4.002 | 3.808 |
| 短轴 mm | 1.152 | 1.211 | 1.407 | 1.126 |
| CNR | 0.327 | 0.386 | 0.393 | 0.362 |
| rel_echo | 0.185 | 0.098 | 0.162 | 0.111 |
| 面积 mm² | 16.847 | 16.771 | 23.670 | 14.778 |
| Dice | 0.889 | 0.932 | 0.915 | 0.928 |
| HD mm | 0.618 | 0.469 | 0.807 | 0.424 |

### 4.4 Spearman 相关（库内，仅作诊断，不作显著性结论）

| 变量对 | SPARC rho (p) | Zhongnan rho (p) |
|---|---|---|
| 长宽比 ~ Dice | −0.246 (0.002) | −0.056 (0.449) |
| 短轴 mm ~ Dice | 0.100 (0.210) | 0.200 (0.006) |
| CNR ~ Dice | 0.142 (0.074) | −0.024 (0.741) |
| rel_echo ~ Dice | 0.034 (0.665) | −0.048 (0.513) |
| 面积 mm² ~ HD | 0.531 (0.000) | 0.546 (0.000) |
| 短轴 mm ~ HD | 0.457 (0.000) | 0.434 (0.000) |
| 长宽比 ~ HD | 0.110 (0.165) | 0.166 (0.023) |

## 5. 误差几何：能修什么、不能修什么

| 诊断量（逐切片中位数） | SPARC | Zhongnan | 含义 |
|---|---|---|---|
| 漏检像素占比 `miss_frac` | 7.8% | 7.4% | 每张切片都有漏检 |
| 漏检带深度 `miss_depth_mm` | 0.130 | 0.095 | ≈ 1 个原始像素 |
| 多检像素占比 `extra_frac` | 8.0% | 6.6% | 与漏检同量级 |
| 连通域个数 `n_cc` | 1 | 1 | 无孤立假阳性块 |
| 边界整体偏移 `abs_offset_mm` | 0.130 | 0.134 | 同样 ≈ 1 px |
| 全局固定偏移最优 k | 0 | 0 | 无系统性加粗/收缩 |
| 逐切片最优偏移的 Dice 增益 | +0.0087（低分组 +0.0171） | +0.0075（低分组 +0.0164） | 修正上界很低 |
| 高 HD 组面积中位数 | 25.8 mm² | 23.7 mm² | 大斑块，单点越界拉高 HD |

已排除的方向（都有实测，不要再试）：全局阈值 0.4/0.6、开闭运算后处理、面积一致性纠偏、把边界带单独加权（`BoundaryWeightedCELoss` 实测仅 +0.41 未过 CLS 门槛）、4 向翻转 TTA（SPARC 0.9049→0.9097，用户已决定不上方法）。在能可靠估计的切片上，SPARC 的多检/漏检落在暗侧的比例是 0.44 / 0.51（接近对称），说明没有可利用的「系统性偏暗偏好」；Zhongnan 这一项分母多为 0，不参与判断。

## 6. 针对低分特性的模块：224 全分辨率证据支路（开关 `--head_hr`）

### 6.1 为什么补这里

实测的网络形状（`R50-ViT-B_16`、`n_skip=3`、`vit_layers=0`、输入 224²）：

```
解码器四级：(14², 512) → (28², 256) → (56², 128) → (112², 64) → (224², 16)  → 1×1 头 → logits
skip 来源  ：          28² R50       56² R50       112² R50      （无）
```

- 最细一级 `224²/16ch` 的输入是 `112²/64ch` 的**双线性上采样**结果，该级 `skip_channels` 被 `n_skip=3` 置 0，**完全没有原图证据**；
- 网络里能看到的最高分辨率图像证据停在 112²，而标签回原网格后是 186×280 之类，**比标签网格粗约 2 倍**；
- 低分样本恰好是短轴 1.1 mm（≈1–2 个原始像素）的薄结构，误差又恰好是 1 个原始像素宽的边界带（§5）。两件事在尺度上吻合：**边界定位缺的正是标签分辨率这一级的图像证据**。

注意表述边界（不要越界）：这不等于「边界是锐利高频」。本项目实测 GT 边界带梯度只有全图平均的 1.90 倍、输入级边缘图零增益，所以方案里**不使用** Sobel / Laplacian / 小波高频等边缘算子，只用原图灰度与其多尺度局部统计量。

### 6.2 结构

在 224² 上把解码器末级特征 `x`（16ch）与由原图派生的 4 个通道拼接，做两层 3×3 卷积后输出 2 通道 logits 残差，加回主 logits：

```
输入通道（均 224²，全部是低频/统计量）：
  I     归一化后的原图（数据集已做 1%~99% 截断 + z-score）
  m5    5×5 均值      —— 局部亮度基准
  m21   21×21 均值    —— 更大尺度的腔/斑块背景亮度
  s9    9×9 局部标准差 —— 散斑强度（低对比度处的可靠性）

h1 = ReLU(BN(conv3x3([x, I, m5, m21, s9])))     20 → 16
h2 = ReLU(BN(conv3x3(h1)))                      16 → 16
logits = segmentation_head(x) + conv1x1(h2)     16 → 2，末层零初始化
```

- **恒等起步**：`conv1x1` 权重与偏置零初始化，起始输出严格为 0，同 seed 下与基线逐位一致（与项目里其他模块同一纪律）。
- **参数量**：5 282（2880 + 32 + 2304 + 32 + 34）。
- **算力**：224² 上约 0.55 GFLOPs/前向，相对骨干可忽略；显存增加约 20–30 MB（16ch × 224² 的几份激活）。
- **为什么用局部均值/标准差而不是边缘算子**：低分切片 CNR 只有 0.21/0.33，边界处的信息是「一阶灰度台阶 + 散斑统计」而不是高频纹理；局部均值给出台阶两侧的基准亮度，局部标准差给出该处可不可信，正对 §4 的观察。

### 6.3 精确改法（4 处，已落地）

**(1) `networks/vit_seg_modeling.py`：加 `F` 导入**

文件顶部 `import torch.nn as nn` 之后加一行：

```python
import torch.nn.functional as F
```

**(2) `networks/vit_seg_modeling.py`：新增模块类**

紧接在 `class DecoderCup(nn.Module):` 之前（即 `SegmentationHead` 之后）插入：

```python
class FullResEvidenceHead(nn.Module):
    """224² 全分辨率证据支路：把原图的多尺度局部统计量接到解码器最细一级。

    解码器末级（224²/16ch）既无 skip、也不接触原图，最细图像证据只到 112²。
    本模块在该级补 4 个原图派生通道（I / 5×5 均值 / 21×21 均值 / 9×9 局部标准差），
    输出 logits 残差；末层零初始化 → 起点与基线逐位等价。
    """

    def __init__(self, dec_ch, num_classes=2, mid_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(dec_ch + 4, mid_ch, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, num_classes, 1)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, img):
        m5 = F.avg_pool2d(img, 5, 1, 2)
        m21 = F.avg_pool2d(img, 21, 1, 10)
        avg9 = F.avg_pool2d(img, 9, 1, 4)
        var9 = F.avg_pool2d(img * img, 9, 1, 4) - avg9 * avg9
        s9 = torch.sqrt(var9.clamp_min(1e-6))
        h = self.act(self.norm1(self.conv1(torch.cat([x, img, m5, m21, s9], dim=1))))
        h = self.act(self.norm2(self.conv2(h)))
        return self.conv3(h)
```

**(3) `networks/vit_seg_modeling.py`：`VisionTransformer` 三处小改**

- 签名（现 L576–582）末尾加参数：`fd_mfea=False, fd_fgbr=False, head_hr=False)`；
- `__init__` 最末尾（现 L697–701，`if edge_fusion:` 块之后）加：

```python
        self.hr_head = None
        if head_hr:
            # 最后创建：RNG 消耗在全部基础权重之后，同 seed 与基线逐位对齐
            self.hr_head = FullResEvidenceHead(config['decoder_channels'][-1],
                                               config['n_classes'])
```

- `forward`（现 L729–734）改为：

```python
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        if self.hr_head is not None:
            logits = logits + self.hr_head(x, x_in[:, :1])
        return logits
```

**(4) 训练 / 评估接线**

- `train.py` 参数区（`--fd_fgbr` 块 L242–246 之后）加：

```python
parser.add_argument('--head_hr', type=int, default=0,
                    help='1 = 解码器 224² 级接原图全分辨率证据支路（I/5×5 均值/21×21 均值/'
                         '9×9 局部标准差 4 通道 → 两层 3×3 → logits 残差，零初始化恒等起步）')
```

- `train.py` 快照命名（当前 L646 `snapshot_path+='_c2d'` 之后）加：

```python
        if args.head_hr:
            snapshot_path += '_hr'
```

- `train.py` 构造 `ViT_seg`（当前 L746–765）加 `head_hr=bool(args.head_hr)`；
- 参数互斥校验（当前 L306 的 `--fd_mfea/--fd_fgbr` 检查块）加：`head_hr` 只允许 `--arch baseline`，且与 `--fgr/--ffusion/--bdfr_v2/--edge_fusion/--r50_fdf/--fd_mfea/--fd_fgbr` 互斥（单模块臂，只变一个变量）；
- `scripts/eval_carotid.py`：参数区（当前 L198 附近）加同义 `--head_hr`，`build_model`（当前 L74–77）加 `head_hr=bool(args.head_hr)`；否则加载新快照会在 `load_state_dict` 报 unexpected keys（L86–88 是严格校验）。

### 6.4 训练与判定

```powershell
# 冒烟（先确认恒等起步与接线）
.\venv\Scripts\python.exe train.py --dataset Carotid --arch baseline --vit_name R50-ViT-B_16 `
  --vit_layers 0 --n_skip 3 --img_size 224 --subset SPARC --root_path E:\mkl\code\datasets\Carotid2D `
  --list_dir lists/lists_Carotid2D --head_hr 1 --max_epochs 1 --batch_size 2 --num_workers 0 --tag smokehr

# 正式（单 seed 1234，与基线同协议）
.\venv\Scripts\python.exe train.py --dataset Carotid --arch baseline --vit_name R50-ViT-B_16 `
  --vit_layers 0 --n_skip 3 --img_size 224 --subset SPARC --root_path E:\mkl\code\datasets\Carotid2D `
  --list_dir lists/lists_Carotid2D --head_hr 1 --max_epochs 200 --batch_size 4 --n_gpu 1 `
  --eval_test_every 5 --seed 1234
```

判定（不做多 seed）：按整测试集均值，**Dice ≥ +0.5 且 ASSD / HD 同向不变差**才算通过；Zhongnan 只做方向性验证。SPARC 单臂参考耗时约 45–60 min（同协议 R50 单臂的历史实测）。

## 7. 可行性评估（诚实版）

**支持做的证据**

- 该接点是结构上唯一「标签分辨率上没有任何图像证据」的位置，属于补管道，不是再叠一个先验；
- 低分样本的误差尺度（1 原始像素）与该级缺失的证据尺度严格对应；
- 成本极低（+5.3k 参数、+0.55 GFLOPs、恒等起步不破坏基线），失败也只损失一次单臂训练。

**压制预期的证据（必须先说清）**

- 修正上界本来就不高：即使每个切片都能找到最优均匀偏移，Dice 也只有 +0.87/+0.75（低分组 +1.71/+1.64）；本项目门槛是 +0.5，因此**过线空间很窄**，模块必须真的把边界贴准而不是只做偏移；
- 1 像素误差里有一部分是**标注歧义**：低分切片 CNR 0.21/0.33，边界带梯度仅为全图 1.9 倍，单标注者 GT 本身在 ±1 px 内就不唯一；这部分是天花板，任何模块都拿不到；
- 224² 级特征是 112² 双线性上采样的结果，存在混叠；把原图直接接进来也可能只是把散斑噪声带进决策，BatchNorm + 小通道是缓解而非消除；
- 与既有负结果的关系：输入级边缘图零增益、五算子横评判负、边界加权 CE +0.41 未过线、DDBE 增益被归因于频率内容而非结构、FGR +0.305 未过线。本方案的差别只在「接点」而不在「先验」，因此**它是这些失败方向之外的一条，但不是高把握的一条**。

**判断**：可以做，值得跑一个单臂；预期 SPARC Dice +0.2~0.8、ASSD / HD 同向小幅改善，**不保证过线**。若不过线，就按 §5 的结论收口——低分尾部中相当一部分是标注与成像极限，不再往「边界模块」方向追加投入。

## 8. 产物清单

| 路径 | 内容 |
|---|---|
| [assets/low-score/SPARC/](assets/low-score/SPARC) | SPARC 12 张，**每样本一张**：全图轮廓 / 局部放大轮廓 / 局部误差图，标题带指标 |
| [assets/low-score/Zhongnan/](assets/low-score/Zhongnan) | Zhongnan 12 张（同上） |
| [assets/low-score/shape_scatter.png](assets/low-score/shape_scatter.png) | Dice ~ 长宽比 / 短轴 / CNR 散点（两库） |
| [logs/analysis/low_score_tables.md](../logs/analysis/low_score_tables.md) | 本文 §4 表格的自动生成版 |
| [scripts/visualize_low_score_cases.py](../scripts/visualize_low_score_cases.py) | 生成上述图表的只读脚本 |
| [logs/analysis/error_slices_{SPARC,Zhongnan}.csv](../logs/analysis) | 逐切片特征与指标（§4/§5 的原始数据） |
| [logs/analysis/pred_geometry_*.csv](../logs/analysis) | 误差几何（漏/多检占比、深度、连通域） |
| [logs/analysis/oracle_offset_*.csv](../logs/analysis) | 逐切片最优偏移上界 |
