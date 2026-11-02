# 会议与论文调研：医学图像分割（颈动脉超声 2D）

> 目的：为「U-Net 编解码器改造 + 小数据 + 单卡 3060 + 无文本提示」找可借鉴的思路。
> 约束：以本项目数据为主（SPARC + Zhongnan），Ar-PlaqSegm1 作为外部数据集候选；不考虑 SAM 系大模型；本科生投稿。
> 收录标准：CCF 2022 目录 B 类及以上为主；个别 C 类/期刊/预印本因思路直接相关而单列并标注。
> 状态：2026-11-01 首版，条目按「思路」分组，不按会议分组。

## 收录优先级（2026-11-01 修订）

后续所有批次按此排序，不再按主题平铺：

1. **2025–2026 顶会**：CVPR、ICCV、ECCV、NeurIPS、ICLR、ICML、AAAI
2. **CCF B 会议**：MICCAI、IPMI、BIBM、ICME、ICASSP、ICMR
3. **期刊**：TMI、MedIA、UMB、CMPB、Pattern Recognition、IEEE TIP
4. **高价值 arXiv**：仅限 2024 年后、可直接落地、且尚无会议版本

**剔除规则**：2022 年前的 arXiv、无会议版本的老预印本、低质量预印本不收录；2021–2022 的经典工作只作基线对照保留。

**按此规则降级/可删的既有条目**（保留编号以防引用错位，但不再作主推）：

| # | 原条目 | 处理理由 |
|---|---|---|
| 9 | IAUNet（arXiv 2025） | 无会议版本，同方向已有 BIBM 2024 的门控线性注意力（#37）覆盖 |
| 11 | RWKV-UNet（arXiv 2025） | 训练开销大，3060 上不可行，不满足我们的硬约束 |
| 30 | UNet#（arXiv 2022） | 2022 年预印本，跳跃连接重设计已被后续会议工作覆盖 |
| 24 | DACL（arXiv 2026） | 价值待评估，暂不作为主线 |

> 说明：本页早期条目（#1–#8 等）之所以偏 2021–2022，是因为按「思路溯源」组织，先找提出者再找改进者。从第二批（§10）起已改为按年份+会议等级排序。

---

## 0. 先说三个检索结论

1. **颈动脉斑块超声分割在 CCF B+ 会议里非常稀薄**。检索下来绝大多数落在期刊（Ultrasound in Medicine & Biology、Medical Image Analysis、TMI、Computers in Biology and Medicine）和非 CCF 会议。会议线上真正集中的只有 MICCAI 的 LNCS 论文集，外加少量 BIBM / ICME。
2. **能发上 A 类会议的超声工作，卖点都不是「分割得更准」**，而是换了一个问题设定：半监督、域泛化、不确定性、拓扑约束。纯精度提升在 CCF B+ 会议上很难过审。
3. **U-Net 编解码器改造这条线在 CVPR 2024 之后已成体系**（EMCAD 是代表），单卡 3060 完全跑得动。这是最贴合我们资源条件的路线。

> 等级说明：MICCAI / BIBM / ICME / ICASSP / IPMI = CCF B；CVPR / AAAI / NeurIPS / ICCV = CCF A；WACV = CCF C；ISBI、MIDL 不在 CCF 目录。标注「LNCS·待确认」的条目，会议名需下载 PDF 后核对页眉。

---

## 1. 直接相关：颈动脉斑块超声分割

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 1 | MBFF-Net: Multi-Branch Feature Fusion Network for Carotid Plaque Segmentation in Ultrasound | LNCS（疑 MICCAI 2021） | 多分支特征融合，针对斑块形态差异大 | 与我们「双骨干」思路同类，可作为「别人怎么做多分支」的对照；注意它的融合发生在特征级而非决策级 |
| 2 | Boundary-Supplementary Network for Carotid Plaque Segmentation in Ultrasound Images | IEEE SPL 2025（期刊） | 显式补一条边界分支来补主分支丢失的边界 | 与我们失败的「边缘图输入」同一动机，重点看它为什么把边界放在**分支**而不是**输入通道** |
| 3 | DPSMUNet: dual-pooling self-attention module | J. Supercomputing 2024（期刊） | 双池化自注意力模块嵌入 U-Net | 轻量注意力插入方式，参数开销小，适合 3060 |
| 4 | HRU-Net: Transfer Learning for Carotid Artery Plaque Segmentation | Diagnostics 2022（期刊） | 迁移学习 + 残差 U-Net | 反驳/支撑「无预训练从头训」的选择，看它的数据量与增益 |
| 5 | A new network for carotid artery plaque segmentation in ultrasound images | BIC 2024（非 CCF） | 新网络结构 | 低价值，仅作同题对照 |
| 6 | PA-Net: parallel network for carotid vulnerable plaque segmentation | ICCVDM 2023（非 CCF） | 并行双网络 + 后处理 | 与我们的双骨干/后处理策略同构 |

**读法**：这一组不是用来抄的，是用来确认「这个题在会议上没人做透」——所以我们的机会在**问题设定**而不是结构本身。

---

## 2. 可落地的创新方法（不限结构改造）

> 口径：只要同时满足「我们数据集上能涨点 + 3060 能训 + 有可发表的创新点」，形式不限：
> - 结构：编码器 / 解码器 / 跳跃连接 / 上采样算子 / 注意力块
> - 训练：损失函数、半监督、自监督、课程学习、伪标签
> - 数据：增强、合成、频域变换、跨中心域泛化
> - 输出：不确定性建模、边界细化、拓扑约束、后处理
>
> 唯一硬约束是**能不能在我们现有数据和单卡 3060 上跑出涨点**，不限定必须动 U-Net 结构。

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 7 | **EMCAD: Efficient Multi-Scale Convolutional Attention Decoding** | **CVPR 2024 (A)** | 整个解码器换成多尺度卷积注意力块，参数更少、精度更高 | **首选精读**。证明「同层次替换一个部件」就能发 A 类，不必推翻整个 U-Net；直接对标我们 GobletNet 复现失败的原因（双编码器重、难训） |
| 8 | MIST: Medical Image Segmentation Transformer with Convolutional Attention Mixing (CAM) Decoder | WACV 2024 (C) | CNN-Transformer 混合编码 + 轻量 CAM 解码器 | 解码器轻量化写法，适合小数据 |
| 9 | IAUNet: Instance-Aware U-Net | arXiv 2025 | 保留完整 U 形，加轻量像素解码器 + 多尺度 Transformer 解码器 | 「完全改掉解码器」的另一种做法：query 式解码 |
| 10 | Rethinking Decoder Design: Depth-to-Space Restoration and Residual Linear Attention | arXiv 2025 | 用 depth-to-space 上采样替代转置卷积 + 残差线性注意力 | 只动上采样算子，改动小、易消融，适合本科生第一篇 |
| 11 | RWKV-UNet: Improving UNet with Long-Range Cooperation | arXiv 2025（待确认会议） | 用 RWKV 结构替代 Transformer 做长程建模 | 显存友好，3060 可跑；但工程量大，慎选 |
| 12 | LKM-UNet: Large Kernel Vision Mamba UNet | LNCS 2024（疑 MICCAI 2024） | 大核卷积 + Mamba 编码 | Mamba 在 3060 上训练慢，仅作了解 |
| 13 | Beyond Self-Attention: Deformable Large Kernel Attention | WACV 2024 (C) | 可变形大核注意力替代自注意力 | 与我们已经跑过的 `deform` 算子同源，可解释我们为什么没涨 |
| 14 | Half-UNet | Frontiers in Neuroinformatics 2022（期刊） | 砍掉整个解码器，用统一通道聚合 | 极端简化的对照组，成本极低，适合做消融基线 |

**读法**：7 → 10 → 9，先看「只改解码器能到什么程度」，再看「最省改动的版本」。

---

## 3. 弱边界 / 不确定性（对应「图像比较糊」）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 15 | **Uncertainty-aware Cascade Network for Ultrasound Image Segmentation with Ambiguous Boundary** | LNCS 2022（疑 MICCAI 2022） | 级联预测 + 显式建模边界模糊区域 | 与「糊」直接对应；**候选主线之一**，可解释我们边界指标 HD 上不去的现象 |
| 16 | BGRA-Net: Boundary-Guided and Region-Aware CNN | **BIBM 2021 (B)** | 边界分支引导区域分支 | 双分支但分工明确，比我们的「双骨干同构并联」更有理论依据 |
| 17 | MGPM: Multilevel perception boundary-guided network（乳腺超声） | Medical Physics（期刊） | 用池化的膨胀/腐蚀提边界，再用边界引导高低层融合 + 多级边界损失 | 边界引导融合 + 多级边界损失，实现成本低，可先在我们数据上试 |
| 18 | Boundary-aware convolutional attention network（肝脏超声） | Scientific Reports 2024（期刊） | 边界感知注意力 | 同 17，次级参考 |
| 19 | A Noise Robust Framework via Uncertainty Guidance for Medical Image Segmentation with Noisy Label | **ICME 2024 (B)** | 用不确定性筛噪声标签 | 若考虑「多标注者口径不一致」，这是最近的可引用会议工作 |

---

## 4. 半监督 / 小样本（小数据的现实解法）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 20 | **ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling** | **AAAI 2026 (A)** | 提示引导的伪标签生成，面向通用超声 | 最新 A 类，**必读**；虽用「prompt」但不是文本大模型，需确认是否可在无文本条件下改造 |
| 21 | Progressive Learning for Semi-Supervised Ultrasound Image Segmentation | **BIBM 2025 (B)** | 渐进式从易到难利用无标注数据 | 与我们数据规模匹配，工程成本中等 |
| 22 | High-Frequency Guided Feature Refinement for Semi-supervised Ultrasound Image Segmentation | LNCS 2026（疑 MICCAI 2026） | 高频特征引导细化，配合半监督 | **直接呼应我们的小波/高频线**，回答「为什么我们的高频没涨」 |
| 23 | Deep Semi-Supervised Ultrasound Segmentation: Shadow Aware Network with Boundary Refinement | TMI 2023（期刊） | 声影感知 + 边界细化 | 声影是超声特有伪影，我们的斑块图里也常见 |
| 24 | DACL: Dual Agreement Consistency Learning | arXiv 2026 | 轻量 CNN + Transformer 双网络一致性 | 双骨干终于有合理用法：做一致性而不是做融合 |

**读法**：22 对我们最有解释力，20 决定我能不能蹭上 A 类。

---

## 5. 频域 / 小波（我们试过、没涨）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 25 | Wavelet-Driven Spatial and Frequency Domain Representation Learning | LNCS 2026 | 空域 + 频域双表示学习 | 对照 XNet 的失败，看它多了什么监督信号 |
| 26 | FWMamba-UNet | LNCS 2026 | 频域增强的 Mamba UNet | 了解趋势即可 |
| 27 | Frequency-Mixed Single-Source Domain Generalization for Medical Image Segmentation | LNCS 2023（疑 MICCAI 2023） | 频域混合做单源域泛化 | **与我们的两中心场景最匹配**，可作为跨中心卖点的方法外壳 |

---

## 6. 跨中心 / 域泛化（我们有两个中心，天然优势）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 28 | **MI-SegNet: Mutual Information-Based US Segmentation for Unseen Domain Generalization** | LNCS 2023（疑 MICCAI 2023） | 互信息约束解耦域相关/域无关特征 | **候选主线之二**：SPARC→Zhongnan 正好是天然的两域实验 |
| 29 | SHAN: Shape Guided Network for Thyroid Nodule Ultrasound Cross-Domain Segmentation | LNCS 2024（疑 MICCAI 2024） | 形状先验引导跨域 | 斑块形状不固定，形状先验可能不适用，但思路可借 |
| 30 | Domain generalization for prostate segmentation in TRUS: A multi-center study | Medical Image Analysis 2022（期刊） | 多中心经直肠超声域泛化 | 超声域泛化最完整的一篇，实验设计值得抄 |

---

## 7. 拓扑 / 形状约束（备选，风险偏高）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 31 | Topology Preserving Compositionality for Robust Medical Image Segmentation | CVPRW 2023 | 拓扑保持约束 | 斑块可能多发、形态不规则，拓扑约束未必成立，优先级低 |
| 32 | Topology Optimization with Fast χ Euler Characteristic | TMI 2025（期刊） | 快速欧拉特征做拓扑优化 | 计算便宜，可作为小创新点 |

---

## 8. 建议的下载顺序（先 8 篇）

| 顺序 | 编号 | 理由 |
|---|---|---|
| 1 | 7 EMCAD (CVPR 2024) | 同层次整体替换解码器就中 A 类的样板，证明不必推翻整个 U-Net |
| 2 | 15 Uncertainty-aware Cascade (MICCAI 2022) | 直接对「糊」这个问题 |
| 3 | 22 HF-Guided Refinement (MICCAI 2026) | 解释我们高频/边缘线为什么没涨 |
| 4 | 28 MI-SegNet (MICCAI 2023) | 两中心域泛化的现成方法外壳 |
| 5 | 20 ProPL (AAAI 2026) | 摸 A 类的边，判断能不能蹭 |
| 6 | 2 Boundary-Supplementary Network (SPL 2025) | 同题、同动机，看别人怎么放边界 |
| 7 | 16 BGRA-Net (BIBM 2021) | 双分支分工的规范写法 |
| 8 | 10 Rethinking Decoder Design (2025) | 改动最小、消融最干净，适合先落地 |

**3060 可行性分级**：

- 直接能跑：7 EMCAD、10 Rethinking Decoder、14 Half-UNet、8 MIST、16 BGRA-Net、2 Boundary-Supplementary
- 要控 batch / 加梯度累积：9 IAUNet、11 RWKV-UNet
- 不建议：12 LKM-UNet（Mamba 训练慢），20 ProPL 需先确认其「prompt」是否依赖文本

---

## 9. 待确认

1. 编号 1/15/22/27/28/29 的 LNCS 条目，需下载 PDF 核对具体是 MICCAI 哪一届（影响引用与等级）。
2. Ar-PlaqSegm1 的 76% Dice 出处未核实，数据集页面上没有该数字；找到原论文后补进 [dataset.md](dataset.md)。
3. Ar-PlaqSegm1 的定位（第三中心外部测试 / 参与训练 / 暂不用）尚未确定。
4. 本页 §10 中标注「待核」的条目，需下载 PDF 核对页眉确认会议与年份。

---

## 10. 补充清单：低成本、易验证的方向（2026-11-01 第二批）

> 口径放宽到「能涨点 + 3060 能跑 + 有创新点」后补收。这一组的共同点是**不改主干网络也能做**，适合先快筛。

### 10.1 损失函数（改一行代码，最快验证）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 33 | **Centerline Boundary Dice Loss for Vascular Segmentation (cbDice)** | MICCAI 2024（待核） | 把中心线约束与边界 Dice 合成一个损失 | 颈动脉是管状结构，斑块沿管壁分布；**最值得先试**，且直接对着 HD/ASSD 这两个我们上不去的指标 |
| 34 | Contour-Weighted Loss for Class-Imbalanced Image Segmentation | ICIP 2024 (C) | 形态学腐蚀提轮廓 → 轮廓加权 CE + 可分离 Dice | 斑块面积远小于背景，现成的类别不平衡解法，改动极小 |
| 35 | Skea-Topo: Skeleton-Aware Loss for Topologically Accurate Boundary Segmentation | Pattern Recognition 2025（期刊） | 骨架加权 + 边界校正项，惩罚拓扑关键像素 | 与 33 同类但更重，作备选 |
| 36 | BKSeg-Net: Segmentation of Ovarian Tumors with Boundary Keypoints Loss | MAPR 2024（非 CCF） | 用边界关键点做损失约束 | 斑块边界不连续时比逐像素边界损失更稳 |

### 10.2 轻量与注意力（不改整体架构）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 37 | A Tiny Efficient U-Net with Gated Linear Attention | **BIBM 2024 (B)** | 门控线性注意力，参数量很小 | 与 3060 12G 最匹配的一档，可直接替换我们 U-Net 的瓶颈块 |
| 38 | MAGRes-UNet / GAUNet（多注意力门控残差 U-Net） | IEEE Access 2024 / 2023（期刊） | 注意力门控残差块堆叠 | 写法常规、风险低，适合做消融对照 |

### 10.3 训练策略：课程学习与半监督

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 39 | Curriculum Learning for Self-Iterative Semi-Supervised MIS | **BIBM 2024 (B)** | 由易到难自迭代生成伪标签 | 与 21 同源，会议等级够、成本中等 |
| 40 | UA-UNet: Uncertainty Aware Pseudo-Label Generation in Residual U-Net | EMBC 2025 | 残差 U-Net + 不确定性筛伪标签 | 超声域常用写法，实现简单 |

### 10.4 测试时增强 / 域偏移（对应我们两台设备、两个中心）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 41 | SaGTTA: Saliency Guided Test Time Augmentation Across Vendor Domain Shift | **ISBI 2023**（目录外） | 用显著性引导的 TTA 应对**厂商差异** | Ar-PlaqSegm1 正好是 Esaote + Vinno 两台设备，这条最贴合 |
| 42 | TAAL: Test-Time Augmentation for Active Learning | LNCS（疑 MICCAI 2022） | TTA 结果差异作为主动学习采样依据 | 若后续要挑图标注可用 |
| 43 | Test-time generative augmentation for medical image segmentation | Medical Image Analysis 2025（期刊） | 生成式 TTA | 计算重，3060 需谨慎 |

### 10.5 自监督预训练（我们目前没有任何可载权重）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 44 | Adaptive Batch-Fusion Self-Supervised Learning for Ultrasound Image Pretraining | 2025（待核） | 超声域自监督预训练 | 用 SPARC+Zhongnan 无标注数据自建预训练，绕开「无公开权重」的问题 |
| 45 | SOTS: Self-supervised Pretraining for Label-Efficient Ovarian Tumor Segmentation | 2026（待核） | 自监督 + 少标注 | 与 44 同路，作方法参考 |

### 10.6 同题补充（颈动脉，供写作时引用）

| # | 论文 | 出处 | 核心做法 |
|---|---|---|---|
| 46 | FRDD-Net: Carotid Plaque Segmentation Using Feature Remapping and Dense Decoding | 2022（期刊，待核） | 特征重映射 + 稠密解码 |
| 47 | Automated Lumen Segmentation in Carotid Artery Ultrasound Based on Adaptive Generated Shape Prior | 2024（待核） | 自适应形状先验 |
| 48 | CAIMTUSNet: Comparison of DL Models in Carotid Intima-Media Thickness Ultrasound | 2022（待核） | 多模型横向比较（写作时可借其实验组织方式） |
| 49 | A Polar Transformation Augmentation for Mammary Gland Segmentation in Ultrasound | 2024（待核） | 极坐标变换增强，适配弧形/管状结构 |

### 成本排序（先做哪个）

| 档 | 内容 | 说明 |
|---|---|---|
| 当天可验证 | 33 cbDice、34 轮廓加权损失 | 只改损失，复用现有训练脚本，直接看 Dice/HD |
| 一周内 | 37 门控线性注意力、41 SaGTTA | 换瓶颈块 / 加推理期增强，各自独立消融 |
| 需新流程 | 44 自监督预训练、39 课程学习半监督 | 要加预训练阶段或伪标签循环，工程量大但故事完整 |

---

## 11. 补充清单：按新优先级重检（2026-11-01 第三批）

> 口径：`from_publication_date:2025-01-01`，剔除 arXiv 与无出处条目，按「顶会 > B 会 > 期刊」排序。

### 11.1 顶会（2025–2026）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 50 | **Pathology-Centric Region Exchange: Joint Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound** | **ISBI 2026** | 斑块分割与易损性评估联合，病理中心区域交换 | **同题最新（2026）**，可直接对表；ISBI 不在 CCF 目录，但医学影像圈认可 |
| 51 | **DPGS-Net: Dual Prior-Guided Cross-Domain Adaptive Framework for Ultrasound Image Segmentation** | **MICCAI 2025 (B)** | 双先验引导 + 跨域自适应 | 对应我们 SPARC→Zhongnan 的域偏移，CCF B 且有代码风格的写法可循 |
| 52 | RDTE-UNet: A Boundary and Detail Aware UNet | ISBI 2026 | 边界 + 细节双路 UNet | 与我们边界/细节支线直接同名，可作对照 |
| 53 | ED-Mask2Former: Dynamic Hierarchical Query and Edge-Guided Enhancement | LNCS 2026（待核，疑 MICCAI） | 边缘引导的 query 式分割头 | query 式解码器路线，若走「换解码器」可参考 |

> 说明：CVPR 2025 的医学分割工作基本转向 SAM + 文本/偏好优化（如 Enhancing SAM with Efficient Prompting），与「无文本」约束冲突，本批不收录。

### 11.2 CCF B / 目录外但在范围内的会议

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 54 | CFPFormer: Cross Feature-Pyramid Transformer Decoder | IJCNN 2025 (C) | 跨层特征金字塔 Transformer 解码器 | 解码器改造的 B/C 档次模板，改造成本中等 |

### 11.3 期刊（2025–2026，按相关性排序）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 55 | **Boundary-Aware Attention with Dual-Stream Frequency Fusion for Medical Image Segmentation** | **JBHI 2026** | 边界注意力 + 双流频域融合 | **同时命中我们的边界支线与频域支线**，最贴合的一条，优先读 |
| 56 | **Frequency-Enhanced Wavelet Transformer Based Decoder** | **Pattern Recognition 2026** | 小波增强的 Transformer 解码器 | 站在我们已试过的小波线上，给出「为什么我们没涨」的另一种解释 |
| 57 | **Contourlet-Informed Prior Controllable Adaptation of Ultrasound Foundation Model** | **Pattern Recognition 2026** | Contourlet 频域先验 + 超声基础模型适配 | 频域先验用于超声域适配，且不改主干，3060 可行 |
| 58 | DPGNet: Boundary-Aware Framework via Uncertainty Perception | JBHI 2025 | 不确定性感知的边界建模 | 与 #15 同一思路的新版本，可替换旧引用 |
| 59 | Boundary-Guided Contrastive Learning for Semi-Supervised MIS | TMI 2025 | 边界引导对比学习做半监督 | 边界 + 半监督组合，两个低成本模块叠加 |
| 60 | Segment Together: A Versatile Paradigm for Semi-Supervised MIS | TMI 2025 | 统一半监督范式 | 若走半监督主线，这是最新基线 |
| 61 | PICK: Predict and Mask for Semi-Supervised MIS | IJCV 2025 | 预测-掩码自训练 | 实现简单，适合先复现 |
| 62 | Discriminative Consistency Semi-Supervised Carotid Ultrasound Plaque Segmentation by Exploiting Global Context | IJIST 2025 | 全局上下文 + 判别一致性半监督 | **同题同模态的半监督**，直接对标 |
| 63 | Challenge-aware U-Net for Breast Lesion Segmentation in Ultrasound | Pattern Recognition 2025 | 针对超声难点自适应加权 | 思路是「按样本难度调权」，与课程学习可结合 |
| 64 | TinyUSFM: Towards Compact and Efficient Ultrasound Foundation Models | JBHI 2026 | 轻量超声基础模型 | 唯一 3060 可训的超声基础模型路线 |
| 65 | Automatic Multi-Task Segmentation and Vulnerability Assessment of Carotid Plaque on CEUS | JBHI 2025 | 分割 + 易损性多任务 | 若要与 #50 组合成「分割+评估」故事 |

### 本批结论

1. **CVPR/ICCV/ECCV 的 25–26 医学分割几乎都在做 SAM/文本提示**，与「无文本」约束正面冲突。因此 A 类线暂时不追，主攻 MICCAI 2025/2026 + 期刊。
2. **最值得先读的三篇：55（JBHI 2026 频域+边界）、56（PR 2026 小波解码器）、51（MICCAI 2025 跨域）**。前两篇正好解释我们小波/边界线为什么没涨，第三篇给两中心问题一个 CCF B 的方法外壳。
3. **同题最新是 #50（ISBI 2026）和 #62（IJIST 2025）**，写作时的「相关工作」和「对比方法」应优先引这两篇。
