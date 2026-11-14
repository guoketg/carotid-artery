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

---

## 12. 补充清单：特征融合方法专批（2026-11-05 第四批）

> 口径：只收「怎么融合」的方法——融合位置、融合算子、频域/小波分解、跨模态缝合；2024 年后为主。
> 证据级别：`【全文】`=已读正文；`【摘要】`=仅摘要；`【检索】`=仅元数据。标 `待核` 的**不得直接写进参考文献**。
> 检索说明：OpenAlex 本机返回 503、WebFetch 返回空，本批 venue 由 ACM DL / OpenReview / PubMed / arXiv 页眉核对。

### 12.1 顶会（2024–2026）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 66 | **FreqFusion** | **NeurIPS 2024** | 【全文】在「深层上采样 + 浅层」融合点做三件事：**ALPF** 为每个输出坐标预测空变低通核，压掉类内高频、降低类内不一致；**offset generator** 先算局部相似度，再向高相似度邻域重采样，替换掉不一致像素；**AHPF** 对浅层特征预测高通核，补回 downsampling 丢掉的 >Nyquist 边界细节 | 开源、纯算子、不动骨干；SegFormer-B1 +2.8 mIoU、SegNeXt-T +2.0。**我们的 112²/56² 上采样接点可直接把 concat 换成它**，且能严格同配置对照 |
| 67 | **DB-SAM** | **MICCAI 2024 Oral** | 【摘要】冻结 ViT 分支 + 轻量 CNN 分支并列；ViT 每个 attention block 后挂可学习通道注意力，两路逐级交互，而非末端拼接 | 双分支不是「各跑一个编码器再融合」；它每级都交互，可解释我们双骨干末端融合为何收益小 |
| 68 | **StitchFusion** | **ACM MM 2025 Oral** | 【摘要】放弃末端融合；用 **MoA（multi-directional modality adapter）** 在编码器内部把其他模态特征 stitch 进来，**只训一个骨干**，让编码器自己完成融合 | 直接回答「能否省掉 121M 双骨干」；同图派生视图（原图 + HF）是否适用需核 |
| 69 | **FMISeg** | **MICCAI 2025** | 【摘要】频域多模态**晚融合**：FFBI 做频域特征双向交互；LFFI 用语言引导抑制语义无关的视觉频域特征 | 结构顺序值得借：先把频域特征双向融合，再让语义信号做门控；无文本约束下只用 FFBI |
| 70 | **Omni-Fuse** | **MICCAI 2025** | 【摘要】跨维**双向**注意力增强空间/谱特征；谱引导的空间 query 选择；两阶段跨维解码器 | 双向注意力不是对称 concat；「先选 query 再解码」可借到主路 query—侧路 key |

### 12.2 CCF B / 超声专向

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 71 | **FABRF-Net** | **Information Fusion 2025**（期刊，DOI 10.1016/j.inffus.2025.103299） | 【摘要】HWDM：Haar 小波分解出多尺度频率特征；再用「边界感知 + 区域感知」双路融合，CNN 与 Transformer 互补，解类内不一致 | 与老师指定的 BDFR-Net **DDBE 同族**（DWT 分带→分处理→融合），可作为已发表支撑；注意它的任务是乳腺超声 |
| 72 | **FSS-Net** | arXiv 2606.10378（2026-06，无 venue） | 【全文】CSWA 通道-空间-小波注意力抑噪；WEB 小波增强瓶颈；**LAEF：Laplacian 引导的自适应边缘融合**，补高频并保边界连续 | **同题（颈动脉）**。精读后：U-Net 96.08 → 全模型 **96.46**，**净增仅 +0.38**，不是「96.46 很 SOTA」；1100 张来自 11 人、随机 5 折，**受试者级泄漏风险高**，且无外部测试。详见 §12.8 |
| 73 | **TGMS-UNet** | **Digital Health 2026**（SAGE，DOI 10.1177/20552076261467802） | 【摘要】双分支：轮廓转序列 + 全局注意力/扫描做跨模态对齐；特征校正与自适应融合用**门控**纠多尺度错位 | 「门控用来纠错位」而不是简单加权，对应我们 skip 的语义错位 |
| 74 | **DAF-UNet** | IEEE 会议（**出处待核**） | 【摘要】DDAM 在编码器并联空间/通道注意力提纯；FFB 在解码器自适应融合、桥接语义差 | 位置分工明确：编码器提纯、解码器融合；可作「融合位置」对照 |
| 75 | **Dual-Branch Hybrid Attention Network** | IEEE 会议（**出处待核**） | 【摘要】多尺度空洞卷积+通道注意力 与 轴向自注意力 双支；**通道-空间联合门控**动态融合 | 门控是通道+空间联合，不是单 sigmoid；可对照我们的 gate 算子 |

### 12.3 预印本（细节值得读，引用前必须核）

| # | 论文 | 出处 | 核心做法 | 可迁移点 |
|---|---|---|---|---|
| 76 | **FreqDINO** | arXiv 2512.11335（2025-12） | 【全文】Haar 两级分解：`F_Hf=φ_H([LH;HL;HH])`、`F_L=φ_L(LL)`；边界注意力 `A_b` 与结构注意力 `A_s` 固定 α=β=0.5，`F_enh=F+λ(F⊙(αA_b+βA_s))`，λ=0.3；**FGBR** 把高频压成 64 维边界原型，用 `F_enh` 作 Q、原型作 K/V（8 头 d=128，残差 ω=0.2）；MBGD 边界头先出、再拼回 mask 头 | **我们 QKV 负结果的正解**：K/V 不是全图侧路，而是压缩后的边界原型。既解释「全图跨支注意力涨幅与 HF 无关」，又是可写成 novelty 的改法 |
| 77 | **WDFFU-Mamba** | arXiv 2512.17278（2025-12） | 【摘要】WHF 用小波去噪后的高频引导低层特征；DAFF 双注意力融合 skip 与语义特征 | 与我们的「HF 输入」和「skip 融合」两条线都重合，literature.md 已有其消融数字 |
| 78 | **DCAU-Net** | arXiv 2603.09530（2026-03） | 【摘要】DCA 用**窗口级 summary token** 替代逐像素 K/V 来降复杂度；CSFF 在 skip/上采样融合点用通道-空间串行注意力重标定 | 直接解决「跨支注意力太贵、小数据易过拟合」；CSFF 是「作用在融合点的 CBAM」 |
| 79 | **HyM-UNet** | arXiv 2511.17988（2025-11） | 【摘要】MGF-Skip：用语义强的解码特征 `D_i` 作**门控信号**过滤噪声编码特征 `E_i`（先上采样对齐再算门） | 方向与「编码门控解码」相反，可作对照；门控比注意力便宜，3060 友好 |
| 80 | **CCFNet** | 出处**待核** | 【摘要】CNN↔Transformer **持续互融**：CNN→Transformer 用协同自注意力对齐 local/global token；Transformer→CNN 用空间特征注入器 | 双向融合要分方向设计，不能同一个模块跑两遍 |
| 81 | **MD-RWKV-UNet** | arXiv 2603.27261（2026-03） | 【摘要】选择性核注意力自适应感受野 + **跨阶段双注意力融合**聚合多级编码特征 | 跨阶段融合兼顾低层结构，对应我们的 skip 多级聚合 |

### 12.4 别人怎么做的：算子速查（写作时可直接改写进 Related Work）

| 算子 | 数学形式 / 机制 | 代表 | 什么时候有效 / 坑 |
|---|---|---|---|
| Concat + 1×1 | `F=φ_{1×1}([F1;F2])` | 我们已实现的 `BidirConcat`、BDFR-Net DDBE | 最便宜；1×1 只做通道重配，**不解决空间错位**；强基线上常只有 +0.005 |
| 加权和 | `F=Σ w_i F_i`，w 经 fast-normalized | BiFPN、跨尺度聚合 | 标量权重无法逐像素；旧作，只作基线 |
| 通道注意力定标 | `F=F1⊙σ(MLP(GAP(F1+F2)))+F2⊙(1-σ(·))` | SE / CBAM | GAP 丢空间信息，**边界无感**；literature.md 已记录强基线上零增益 |
| 多尺度通道注意力 | MS-CAM = 全局 GAP 分支 + 局部 point-wise conv 分支相加；iAFF 迭代两次 | AFF / iAFF（WACV 2021） | 针对「初始融合即瓶颈」；参数小，但增益易被容量混淆 |
| 空间/通道门控 | `F=F1⊙σ(Conv_{1×1}([F1;F2]))` | TGMS-UNet、DAF-UNet、双支混合注意 | 便宜；门控图可能退化成常量，需可视化验证 |
| 交叉注意力 | `F=F1+ω·W_O(Attn(Q=F1,K=F2,V=F2))` | DCAU-Net（窗口 token）、FreqDINO（原型 K/V） | 全图 K/V 在小数据上过拟合、增益不可归因（**我们已实测**）；压缩 K/V 才稳 |
| 可变形 / 偏移重采样 | 学 offset，把低相似度像素换成高相似度邻域 | FreqFusion offset、D-LKA 式可变形大核 | 训练不稳，需可视化/正则；对细边界与大片不一致同时有效 |
| 频域滤波融合 | ALPF 预测空变低通核、AHPF 预测高通核 | FreqFusion | **与我们最贴**：融合点上直接做频率分工，不依赖固定小波基 |
| 小波分带 | DWT→LL/LH/HL/HH→分处理→IDWT 或 concat+1×1 | FABRF-Net、FSS-Net、FreqDINO、XNet、GobletNet、Syn-Net | 高频单独无语义（XNet HF only 54.5）；固定小波基/级数差异 <1%（GobletNet） |
| 边界原型 | 高频特征压成 64 维 prototype，作 K/V | FreqDINO FGBR | 容量小，可能丢多斑块；需按病变数评估 |
| 编码器内缝合 | MoA adapter 在编码器各级注入另一路特征，单骨干 | StitchFusion | 需大预训练骨干；小数据从头训风险高 |
| 边界多任务融合 | 边界头先预测，再转成边界特征拼回 mask 头 | FreqDINO MBGD | 与边界监督损失正交，可作为 decoder 的轻改法 |

### 12.5 本批结论：先做哪三条

1. **FreqFusion（#66）替换解码器融合算子**。全文已读、有开源、不动骨干，能和现有 concat 严格同配置对照；最像「方法贡献」且风险最低。落点就是我们 112²/56² 的上采样接点。
2. **FSS-Net 的 LAEF（#72）与我们 edge 队列对照**。同题同动机（Laplacian 高频补边界），但它在特征级、我们在输入级，两者天然构成「融合位置」消融。
3. **FreqDINO 的边界原型 K/V（#76）改造我们的 QKV**。我们已证明全图跨支注意力的涨幅与 HF 内容无关；把 K/V 换成 64 维边界原型，正好把「重组噪声」换成「边界先验注入」，可作为论文的机制解释 + 新贡献。
   - 备选：**StitchFusion（#68）** 省双骨干；**FABRF-Net（#71）** 给老师的 DDBE 找已发表支撑。

### 12.6 写「特征融合」论文必须补的三类证据

1. **参数 / FLOPs 匹配**：双骨干 vs 单骨干+侧路，否则增益会被归给容量（literature.md 已多次记录该坑）。
2. **3 seeds 报告分布**：现有单 seed 结论只能写「未观察到增益」，不能写「更差/更优」。
3. **机制指标**：借 FreqFusion 的 **IntraSim / similarity margin**，报融合前后类内一致性与边界位移；比只报 Dice 更能过审。
4. **零输入 / 随机侧路对照**：我们已经做过（侧路全零反而更高），这条写进正文会显著增强说服力。

### 12.7 待下载 / 待核清单

| 优先级 | 文献 | 获取方式 | 拿到后先核 |
|---|---|---|---|
| 高 | FreqFusion | arXiv 2408.12879 + GitHub `Linwei-Chen/FreqFusion` | 融合点实现、参数量/耗时、医学数据有无结果 |
| 高 | FSS-Net | arXiv 2606.10378 PDF | 数据划分与 Dice 96.46 口径；LAEF 公式 |
| 高 | FreqDINO | arXiv 2512.11335 PDF | 原型维度、K/V 构造、边界头损失权重 |
| 中高 | FABRF-Net | DOI 10.1016/j.inffus.2025.103299 | HWDM 与 DDBE 的差异、边界/区域双路公式 |
| 中 | StitchFusion | ACM DL 10.1145/3746027.3755110 | MoA 是否支持同图派生视图；单骨干参数量 |
| 中 | FMISeg | 10.1007/978-3-032-05114-1_27 | FFBI 在无文本时是否仍成立 |
| 中 | DB-SAM | arXiv 2410.04172 | 两路交互是门控还是注意力；参数匹配 |
| 中 | DCAU-Net | arXiv 2603.09530 | 窗口 token 的窗口大小与开销 |
| 低 | TGMS-UNet | PubMed 42436892 | 期刊名、门控公式 |
| — | DAF-UNet / Dual-Branch Hybrid Attention / CCFNet | 待核 | venue 与年份，核不到就不引 |

---

## 13. 精读回填：11 篇的融合位置 / 公式 / 增量 / 消融（2026-11-05）

> 口径：以 `pdf/` 内 11 个 PDF 的**表格数字**为准，覆盖 §12.1–12.3 的摘要级描述。
> 关键提醒：**「论文宣称」与「PDF 表格可核对值」多处不一致**，下表只抄表格，引用前按表号复核。
> 已读全文：FreqFusion、FreqDINO、FSS-Net、StitchFusion、WDFFU-Mamba、FMISeg、DB-SAM、DCAU-Net、HyM-UNet、MD-RWKV-UNet、TGMS-UNet。

### 13.1 逐篇精读表

| # | 论文（文件） | 出处核实 | 融合位置 | 核心算子 | 模块/整模增量 | 消融净增益（PDF 表格） | 可信度 / 风险 |
|---|---|---|---|---|---|---|---|
| 66 | FreqFusion（`NeurIPS2024_FreqFusion.pdf`） | NeurIPS 2024（OpenReview 确认）；**下载 PDF 实为 arXiv 2408.12879v1 的 IEEE 排版版，无 NeurIPS 页眉** | 解码器/FPN 每个上采样融合点 | ALPF 空变低通核 + offset 高相似度重采样 + AHPF 空变高通核 | SegFormer-B1 **+0.34M / +2.35G**；FPS 26.5→23.0（**−13%**） | ADE20K mIoU 41.1→42.0(+ALPF)→42.9(+AHPF)→43.5(+Offset)；强基线 Mask2Former-Swin-B **+1.4**、Swin-L **+0.7** | 结构最可信、有开源；但只验证自然图像；**强基线上增益 <1 mIoU**，不能按 +2.4 预期 |
| 67 | DB-SAM（`MICCAI2024_DB-SAM.pdf`） | MICCAI 2024（ACM DL 确认）；PDF 为 arXiv 2410.04172 版 | 编码器每层 ViT block 后 + 末端融合 | `F=F_vit+Conv1x1(SE(DWConv3x3(LN(F_vit))))`；双臂可变形 cross-attn + 逐位置软选择 `M=σ(Λ_d+Λ_s)`，`F=F_od⊙M+F_s⊙(1−M)` | **未报告** | 3D 81.02→85.25(+CA)→86.60(+双边)→**87.05(+末端融合)**；2D 77.22→82.00；**末端融合仅 +0.45 / +2.45** | 增益主要来自新卷积分支与通道注意力；需 bbox prompt、16 万训练图，小数据不可比 |
| 68 | StitchFusion（`ACMMM2025_StitchFusion.pdf`） | **ACM MM 2025**（页眉 + DOI 10.1145/3746027.3755110 确认） | 编码器 4 个 stage 的每个 block（MHSA 后、FFN 后各一次） | 低秩 MLP adapter：`x + W_up·GELU(W_mid·W_down·x)`，再把另一模态特征加回 | MiT-B2 **+0.86M / +4.06G**；r=8 时 MoA 占 **0.89%** 参数 | mIoU：MCUBES-RGBN 47.97→49.72、PST900 82.09→83.43；再加 FFM 51.70→53.92 | 多模态需每模态一个预训练编码器，GFLOPs 随模态数线性涨；**单模态超声不能直接套** |
| 69 | FMISeg（`MICCAI2025_FMISeg.pdf`） | MICCAI 2025（ACM DL 确认）；PDF 为 arXiv 2509.19719 版，无会议页眉 | 瓶颈 FFBI + 解码器逐级 LFFI（晚融合，双分支各自解码头） | 双向 MHCA：`F̂_HF=LN(F_HF+MHCA(F_HF,F_LF,F_LF))`；LFFI 用文本特征滤波频域特征 | **未报告** | QaTa Dice：concat 90.61 → **FFBI 91.21（+0.60）**；无文本 87.63→91.21 | 相对 concat 仅 +0.60；大增益来自双频输入/双分支容量；**需文本报告**，与无文本设定冲突 |
| 72 | FSS-Net（`arXiv2026_FSS-Net.pdf`） | arXiv 2606.10378，**无 venue** | 编码器 CSWA + 瓶颈 WEB + 解码器 skip LAEF | db4 DWT/IWT；`F_out=F_enc·b+F_edge·(1−b)`（b 为 SK 式通道门控） | 整模 **5.01M**；WEB 0.53M/134.30M MACs（ASPP 2.23M/554.04M） | 颈动脉 DSC：U-Net 96.08 → **全模型 96.46（净增 +0.38）**；去 CSWA 96.28、LAEF 换普通 skip 96.38、去 WEB 96.39 | **泄漏风险最高**：1100 张/11 人、随机 5 折；无外部测试；BUSI 迁移仅 75.86；**同题唯一一篇，但不可当 SOTA 论据** |
| 73 | TGMS-UNet（`DigitalHealth2026_TGMS-UNet.pdf`） | **Digital Health 2026**（首页卷期 + DOI 确认） | 编码器各级 TIFM + 瓶颈 FCAM + skip 侧 GLFM | 轮廓序列 cross-attn；门控校正 `F=β⊙F̂_m+(1−β)⊙F̂_o` 后接 CBAM | 整模 **62.2M / 83.6G / 45.2ms**；相对 U-Net 约 **+31.2M / +33.3G** | baseline 0.8624→全量 **0.9184 Dice**；无序列 0.8848→有序列 0.9201（**+3.53 Dice**） | **推理需 GT 轮廓序列 = 输入即标签**；同一模型出现 3 个 Dice 值；划分表述矛盾（泄漏） |
| 76 | FreqDINO（`arXiv2025_FreqDINO.pdf`） | arXiv 2512.11335，**无 venue** | 编码器输出频域增强 + 解码器双头（非跨层 skip） | Haar 两级分解；`F_enh=F+λ(F⊙(αA_b+βA_s))`，λ=0.3、α=β=0.5；64 维边界原型作 K/V（8 头，ω=0.2） | **未报告**（冻结 DINOv3-Large） | BUSI Dice 82.35→84.17(+MFEA)→85.13(+FGBR)→**86.52**；HD 47.59→44.59→43.02→**39.63** | 正文称 MFEA +2.21，表为 **+1.82**；依赖大预训练骨干；基线已 82.35，增益小 |
| 77 | WDFFU-Mamba（`arXiv2025_WDFFU-Mamba.pdf`） | arXiv 2512.17278，**无 venue** | 编码器 WHF + 瓶颈 DAFF | 小波去噪后高频引导；DAFF 通道+空间双注意力 | 整模 5.65M；CSAM 41.7K/116.2M vs CBAM 111.1K/115.8M | BUSI：81.42→+WHF 83.10→+DAFF 82.41→全量 **84.48**；HD95 26.78→18.93→16.08→13.28 | 正文称 WHF +3.5%，表为 **+1.68%**；BUS 上 HD95 **反而 +2.65**；正文表格多处不一致 |
| 78 | DCAU-Net（`arXiv2026_DCAU-Net.pdf`） | arXiv 2603.09530，**无 venue** | 解码器每个上采样点的 skip（CSFF） | DCA 用 7×7 窗口摘要 token 替代逐像素 K/V；CSFF 通道-空间串行重标定 | 整模 **21.56M / 4.67G**（增量未报告） | Synapse 79.76→**81.25（+1.49 DSC，HD −2.87）**；删 SA 79.63、删 CA 80.83 | 消融为随机初始化，主表 83.29 是 ImageNet 预训练（预训练本身 +2.04）；HD 对删 CA 反而变差 |
| 79 | HyM-UNet（`arXiv2025_HyM-UNet.pdf`） | arXiv 2511.17988，**无真实 venue**（页眉是 LaTeX 模板占位） | skip（每层） | `G=σ(Conv1x1(ReLU(Conv3x3(Up(D_i))))`；`E_f=E⊙G`；`F=Concat([E+E_f],Up(D))` | **未报告** | ISIC2018：U-Net 79.32→Ours **81.82**（+2.50） | **核心 MGF-Skip 无消融**；表内有重复行、正文与表冲突 → 只借结构，不引数字 |
| 81 | MD-RWKV-UNet（`arXiv2026_MD-RWKV-UNet.pdf`） | arXiv 2603.27261，**无 venue** | 编码器跨 stage（非 skip） | `F=(α_l·s)⊙F_l+(α_h·(1−s))⊙F_h`，通道权重×空间掩码互补 | **未报告** | Synapse 84.02→仅 CSF 84.56→全量 **85.07**（CSF **+0.54**）；ACDC 92.17→93.15 | 基线表与 DCAU-Net Table I **完全一致**，疑复用他表；用 TTA 而基线未说明；肝 HD95 变差 |
| 71 | FABRF-Net | Information Fusion 2025（**未下载**） | 摘要级：HWDM 小波分解 + 边界/区域双路 | — | — | 未核 | 需订阅；暂只能作动机引用 |
| 70 | Omni-Fuse | MICCAI 2025（**未下载**） | 摘要级：跨维双向注意力 + 谱引导 query | — | — | 未核 | 高光谱任务，参考价值有限 |

### 13.2 精读后的硬结论

1. **11 篇没有一篇做决策级融合**。位置集中在三类：解码器上采样/skip（FreqFusion、DCAU-Net、FSS-Net、HyM、TGMS）、编码器内（DB-SAM、StitchFusion、MD-RWKV）、瓶颈（FMISeg、WDFFU）。这给 Related Work 一个干净的分类轴：**融合位置 × 融合算子 × 权重来源**。
2. **融合算子本身的净增益普遍只有 0.4–1.5 个点**：FMISeg +0.60 Dice、DB-SAM 末端 +0.45/+2.45、DCAU CSFF +1.49、MD-RWKV CSF +0.54、FSS-Net 相对 U-Net +0.38。**大增益几乎都来自加分支、加输入、加预训练或加注意力**。所以我们必须做「同参数量 concat」对照，否则审稿人会把增益归给容量。
3. **主流的融合写法高度同构**：`F=β⊙F_a+(1−β)⊙F_b`，区别只在 β 从哪来——通道 SE（DB-SAM）、空间卷积（HyM/MD-RWKV）、cross-attn（FMISeg/TGMS）、边界原型（FreqDINO）。这可以直接支撑我们「统一门控 baseline + 换权重来源」的实验设计。
4. **成本证据严重缺失**：只有 DCAU-Net（21.56M/4.67G）和 TGMS（62.2M/83.6G）报整模，且都没给融合模块增量；FreqFusion 给了 +0.34M/+2.35G 但 FPS 掉 13%；StitchFusion 三模态 GFLOPs 从 102 涨到 184/242G。**「lightweight / parameter-efficient」不能当事实引用。**
5. **可信度分级**：A 级（结构可信、数字自洽）FreqFusion、StitchFusion、DB-SAM、FMISeg、FSS-Net；B 级（口径有问题但结构可借）DCAU-Net、MD-RWKV、TGMS；C 级（正文与表格冲突/核心模块无消融）HyM-UNet、WDFFU-Mamba。
6. **颈动脉证据只有 FSS-Net 一篇**，净增 +0.38 且有受试者泄漏风险。因此论文里不能写「频域融合已在颈动脉被验证」，只能写「已有工作提示该方向」，并把「按受试者划分 + 跨中心验证」写成我们的差异点与贡献。

### 13.3 对现有三条路线的影响

| 原建议 | 精读后修正 |
|---|---|
| FreqFusion 替换解码器融合算子（§12.5-1） | **仍第一优先**。但预期增益按强基线 **+0.7~1.0 mIoU** 量级写，不要按 +2.4；参数 +0.34M 可接受，**FPS 会掉约 13%**，需在 3060 上实测吞吐 |
| FSS-Net 的 LAEF 做同题对照（§12.5-2） | **可对照，不可引用其 96.46 作为 SOTA**。它是 U-Net 96.08→96.46 的 +0.38，且 1100 张/11 人随机 5 折。反过来这是我们机会：用受试者划分 + 中南外部测试重做一遍 |
| FreqDINO 边界原型 K/V 改造 QKV（§12.5-3） | **机制可借，数字用表**：MFEA +1.82 Dice / HD −3.00；全模型 +4.17 Dice / HD −7.96。正文宣称值不可引；且它依赖 DINOv3-Large，我们只能借「压缩 K/V」的思想，不能借规模 |
| （新增）StitchFusion 省双骨干 | 若第二路只是同图派生视图，MoA 思路可行；但它需要每模态一个预训练编码器，**从头训的小数据场景增益未知**，先做参数匹配的可行性验证 |
| （新增）统一门控 baseline | 把 `β⊙F_a+(1−β)⊙F_b` 作为主对照，分别用 SE、空间卷积、cross-attn、边界原型生成 β，一次性回答「权重来源比算子更重要吗」——这比再堆模块更像论文贡献 |

### 13.4 引用时必须带的口径说明

- FreqFusion：NeurIPS 2024 是会议出处；本地 PDF 是 arXiv v1 的 IEEE 排版版，页码/表号与会议版可能不同。
- FMISeg / DB-SAM：会议出处已由 ACM DL 确认，但本地 PDF 是 arXiv 版，引用页码需用会议版。
- FreqDINO / FSS-Net / WDFFU-Mamba / DCAU-Net / HyM-UNet / MD-RWKV：**均为 arXiv 预印本，无正式 venue**，只能写 arXiv 编号与日期。
- TGMS-UNet：Digital Health 2026 已确认，可正式引用。
- 所有 HD 与 HD95 不可跨表比较；Dice 的小数位统一后再比较。

---

## 14. 补充清单：距离指标（ASSD / HD）专批（2026-11-14）

> 筛选口径：只收「能直接改 ASSD / HD 尾部」的工作，2024 年后；优先超声 / 边界监督 / Hausdorff 损失 / SDF 四条线；结构类（换解码器、加注意力）本批不收，已在 §12–§13 覆盖。
> 检索方式：WebSearch 为主；OpenAlex 本次多次返回空正文，只作为交叉核对，未作为唯一来源。
> 编号接 §13 的 81，起 82。带「无 venue」的只能写 arXiv 编号与日期，不得写成会议论文。

| # | 论文 | 出处（年份+会议） | 核心做法 | 可迁移点（对应本项目哪条线） |
|---|---|---|---|---|
| 82 | **Regional Hausdorff Distance Losses** | MLMI 2025（MICCAI workshop，Springer LNCS，DOI 10.1007/978-3-032-09513-8_7） | 可微软腐蚀 `(Y⊖_s k)_n=∏_{j∈N^k(n)}Y_j` 造距离剖面 `D+(G)=Σ_i[(1−G)⊖^i k]`；最大值项 `L_H=smax(smax(D+(G)∘y), smax(d+(y)∘G))`；平均项 `L_AH_sym=0.5(Σ D+(G)∘y/|Y| + Σ d+(y)∘G/|G|)`；复杂度 `O(I·b·c·V)`，I 取 2–5 | **第一优先**。同时压漏检侧与多检侧，正对我们「HD 由 miss 主导（108/160）、只压 far-FP 覆盖不到 2/3」的诊断；无新参数、2D 直接可用、只加一项损失 → 天然单变量 |
| 83 | **Entropy-Guided Boundary Supervision** | arXiv 2606.22308（无 venue） | 用预测熵判断边界可分辨性，只在高可信边界带加监督，模糊带降权/不激励 | 解释并修正我们「全边界等权加权无效」：不是边界监督没用，而是**没有按可分辨性门控**。对应 `--loss bndsoft`（对低可分辨带降监督）的同族写法 |
| 84 | Risk-Routed Implicit Boundary Refinement | arXiv 2607.21787（无 venue） | 按风险把像素路由进隐式边界场（SDF 式）细化，超声 9 个数据集报最低 HD95 | 同模态、直接报距离指标，证据最贴题；但需额外隐式头 + 两阶段训练，工程量明显大于 82，作第二梯队 |
| 85 | FocusSDF | arXiv 2511.11864（无 venue） | 以 SDF 为主监督，把学习容量集中到边界邻域 | SDF 回归头路线的代表，对应 ASSD 侧（面到面平均距离）；需改输出头 + 损失，属「一条新臂」而非改一行 |
| 86 | Boundary-Aware Instance Segmentation in Microscopy | arXiv 2603.21206（无 venue） | SDF 边界 + modified Hausdorff distance（MHD）损失联合监督 | 与 85 同类；其中 MHD 项可作为 `L_AH` 的**轻量替代**（不需要软腐蚀迭代），先算代价低 |
| 87 | Geometry-Guided Sampling Operator | arXiv 2608.25819（无 venue） | 几何先验引导可变形采样，把采样点推向边界而不是全图自由偏移 | BraTS HD95 7.1→6.2。与我们已经跑过的 `deform` 同源，可解释「为什么我们的可变形大核没涨」：缺的是「往边界走」的几何约束 |
| 88 | Learning to Upscale 3D Segmentations | MICCAI 2025（待核细节） | 学习式掩码上采样，替代最近邻/三线性回原分辨率 | 直接对应评测里 `ndi.zoom(order=0)` 把 224² 放回 ≈0.065 mm 原网格这一步；ASSD/HD 对量化台阶极敏感，低分辨率上采样的台阶会直接变成距离误差。**属评测侧改动，须固定协议并单独声明**，不进主推 |
| 89 | PatchGAN-guided semi-supervised U-Net for carotid plaque segmentation | 期刊 2026（PubMed 42464134；刊名待核） | PatchGAN 判别器 + 半监督，同题颈动脉斑块超声 | 同题最新期刊工作，补 Related Work；判别器路线可行性参考，不改变距离指标机制 |
| 90 | DAUNet | arXiv 2512.07051（无 venue） | 双注意力 U-Net 变体，同时报 HD95 / ASD 下降 | 仅作「距离指标常与 Dice 同报」的引证；方法本身增益小、与我们已试过的注意力路线重叠，不作主推 |

### 14.1 本批硬结论

1. **距离指标要动，只有两种落点：损失（82/83/86）或输出表示（85/84）**。后处理已被我们自己证伪（5 种回原网格/平滑口径，ASSD 0.1447–0.1448、HD 0.7064–0.7070，全部在噪声内），所以不要再在推理侧找增益。历史 4 条损失臂（farpf / bndsoft / DL 两项）的实测落差只有 ΔHD ≤0.13、ΔASSD ≤0.016，但**这 4 条都是「像素重加权/像素级额外项」写法**——改的是平均梯度，不是距离剖面。82 是直接对距离剖面回传（`D+(G)∘y` 与 `d+(y)∘G` 分别是 miss/extra 侧的距离能量），机制不同，不能用这 4 条的落差预判它。判定仍按单 seed 配对 → 过线后 3 seed 复核。
2. **只有 82 能在不改结构、不加参数、2D 直接跑的前提下同时压两侧**。其余要么加头（84/85/86），要么换采样算子（87），要么改评测（88）。
3. **没有一篇报颈动脉 + ASSD/HD 的 A/B 级证据**：89 是同题但走半监督/PatchGAN，84 是超声但非颈动脉。所以引用时只能写「超声/医学分割上已有工作提示」，不能写「已在颈动脉验证」。
4. **HD 类损失的通病是梯度稀疏、训练不稳**：82 用 `smax` 平滑 + 只在 I=2–5 层剖面内回传，需先确认我们的 batch 内正样本足够（斑块小，bs4/ga2 下可能个别 step 无前景）。实现时先做「无前景 step 跳过该项」的守卫。
5. **#88 已被本项目的实测排除**：GT 走 224→原网格往返的 ASSD 仅 0.0031、HD 0.0318（320/384 更低），量化台阶不是误差来源，学习式上采样无空间。见 [results.md](results.md) §9.3(a)。

### 14.2 与现有结论的接口

| 已有观测 | 本批对应 |
|---|---|
| HD 漏检侧驱动（miss 108 / extra 52） | 82 的 `D+(G)∘y` 项正是惩罚「GT 上有、预测上没有」的像素 → 对得上 |
| 全边界等权加权无效（bndsoft / 边缘分支） | 83 给出解释：应按可分辨性门控，而非全带加权 |
| `deform` 可变形大核没涨 | 87 给出解释：缺几何引导，偏移是自由学的 |
| 5-seed 集成 HD 0.5947，单 seed sd 0.039 | 论文口径改为单 seed 配对（基线 0.7062）；新臂 HD 增益需 ≥ 0.1 mm 且 Dice 不明显倒退，过线后按 3 seed（42/317/1029）复核报均值±sd。5-seed 集成只作内部诊断、不作论文读数 |
| `--head_hr` 已实现未跑 | 属 ASSD 侧的结构对照，与本批不冲突，可在损失臂之后排 |
