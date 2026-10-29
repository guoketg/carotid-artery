# 实验方案

## 1. 消融矩阵

### 1.1 已完成（2026-10-23 旧版小波引导注意力）

| 编号 | 配置 | 验证目标 |
|---|---|---|
| A0 | 单通道原图基线（原图单通道 TransUNet） | 基线参照 |
| A1 | 三通道早融合（原图通道、去噪重建通道、高频能量通道拼成三通道，直接进主干，无额外融合模块） | 预处理本身是否有效 |
| A2 | 伪小波支路对照（原图复制三通道 + 小波引导注意力） | 融合结构本身是否有效 |
| A3 | 三通道＋小波引导注意力（旧版，只在 14×14 单点融合） | 完整方法（结论：无显著增益） |

### 1.2 待跑（2026-10-24 修订，双骨干 + 三级融合）

前提：所有变体统一用固定 epoch 评估（规则见 CLAUDE.md 第 8 条），不再在测试集上调 epoch；
每次对比都同时报 Params / FLOPs，并做病例级 Wilcoxon。

| 编号 | 配置 | 验证目标 |
|---|---|---|
| A4 | **共享权重双输入**（同一 R50+ViT 分别前向原图通道与高频能量通道）+ 深层 28×28 融合 | 零新增参数下，"两路输入"是否有增益 |
| A5 | **独立同构双骨干**（两套 R50+ViT，均 ImageNet21k 初始化）+ 深层 28×28 融合 | 容量翻倍是否有额外收益（对照共享权重双输入（A4），参数不匹配需注明） |
| A6 | 共享权重双输入（A4）基础上**三级全开**：浅 112 + 中 56 + 深 28 | 多尺度融合是否好于单点 |
| A7 | 融合模块替换：AG / CSA(通道+空间) / CGAF(双向跨注意力) 各一次 | 注意力形式本身是否贡献 |
| A8 | 融合位置单点扫描：仅浅(112) / 仅中(56) / 仅深(28) / 仅瓶颈(14) | 哪个尺度是增益来源 |
| A9 | 参数匹配对照：第二路换成容量相同但输入为噪声的编码器 | 排除"增益来自参数变多" |

执行约束：

- 独立同构双骨干（A5）需先做 batch 4 + AMP + 梯度累积(2) 的显存冒烟；OOM 则第二骨干降级为 R50-only / ViT-S，保持层级对称。
- 112×112 与 56×56 不允许全 token 注意力（56×56 全注意力约 45 GFLOP/前向），只允许 AG / CSA / 窗口化注意力。
- 每个变体两个子集（SPARC / Zhongnan）分开跑，不混合。

附加实验：小波级数（1 vs 2）、软阈值开关、CLAHE 替代去噪重建通道、DoG 替代高频能量通道、去噪重建通道是否保留。

### 1.3 第二轮优先级（2026-10-24 检索后重排，**先跑这一组**）

依据见 `literature.md` 末尾「硬结论」：收益来自第二路**信息源**而不是融合模块（DCLA-Net Table 1、MSFA-Net Table 4），跨支**交互**优于 concat+通道注意力（DEFI-Net Table 4），输入级去噪收益不稳定（denoising benchmark Table V），同图派生视图的双输入增益天花板很低（MS-Net Table 1/2）。由此把共享权重双输入与三级全开提前、独立同构双骨干降级、去噪重建通道移出主路径，并新增两个此前没有的对照（原图通道+高频能量通道双通道早融合门槛、共享权重双输入加 concat 回投影）。

| 编号 | 配置 | 对应旧编号 | 验证目标 |
|---|---|---|---|
| **E1** | 原图通道 + 高频能量通道双通道早融合，单骨干、无注意力（第三通道置零以适配 3 通道 patch embedding） | 新增 | **高频能量通道是否存在可被单编码器利用的信息**（最便宜的门槛） |
| **E2** | 共享权重双输入（同一套 R50+ViT 分别吃原图通道与高频能量通道）+ 跨支门控，仅深 28×28 | ≈A4（共享权重双输入） | 双输入 + 交互，单点够不够 |
| **E3** | 共享权重双输入 + **concat + 1×1 回投影**，仅深 28×28 | 新增 | 增益来自"注意力"还是"多一路特征" |
| **E4** | 共享权重双输入 + 跨支门控，**浅 112 + 中 56 + 深 28 三级全开** | ≈A6（三级全开） | 多级是否好于单点 |
| **E5** | 共享权重双输入 + 跨支门控三级全开（E4）的胜出配置搬到 Zhongnan | — | 跨子集是否复现 |

执行顺序与成本（单次约 50–60 min：120 epoch / batch 8 / `--eval_test_every 10`）：

1. 开发集固定用 **SPARC**，先跑 E1（原图通道+高频能量通道双通道早融合门槛）→ E2（共享权重双输入 + 跨支门控单点）→ E3（其 concat 回投影对照）→ E4（三级全开）（约 4 h）；
2. 只有三级全开（E4）或单点（E2）达标，才跑 E5（Zhongnan 跨子集复现确认）；
3. 独立同构双骨干（A5）**降级为最后**：仅在三级全开（E4）证明有效后，用来量化"容量 vs 信息"各占多少；若 E4 无效则直接跳过。

判定规则（写在跑之前，避免事后解释）：

| 检查 | 通过条件 | 不通过时的处置 |
|---|---|---|
| 原图通道+高频能量通道双通道早融合（E1） vs 单通道原图基线（A0） | \|ΔDice\| ≥ 0.005 或 HD95 显著下降（p < 0.05，病例级 Wilcoxon） | 高频能量通道单编码器吃不动 → 双支路增益只能来自容量，补参数匹配的噪声第二路对照（A9）定性，不主张"小波有效" |
| 共享权重双输入 + 跨支门控单点/三级全开（E2/E4） vs 单通道原图基线（A0） | Dice ≥ +0.01 **且** HD95 显著下降 | 判为无增益，转负面结果，融合线停止投入 |
| 三级全开（E4） vs concat 回投影对照（E3） | E4 优于 concat 对照且 p < 0.05 | 说明"跨支门控"本身不贡献，回退到 concat，结论改写 |
| 三级全开（E4） vs 单点（E2） | E4 优于单点 | 多级融合无额外收益 → 只保留深 28×28 单点，省算力 |
| 跨子集复现（E5） vs 三级全开（E4） | 方向一致（不要求幅度相同，两子集设备不同） | 只在 SPARC 成立 → 结论限定为单中心，写进限制 |

预处理增强变体（等上述第二轮门槛与融合对照出结果后再决定是否插入，不预先占队列）：

| 编号 | 增强 | 实现位置 | 预期作用 |
|---|---|---|---|
| P1 | CLAHE（clip 2.0，8×8） | 新 `enhance.py`，替换原图通道或作为第 4 通道的候选 | 拉开斑块内部灰度差 |
| P2 | 二级小波的高频能量（高频能量通道取第 2 级而非第 1 级） | `build_three_channel(detail_level=2)` | 把斑点（第 1 级）与边界（第 2 级）分开 |
| P3 | 同态滤波（log + 高通） | 新 `enhance.py`，替换去噪重建通道 | 修正声影/近远场亮度差，可能改善 DeltaArea |

> 注意：**"把高频能量通道乘一个增益系数"这种高频增强在现有代码里无效**——`_standardize(energy)` 逐样本减均值除标准差，会抹掉全局幅度。要改变高频权重只能换尺度（改用二级小波高频能量，P2）或换通道构成。

### 1.4 第三轮：双骨干强融合（强融合实验组，2026-10-24，**已入队**）

方案见 [technical-design.md](technical-design.md) 的「双骨干交错四级融合：双编码器 + 四级接点」一节，落地实现见 [dual-encoder-design.md](dual-encoder-design.md)。**2026-10-24 22:55 起已实施并进入夜间队列**，规格选定为独立 R50 侧路版（侧路 BN→GN）。

输入契约（已定）：主路只吃原图通道（复制 3 通道），侧路吃 `LH/HL/HH` 三张方向子带，**输入端不拼原图通道/去噪重建通道/高频能量通道**；四个接点都**双向**（112²/56²/28² 并行双向，14² 串联双向）、按 stage 交错执行。

| 编号 | 配置 | 验证目标 | 状态 |
|---|---|---|---|
| **R0** | 探针验证：50 步、不训练、不写权重 | 实测 batch 4/6/8 的峰值显存、单步耗时、BN `running_var` 方差 | ✅ 2026-10-24 完成 |
| **R1** | 双骨干交错四级融合的独立 R50 侧路（到 stage3）+ 28×28 单点融合 | 双骨干在最强接点是否有效 | ⏳ 本队列实为 4 接点 `auto` 档 |
| **R2** | 独立 R50 侧路 + 112×112、56×56、28×28 三个融合接点全开 | 多级是否好于单点 | ⏳ 由单点配置（R1）的 `--levels 1,2,3,4` 一并覆盖 |
| **R3** | 独立 R50 侧路 + 四个融合接点（增加 14×14 接点） | 瓶颈接点是否还有增量 | ⏳ 同上 |
| **R4** | R50+ViT 双骨干 + 四个融合接点 | 容量 vs 信息（对照四点全开（R3），参数量不匹配需注明） | ⏸ 未排期 |
| **R5** | 四点全开配置（R3）的胜出配置搬到 Zhongnan | 跨子集是否复现 | ⏳ 本队列每个变体都跑两子集 |
| **R6** | 112×112、56×56 融合接点的窗内注意力 → 退回逐元素门控 | 窗内注意力的净贡献，防止「只是多了参数」 | ⏳ 即队列中逐元素门控对照 job（`--fuse_op gate`，r2） |
| **R7** | 四点全开配置（R3）只保留「侧→主」，删掉「主→侧」 | 双向是否必要；检验单向是否退化成三通道＋小波引导注意力（A3）式的单向供体 | ⏳ 队列中单向融合 job（`--bidir 0`，r7） |

实际队列（`scripts/night_queue_dual.ps1`，8 job 串行，batch 4 + `--grad_accum 2`，120 epoch，`--eval_test_every 10`）：

| job | 变体 | 数据集 | 估时 |
|---|---|---|---|
| r1_sparc / r1_zhongnan | `--fuse_op auto`（112×112、56×56 接点窗内、28×28、14×14 接点全局，双向） | SPARC / Zhongnan | 72 / 87 min |
| r2_* | `--fuse_op gate`（即窗内注意力退回逐元素门控的对照） | 同上 | 同上 |
| r3_* | `--fuse_op concat`（concat 对照） | 同上 | 同上 |
| r7_* | `--fuse_op auto --bidir 0`（即删掉主路→侧路的单向融合对照） | 同上 | 同上 |

合计约 **10.6 h**。原计划中单点（R1）、三级（R2）、四点（R3）的拆分因实现合并为一次前向（四个接点同时开），改由 `--levels` 事后消融；当前队列先验证「四点全开 vs 单通道原图基线」，层级拆分留待单点配置（R1）达标后补。

强融合追加队列（`scripts/night_queue_dual2.ps1`，6 job，等队列 1 结束后 GPU 空闲 10 min 自动启动，
同超参：batch 4 + `--grad_accum 2`、120 epoch、`--eval_test_every 10`）：

| job | 变体 | 针对的根因 | 数据集 | 融合模块参数 |
|---|---|---|---|---|
| r4_* | `--fuse_op deform`（D-LKA 式可变形大核，见 3.4） | 采样网格固定 | SPARC / Zhongnan | 3.69 M |
| r5_* | `--fuse_op aff`（AFF MS-CAM） | 只调节幅度、不重组内容 | SPARC / Zhongnan | 0.47 M |
| r6_* | `--fuse_op auto --cross_scale 1`（BiFPN 式 top-down） | 只在 4 个点融合，时间维度上不传播 | SPARC / Zhongnan | +44 K |

> 编号说明：queue 2 里三个 job 名中的 **r4 / r5 / r6 是 3.4 节新增的融合算子槽位**（可变形大核 / AFF 幅度调制 / 跨尺度回流），与上表计划项
> R4（R50+ViT 双骨干，⏸ 未排期）、R6（112×112、56×56 接点窗内注意力退回逐元素门控，即 `--fuse_op gate`）**不是同一件事**。
>
> 三个算子均已通过零初始化等价性（`max|Δ| = 0`）与参数量探针；耗时探针因与队列 1 争用 GPU 未采信。

执行约束：

- 开发集固定 SPARC，按单点（R1）→ 三级（R2）→ 四点（R3）顺序跑；R50+ViT 双骨干（R4）与跨子集复现（R5）只在四点（R3）达标后跑。
- 单点配置（R1）起就是双编码器，**必须先过探针验证（R0）**；实测不达标则降 batch + 梯度累积，不接受「估算可行」。
- 侧路 BN 全部换 GroupNorm（主路 BN 与预训练权重不动）。
- 输出目录追加 `_dualR50` / `_dualViT`，禁止覆盖单通道原图基线（A0）的输出。

判定规则（跑之前写死）：

| 检查 | 通过条件 | 不通过时 |
|---|---|---|
| 单点融合（R1） vs 单通道原图基线（A0） | **HD95 下降 ≥ 0.5 mm 且 p < 0.05** | 双骨干在单点无效 → 不追加接点，转负面结果 |
| 三级全开（R2） vs 单点融合（R1） | HD95 再降 ≥ 0.3 mm 且 p < 0.05 | 多级无额外收益 → 只保留 28×28 单点 |
| 四点全开（R3） vs 三级全开（R2） | HD95 或 Dice 至少一项显著 | 瓶颈接点冗余 → 删 14×14 接点，省注意力开销 |
| R50+ViT 双骨干（R4） vs 四点全开（R3） | 差值 < 四点全开（R3）相对单通道原图基线（A0）的增益一半 | 说明增益来自信息而非容量 → 只报独立 R50 侧路版，R50+ViT 双骨干版写进讨论 |
| 跨子集复现（R5） vs 四点全开（R3） | 方向一致（不要求幅度相同） | 只在 SPARC 成立 → 结论限定单中心 |
| 窗内注意力退回门控（R6） vs 四点全开（R3） | 窗内注意力显著更好 | 门控够用 → 退回逐元素门控，省算力 |
| 单向融合（R7） vs 四点全开（R3） | 双向显著更好（HD95 或 Dice） | 双向无差异 → 只保留单向并在文中注明；若单向反而更好，说明侧路被主路污染 |

主指标统一为 **HD95 / ASSD**，Dice 只作参考（与 1.3 一致）。

### 1.5 2026-10-24 夜实际执行顺序（**只跑 SPARC，单子集**）

用户决定：不做小样本快筛、不跑第二个子集，直接把全量 SPARC 跑完。实测全量 dual 为 **0.70 s/iter（107 iter/epoch ≈ 75 s/epoch）**，即单个 120-epoch job ≈ **2.5 h**，一夜（23:50 → 08:00）只能排 3 个。故按优先级只排：

| 顺序 | job | 变体 | 目的 | 预计完成 |
|---|---|---|---|---|
| 1 | r1_sparc | `--fuse_op auto`（112×112、56×56 接点窗内、28×28、14×14 接点全局，双向） | 主方法，同时产出 120 epoch 测试曲线用于定 epoch | ~02:20 |
| 2 | r4_sparc | `--fuse_op deform` | 可变形大核是否强于窗内注意力 | ~04:50 |
| 3 | r5_sparc | `--fuse_op aff` | AFF 式幅度调制是否够用 | ~07:20 |
| 4 | r6_sparc | `--fuse_op auto --cross_scale 1` | 跨尺度回流（顺延，不保证 8 点前出） | ~09:50 |

- 命令：`night_queue_dual.ps1 -Skip r2_sparc,r3_sparc,r7_sparc,*zhongnan` → `night_queue_dual2.ps1 -Skip *zhongnan`。
- 未排：r2（逐元素门控）/ r3（concat 对照）/ r7（单向融合）与全部 Zhongnan 复现，留待四点全开 job（r1）出结果后再决定。
- 对比基准：**单通道原图基线（A0，单骨干，120 epoch，SPARC）已有结果**，r1（四点全开） vs 单通道原图基线即主检验。
- 每晚只改一个因子；`--eval_test_every 10` 让每个 job 都留下 10/20/…/120 的曲线，可在任意公共 epoch 上做配对比较。

> 原「dev 60% 子集快筛」方案（`lists_Carotid_dev` + `night_queue_dev.ps1` + `summarize_dev.py`）已于 23:50 撤销，脚本保留备用，产物已删。

### 1.6 参数与统计口径

- Params / FLOPs：共享权重双输入（同一套 R50+ViT 分别吃原图通道与高频能量通道）与单通道原图基线相同（105M）；共享权重双输入 + concat + 1×1 回投影对照（E3）额外约 1.2M；独立同构双骨干（A5）为 210M。
- 统计：按病例配对 Wilcoxon（同一病例的多张切片先取均值再配对），每子集独立；不跨设备合并。
- 主指标 **HD95 / ASSD**，Dice 为次要门槛；DeltaArea 只报带符号均值与 |DeltaArea|，不做显著性。

### 1.7 2026-10-28 夜：融合算子横评 + DDBE 门槛验证 + 基线补跑（**已入队**）

队列脚本：[scripts/queue_c2d_fuse.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/queue_c2d_fuse.ps1)（第三个接力队列，2026-10-28 00:54 以 WMI 启动，pid 34168；等主队列 `queue_c2d.ps1` 与边缘队列 `queue_c2d_edge.ps1` 全部退出、GPU 空闲后自动开跑）。统一口径 `$common`：SPARC / bs4 + ga2 / 200 epoch / `--eval_test_every 5`，取 ep150，与 1.6 及既有队列完全一致。

**本次改了哪些代码**（DDBE 实现，冒烟全过）：

| 文件 | 改动 |
|---|---|
| [networks/ddbe.py](file:///e:/mkl/code/python_code/carotid-artery/networks/ddbe.py)（新增） | BDFR-Net 式 DDBE：ECA → HF 支（多通道 Haar DWT 取 LH/HL/HH → concat → 1×1 → 3×3+BN → 转置卷积上采样 → SE）/ LF 支（LL → 5×5+BN → 双线性上采样 → SE）/ CTX 支（7×7+BN → 自门控）→ concat → 1×1（**唯一零初始化点**）→ CBAM → 残差加回。level3 参数量 0.440 M |
| [networks/vit_seg_modeling.py](file:///e:/mkl/code/python_code/carotid-artery/networks/vit_seg_modeling.py) | `DecoderCup` 加 `ddbe/ddbe_idx` 挂载点（forward 中在第 N 级 decoder_block 输出后调用）；`VisionTransformer` 加 `ddbe_level` 参数，**在全部基础权重初始化之后**再建 DDBE（保证同 seed 基础权重逐位不变，RNG 对齐） |
| [train.py](file:///e:/mkl/code/python_code/carotid-artery/train.py) | 加 `--ddbe 0~4`（仅 baseline 接了，dual 会 parser.error）；快照名追加 `_ddbe<N>`（拼在 fusion 后缀之后、dual 块之前） |

冒烟结论（临时脚本 `Temp\smoke_ddbe.py`，已跑 ALL PASS）：① 同 seed 基础权重与不挂 DDBE 版本 `torch.equal`（DDBE 仅新增 35 个张量）；② eval 恒等起步 `max|Δ| = 0`（零初始化 → 初始 logits 与基线逐位相等）；③ train 模式前反向正常；④ level 1~4 四级 forward 均通（解码器四级实际输出 28/56/112/224，level3 = 112×112/64ch，与论文「挂在哪级」无关紧要——先用 level3 做门槛验证）。

**队列 6 个 job（顺序 = 优先级）**：

| # | job | 配置 | 快照目录（与 train.py 逐字对齐） | 对照 | 预计完成 |
|---|---|---|---|---|---|
| 1 | `faff1234_sparc` | dual `--fuse_op aff`（AFF MS-CAM，0.47 M） | `..._224_SPARC_dual_L1234_aff_c2d` | d1234 auto（已完成） | ≈ 07:20 |
| 2 | `fgate1234_sparc` | dual `--fuse_op gate`（逐元素门控，R6 对照） | `..._224_SPARC_dual_L1234_gate_c2d` | 同上 | ≈ 09:00 |
| 3 | `fdeform1234_sparc` | dual `--fuse_op deform`（D-LKA 可变形大核，3.69 M） | `..._224_SPARC_dual_L1234_deform_c2d` | 同上 | ≈ 10:40 |
| 4 | `ddbe3_1234_sparc` | baseline `--ddbe 3`（DDBE 挂 112²/64ch 级） | `..._224_SPARC_ddbe3_c2d` | A0 基线 seed1234 | ≈ 12:00 |
| 5 | `b1234_sparc` | A0 单通道原图基线 seed1234 **重建** | `..._224_SPARC_c2d` | —（见下方发现） | ≈ 13:15 |
| 6 | `b3407_sparc` | A0 基线 seed3407 **补跑** | `..._224_s3407_SPARC_c2d` | —（见下方发现） | ≈ 14:30 |

排班依据与两个意外发现：

- **aff / gate / deform 零实现成本**（算子已在 `networks/` 实现且过零初始化等价性），先横评融合算子，与已完成的 `d1234_sparc`（auto）同 seed 配对；
- **发现 1**：A0 基线 seed1234 的新口径产物（`..._SPARC_c2d/` 的 `test_curve.csv`、`metrics.csv`，§3 epoch 选择即依据它）在本机已不存在——全盘（C:/E:）搜不到。ddbe3 的病例级 Wilcoxon 必须有同 seed 配对对照，故**补跑 job 5**恢复之（aggregate 数字 status.md §3.1 已存档，跑完核对是否复现 90.49%/0.1431/0.3896）；
- **发现 2**：`b3407_sparc` 上一轮在 epoch≈50 被中途杀掉（log 止于 iteration 2163、无 `metrics.csv`），但 `queue_c2d.ps1` 的 `-Skip` 把它跳过了、status.md 仍写「排队」——故**补跑 job 6**；
- **本轮不碰 QKV、不上四级全量 DDBE**：DDBE 只做 level3 单级门槛验证，过了门槛再谈全量（写在跑之前，避免事后解释）。

判定规则（跑之前写死）：

| 检查 | 通过条件 | 不通过时的处置 |
|---|---|---|
| aff / gate / deform vs d1234 auto（同 seed 配对） | 某算子 HD95 下降 ≥ 0.5 mm 且 p < 0.05 | 无显著差异 → 融合算子不敏感，维持 auto，横评线关闭 |
| ddbe3（DDBE 单级门槛） vs A0 基线 seed1234 | \|ΔDice\| ≥ 0.005 **或** HD95 显著下降（p < 0.05，病例级 Wilcoxon） | 不过门槛 → DDBE 线停止投入，不排四级全量 |
| b1234 重建 vs status.md §3.1 存档 | ep150 四项指标复现（Dice 差 < 0.005） | 不复现 → 以重建产物为准并核查口径 |

### 1.8 2026-10-28 三个队列跑完：ep150 横评结果与病例级 Wilcoxon（**已收口**）

三个队列（`queue_c2d.ps1` 主队列 + `queue_c2d_edge.ps1` 边缘队列 + `queue_c2d_fuse.ps1` 融合队列）全部执行完毕。统一口径：SPARC 官方划分 / bs4 + ga2 / 200 epoch（poly 绑死）/ `--eval_test_every 5`，判定点 **ep150**（存档 `epoch_149.pth`）。

**故障与修复**：

- `faff1234`（dual `--fuse_op aff`）在 epoch 5 首次测试时崩溃（exit 1，仅存活 1.8 min）：`_MSCAM` 全局支路的 GroupNorm 在 GAP 后 1×1 空间 + eval batch=1 时校验失败（`input size [1, 8, 1, 1]`）。**修复**：[dual_encoder_model.py](file:///e:/mkl/code/python_code/carotid-artery/networks/dual_encoder_model.py) 的 `_MSCAM.glb` 去掉 GroupNorm，改带 bias 的 SE 式 MLP（局部支路保留 GN）；冒烟（eval batch1 前向 + train batch4 反向）通过，批次 0 重跑。
- **时点错位**：`trainer.py` 末尾 `metrics.csv` 是 ep199 的逐切片指标，与 ep150 判定点不符 → 用 `epoch_149.pth` 重算。[scripts/eval_carotid.py](file:///e:/mkl/code/python_code/carotid-artery/scripts/eval_carotid.py) 已扩展：`build_model` 支持三分支（baseline/dual/fusion + edge_in/ddbe），新增病例级聚合 `_group_by_case`（同病例多切片先取均值再配对，§1.6 口径）与 CLI 全参数；批量驱动 [scripts/eval_batch_ep149.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/eval_batch_ep149.ps1)。全部 11 个有效快照重算完毕，每个 ~0.3 min，逐切片汇总与 `test_curve.csv` ep150 完全吻合（交叉验证通过）。

**ep150 逐切片汇总**（n=160，SPARC test；Dice % / ASSD mm / HD95 mm）：

| job | 配置 | Dice | ASSD | HD95 |
|---|---|---|---|---|
| b1234 | A0 基线 seed1234 | 89.95% | 0.1498 | 0.3997 |
| b2024 | A0 基线 seed2024 | 90.27% | 0.1498 | 0.4146 |
| b3407 | A0 基线 seed3407 | 90.26% | 0.1508 | 0.4303 |
| dauto1234 | dual auto（L1234，双向）s1234 | 90.40% | 0.1476 | 0.4064 |
| dauto2024 | dual auto s2024 | 90.11% | 0.1468 | 0.4036 |
| dauto3407 | dual auto s3407 | 89.85% | 0.1570 | 0.4461 |
| gate1234 | dual gate s1234 | 90.35% | 0.1434 | 0.4034 |
| deform1234 | dual deform s1234 | **90.61%** | **0.1412** | **0.3788** |
| ddbe3 | baseline `--ddbe 3` s1234 | 90.08% | 0.1513 | 0.4404 |
| edge1234 | baseline `--edge_in 1` s1234 | 90.20% | 0.1476 | 0.3973 |
| edgedual | dual auto + edge_in s1234 | **90.72%** | 0.1416 | 0.3895 |
| faff1234 | dual aff s1234 | ❌ 崩溃（见上），批次 0 重跑 | | |

**病例级配对 Wilcoxon**（43 例，同病例切片先均值再配对；基线 = 对应 seed 的 A0）：

| 比较 | ΔDice（病例级） | p(Dice) | p(ASSD) | p(HD95) |
|---|---|---|---|---|
| dauto1234 vs b1234 | +0.38% | 0.209 | 0.294 | 0.385 |
| dauto2024 vs b2024 | +0.00% | 0.816 | 0.586 | 0.385 |
| dauto3407 vs b3407 | **−0.49%** | **0.0036** ✗ | **0.0097** ✗ | 0.104 |
| gate1234 vs b1234 | +0.27% | 0.663 | 0.742 | 0.333 |
| deform1234 vs b1234 | +0.59% | 0.113 | 0.154 | 0.276 |
| ddbe3 vs b1234 | −0.03% | 0.680 | 0.881 | 0.093 |
| edge1234 vs b1234 | +0.27% | 0.595 | 0.447 | 0.891 |
| edgedual vs b1234 | +0.62% | 0.069 | 0.223 | 0.791 |

（各 log 中还出现 `delta_area` p<0.001 的行：A0 基线系统性低估面积（−0.90 mm²），多数变体向 0 靠拢——这是**面积偏差校正**方向的真实差异，但反映的是偏置而非边界精度，不计入判定。）

**结论（对照 §1.7 判定规则）**：

1. **融合算子横评（aff/gate/deform vs auto）：全部未通过**（无算子达 HD95 降 ≥0.5 mm 且 p<0.05）→ 融合算子不敏感，维持 auto；deform 是唯二方向一致为正的点（ΔDice +0.59%、HD95 −0.019 mm）但 p=0.113。
2. **双骨干 3-seed：无增益**。A0 三 seed 均值 90.16% vs dual auto 90.12%；且 s3407 上 dual **显著变差**（Dice p=0.0036、ASSD p=0.0097，方向为负）。在当前融合设计下双骨干线判负——增益若存在也掩埋在 seed 方差之下。
3. **ddbe3 单级挂载：未过门槛**（ΔDice −0.03%，HD95 反而 +0.078 mm）→ 与用户判断一致：别人的论文（BDFR-Net）是**四级全挂 + 每级 ACAM**，单级挂载证明不了什么，转批次 1（§1.9）。
4. **edge 线索保留**：edgedual 是全场最佳点（90.72%/0.1416/0.3895，p=0.069 边缘），deform 次之；两者都只是方向证据，不作为结论。

## 1.9 文献复现计划（2026-10-28 用户批准，批次 0–4）

**复现口径**：本项目只有 Carotid2D 数据（SPARC/Zhongnan 两子集），无法复现论文原数据集 → 「复现」定义为**把论文核心模块移植进现有 TransUNet 管线**，与 A0 基线同口径（官方划分 / bs4+ga2 / 200 epoch poly / 判定点 ep150 / SPARC / seed1234），用 §1.8 的病例级配对 Wilcoxon 比较。基线锚点：A0 s1234 = 89.95% / 0.1498 / 0.3997。

**判定规则（跑之前写死，每批次同一模板）**：

| 通过条件 | 不通过时的处置 |
|---|---|
| \|ΔDice\| ≥ 0.005（病例级）**或** HD95 下降 ≥ 0.5 mm 且 p < 0.05 | 该论文的模块在本任务无效 → 记录负面结果，该线停止投入 |

| 批次 | 内容 | 出处（pdf/） | 实现要点 | 状态 |
|---|---|---|---|---|
| **0** | faff 重跑 + 全量 ep149 病例级统计 | 本项目 | AFF GroupNorm 修复（§1.8）；`eval_batch_ep149.ps1` 已跑完 11 快照 | 统计 ✅ 2026-10-28；faff 重跑已入队（队列第 3 位） |
| **1** | ① `ddbe4`：解码器**四级全挂** DDBE ② `ddbefull`：DDBE×4 + ACAM skip 对齐 | 3612 Submission.pdf（BDFR-Net）p.3「four cascaded stages, each comprising … DDBE … ACAM」、p.5「repeats across all four resolution levels」 | 复用 [networks/ddbe.py](file:///e:/mkl/code/python_code/carotid-artery/networks/ddbe.py)（零初始化恒等起步，保同 seed 等价）；ddbe4 = `--ddbe 4` 语义扩展为全四级；ACAM = 解码器逐级对 skip 特征做跨级语义对齐再拼接 | 已实现 ✅（冒烟 5 项全过，参数量 105.3M→114.8M）；已入队在训 |
| **2** | GobletNet 式：单主干 + 轻量 HF 支 + FAM 逐层融合 | GobletNet.pdf Eq.5 / Table VI / Table X | 单 R50+ViT 主干（不翻容量）：HF 图 `H=HL+LH+HH+λ·LL`（λ=0.3，forward 内现算）→ 轻量 HF 支（32@112/64@56/128@28，+0.29M）→ FAM 逐级与 skip 互标定（官方通道不等适配版，+0.84M）→ `skip = x1_fa + inject(x2_fa)`（inject 零初始化，对应官方 decoder cat(x2_fa)） | 已实现 ✅（冒烟 5 项全过：同 seed 对齐 / 恒等起步 max\|Δ\|=0 / 门控时序验证 / 参数 105.3M→106.5M / λ 回灌行为；详见 §1.9 执行记录②）；待批次 1 结论后入队 |
| **3** | Syn-Net FrCFusion 融合算子 | Syn-Net.pdf | 落地为 `--fuse_op dct`（频域通道混合），接入现有 dual 四接点框架 | 待实现 |
| **4** | WMREN WRDM 双分支 DWT | WMREN.pdf | 双分支（原图 + DWT 子带）+ WRDM 式权重残差蒸馏融合 | 待实现 |
| 搭车 | XNet 便宜变体 `--main_input ll`（主路吃 LL 低频图） | XNet.pdf | 零新代码，仅改输入构成 | 已入队（队列第 4 位） |

不复现：WeaveSeg / FcaNet / MFDF / XNet 全量（模块重、与现有框架重叠度高、预期增益低，理由见 [literature.md](literature.md)）。

执行顺序：批次 0 收尾（faff 重跑）→ 批次 1（ddbe4 → ddbefull，先四级 DDBE 消融出「挂几层」结论，再加 ACAM）→ 视结果决定批次 2–4 排期。队列脚本沿用 [queue_c2d_fuse.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/queue_c2d_fuse.ps1) 模板（接力等待 + jobs 数组 + WMI 启动），WMI 启动纪律不变。

> **执行记录**：批次 0+1 已合并为一个队列 [queue_c2d_repro.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/queue_c2d_repro.ps1)，2026-10-28 17:16 经 WMI 启动（pid 26772），4 job 顺序：ddbe4 → ddbefull → faff 重跑 → mainLL 搭车，约 12 h；快照名 `_ddbe4` / `_ddbe4_acam`；status CSV `logs/queue/queue_c2d_repro_status.csv`。队列跑完后按 §1.9 判定规则对 4 个快照跑 ep149 eval + 病例级 Wilcoxon。
>
> **执行记录②（批次 2 实现，2026-10-28 18:00）**：GobletNet 移植完成，趁 GPU 忙碌先行实现（不占队列）。改动：[networks/hf_branch.py](file:///e:/mkl/code/python_code/carotid-artery/networks/hf_branch.py)（新建，HFBranch + FAM + `HF_DET_CHANNELS=(32,64,128)`）；[networks/wavelet.py](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py) `build_hf_image` 加 `lf_weight=0.0` 参数（λ·LL 回灌，默认 0 与 edge_in 逐位一致）；[networks/vit_seg_modeling.py](file:///e:/mkl/code/python_code/carotid-artery/networks/vit_seg_modeling.py) `VisionTransformer` 加 `hf_branch/hf_lambda`（挂载在 ddbe/acam 之后，RNG 纪律；forward 浅→深交错：det 被 FAM 重标定后继续下采样，贴官方「FA 后特征继续进下一层」）；[train.py](file:///e:/mkl/code/python_code/carotid-artery/train.py) `--hf_branch / --hf_lambda 0.3`（仅 baseline，快照后缀 `_hfbr3`）；[scripts/eval_carotid.py](file:///e:/mkl/code/python_code/carotid-artery/scripts/eval_carotid.py) 同名参数。冒烟 `Temp\smoke_hfbr.py` 5 项全过：①同 seed 基础权重对齐（96 新张量，0 missing/mismatch）②恒等起步 max|Δlogits|=0 ③门控时序：step 0 仅 gamma/inject 有梯度（det 链在两道零门后，LoRA B=0 同款一步延迟，预期行为），3 步 SGD 后全链解锁 ④参数 105.276M→106.482M（+1.206M）⑤λ=0.3 回灌改变 HF 图且有限值。入队命令（批次 1 结论后）：`--hf_branch 1 --hf_lambda 0.3`，快照 `*_SPARC_hfbr3_c2d`。
>
> **执行记录③（评测流水线就绪，2026-10-28 17:55）**：`eval_batch_ep149.ps1` 追加 4 个新 job 条目（ddbe4 / ddbefull / faff / mainll，Baseline 均配 b1234 的 metrics_ep149.csv，DryRun 验证快照路径全部正确）；新增 [watch_eval_repro.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/watch_eval_repro.ps1)（WMI 启动，pid 16200）每 5 min 轮询 `queue_c2d_repro_status.csv`，某 job 行落地（exit 0）即自动跑其 ep149 eval + 病例级 Wilcoxon，4 job 处理完自动退出；watcher 日志 `logs/queue/watch_eval_repro.log`，各 job 评测日志 `logs/queue/watch_eval_<job>.log`。进度实测：ddbe4 ~120 iter/min（比 ddbe3 的 290 iter/min 慢，四级挂载计算量更大），17600 iter 预计 ~19:45 跑完，全队列 ~01:30。


## 2. 评估协议

- **两个子集分开**：SPARC 与 Zhongnan 各自训练、各自测试，不混合。
- 暂不设验证集，训练固定轮数后直接用测试集评估；最终标准待老师给出后按其定义重跑。
- 指标：Dice、ASSD、HD95、DeltaArea（预测面积 − 真值面积）。**2026-10-26 起切换物理单位：ASSD/HD95 为 mm、DeltaArea 为 mm²（预测重采样回原分辨率网格后按 NIfTI 头 spacing 计算，明细见 [dataset.md](dataset.md) §6）；此前全部数字为 224 网格像素口径，与 mm 口径不可直接比。**
- 显著性：paired t-test 或 Wilcoxon signed-rank，p < 0.05。
- 必须同时报告 Params、FLOPs、单张推理时延，排除"涨点来自更大模型"。
- 斑块为小目标：损失用 `Dice + BCE`（或 CE+Dice），重点看 HD95 与 DeltaArea 的偏差方向。
- 失败案例：低回声斑块、声影区、钙化伴声影、增益过曝。

## 3. 训练超参（以夜间队列实际跑过的配置为准）

训练入口统一使用官方 [train.py](file:///f:/code/MyPython/TransUNet/train.py) + [trainer.py](file:///f:/code/MyPython/TransUNet/trainer.py)，只把数据换成了颈动脉数据集。

| 项 | 值 |
|---|---|
| 输入 | 224×224，单通道原图（单通道原图基线） |
| batch | **8**（`--batch_size 8`，8GB 内的实测峰值约 3.3GB） |
| epoch | **120**（`--max_epochs 120`；代码默认 200，队列显式覆盖） |
| 优化器 | **SGD**，lr 0.01，momentum 0.9，weight_decay 1e-4，poly 衰减（power 0.9） |
| 损失 | 0.5 × CE + 0.5 × Dice |
| 预训练 | **启用**：`R50+ViT-B_16.npz`（ImageNet21k） |
| 数据加载 | `num_workers=1`（Windows 内存限制），`persistent_workers=True`（避免每 epoch 重建 worker 造成 GPU 空转） |
| AMP | **尚未接入**（trainer 里没有 autocast/GradScaler）；batch 8 显存够用，暂不引入 |
| test 曲线 | 每 **10** 个 epoch 在测试集上算一次四项指标（`--eval_test_every 10`，0 关闭；代码默认 5） |
| 单次耗时 | 约 50–57 min（120 epoch / batch 8） |

### epoch 选择策略（用户指定）

1. 先用一轮 120 epoch 的训练，每 10 epoch 记录一次测试集指标，写入 `test_curve.csv`；
2. 从曲线上挑一个**整数 epoch**（Dice 高、HD95/ASSD 不过差的区间中点，避免贴住偶然峰值）；
3. 之后所有消融实验统一固定该 epoch 训练，不再在测试集上早停或调 epoch：原图通道+高频能量通道双通道早融合（E1）、共享权重双输入 + 跨支门控单点（E2）、其 concat 回投影对照（E3）、三级全开（E4）、Zhongnan 跨子集复现（E5），两个子集均同。

### epoch 选择（**2026-10-26 新口径已定：150**，全部后续实验复用）

依据：新口径（官方划分、mm 单位）单通道原图基线 `test_curve.csv`（200 epoch、每 5 epoch 一测，文件 `../model/TU_Carotid224/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_c2d/`）：

- 曲线 150 epoch 即饱和：dice 25→150 从 88.8% 缓升到 90.49%，150~200 区间在 90.4%±0.1% 内波动（噪声水平）；assd/hd95 在 150（0.1431/0.3896 mm）与 155（0.1427/0.3852 mm）达最优，之后不再改善；
- **固定 epoch = 150**（对应存档 `epoch_149.pth`，权重文件名 0 基）：四项指标在可选存档点（149/199）中全面更优，且不贴住单点偶然峰值（155 与 150 差异在噪声内）；
- 结论：200 epoch 中后 50 个 epoch 无实质贡献（dice 仅 +0.05%），**属曲线余量而非必要训练量**。后续 job 保持 `--max_epochs 200` 不变（poly 衰减总长绑死 max_epochs，A 在 200 调度下跑的 epoch 150，B/C/消融必须在同一调度下取 epoch 150 才可比），**但每个 job 训到出现 `epoch_149.pth` 即可手动停止**（省约 53 min/job），或任其跑完、读数仍取 150。


## 4. 运行命令

```powershell
# 生成病例级划分（每个子集独立，train/test）
.\venv\Scripts\python.exe datasets\dataset_carotid.py

# 夜间队列（顺序执行，一次只跑一个）
powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1 -DryRun   # 只打印计划
powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1            # 正式执行

# 融合算子横评 + DDBE 门槛 + 基线补跑（第三个接力队列，2026-10-28 00:54 已 WMI 启动 pid 34168，详见 §1.7）
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fuse.ps1 -DryRun   # 只打印 6 条计划
powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fuse.ps1            # 正式执行（等主/edge 队列全退出后自动接力）

# dev 快筛（**已撤销，仅保留备用**；16 个单变量 job，约 4 h）
# .\venv\Scripts\python.exe scripts\build_dev_subset.py --subset SPARC --ratio 0.4
# powershell -ExecutionPolicy Bypass -File scripts\night_queue_dev.ps1 -DryRun
# powershell -ExecutionPolicy Bypass -File scripts\night_queue_dev.ps1
# .\venv\Scripts\python.exe scripts\summarize_dev.py      # 汇总排序表

# 单通道原图基线（A0）：两个子集分别跑（官方 train.py，120 epoch / batch 8 / SGD / workers 1）
.\venv\Scripts\python.exe train.py --subset SPARC --max_epochs 120 --batch_size 8 --eval_test_every 10
.\venv\Scripts\python.exe train.py --subset Zhongnan --max_epochs 120 --batch_size 8 --eval_test_every 10
```

## 5. 日志与产物

- 输出目录：`../model/TU_Carotid224/TU_pretrain_R50-ViT-B_16_skip3_epo120_bs8_224_<subset>/`
  - `log.txt`：完整训练日志
  - `epoch_*.pth`：**后半程每 50 个 epoch** + 最后一轮（trainer.py 的 `save_interval = 50`）
  - `metrics.csv`：测试集逐样本四项指标（trainer 末尾自动执行）
  - `log/`：tensorboardX 事件
- 禁止复用单通道原图基线（baseline）的目录名，避免权重互相覆盖。
