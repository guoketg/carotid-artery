# 论文模块套件设计：频带改写家族的位置分工（2026-11-10）

配套代码：[networks/fgr.py](../networks/fgr.py)、[networks/wavelet.py](../networks/wavelet.py)（`laplacian_split_channels`）、
挂载点 [networks/unet.py](../networks/unet.py#L260-L269)（unet）与 [networks/vit_seg_modeling.py](../networks/vit_seg_modeling.py#L636-L640)（R50 解码器）；
设计守则：[study/fusion-design-rules.md](study/fusion-design-rules.md)；FGR 讲解：[study/fgr-explained.md](study/fgr-explained.md)。

## 0. 一页版

**问题**：两个过线模块 FGR（解码器四级输出门控改写）与 DDBE+lap（同点分带改写）在 R50 上串联互覆
（90.632 < 单臂 90.795 / 90.996）。**解法**：合并成一台机器——FBGR（分带门控改写器，M1），
参数量与 FGR 持平（每级 8C²+4C），再配一个位置正交的输入级边缘通道（M2，已有开关 `--edge_in`）。
论文消融阶梯：base → +M2 → +FGR(整带) → +FBGR(分带) → full。队列排程见 §7。

## 1. 冲突诊断：为什么 FGR + DDBE 同点串联互覆

**事实**（results.md §5(16)(17)、§7(4)）：

| 骨干 | DDBE-lap 单挂 | FGR 单挂 | 同点串联 | 组合−最佳单臂 |
|---|---|---|---|---|
| R50（预训练，饱和） | **90.996**（+0.506） | 90.795（+0.305） | 90.632（+0.142） | **−0.364，互覆** |
| UNet（从头，欠饱和） | —（abf 臂另证） | **88.248**（+1.703） | 88.386（+1.841） | +0.138，小幅互补 |

**机制**（三条，全部有实验证据）：
1. **同族算子**：DDBE 与 FGR 都是「拉普拉斯分带 → 通道重加权 → 融合」。串联 = 同一位置对同一特征
   连做两次改写，第二个机器看到的输入统计已被第一个机器改变（统计失配）。
2. **残差预算竞争**：两台机器各自要吃同一份「改写残差」。R50 特征已饱和（§6：预训练强骨干频带内容
   补不进空间），预算趋零，两机器互相拉扯 → 比单挂更差；UNet 欠饱和、预算大 → 各吃一点、小幅互补。
3. **门控串叠**（守则 6）：每个模块自带一个 sigmoid 门，串联后梯度要穿两个门，有效梯度衰减。

**推论**：冲突的根因是「同点两台机器」，不是「两个模块都有用」。解法要么合并成一台（M1），
要么位置切分互不重叠（B 方案，见 §3.5）。

## 2. 设计约束（来自守则文档，逐条遵守）

- 通道轴逐位置 > 空间轴注意力（守则 4）→ 新模块全部走通道轴；
- 改写式 > 注入式（守则 3）→ 不做「把频带图回注到主干」类算子（R-GEF/MREF/FEM/ABF 全判负）；
- 拉普拉斯幅值 > Haar 定向子带（守则 8）→ 分带继续用 `laplacian_split_channels`，hf = |LH|+|HL|+|HH|；
- 一条链只留一个零（守则 7）→ 每个挂载点全链只允许一组零初始化融合核；
- 门控有代价（守则 6）→ 不新增门，复用 FGR 现有单门；
- 轻机器够用（守则 5）→ M1 参数增量 ≈ FGR（+2.79M on UNet），M2 为 0（换输入通道）。

## 3. M1：FBGR —— 分带门控改写器（主模块，统一 DDBE+FGR）

### 3.1 定位与公式

挂载点不变（解码器四级输出，与 FGR 相同），只改 FGR 内部的融合核粒度——**整带融合 → 分带融合**：

```text
FGR（现状，fgr.py L73）：  y = x + fuse( w ⊙ [hf, lf] )          fuse: 2C→C，两带混在同一个核里
FBGR（新）：               y = x + fuse_hf( w_hf ⊙ hf ) + fuse_lf( w_lf ⊙ lf )
                           w_hf, w_lf = σ(Linear(z)) 按 C 切开；z = DCT 谱描述子([x,hf,lf])（不变）
```

差异只有一个：**hf 与 lf 各走独立的 1×1 改写核，重组时保持频带纯度**（DDBE 的「分带各自处理」思想
并入），门与谱描述子完全复用 FGR 的（不新增门，守则 6）。

### 3.2 初始化与恒等起步

- `gate`：零权重 + bias 2.0（w≈0.88 全开，继承 FGR，[fgr.py L71-72](../networks/fgr.py#L71-L72)）；
- `fuse_hf` / `fuse_lf`：全零初始化 → y = x 逐位恒等（唯一零点 = 一组并联零核，非串联双零）；
- **冒烟已全绿**（2026-11-10，[scripts/smoke_fbgr_band.py](../scripts/smoke_fbgr_band.py)，CPU-only）：
  UNet fgr_whole / fgr_band、R50 fgr_band、full 臂（hfe+band）与各自基线 eval 下 logits 逐位相等
  （恒等比对必须 eval——ViT 嵌入层 Dropout(0.1) 在 train 模式下随机 mask 会造成假失配）；
  参数差 == ΣC（UNet band−whole +960：60.299M → 63.086M；R50 网络差 == 四级模块和：
  20.222M → 20.914M）；梯度第 0 步直达 fuse_hf/fuse_lf（gate 数学上为 0，fuse 移动后第 1 步起
  gate 有梯度，门不被冻死）；旗标接线（band 不带 --fgr 1 → parser.error）。

### 3.3 参数增量

| 项 | FGR（整带） | FBGR（分带） |
|---|---|---|
| gate Linear(3C→2C) | 6C²+2C | 6C²+2C（共用） |
| fuse | 2C²+C（一枚 2C→C） | 2C²+2C（两枚 C→C） |
| 每级合计 | 8C²+3C | 8C²+4C |

UNet 四级（C=512/256/128/64）共 +2.789M，比 FGR 多 960 个 bias，参数量叙事不变。

### 3.4 对照档位（论文表直接可用）

| 档 | 配置 | 回答的问题 |
|---|---|---|
| plain concat | 无模块 | 基线 |
| DDBE-lap | `--bdfr_v2 1 --bdfr_wave_op lap` | 分带改写（文献算子） |
| FGR 整带 | `--fgr 1` | 频带门控改写（自研 v1） |
| FBGR 分带 | `--fgr 1 --fgr_mode band` | **频带纯度是否是缺失的那块收益**（自研 v2） |
| FBGR+DDBE 串联 | 两者同开 | 复核互覆是否消失（预期仍判负，作反证） |

### 3.5 风险与 B 方案

- **风险**：FBGR ≈ FGR（分带不带来增益）→ 论文退回 FGR 单模块 + 骨干饱和度分析，故事仍成立但弱一档。
- **B 方案（位置切分）**：DDBE 只挂最深 2 级（28²/14²）、FGR 只挂浅 2 级（112²/56²），零重叠。
  实现便宜（levels 掩码），但「两台机器」叙事不如一台统一机器，仅作 FBGR 失败后的备选。

## 4. M2：输入级高频边缘通道 HFE（位置正交的第二个模块）

- **做法**：stem 吃 4 通道 = 原图 3 通道 + 1 张 Haar 高频重建边缘图。开关已实现：
  `--edge_in 1 --edge_op wavelet`（[networks/wavelet.py](../networks/wavelet.py) `build_hf_image`），
  **零新增参数**，只换输入通道数。
- **证据**：先行诊断（小网络 = 弱骨干口径）SPARC +1.81 / Zhongnan +1.83，双子集一致（results.md §3）；
  R50 正式训练 −0.01（强骨干不需要，预训练 stem 已学到该信息）；**UNet 已测 87.42（+0.87，
  归档 A.4 C 组，与 §0 同口径：unet base 86.55 / 结构参数量完全一致 / 只换第 2 通道算子）——
  单档已过 Dice 门槛（HD 1.16 未改善）**。所以 M2 不重跑，第三臂改跑 full 档（§7）。
- **与 M1 的正交性**：作用位置是 stem（输入级）vs 解码器级输出，无共享特征流、无门控串叠，
  不构成同点串联，冲突机制（§1）在此不适用。
- **判定**：UNet 上 |ΔDice| ≥ 0.5 或 HD 降 ≥ 0.1 mm 才入论文表；R50 结果作「骨干饱和度」分析的对照行。

## 5. 备选 M3：跨级频率上下文（暂缓，不实现）

把深层级的谱描述子 z 传给浅层门控作额外输入（跨尺度频率语境）。位置/轴均与 M1 不冲突，
但**无任何既有证据**且加深门控输入（守则 6 风险）。M1/M2 落地后再议。

## 6. 论文消融阶梯与故事线

故事线一句话：**「频带改写机器的收益由骨干饱和度决定：欠饱和骨干吃频带内容，饱和骨干只吃分带纯度」**
——§1 表格本身就是论文的分析性证据（互覆 vs 互补随骨干切换）。

UNet 底座消融阶梯（每行都是已排期或已有数字）：

| 行 | 配置 | 数字状态 |
|---|---|---|
| base | U-Net | 86.5455（已有） |
| +M2 HFE | `--edge_in 1` | 87.42（已有，归档 A.4 C 组；+0.87 过 Dice 门槛） |
| +FGR 整带 | `--fgr 1` | 88.2484 单 seed；多 seed 2024/3407 在跑 |
| +FBGR 分带 | `--fgr 1 --fgr_mode band` | 已实现，FBGR-U 在队 |
| full | HFE+FBGR | FULL-U 在队 |

跨骨干表（R50 底座）：DDBE-lap / FGR / FBGR / 串联互覆四行，支撑故事线后半句。
跨数据集：Zhongnan（unet 基线由 follow-up 队列补，FGR 臂在跑）。

## 7. 排程与判定门槛

1. **在跑**（不等设计）：`queue_fgr_multiseed_zn` 主队列收口约 23:10，follow-up Zhongnan unet 基线约 00:30。
2. **实现**（已完成，2026-11-10）：[networks/fgr.py](../networks/fgr.py) 加 `mode="band"` 分带档 +
   [train.py](../train.py) `--fgr_mode` 旗标（快照段名 `_fgrb` 与历史 `_fgr` 分目录）+ 两个挂载点透传
   （[unet.py](../networks/unet.py#L268-L272)、[vit_seg_modeling.py](../networks/vit_seg_modeling.py#L639-L643)）；
   冒烟全绿（§3.2，恒等起步 / 参数差 == ΣC / 梯度链路 / 旗标接线）。
3. **入队**（已完成，2026-11-10 21:09 WMI 启动 `queue_fbgr_full.ps1`，PID 39212，等 ZUB 收口后自动接力；
   3 job 全 SPARC s1234，全走 band 档）：FBGR-U（快照 `..._SPARC_fgrb_unet_c2d`）、
   FBGR-R50（`..._vitL0_..._SPARC_fgrb_c2d`）、**FULL-U = HFE+FBGR 同开**
   （`..._SPARC_edge_fgrb_unet_eowavelet_c2d`）。第三臂原定 HFE-UNet，因其同口径结果已存在
   （87.42 / +0.87，归档 A.4 C 组）经用户决策改跑 full 档。
4. **判定门槛**（判定协议不变）：单 seed |ΔDice| ≥ 0.5 或 HD(medpy) 降 ≥ 0.1 mm。
   - FBGR-UNet 对照 88.2484（FGR）：过线 → 主模块换 FBGR，不过线 → 退回 FGR（§3.5 风险）；
   - FBGR-R50 对照 90.996（DDBE-lap）：不劣于最佳单臂即算统一成功，过线则为强证据；
   - FULL-U 对照 87.42（+HFE 单独）：FBGR 过线且 full > 单 FBGR → 两模块互补叠加成立。

## 8. 结论边界

- 能说：互覆机制有 R50/UNet 双骨干数字支撑；FBGR 与 FGR 参数量持平、恒等起步可证；
  M2 在 UNet 上已有单 seed 正式结果（87.42 / +0.87 过 Dice 门槛，HD 未改善，归档 A.4 C 组）。
- 不能说：FBGR 一定能吃到 DDBE 的分带收益（FBGR-U / FBGR-R50 在队）；
  HFE 与 FBGR 是否互补叠加（FULL-U 在队）。
- 本设计不改判定协议、不加门、不新增空间轴算子，全部构件都有已跑通的代码路径。
