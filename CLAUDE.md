# CLAUDE.md — 颈动脉斑块超声分割

> 本文件只保留**必须遵守的硬约束**。设计细节、数据集说明、实验方案分别在 `docs/` 下，避免主文件膨胀。

## 1. 项目

- 基座：TransUNet（R50-ViT-B_16 + U-Net 解码）。
- 任务：**B 型超声颈动脉斑块分割**（2D 切片，二值前景）。
- 路线：小波预处理分支 + 原图分支 → 注意力融合 → 分割，验证指标是否显著提升。

## 2. 环境（默认本地 venv）

```powershell
.\venv\Scripts\python.exe <script.py>        # 直接调用，无需激活
.\venv\Scripts\Activate.ps1                  # 或激活后使用
```

- Python 3.11.5，torch 2.14.1+cu132，RTX 3060 12GB。（旧机器 RTX 5060 Laptop 8GB 已弃用，docs 中标注"估算/8GB"的历史实测均来自旧机器）
- 不新建 conda 环境，不使用 Docker。
- 新依赖只装进 `venv`，并同步更新 `requirements.txt`。
- **禁止** 用旧版 `torch==1.4.0` 覆盖现有环境。

## 3. 硬约束

1. 数据划分**必须按病例**（patient-level），禁止切片级随机划分。
2. 实验输出不得覆盖单通道原图基线权重，目录名带方法后缀。
3. 训练显存上限 12GB（RTX 3060）：`img_size=224`、`batch<=8`；不够就用梯度累积。注意 Windows WDDM 换页：峰值逼近专用显存上限时单步耗时会翻倍级劣化，别把显存吃到顶。
4. 数据加载固定 `num_workers=1`，不要调大：本机内存紧张，WSL 同时开启时会冲到 95%，需要用户手动杀进程。
5. 指标口径固定：**Dice、ASSD、HD95、DeltaArea**（预测面积 − 真值面积）。
6. 冒烟测试样本 ≤ 50，避免无意义开销。
7. 关键决策、路径、超参变更必须回写 `docs/` 并在本文件索引。
8. epoch 选择：先用 `--eval_test_every 5` 跑出 test 曲线，挑一个整数 epoch 固定；后续所有消融复用该 epoch，不再反复在测试集上调。
9. **严禁用代号**（如 `A1`、`A3`、`C1/C2/C3`、`D11`）指代方案或通道：文档、代码注释、对话回复、提交说明里都要先写清具体做法，例如「原图＋去噪重建图＋高频能量图三通道、无额外融合模块」。实验编号只能出现在附录/对照表中，且同一处必须给出对应的自然语言描述；单独出现一个编号视为不合格。
10. **代码修改默认由用户自己完成，agent 不直接改**：涉及 `*.py` / `*.ps1` 等代码文件的修改，agent 的默认职责是给出精确到行号与完整内容的改动步骤，由用户在 IDE 里自行修改；除非用户明确要求「由 agent 来改」，否则 agent 不得动任何代码文件。**任何代码、脚本、训练或评估命令同样未经用户明确允许不得执行**；只读查看（读文件、搜索、列目录）不受限。`docs/` 与本文件的文档回写按第 7 条照常进行。此约束的原因：用户的目的是真正掌握代码——即使改动是自己复制粘贴的，亲手执行每一步才知道它做了什么；若 agent 完全代改，用户什么也学不到。
11. **禁用专家类、可视化类 skill 调用**：不调用深度研究/专家类技能，也不调用可视化类技能（PureShowWidget、dynamic-ui、图表渲染等）；所有回答一律直接用文字与 markdown 表格给出，除非用户当次明确要求。原因：这类调用启动与渲染开销大、浪费时间。

## 4. 文档索引

| 文档 | 内容 |
|---|---|
| [docs/imaging-basics.md](docs/imaging-basics.md) | 图像类型入门：B 型图、FFT、小波、低频/高频、原图通道、去噪重建通道、高频能量通道的图文解释 |
| [docs/technical-design.md](docs/technical-design.md) | 小波预处理、三通道、分支结构、注意力融合的设计决策与理由 |
| [docs/dual-encoder-design.md](docs/dual-encoder-design.md) | **双骨干交错四级融合落地版**：双骨干交错执行、四级双向融合的数据流、公式、skip 替换、参数/显存/耗时实测 |
| [docs/dataset.md](docs/dataset.md) | 数据集结构、划分规则、预处理与归一化 |
| [docs/literature.md](docs/literature.md) | 文献正式条目（venue / 表号 / 与本项目关系）+ 负面结果汇总 + 待读清单 |
| [docs/paper-reading.md](docs/paper-reading.md) | 频域论文速查笔记；第五章为 2026-10-26 对本地 8 篇 PDF 的精读校正、结论冲突拆解、融合六档与推进顺序 |
| [docs/experiments.md](docs/experiments.md) | 消融矩阵、评估协议、统计检验、运行命令 |
| [docs/status.md](docs/status.md) | 环境与权重校验记录、进度清单、已知问题 |

## 5. 当前状态

- 环境、预训练权重（R50+ViT-B_16）、依赖、Windows `pjoin` 修复：已完成，见 `docs/status.md`。
- 单通道原图基线与三通道早融合、三通道＋小波引导注意力的消融已完成；**三通道＋小波引导注意力相对单通道原图基线无显著增益**（p > 0.29），旧轻量侧支路线已否。
- **双骨干交错四级融合**已实现并通过探针验证。融合分支此前因「输出投影零初始化 + γ 零初始化」双重零相乘永久冻结，2026-10-25 已修；最深 14² 接点「侧路无下游消费者」的结构性死支路也已改为串联双向（先更新侧路、主路再用 `qkv_b(B2)` 重取 K/V），探针 `max|Δ|=0`、`gamma_b` 第 1 步起有非零梯度。修复后跑完的首个活融合配置（`--fuse_op window`，前三个接点并行双向、最深接点当时仍是 A→B 单向）Dice 0.9051，**相对单通道基线 0.9092 无增益、四项指标全差**，但存在 `bs8` vs `bs4+ga2` 口径不一致与单 seed 两个混杂，详见 `docs/status.md` §4.5.6。
- **2026-10-24 夜队列（只跑全量 SPARC 单子集）**：`scripts/night_queue_dual.ps1 -Skip r2_sparc,r3_sparc,r7_sparc,*zhongnan` → `scripts/night_queue_dual2.ps1 -Skip *zhongnan`，实际顺序为四点全开·窗内/全局自适应融合 → 可变形大核 → AFF 式幅度调制 → 自适应融合＋跨尺度回流（队列中依次记作 `r1` / `r4` / `r5` / `r6`）。
- **实测耗时**：全量 SPARC 0.70 s/iter（107 iter/epoch ≈ 75 s）→ **2.5 h/job**；瓶颈是 dataloader（`num_workers=1`），GPU 仅 63% util。旧文档“34.6 s/epoch、72 min”是错的。
- **dev 快筛子集（已撤销）**：`lists_Carotid_dev` + `scripts/night_queue_dev.ps1` + `scripts/summarize_dev.py` 保留备用，当晚未使用，中间产物已删。
- 训练超参变更：**batch 4 + `--grad_accum 2`**（等效 8）；batch 8 触发 WDDM 换页，单步 2.6×，禁用。
- **2026-10-26 频域文献复盘**：本地 8 篇 PDF 精读完毕，XNet（ICCV2023）与 GobletNet（TMI2025）结论不矛盾——XNet 的 raw 基线是单编码器（增量混容量），GobletNet 参数量对齐后只剩「高频有用、低频有害」。据此确定：**变换种类不是主要矛盾，注入位置才是**；不再新增更重的融合算子，下一轮按 Step 0（打印 `γ` + HF 图改 IDWT 重建）→ Step 1（`bs4+ga2` 基线 3 seed）→ Step 2（stem 加一张 HF 通道，+27 参数）推进，详见 `docs/status.md` §4.6 与 `docs/technical-design.md`「频率注入位置」。
- **2026-10-26 老师指示（改排期，代码未动）**：① 新数据集口径——根目录 `F:\code\datasets\Carotid2D`（`_S` 目录）＋官方划分 `datasets/split/`（SPARC：训练 350 张/约 101 例、测试 160 张/约 43 例；Zhongnan 448/187 张）。旧划分弃用，**全部历史数字只在旧口径下有效**；接入需小改代码（`train.py` 的 root_path 写死 + 划分文件转成 `<list_dir>/<Subset>/train.txt` 结构），明细见 `docs/dataset.md` §7，待批准。② 下一结构实验先做**边缘特征融合**，插队到「stem 加 HF 通道」与「特征级 DWT 解耦」之前，读法与最小设计见 `docs/technical-design.md`「频率注入位置」待决策第 4 条，实施待批准。
- **2026-10-26 新口径落地（冒烟通过，基线重跑中）**：① Carotid2D 接入完成——root/list_dir 已切、划分四文件就位、快照带 `_c2d` 后缀，冒烟 n=160 与官方一致；② **指标物理单位化**——预测经 `original_grid_pair` 重采样回原分辨率网格后用 NIfTI 头 spacing 算 medpy 距离，**assd/hd95 单位 mm、DeltaArea 单位 mm²**，历史数字均为像素口径不可直接比（口径断代见 `docs/dataset.md` §6、`docs/status.md` §4.6.6）；③ XNet 式低频主路已实现（`--main_input ll`，快照带 `_mainLL`），待基线跑完后与双骨干原图对照。新口径实验命令：`--batch_size 4 --grad_accum 2 --max_epochs 200 --eval_test_every 5`。
- **2026-10-28 三队列收口（SPARC，官方划分，ep150 判定，43 例 Wilcoxon）**：① 双骨干 QKV 跨支注意力三 seed 判负——3-seed 均值 0.9012 vs 单骨干基线 0.9016，seed3407 显著变差（Dice p=0.0036）；② 融合算子五档横评齐：deform 0.9061 唯一三项全优但未过门槛（p=0.113/0.154/0.276），gate 与 auto 同噪声带，aff 垫底 0.9017；③ BDFR-Net 复现：单级挂载判负，四级全挂 0.9033、+ACAM 完整版 0.9053 方向为正（Wilcoxon p=0.462/0.288/0.876 与 0.334/0.419/0.731，均未过门槛，已补齐——eval 脚本 `-Only` 下 `$snapOf` 构建在过滤之后的 bug 已修）；④ 主路换 LL 低频图与原图无差异（0.9043 vs 0.9040，p=0.447/0.742/0.426）；⑤ 输入级边缘图单用无效、配双骨干 0.9072 全场最佳点（p=0.069 边缘）。详见 `docs/status.md` §3、`docs/study/fusion-operators-explained.md` 与 `deform-fusion-explained.md`。
