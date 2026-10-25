# 状态与环境校验记录

## 2026-10-23/24 夜间队列

- 已实现：`networks/wavelet.py`（Haar DWT/IDWT + 软阈值 + 三通道）、`networks/wavelet_model.py`（小波分支 + 小波引导的交叉注意力 / SE 对照）；`train.py` 新增 `--use_wavelet`、`--fusion {none,se,cross}`、`--heads`，默认 `batch_size=8`。
- 融合位置为 **ViT 输出瓶颈（14×14，196 token）**，不是原设计的 stage2/3 skip：实现简单、注意力开销低，单通道原图基线完全不受影响。偏差已记入 `technical-design.md`。
- batch=8 实测：峰值显存 3.3GB，纯模型单步 0.36s，含数据约 0.5s，SPARC 约 27s/epoch。
- 夜间队列 6 个任务，统一 `--max_epochs 120 --eval_test_every 10`，顺序 a0_sparc → a3_sparc → a1_sparc → a0_zhongnan → a3_zhongnan → a1_zhongnan（依次为单通道原图基线、三通道＋小波引导注意力、三通道早融合，SPARC 与 Zhongnan 各一轮），预计 6 小时。
- 冒烟（1 epoch，仅验证链路）：三通道早融合 dice 0.745，三通道＋小波引导注意力 dice 0.756。
- 2026-10-24 12:30 起训练 DataLoader 加 `persistent_workers=True`：实测 GPU 每 epoch 空转 6–7 s（GPU 利用率 0%、功耗 6W，worker 进程被销毁重建），约占 wall time 25%。**早期的单通道原图基线、三通道＋小波引导注意力、三通道早融合三个 job（`a0_sparc` / `a3_sparc` / `a1_sparc`）为旧配置，`a0_zhongnan` 及之后为新配置**；该改动使增强随机序列跨 epoch 变化（旧配置每 epoch 重置 seed，增强序列完全相同）。

## 1. 环境

| 项 | 值 |
|---|---|
| 解释器 | `f:\code\MyPython\TransUNet\venv\Scripts\python.exe` |
| Python | 3.12.10 |
| torch / torchvision | 2.14.1+cu132 / 0.29.1+cu132 |
| GPU | RTX 5060 Laptop，8GB |
| 关键依赖 | numpy 2.5.2、scipy 1.18.1、h5py 3.16.0、nibabel 5.4.2、SimpleITK 2.5.6、ml-collections 1.1.0、MedPy 0.5.2、tensorboard 2.21.0、tensorboardX 2.6.5、tqdm 4.70.1 |

`requirements.txt` 已按上述版本重写，不再固定 `torch==1.4.0`。

## 2. 预训练权重

- 路径：`f:\code\MyPython\model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz`
- 大小 461,217,452 字节；358 个数组；键前缀 `conv_root / block1~3 / embedding(1,1,1024,768) / Transformer`，确认 R50 混合版。
- 下载源：`https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz`（README 称失效，实测可用）。
- 校验：Params 105.28M，输入 `2×3×224×224` 输出 `(2,2,224,224)`，CPU 前向 0.59s。

## 3. 已修复问题

| 问题 | 处理 |
|---|---|
| TransUNet 在 Windows 下加载 npz 报 `KeyError: 'Transformer/...\kernel'` | `vit_seg_modeling.py` 与 `vit_seg_modeling_resnet_skip.py` 的 `pjoin` 改为 `"/".join(args)` |
| `requirements.txt` 固定 torch 1.4.0 | 按实际 venv 重写 |
| git origin 指向原仓库 | 改为 `https://github.com/guoketg/carotid-artery` |
| 训练结束写 `metrics.csv` 时重建 `CarotidDataset` 漏传 `use_wavelet`，导致小波模型用非小波输入评估（SPARC_w3 0.8906 vs 实际 0.9079；Zhongnan_w3_cross 0.8579 vs 0.9077） | 2026-10-24 15:18 在 `trainer.py` 末尾评估处补上 `use_wavelet=bool(args.use_wavelet)`；新增 `scripts/eval_carotid.py`（支持 `--use_wavelet/--fusion` 与 `--baseline_csv` 配对检验），已重算全部 6 个模型的 `metrics.csv`（`test_curve.csv` 不受影响，可信） |

## 4. 进度

- [x] 确认代码基座与环境
- [x] 下载并校验预训练编码器
- [x] 确认数据集位置与格式（641 例 / 1148 张）
- [x] 补齐 venv 依赖
- [x] 重写 `requirements.txt` 并切换 origin
- [x] 修复 Windows 路径分隔符 bug
- [x] 接入数据集与病例级划分
- [x] 单通道原图基线跑通并产出四项指标（SPARC / Zhongnan 分开）
- [x] 训练入口已切回官方 `train.py`（**120 epoch / bs8 / `--eval_test_every 10`** / SGD / workers1 / 预训练启用）
- [x] 实现 `networks/wavelet.py`（原图通道、去噪重建通道、高频能量通道）
- [x] 实现旧版小波引导注意力（`networks/wavelet_model.py`，仅瓶颈 14×14 单点融合）
- [x] 完成单通道原图基线、三通道早融合、三通道＋小波引导注意力三组消融与统计检验（三通道＋小波引导注意力相对基线：SPARC −0.0005、Zhongnan +0.0017，p > 0.29，**无显著增益**）
- [x] 文献检索第一轮（**11 篇**，多数已读正文与消融表，结论落盘 `docs/literature.md`）
- [x] 文献补读：**MBFF-Net**（MICCAI 2021）与 **MS-Net**（ISBI 2021）PDF 全文，含 Table 1/2/3，条目已写入 `literature.md` 的「双骨干 / 双输入融合」与「skip 融合与注意力门控」两节

## 4.1 下一轮待办（2026-10-24 记录，**代码尚未改动**）

详见 `technical-design.md` 的「代码改动清单」与 `experiments.md` 1.3。

- [ ] `networks/wavelet.py`：`build_three_channel` 支持 `layout`（`c1c3` 即原图＋高频能量两通道、`c1c2c3` 即原图＋去噪重建＋高频能量三通道）与 `detail_level`（1/2），两通道时第三通道置零
- [ ] `datasets/dataset_carotid.py`：`use_wavelet` → `wavelet_layout` + `detail_level`
- [ ] 新建 `networks/dual_input_model.py`：共享权重双输入 + 浅层 112×112、中层 56×56、深层 28×28 三级跨支门控（γ 零初始化），`fuse ∈ {gate, concat}`
- [ ] `train.py`：新增 `--wavelet_layout`、`--detail_level`、`--dual_input`、`--levels`、`--fuse`，输出目录追加方法后缀
- [ ] 冒烟四件套：导入/参数量、通道契约（`c1c3` 第三通道全零、`c1c2c3` 与旧输出逐元素一致）、显存（batch 4 双输入 ≤ 5.5GB）、等价性（γ=0 时与单通道原图基线误差 < 1e-5）
- [ ] 夜间队列依次跑：原图＋高频能量两通道早融合（单骨干、无注意力，第三通道置零）→ 共享权重双输入＋跨支门控（仅深层 28×28 接点）→ 共享权重双输入＋拼接与 1×1 回投影（仅深层 28×28 接点）→ 共享权重双输入＋跨支门控（浅/中/深三级全开）（SPARC），达标后把胜出配置搬到 Zhongnan
- [ ] 视结果决定是否实现 `networks/enhance.py`（CLAHE 对比度增强 / 同态滤波）与二级小波的高频能量
- [ ] 检索第二轮：**HDL、Zhou 2021、MF-Net、MulFF-Net、Wavelet-enhanced boundary adaptation network（共 5 篇，清单见 `literature.md`）**——MBFF-Net 与 MS-Net 的 PDF 用户已提供并读完

## 4.2 2026-10-24 方案修订：双骨干强融合（**讨论记录，代码未改**）

- 触发：用户判断「0.9M 轻量侧支 + 单点门控」不足以支撑结论，要求双骨干与更强融合。
- **事实修正**（写入 technical-design.md 双骨干方案的「触发原因」一节）：旧实现主路吃原图＋去噪重建＋高频能量三通道、侧路又只吃高频能量通道，问题在于**第二路没有新信息**，不是「双骨干参数冗余」。同时更正此前「两条路输入重复」的表述——LH/HL/HH 与原图通道不是同一份输入，真冗余的只有高频能量通道。
- 已落文档：
  - [technical-design.md](technical-design.md) **双骨干交错四级融合**方案（输入契约、三种骨干配置方案、四个接点、融合算子五档分级、工程量级）
  - [experiments.md](experiments.md) **1.4**（探针验证、各接点与对照项的判定规则，未排期）
- **2026-10-24 二次修订（用户要求）**：
  - 输入端**不再拼原图、去噪重建、高频能量三通道**：主路只吃原图通道（复制 3 通道），侧路只吃 `LH/HL/HH`；去噪重建通道与高频能量通道退出输入，仅作消融。
  - 四个接点改为**双向 + 按 stage 交错执行**：前稿「侧路只供 K/V、只替换 skip」作废，因为那是旧版三通道＋小波引导注意力式的单向供体。
  - 新增对照「只保留侧路到主路的单向融合」验证双向必要性。
- 已改的旧条目：「两套独立 R50+ViT 双骨干」由「最后做、只量化容量」升级为候选主线（technical-design.md 编码条目里的第二项）；「浅层 112×112、中层 56×56 只能门控」的旧限制被窗内注意力推翻（融合条目）。
- 关键量化依据：MSFA-Net Table 4（PDF p.12）「加第二路」+8.56 Dice，「加/换融合模块」仅 +0.45~2.18 → **第二路优先于融合算子**。
- **尚未做**：任何代码改动；显存探针验证未跑；骨干配置方案未选定。

## 4.3 2026-10-24 双骨干交错四级融合落地（**已实现，夜间队列运行中**）

- 新增 `networks/dual_encoder_model.py`：按 stage 交错的双骨干 + 四级双向跨支融合（112×112/64、56×56/256、28×28/512、14×14/768 四个融合接点），主路只吃原图通道、侧路吃 LH/HL/HH 三个方向子带，侧路 BN→GN，γ 零初始化。
- `train.py` 新增 `--arch dual`、`--levels`、`--fuse_op {auto,gate,concat}`、`--window`、`--dk_ratio`、`--bidir`、`--side_pretrained`、`--grad_accum`；输出目录追加 `_bs{batch}` / `_ga{accum}`，不覆盖单通道原图基线权重。
- `trainer.py` 接入梯度累积：`grad_accum=1` 时与原行为完全一致，保证单通道原图基线、三通道早融合、三通道＋小波引导注意力三组结果可复现。
- 详细数据流、融合公式、skip 替换与实测表见 [dual-encoder-design.md](dual-encoder-design.md)。
- **探针验证**：总 121.66M（主 105.28 / 侧 12.68 / 融合 3.70；112×112 接点 0.017、56×56 接点 0.264、28×28 接点 1.053、14×14 接点 2.366）；`γ=0` 与单通道原图基线逐元素 `max|Δ| = 0`。
- **显存/单步**：batch 2/4/6/8 = 2.13GB/267.9ms、3.32GB/319.8ms、4.21GB/416.3ms、5.01GB/1101.6ms。
  batch 8 超过 8GB 卡的可用显存后进入 WDDM 换页（共享显存走 PCIe），单步 2.6× 惩罚，**不是算力瓶颈**。结论：**batch 4 + `--grad_accum 2`**（等效 8，0.64 s/optimizer-step，峰值 3.32GB）。
- **冒烟**：SPARC 1 epoch 57 s（含 82 张测试推理），dice 0.7948 / assd 9.92 / hd95 24.50 / abs_delta_area 4028.4，`metrics.csv`、`test_curve.csv` 正常产出。
- **耗时（2026-10-24 实测修正）**：早先按 34.6 s/epoch 估的 72 min/job 是错的。全量 SPARC 实测 **0.70 s/iter × 107 iter ≈ 75 s/epoch → 120 epoch ≈ 2.5 h/job**。`nvidia-smi` 同期只有 63% util / 34 W / 1560 MHz，瓶颈是 dataloader（`num_workers=1` + `nib.load`），不是算力；`num_workers` 受 CLAUDE.md 约束不能调大。
- **夜间队列** `scripts/night_queue_dual.ps1`：四点全开·窗内/全局自适应融合（`r1`）→ 跨支门控（`r2`）→ 拼接回投影对照（`r3`）→ 仅侧路到主路单向（`r7`），各跑 SPARC + Zhongnan，共 8 job（按实测口径 ≈ 20 h）。状态写 `logs/queue/queue_dual_status.csv`。
- 已知坑：`fuse_op=auto` 的档位判据必须是 `res >= 56` 用窗内、否则全局；写反时 112×112 融合接点误用全局注意力，单次 265ms（正确应 6.0ms）。
- 已知坑：`ResNetV2.maxpool(3,2,0)` 使 112→55，skip 侧零填充到 56 再进解码器，主干继续用 55²。

### 4.3.1 强融合算子追加（同日，**已实现并排队**）

- 新增 `BidirDeform`（D-LKA 式可变形大核，`--fuse_op deform`）、`BidirAFF`/`_MSCAM`（AFF CVPR 2021，`--fuse_op aff`）、跨尺度回流 `--cross_scale 1`（BiFPN 式 14²→28²→56²→112²）。公式与归因见 [dual-encoder-design.md](dual-encoder-design.md) 3.4。
- 零初始化等价性全部通过：deform 121.65M / aff 118.43M / auto+cs 122.20M，`max|Δ| = 0.000e+00`。
- 修过一个实现 bug：`BidirDeform._dir` 原把残差加在降维后的 `d` 维张量上（32 vs 64 通道不匹配），改为加回原始 `dim` 维特征。
- `scripts/night_queue_dual2.ps1`：等队列 1 结束后 GPU 空闲 10 min 自动启动，可变形大核（`r4`）→ AFF 式幅度调制（`r5`）→ 自适应融合＋跨尺度回流（`r6`），先 SPARC 后 Zhongnan，6 job；状态写 `logs/queue/queue_dual2_status.csv`。

### 4.4 2026-10-24 夜：**只跑全量 SPARC 单子集**（dev 快筛方案已撤销）

- 用户意图修正：原计划“SPARC + Zhongnan 两子集全跑”改为**只用一个子集（SPARC）**，问的“一个数据集”指的是子集，不是 dev 快筛。
- **dev 快筛方案（23:25 提出、23:50 撤销）**：`scripts/build_dev_subset.py` / `lists/lists_Carotid_dev` / `scripts/night_queue_dev.ps1` / `scripts/summarize_dev.py` 均保留备用，但**今晚不使用**；已产出的 `_dev` snapshot、`queue_dev_status.csv`、`dev_screening_summary.md` 全部删除。撤销原因：全量 SPARC 单 job 只需 2.5 h，一晚上够跑 3 个完整方案，不需要降级到子集。
- `train.py` 的 `--list_dir`（默认空 → 回退数据集配置）与 `_dev` 后缀逻辑保留，对全量路径无影响。
- **今晚实际队列（23:50 启动）**，按实测 2.5 h/job 排期：
  | 顺序 | job | 命令要点 | 预计完成 |
  |---|---|---|---|
  | 1 | `r1_sparc` | `--fuse_op auto`（窗内/全局自适应） | ~02:20 |
  | 2 | `r4_sparc` | `--fuse_op deform`（可变形大核） | ~04:50 |
  | 3 | `r5_sparc` | `--fuse_op aff`（AFF 式幅度调制） | ~07:20 |
  | 4 | `r6_sparc` | `--fuse_op auto --cross_scale 1`（自适应融合＋跨尺度回流） | ~09:50（顺延） |

  启动方式：`night_queue_dual.ps1 -Skip r2_sparc,r3_sparc,r7_sparc,r1_zhongnan,r2_zhongnan,r3_zhongnan,r7_zhongnan` → 结束后 `night_queue_dual2.ps1 -Skip r4_zhongnan,r5_zhongnan,r6_zhongnan`。
- 未排：跨支门控、拼接回投影、仅侧路到主路单向三个对照（队列中记作 `r2` / `r3` / `r7`）与全部 Zhongnan job，等白天的结果决定是否补跑。
- 对比基准：已有的单通道原图基线（全量 SPARC，120 epoch）。
- **2026-10-25 08:30 重启**：宿舍夜间断电，23:50 那轮只跑到 epoch 20（`test_curve.csv` 有 epoch 10/20 两点，未存权重）即中断。残目录 `..._SPARC_dual_L1234_auto` 已删，队列按同样顺序重跑（自适应融合 → 可变形大核 → AFF 式幅度调制 → 自适应融合＋跨尺度回流）。白天实测 0.62 s/iter ≈ 67 s/epoch → **2.2 h/job**，四个 job 预计 ~17:30 全部结束。
- 事故记录：23:22 停 wrapper 时整棵进程树被杀，正在跑的自适应融合 job（`r1_sparc`，约 7 epoch）丢失，已重启；23:47 又因两子集方案重启一次，第一版 `..._auto` 残目录已删。

## 5. 已知风险

- 8GB 显存：ViT-B 在 224 分辨率下 batch 超过 8 可能 OOM；当前 batch 8 峰值约 3.3GB。**AMP 尚未接入**（trainer 无 autocast/GradScaler），双输入若超限优先降 batch + 梯度累积。
- 数据集样本量小（1148 切片 / 641 例），从零训练容易欠拟合，基线指标可能偏低。
- 两子集设备与增益不同，需关注域差异带来的指标波动。
