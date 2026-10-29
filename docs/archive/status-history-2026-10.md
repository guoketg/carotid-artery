# 状态与环境校验记录

## 2026-10-27 换机后的环境重建（E: 盘工作副本）

- **pip 镜像源已确认是清华源**：`C:\Users\ranzh\AppData\Roaming\pip\pip.ini` 里 `global.index-url = https://pypi.tuna.tsinghua.edu.cn/simple`（user 级，venv 内 pip 同样生效；本次安装全部走该源）。
- **新 venv 位置与 Python 版本变了**：工作副本在 `E:\mkl\code\python_code\carotid-artery`，venv 为 `E:\mkl\code\python_code\carotid-artery\venv`，**Python 3.11.5**（原机器是 `f:\code\MyPython\TransUNet\venv`，Python 3.12.10）。
- **版本回退（因 Python 3.11 的 Requires-Python 限制）**：`numpy==2.5.2` 与 `scipy==1.18.1` 均要求 `>=3.12`，在 3.11 上装不了；改用 **numpy 2.4.6 / scipy 1.17.1**，并已同步 `requirements.txt`。若后续在 Python 3.12 上重建环境，再改回 2.5.2 / 1.18.1。
- **本轮新装的库**（原 venv 只有 torch/torchvision/numpy）：h5py 3.16.0、nibabel 5.4.2、SimpleITK 2.5.6、ml-collections 1.1.0、MedPy 0.5.2、tensorboard 2.21.0、tensorboardX 2.6.5、tqdm 4.70.1（连带 grpcio/protobuf/werkzeug/PyYAML/absl-py/importlib-resources 等）。
- **校验结果**：`pip check` 无冲突；AST 扫描全部 `.py` 的外部 import 共 10 个（SimpleITK/medpy/ml_collections/nibabel/numpy/scipy/tensorboardX/torch/torchvision/tqdm）全部可导入；项目 11 个模块 import 通过；TransUNet 前向 105.28M 参数、输出 `(1,2,224,224)`；合成 NIfTI 走通 `CarotidDataset(use_wavelet=True)` → `(3,224,224)`，medpy 指标链路正常（dice 1.0 / assd 0）。
- **本机差异**：GPU 为 **RTX 3060 12GB**（原机器 RTX 5060 Laptop 8GB）；**F: 盘不存在**，数据根与权重原本不可达。→ **已在下一节「2026-10-27 补齐」全部处理完毕**（数据、权重、路径、隐私、队列均已就位）。

## 2026-10-27 补齐（数据 / 权重 / 路径 / 隐私 / 队列启动）

### 4.0.1 数据与权重已落位

- **数据**：`C:\Users\ranzh\Downloads\Carotid2D.zip等2个文件.zip` 已解压到 `E:\mkl\code\datasets\Carotid2D`。
  - `PlaqueDataNii_SPARC_S`：144 例 / **510 图**，images 与 labels 数量一一对应（510/510）。
  - `PlaqueDataNii_Zhongnan_S`：497 例 / **636 图**（636/636）。
  - **官方划分**（`lists.zip`）→ `lists/lists_Carotid2D/{SPARC,Zhongnan}/{train,test}.txt`：SPARC 350 张/101 例 训练 + 160 张/43 例 测试；Zhongnan 448 张/348 例 + 187 张/148 例。
  - 校验：两子集**病例级泄漏 0 例**，图像/标签**缺失 0 个**。
- **预训练权重**：`R50+ViT-B_16.npz` → `E:\mkl\code\python_code\model\vit_checkpoint\imagenet21k\`，**461,217,452 字节 / 358 个数组**，键前缀 `conv_root / block1~3 / embedding / Transformer / head / pre_logits`，确认 R50 混合版。
  - 下载方式：Google Storage 对本机单连接限速 ~140 KB/s，且 **`curl --speed-limit` 会把「低速但正常」的连接误杀**（本轮踩坑）。最终方案：24 分片 × 4 子区间并行 + 不设 speed-limit，约 25 分钟下完。

### 4.0.2 路径改造（全部硬编码 `F:\code\...` 已切到本机）

| 文件 | 改动 |
|---|---|
| `train.py` L92 | `root_path` → `E:\mkl\code\datasets\Carotid2D` |
| `datasets/dataset_carotid.py` L10/L12/L14 | `DATA_ROOT` → 新根；`SUBSETS` 的 `_M` → **`_S`**；`LIST_DIR` → `lists_Carotid2D` |
| `scripts/eval_carotid.py` L104/L106 | `--root_path` / `--model_root` 默认值 |
| `scripts/summarize_dev.py` L68 | `--model_root` 默认值 |
| `scripts/build_dev_subset.py` L17 | `SRC_ROOT` → `lists_Carotid2D`（原指向已废弃的 `lists_Carotid`） |
| 4 个 `scripts/night_queue*.ps1` | `$Python` / `$Root` / `$modelDir` 全部切本机 |

**顺带修掉一个潜伏 bug**：4 个队列脚本的 `Snapshot` 字段缺 `_c2d` 后缀，而 `train.py` 会给快照目录加 `_c2d` → 归档上一轮产物的 `Move-Item` 从来没匹配上，重复跑同一个 job 会把新旧权重混在同一目录。已补齐后缀（20 处）。

**踩坑**：`.ps1` 文件含中文注释时必须存成 **UTF-8 with BOM**，否则 Windows PowerShell 5.1 按 ANSI 解码导致 `ParserError`（`queue_c2d.ps1` 已按 BOM 存）。

### 4.0.3 病人隐私：已全部忽略

- `.gitignore` 新增：`datasets/split/`、`*.nii`、`*.nii.gz`、`*.dcm`、`patient*`、`case*`、`*.pth`、`*.npz`、`model/`（`lists/`、`logs/`、`runs/` 本来就在忽略列表）。
- **`git rm --cached` 已从索引移除 4 个含病人标识的历史跟踪文件**：`datasets/split/filelist_train101`、`filelist_test43`、`trainfile_zhongnan.txt`、`testfile_zhongnan.txt`（内容含 `667532_04-30-2019`、`1334`、`107726_02-17-2015` 等病例号）。**文件仍留在工作区，只是不再进版本库。**
- 原始影像在 `E:\mkl\code\datasets\`（仓库外），仓库内只保留相对路径清单，且 `lists/` 已被忽略。

### 4.0.4 本机性能标定（RTX 3060 12GB vs 老机器 RTX 5060 Laptop 8GB）

| 配置 | 峰值显存 | 单步耗时 | 备注 |
|---|---|---|---|
| baseline bs4 | 2.13 GB | 159.5 ms | — |
| baseline bs8 | 3.32 GB | 278.8 ms | — |
| baseline bs12 | 4.53 GB | 401.3 ms | — |
| **dual L1234 auto bs4** | **3.06 GB** | **245.9 ms** | 队列口径 |
| **dual L1234 auto bs8** | **4.97 GB** | **428.3 ms** | **不再触发 WDDM 换页**（老机器同配置 1101.6 ms，慢 2.6×） |

- **12GB 卡上 batch 8 有 7GB 余量，Windows 换页惩罚消失**，这是本机相对老机器最大的吞吐改善。
- 实测训练节拍（SPARC，bs4+ga2）：训练 epoch ≈ **20.3 s**，测试评估（160 张）≈ **24 s**。
- 折算：baseline 200 epoch ≈ **1.4 h/ job**；dual 200 epoch ≈ **1.5 h/ job**；Zhongnan（448/187 张）≈ **1.8~2.0 h/ job**。老机器 dual 120 epoch 就要 2.5 h。

### 4.0.5 新口径补齐队列已启动

- 新脚本 `scripts/queue_c2d.ps1`（新口径：官方划分 + `--max_epochs 200` + `--eval_test_every 5` + `--batch_size 4 --grad_accum 2`，取 `epoch_149.pth`），**2026-10-27 19:59 已用独立 powershell 进程启动**（脱离会话，脚本内 `SetThreadExecutionState` 防休眠）。
- 7 个 job 按优先级串行：

| # | job | 内容 | 估时 |
|---|---|---|---|
| 1 | `d1234_sparc` | 双骨干 auto（最深接点已改串联双向），seed 1234 | 1.5 h |
| 2 | `b2024_sparc` | 单通道原图基线，seed 2024 | 1.4 h |
| 3 | `b3407_sparc` | 单通道原图基线，seed 3407 | 1.4 h |
| 4 | `d2024_sparc` | 双骨干 auto，seed 2024 | 1.5 h |
| 5 | `d3407_sparc` | 双骨干 auto，seed 3407 | 1.5 h |
| 6 | `b1234_zhongnan` | 单通道原图基线，Zhongnan | 1.8 h |
| 7 | `d1234_zhongnan` | 双骨干 auto，Zhongnan | 2.0 h |

- 合计 ≈ **11 h**；状态写 `logs/queue/queue_c2d_status.csv`，单 job 日志写 `logs/queue/<job>_<ts>.log`。
- **口径提醒**：poly 衰减总长绑死 `max_epochs`，所以基线与消融**必须同在 200 epoch 调度下取 epoch 150** 才可比；本队列沿用历史 seed1234 基线的同一套参数。
- **队列首 job 现场验证**：`d1234_sparc` epoch 1 的 γ 实测 `fuse.1 a/b = 0.098/0.0055`、`fuse.2 = 0.0129/0.0030`、`fuse.3 = 0.0323/1.9e-4`、**`fuse.4 = 0.0063/1.24e-4`**。最深接点的 `gamma_b` **已非零**（4.5.3 结构性死支路修复 + 4.5.4 串联双向修复均生效；修复前它 30 个 epoch 恒为 0.0）。快照目录名与队列脚本的 `Snapshot` 字段已对齐校验一致。

### 4.0.6 启动方式：必须用 WMI 脱离进程树（重要）

**事故记录**：第一次用 `Start-Process powershell -WindowStyle Hidden` 启动队列，训练跑到 iteration 71（约 2 min）被**无痕杀掉**——日志没有任何 Python traceback，队列 powershell 进程也一起消失。原因是该子进程仍在启动它的终端进程树的 Job Object 里，终端命令一返回，整棵树被回收。

**正确做法**（WMI 创建，父进程是 WmiPrvSE.exe，不受调用方进程树影响）：

```powershell
$cmd = 'powershell -ExecutionPolicy Bypass -NoProfile -File "E:\mkl\code\python_code\carotid-artery\scripts\queue_c2d.ps1"'
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
```

2026-10-27 20:02 已用此方式重启，pid 30472，已验证跨多次独立 shell 调用存活、GPU 稳定 85% / 4.98 GB / 137 W。

- 队列脚本自带 `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`，机器不会因空闲休眠而中断。
- **人工重启方式**：先 `Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -like "*train.py*" }` 确认没有残留训练，再执行上面的 WMI 命令；已完成的 job 用 `-Skip` 跳过。
- **注意**：如果机器重启或断电，队列不会自动续跑（脚本没有断点续跑逻辑），需要按 `logs/queue/queue_c2d_status.csv` 里已完成的 job 手工 `-Skip` 后重启。

## 2026-10-23/24 夜间队列

- 已实现：`networks/wavelet.py`（Haar DWT/IDWT + 软阈值 + 三通道）、`networks/wavelet_model.py`（小波分支 + 小波引导的交叉注意力 / SE 对照）；`train.py` 新增 `--use_wavelet`、`--fusion {none,se,cross}`、`--heads`，默认 `batch_size=8`。
- 融合位置为 **ViT 输出瓶颈（14×14，196 token）**，不是原设计的 stage2/3 skip：实现简单、注意力开销低，单通道原图基线完全不受影响。偏差已记入 `technical-design.md`。
- batch=8 实测：峰值显存 3.3GB，纯模型单步 0.36s，含数据约 0.5s，SPARC 约 27s/epoch。
- 夜间队列 6 个任务，统一 `--max_epochs 120 --eval_test_every 10`，顺序 a0_sparc → a3_sparc → a1_sparc → a0_zhongnan → a3_zhongnan → a1_zhongnan（依次为单通道原图基线、三通道＋小波引导注意力、三通道早融合，SPARC 与 Zhongnan 各一轮），预计 6 小时。
- 冒烟（1 epoch，仅验证链路）：三通道早融合 dice 74.5%，三通道＋小波引导注意力 dice 75.6%。
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
| 训练结束写 `metrics.csv` 时重建 `CarotidDataset` 漏传 `use_wavelet`，导致小波模型用非小波输入评估（SPARC_w3 89.06% vs 实际 90.79%；Zhongnan_w3_cross 85.79% vs 90.77%） | 2026-10-24 15:18 在 `trainer.py` 末尾评估处补上 `use_wavelet=bool(args.use_wavelet)`；新增 `scripts/eval_carotid.py`（支持 `--use_wavelet/--fusion` 与 `--baseline_csv` 配对检验），已重算全部 6 个模型的 `metrics.csv`（`test_curve.csv` 不受影响，可信） |

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
- **冒烟**：SPARC 1 epoch 57 s（含 82 张测试推理），dice 79.48% / assd 9.92 / hd95 24.50 / abs_delta_area 4028.4，`metrics.csv`、`test_curve.csv` 正常产出。
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

## 4.5 2026-10-25 融合分支零初始化死锁 + 最深接点死支路（**两者均已修复**）

### 4.5.1 零初始化死锁（已修，用户手工改 `networks/dual_encoder_model.py`）

- 机理：`A2 = A + γ·GN(Proj(Attn(...)))`，`Proj` 权重与偏置全零 → 残差项恒 0；而 GroupNorm 对全零输入输出恰为 0，于是 `∂L/∂γ = ⟨∂L/∂A2, 0⟩ = 0`，同时 `∂L/∂Proj = γ·(...) = 0`。两个零乘因子互为前提 → `γ` 与 `Proj` 永久停在 0，`qkv`、`norm` 也因 `Proj≡0` 拿不到梯度。整个融合块退化为恒等映射，且**不是"学得慢"，是永远学不动**。
- 判据：一条乘法链上只能有一个零，且这个零的梯度不能依赖它自己。ReZero 之所以成立，是因为只保留 γ=0、`Proj` 正常初始化，`h(x)≠0`，γ 一步就能离开 0。
- 修改：删除 `BidirCrossAttention` 的 `proj_a/proj_b` 与 `BidirConcat` 的 `conv_a/conv_b` 零初始化；保留 `BidirGate.conv`（`sigmoid(0)=0.5`，把零变成非零，本身不死锁）、`BidirDeform.off`（偏移为 0 等价固定网格采样，DCN 标准做法，且输出非零）与全部 γ（换恒等起点）。
- 验证：五档（窗内、全局、拼接、可变形、门控）第 0 步 `max|Δ|=0`，γ 第 0 步即有非零梯度，`proj/qkv/conv/off` 第 0 步为 0 属正常（梯度 ∝ γ），第 1 步全部解锁。
- 小的坑：`Proj` 零初始化本身**没有换来任何东西**（恒等起点由 γ 就能提供），只是多加了一道梯度闸门。每加一个零初始化就该问一句"这个零换来了什么"。

### 4.5.2 由此带来的口径问题（**此前所有双骨干结果都要重新解读**）

死锁修复时间是 2026-10-25 15:13，因此下面这些 job 跑的都是冻结版代码：

| 配置 | 跨支注意力/可变形分支 | 跨尺度回流 | epoch 120 Dice |
|---|---|---|---|
| 单通道原图基线 | — | — | 90.92% |
| 原图＋去噪重建＋高频能量三通道早融合 | — | — | 90.79% |
| 三通道＋小波引导注意力（旧轻量路线） | — | — | 90.87% |
| 双骨干·窗内/全局自适应融合 `auto` | **冻结** | — | 91.17% |
| 双骨干·可变形大核 `deform` | **冻结** | — | 90.88% |
| 双骨干·自适应融合＋跨尺度回流 `auto_cs` | **冻结** | 正常学 | 90.93% |

- 前三个是旧路线（本来就没有该融合块），后三个里 `auto` / `deform` 的融合分支全程冻结，模型等价于「基线 + 一批不参与计算、也不更新的参数」；它们与基线的差异（90.79%~91.17% 全距 0.38%）落在初始化/随机数差异的噪声带内，**不能作为"融合有/无增益"的证据**。
- `auto_cs` 是唯一部分有效的：`cs_gamma` 乘的是非零的 `cs43(ta)`，跨尺度回流确实在训；但它的四级跨支融合仍是冻结的。
- `--fuse_op aff` 不受死锁影响（该档没有输出投影，γ 乘的是非零的 `A*m`）；但白天 11:03 那次没有产出 `test_curve.csv`，15:05 的 `aff` 重跑 3 分钟后中断，等于**AFF 档至今没有一次完整结果**。

### 4.5.3 最深接点（14²/768）的 b 方向是结构性死支路（**已修，见 4.5.4**）

- 位置：`networks/dual_encoder_model.py` 中 `ta, b = self._fuse(4, ta, b)` 之后，`b` 再没有被任何算子引用，解码器只拿 `t` 和 `[f3, f2, f1]`。
- 后果：`∂L/∂B2` 不存在 → `gamma_b` 从第 0 步起就是精确 0（实测 `fuse.4.gamma_b` 四个 epoch 全为 `0.0`，同层 `gamma_a` 为 0.009→0.0125）；`qkv_b`、`proj_b`、`norm_b` 一起悬空，合计约 118 万参数零梯度。
- 与前三级的区别：`fuse.3.gamma_b` 只有 1e-4 量级但非零——F3 的 `B2` 还能经 `body[2] → side_proj → F4 的 K/V → out_a` 这条长链间接影响 loss，属"长链稀释"，不是"断路"。
- 这与 4.5.1 的死锁是两回事：死锁是两个零相乘互锁（数值路径），这里是计算图上少了一条边（结构问题），`gamma_b` 初值给 0.1 也会停在 0。
- 自检方法：一次 backward 后看 `.grad` 是否为 `None`（悬空子图连梯度都不发），而不是看是否为零张量。

### 4.5.4 落地：最深接点改串联双向（**已实现并验证**）

做法：先更新侧路、再让主路去查**更新后的**侧路，而不是两支各自用融合前的输入并行更新。

1. `BidirCrossAttention.__init__` 增加 `chain=False`，存为 `self.chain = chain and bidir`（`bidir=False` 时自动失效）。
2. `forward` 重排：窗内/全局两条分支各自合成 `to_tok` / `back` 闭包；`chain=True` 时先算 `B2 = B + γ_b·GN(Proj_b(Attn(qb, k_a, v_a)))`，再用 `qkv_b(B2)` 重取 K/V 算 `A2`，直接返回 `(A2, B2)`；`chain=False` 保持原并行路径，逐元素不变。
3. `make_fusion` 增加 `chain` 透传，只有 `window` / `global` 两档接受。
4. 模型构造处 `chain = bool(bidir) and lv == max(self.levels) and kind in ("window", "global")`，前三级仍并行双向。

验证（`scripts\probe_dual.py --fuse_op window --levels 1,2,3,4 --check_grad`，`--check_grad` 为本次新增的开关）：

- **等价性**：`max|baseline-dual| = 0.000e+00`，PASS。总参数 121.66M 不变（F1 0.017 / F2 0.264 / F3 1.053 / F4 2.366 M）；batch 4 峰值 3.34 GB、单步 335 ms，与改前一致。
- **梯度**：`fuse.4.gamma_b.grad` 不再是 `None`。但**第 0 步仍为 0**——`A2` 的残差项乘的是 `γ_a`，`γ_a = 0` 时 `∂L/∂B2 = 0`，`γ_b` 拿不到梯度；**第 1 步起非零**。前稿写的"`γ_b` 第 0 步即有非零梯度"是错的。
- **量级小**：`γ_b` 的梯度是 `γ_a·γ_b` 的二阶小量。实测 120 步（batch 4、随机数据、SGD lr 0.01）F4 `γ_b` 才走到 −1.7e-5，F3 到 +7.8e-7，而同层 `γ_a` 已到 −1.4e-3；前三接点的 `γ_b`（直接进 loss）到 −2e-4 / +1e-4。结论是"结构上通了、学得慢"，日志里 F4 `γ_b` 长期停在 1e-5 量级属预期，不要再当断路。
- 其余注意：`chain` 与 `bidir=False` 互斥；`gate` / `concat` / `aff` / `deform` 四档没有 chain 分支，四级配置下最深接点的 b 方向仍是死支路。

### 4.5.5 观测手段（已加）

`trainer.py` 在每个 epoch 训练循环结束后遍历 `named_parameters()`，把名字含 `gamma` 的参数取 `abs().max()` 写入 `log.txt`，并记到 tensorboard 的 `fusion/<参数名>`。这样第 1 个 epoch（约 30 s）就能判断门控是否在学，不必等 epoch 99 的第一个权重落盘。判读：从 1e-4~1e-2 量级起步且逐 epoch 变大 = 正常；恒为精确 0 = 断路或死锁。

**`--tag rerun1` 已跑完**（2026-10-25 15:29 起跑，全量 SPARC，120 epoch）：跑的是**死锁已修、最深接点串联未改**的代码，即"前三个接点并行双向 + 最深接点仅 A→B 单向、融合真在学"的第一个有效配置，结果见 4.5.6。

### 4.5.6 活融合结果：**无增益，四项指标全差**（2026-10-25 收尾）

配置：`--fuse_op window`（112² 窗内 8×8 / 56² 7×7 窗内 / 28² 7×7 窗内 / 14² 全局）、`--levels 1,2,3,4`、`--bidir 1`、`--cross_scale 0`、`bs4 + grad_accum 2`、seed 1234，目录 `..._SPARC_dual_L1234_window_rerun1`。

| 配置 | Dice | ASSD | HD95 | \|ΔArea\| | 融合分支状态 |
|---|---|---|---|---|---|
| 单通道原图基线（bs8） | **90.92%** | **3.809** | **10.022** | **945.1** | — |
| 三通道早融合（bs8） | 90.79% | 3.818 | 9.997 | 1017.9 | — |
| 三通道＋小波引导注意力（bs8） | 90.87% | 3.892 | 10.041 | 1065.0 | — |
| 自适应融合 `auto`（bs4+ga2） | 91.17% | 3.714 | 9.981 | 943.5 | 冻结 |
| 可变形大核 `deform`（bs4+ga2） | 90.88% | 3.861 | 10.163 | 986.2 | 冻结 |
| 自适应＋跨尺度回流 `auto_cs`（bs4+ga2） | 90.93% | 3.877 | 10.175 | 973.8 | 融合冻结、回流正常 |
| **窗内融合 `window_rerun1`（bs4+ga2）** | **90.51%** | 3.940 | 10.127 | 1024.8 | **正常学** |

- 相对单通道基线：Dice **−0.41%**、ASSD +0.131、HD95 +0.105、\|ΔArea\| +79.7，**四项全部变差**；相对三通道早融合 Dice −0.28%。
- 不是"后期运气差"：12 个评估点里 10 个低于基线同步 epoch（差最大在 ep50，−1.34 个点；ep70 后收敛到 0.25 个点以内，ep120 为 −0.41），只有 ep20（+0.20）、ep110（+0.04）略高。注意这些 epoch 强相关，不能当独立样本做符号检验，只作方向性参考。
- γ 实测（`log.txt`，ep1–ep30）：F1 0.07~0.10 / 0.01~0.03，F2 0.02~0.06 / 0.02~0.04，F3 0.05~0.09 / 1e-4~5e-4，F4 0.01~0.02 / **0.0**。前三级双向均在学，最深接点 `gamma_b` 30 个 epoch 一次没离开 0，与 4.5.3 的死亡支路判断一致。
- **结论口径**：严格表述为「**没有观察到增益，方向上偏向变差**」。不能说"显著变差"——单 seed 无法估计方差，−0.41% 落在 4.5.2 测得的配置间噪声带量级内。
- **两个未消掉的混杂**（下一步必须先解掉，否则这条结论仍然不干净）：
  1. **基线口径不一致**：基线/三通道那批是 `bs8` 直接一步，dual 全部是 `bs4 + grad_accum 2`。等效 batch 相同但梯度路径不同（BN 统计、优化器步频），差值的 0.3~0.4 个点里有多少来自这里未知。需要补一个 `bs4+ga2` 的单通道基线。
  2. **单 seed**（1234）。要给"变差"下统计结论，至少 1234 / 2024 / 3407 三个 seed。
- 关于"为什么活起来反而更差"，目前只有假设、无证据：侧路吃的是无低频的方向子带（LH/HL/HH），语义弱，K/V 注入的更可能是噪声；F1/F2 的 γ 稳定在 0.02~0.09，相当于每层叠加 2~9% 的扰动，而超声斑块边界本身模糊、容错低。等 4.5.4 的串联方案与 `bs4+ga2` 基线同时具备后再判断。

## 4.6 2026-10-26 频域文献复盘：结论冲突拆解 + 下一轮路线（**仅结论，代码未改**）

触发：本地新增 8 篇 PDF（`C:\Users\guoke\Desktop\医学图像`，含待提交的 `3612 Submission.pdf`）逐篇精读；导师提出「现在不局限于两种变换」；XNet（ICCV 2023）「LF+HF 融合 SOTA」的结论与 GobletNet（TMI 2025）不一致。

### 4.6.1 XNet 与 GobletNet 不矛盾

四个维度不同，详见 [literature.md](literature.md)「频域融合：2026-10-26 精读的 8 篇」：

| 维度 | XNet | GobletNet |
|---|---|---|
| 基线容量 | raw-only 是**单编码器**，全网 326M / 83G | **参数量对齐** |
| 「低频」含义 | `LL` 近似图 = 模糊版原图 | 主分支已吃原图，再灌 `λ·LL` 是冗余 + 噪声 |
| 结论 | LF+HF+融合 84.77 > raw 82.03（Table 7） | **只有高频有用，再加低频反而变差** |

补充事实（同一批）：XNet Table 7 里 **HF only 只有 54.50**、LF+HF 不融合 75.82，都低于 raw——增益来自**两支互补**，不是「多一个频带」。

### 4.6.2 对 4.5.6「无增益」的六条解释（均未验证）

1. 融合分支可能仍然没学起来（`W_o` 接近零 + 120 epoch）。**低成本判定：读 checkpoint 打印 4 个接点 `|γ_A| / |γ_B|`**。
2. **HF 图构造方式可疑**：现状是三个半分辨率子带 `concat` 后双线性上采样到 224²，有 2×2 块状伪影、三通道冗余；XNet / GobletNet 都是 **IDWT 重建一张全分辨率 HF 图（`LL` 置零）**。
3. 口径未消（baseline `bs8` vs 融合 `bs4 + grad_accum 2`）。
4. 单 seed + 小测试集，检测 +0.5 Dice 约需 128 对病例（α=0.05, power=0.8）。
5. 天花板效应：Dice 90.9% / HD95≈10，残差集中在边界几像素；融合改善的是全局语义一致性。
6. 颈动脉回声特性：真正难点是钙化后方声影导致的**信号缺失**，频域增强补不回来。

### 4.6.3 下一轮顺序（**均未开始**）

| 步骤 | 内容 | GPU 代价 |
|---|---|---|
| Step 0 | 打印 4 个接点 `|γ|`；HF 图改为 IDWT 重建 | 0 |
| Step 1 | baseline 用 `bs4 + grad_accum 2` 重跑，seed 1234 / 2024 / 3407 | 3 job |
| Step 2 | 最小验证：stem 输入 3 → 4 通道（原图 3 + IDWT 重建 HF 图 1），新增 27 参数 | 1 job |
| Step 3 | 若有效：回单骨干，3–4 级特征做 DWT 解耦（高频支 `1×1 → 3×3 → 上采样 → 通道注意力`；低频支 `5×5 → 上采样`），skip 加轻量空间-通道对齐，融合用门控（GobletNet FAM 式）而非交叉注意力 | 1–2 job |
| Step 4 | 可选：LF/HF 双视图一致性正则 / 半监督 | — |

### 4.6.4 本轮确定的取舍

- **不再新增更重的融合算子**（可变形大核、双向跨支交叉注意力均已试过且无增益），先把 Step 1 的口径补齐。
- **不再追加小波基对比**：GobletNet 实测小波基差异多数 <1%、二级小波无显著收益；WMREN 表明小波单独（81.29）不如纯卷积（82.59）。
- **若 Step 2 无效**，转 **特征级 DWT 解耦（BDFR-Net 式，同域超声，+2.09）** 或 **显式边界监督（MBFF-Net BPS 单项 +0.044）**，而不是继续加支路。
- 「不局限于两种变换」的工具箱已列入 [paper-reading.md](paper-reading.md) 5.6 节；结论是**变换种类不是主要矛盾，注入位置才是**。

### 4.6.5 2026-10-26 老师两项指示（**改排期，代码未动**）

1. **新数据集口径**：数据根 `F:\code\datasets\Carotid2D`（`_S` 目录），官方划分在 `datasets/split/`——SPARC 训练 350 张/约 101 例、测试 160 张/约 43 例；Zhongnan 448/187 张。旧划分（`lists_Carotid`，测试 28 例）弃用，**4.5 节全部历史数字只在旧口径下有效**。原 Step 1（补口径）相应升级为：**新划分上重跑单通道原图基线**（`bs4 + grad_accum 2`，3 seed）。接入需要 2 处代码/脚本改动（`train.py` root_path 写死、划分文件转 `<list_dir>/<Subset>/train.txt` 结构），明细见 [dataset.md](dataset.md) §7，按硬约束 10 待批准。
2. **先尝试边缘特征融合**：下一结构实验的优先方向，插队到「stem 加 HF 通道」与「特征级 DWT 解耦」之前。两种落地读法（输入级 4 通道边缘图 / 特征级边缘支路）与最小设计见 [technical-design.md](technical-design.md)「频率注入位置」待决策第 4 条，实施待批准。

统计功效备注：新测试集约 43 例，对 +1.0 Dice（约需 32 对）勉强够，对 +0.5（约需 128 对）仍不够；结论表述仍需谨慎。

### 4.6.6 2026-10-26 新口径落地：Carotid2D 接入 + 指标物理单位化（**冒烟通过，基线重跑中**）

**数据接入（已实施）**：`train.py` root_path/list_dir 切 `F:\code\datasets\Carotid2D` + `lists/lists_Carotid2D`；划分四文件由用户转换；快照目录带 `_c2d` 后缀。冒烟 2 epoch 全链路通过，测试集 n=160（与官方 test43 一致）。明细见 [dataset.md](dataset.md) §7.3。

**指标物理单位化（已实施 + 验证）**：距离指标不再在 224 网格上按像素计——`metrics_carotid.py` 新增 `original_grid_pair()`，把预测最近邻重采样回原分辨率网格，与原分辨率 GT、NIfTI 头 spacing（实测 SPARC 0.065~0.068、Zhongnan 0.067 mm/px，非正方形约 2:3）一起交给 medpy `assd`/`hd95` 的 `voxelspacing` 参数。**assd/hd95 单位 mm，DeltaArea 单位 mm²，Dice 改在原分辨率网格上算（与 224 网格差 <0.6%）**。验证脚本（临时目录，已删）：原网格 GT 膨胀 3px 造模拟预测，校正后相对物理参考 assd 误差 ≤9.2%、hd95 ≤17.4%；**直接传头 zooms 到 224 网格会失真 0.83×~6.67×（随图像尺寸翻转方向），故必须重采样回原网格或逐轴换算 spacing**。改动点：`metrics_carotid.py` 全量、`trainer.py` `_eval_test` 签名/调用/末尾最终评估、`scripts/eval_carotid.py` import 与循环。

**口径断代**：4.5 节全部数字为「旧划分 + 像素口径」双重旧口径；2026-10-26 起的实验为「官方划分 + mm/mm² 口径」。冒烟 summary（2 epoch 基线参考起步）：dice 80.6%±11.1%、assd 0.338±0.292 mm、hd95 1.014±1.102 mm、|delta_area| 4.95 mm²（n=160）。

**XNet 式低频主路（已实现，实验未跑）**：`--main_input ll`（train.py L67-68 argparse、L129 快照后缀 `_mainLL`、L157 透传；`dual_encoder_model.py` `build_ll_image` + forward 分支），主路吃小波 LL 低频近似图、高频信息全在侧路。待新一轮基线跑完后与双骨干原图对照。

**基线 A 结果与 epoch 决策**：单通道原图基线 200 epoch 跑完，曲线 150 即饱和（dice 90.49%、assd 0.1431 mm、hd95 0.3896 mm @150；150~200 平台噪声内波动，后 50 epoch 无实质贡献）。**固定 epoch = 150（epoch_149.pth），全部后续实验复用**；job 可训到 epoch_149.pth 即停。详见 experiments.md「epoch 选择」。双骨干原图 job 起跑时曾因 forward 单通道分支漏 `raw = x` 崩溃（UnboundLocalError），2026-10-26 已修复，双骨干原图/低频两模式 1 epoch 冒烟均通过（dice 76.8%/72.5%，融合 γ 梯度非零）后正式开跑。

### 4.7 2026-10-27 换机收尾：队列盘点 + 边缘特征融合落地

#### 4.7.1 新口径队列盘点（用户问「还有哪些队列没跑完」）

**本机起点**：`E:\mkl\code\python_code\model\TU_Carotid224` 下**零快照、零日志、零 runs**——新口径（官方划分 + mm 口径 + `bs4/ga2` + `epo200`）在本机几乎全部实验都要重跑。

| 分类 | 数量 | 明细 | 状态 |
|---|---|---|---|
| 新口径**已跑完** | 1 | 单通道原图基线 SPARC seed1234（dice 90.49% / assd 0.1431 mm / hd95 0.3896 mm @ep150） | 结果在案，**权重不在本机**，需重跑才能拿到 `.pth` |
| 新口径**正在跑** | 1 | `d1234_sparc`（双骨干 auto seed1234） | 队列第 1 个 job |
| 新口径**队列内待跑** | 6 | `b2024_sparc`、`b3407_sparc`、`d2024_sparc`、`d3407_sparc`、`b1234_zhongnan`、`d1234_zhongnan` | 已排期 |
| **状态未知** | 2 | 2026-10-26「双骨干原图 job 正式开跑」的结果未记录在案；基线 seed1234 的权重也不在本机 | 需重跑消歧 |
| **旧队列脚本待按新口径重跑** | 36 | `night_queue.ps1`(6：a0/a1/a3 × 2 子集)、`night_queue_dual.ps1`(8：r1/r2/r3/r7 × 2 子集)、`night_queue_dual2.ps1`(6：r4/r5/r6 × 2 子集)、`night_queue_dev.ps1`(16：d00~d15，已撤销、保留备用) | 未排期 |
| **已实现未跑** | 1 | `--main_input ll`（XNet 式低频主路） | 等基线补齐后跑 |
| **未排期（experiments.md 已写）** | 若干 | §1.4 R4（R50+ViT 同构双骨干，容量 vs 信息）、P1 CLAHE / P2 二级小波 / P3 同态滤波、§1.2 A4–A9 共享权重双输入系列、跨子集复现 | — |
| **本轮新增** | 4 | 边缘特征融合·输入级（`e1234_sparc` / `de1234_sparc` / `e1234_zhongnan` / `de1234_zhongnan`） | 接力队列已启动 |

**两个队列脚本**：

| 脚本 | job 数 | 启动方式 | 预计 |
|---|---|---|---|
| `scripts/queue_c2d.ps1` | 7 | 2026-10-27 20:02 用 `Invoke-CimMethod Win32_Process.Create` 脱离启动（pid 30472） | 全部完成 ≈ 2026-10-28 07:30 |
| `scripts/queue_c2d_edge.ps1` | 4 | 2026-10-27 20:21 同法脱离启动（pid 13204），**接力等主队列放卡后开跑** | 第一个 edge 结论 ≈ 10-28 08:40；4 个全完 ≈ 10-28 14:30 |

合计 ≈ 18 h。实测速度（本机 RTX 3060 12GB）：dual bs4+ga2 **33 s/epoch**（SPARC，88 micro-batch → 44 optimizer step/epoch，含每 5 epoch 一次的测试评估），基线 bs4 约 20 s/epoch，Zhongnan 因训练集大 1.28×（448 vs 350 张）。

接力脚本的等待条件：`无 train.py 进程` **且**（`主队列 status.csv 已写满 7 行` **或** `主队列进程已退出`）→ 后者保证主队列万一被打断时 GPU 也不会空转。**注意：edge 队列是等主队列 7 个 job 全跑完才开跑**（不是插空跑），所以 edge 的第一个结论要到 10-28 早上才有。

**2026-10-27 20:36 实时状态**：`d1234_sparc` 跑到 **epoch 60/200**（iteration 2675 / 8800），曲线 dice 89.86% / assd 0.1562 mm / hd95 0.4323 mm；GPU 90% util / 4874 MiB（12 GB 卡，够再排一个 job）。作为参照，单通道原图基线 seed1234 在 **ep150** 是 dice 90.49% / assd 0.1431 mm / hd95 0.3896 mm —— dual 在 ep60 已基本追平量级，ep150 才能下结论。

#### 4.7.2 边缘特征融合·输入级（已实现 + 冒烟通过）

老师定的「边缘特征融合」按 technical-design.md「频率注入位置」第 4 条落地**输入级最小版**：stem 吃 4 通道。

| 改动点 | 文件 | 内容 |
|---|---|---|
| 边缘图算子 | `networks/wavelet.py` | 新增 `build_hf_image(x)`：对数域 1 级 Haar → **LL 置零 → IDWT 重建** → `expm1` 回原域 → 逐样本标准化；输出 (B,1,H,W) |
| stem 通道数 | `networks/vit_seg_modeling_resnet_skip.py` | `ResNetV2.__init__` 增 `in_channels`，`root.conv` 由 `StdConv2d(3,64,7)` 变成 `StdConv2d(in_channels,64,7)` |
| 装配 | `networks/vit_seg_modeling.py` | `Embeddings`/`Transformer`/`VisionTransformer` 逐层透传 `in_channels`；`VisionTransformer.__init__` 增 `edge_in` 开关，`forward` 里 `cat([x.repeat(1,3,1,1), build_hf_image(x)])`；`load_from` 里 `conv_root/kernel` 3→4 通道**零填充** |
| 双骨干 | `networks/dual_encoder_model.py` | `DualEncoderTransUNet` 增 `edge_in`，主路 `main_in` 追加 HF 通道；侧路契约不变（仍吃 LH/HL/HH） |
| 开关与防撞名 | `train.py` | 新增 `--edge_in 0/1`；快照目录加 `_edge` 后缀（`_SPARC_edge_c2d` / `_SPARC_edge_dual_L1234_auto_c2d`） |
| 复评 | `scripts/eval_carotid.py` | 新增 `--edge_in`，重算 checkpoint 时能重建同构模型 |

**验证结果**（`scripts/probe_edge.py` / `probe_edge_train.py`）：

- Haar DWT/IDWT 精确可逆：max|Δ| = 2.98e-07；
- `build_hf_image` 输出 (B,1,224,224)，逐样本标准化后 mean=0.0000 / std=1.0000，与原图相关性接近 0（确实只剩高频）；
- 预训练权重加载成功，`conv_root` 形状 (64,4,7,7)，**第 4 通道 absmax = 0.0**（零填充正确）；
- 参数量：baseline 105.2761M → edge_in **105.2792M**；
- 真实数据 3 个 optimizer step：loss 0.703 → 0.655 → 0.604，峰值显存 **2.13 GB**（与 baseline bs4 完全相同，多一个通道不涨显存）。

**参数增量修正**：technical-design.md 原写「输入级 **+27 参数**（stem 3×3×3→3×3×4）」——那是按 3×3 卷积、3 个输出通道估的。实际 `R50-ViT-B_16` 的 hybrid stem 是 **7×7、64 输出通道**，3→4 通道的真实增量是 **64×7×7 = 3136 参数**。结论（近乎零成本的单变量消融）不变。

#### 4.7.2b 边缘融合先行诊断（`scripts/probe_edge_signal.py`，真实病人数据）

正式 200-epoch job 在接力队列里尚未开跑，为给方向性结论，先用**真实病人数据**做了两段诊断（SPARC + Zhongnan 两个子集各跑一遍）。

**Part A：四个边缘算子与斑块真值边界的吻合度**（test 集；「边界命中率 ×随机」= 算子在边界带上的响应密度 ÷ 随机落点密度；「分离比」= 带内平均响应 ÷ 带外平均响应；AUROC 用秩和法）

| 算子 | SPARC 命中率×随机 | SPARC 分离比 | SPARC AUROC | Zhongnan 命中率×随机 | Zhongnan 分离比 | Zhongnan AUROC |
|---|---|---|---|---|---|---|
| 小波高频 IDWT（**本次采用**） | 1.42x | 2.07 | 0.723 | 1.71x | 1.78 | 0.657 |
| Sobel 梯度幅值（对照） | 1.07x | 1.84 | 0.717 | 1.17x | 1.62 | 0.676 |
| Laplacian 幅值（对照） | **0.44x** | 1.14 | 0.653 | **0.59x** | 0.88 | 0.566 |
| 小波细节能量图（旧方案） | **1.63x** | 2.05 | **0.759** | **2.02x** | 1.73 | **0.715** |

结论：老师定的 **HF-IDWT 算子成立**——在两个子集上都稳定优于 Sobel 与 Laplacian（Laplacian 甚至低于随机水平，说明它跟斑块边界不是一回事）。但**旧的「细节能量图」在纯边界定位上更好**（1.63x/2.02x、AUROC 0.759/0.715）——两者定位目标不同：IDWT 重建图保的是「高频结构」不只是「边界」。**若正式 edge job 无增益，第一备选就是把第 4 通道换成细节能量图**（改动仅 `build_hf_image` 一行）。

**Part B：门槛实验——多一个 HF 通道到底有没有信息增量**（0.061M 小 CNN、20 epoch、Adam 2e-3 + CosineAnnealingLR、单 seed、无增强）

| 子集 | 变体 raw（1 通道） | 变体 raw+HF（2 通道） | 变体 HF 单独（1 通道） | raw+HF − raw |
|---|---|---|---|---|
| SPARC（train 350 / test 160） | 0.4893 | **0.5075** | 0.4930 | **+0.0181** |
| Zhongnan（train 448 / test 187） | 0.6861 | **0.7044** | 0.6783 | **+0.0183** |

结论：**HF 通道确实携带原图里没有的信息**（两个子集 +0.018/+0.018，且 HF 单独的表现已接近原图单独，说明它不是噪声）。

**不可外推的边界（必须随结论一起说）**：

- 小网络 0.061M、20 epoch、无数据增强、单 seed、n=160/187 切片，**不是病例级配对**（同一切片同时出现在两边），无法做病例级配对 Wilcoxon 检验；
- 真模型基线 Dice 已 **90.49%**，小网络门槛实验的 +1.8% 绝对增量在真模型上**大概率远小于 1.8%**，甚至可能被 105M 参数的容量优势淹没；
- 因此 Part B 只回答「信息增不增量」，**不回答「真模型上赚不赚」**——后者必须等正式 200-epoch job 的配对结果（HD95 ↓≥0.5 mm 且 p<0.05）。

#### 4.7.3 隐私处理（病人数据不入仓库）

- `.gitignore` 重写：新增 `*.nii` / `*.nii.gz` / `*.dcm` / `patient*` / `case*` / `*.pth` / `*.npz` / `model/` / `datasets/split/`；图片与 PDF 改为「除 `docs/assets/` 下的脱敏论文插图外一律忽略」。
- `git rm --cached` 移除 4 个**已跟踪**且含病人标识的历史文件：`datasets/split/filelist_train101`、`filelist_test43`、`trainfile_zhongnan.txt`、`testfile_zhongnan.txt`（内容含 `667532_04-30-2019`、`1334`、`107726_02-17-2015` 等）。
- `git ls-files` 复查：现无任何 split / nii / pth / npz 跟踪文件，仓库里只剩代码与文档（49 个文件）。
- **残留提醒**：病人图像仍留在 **git 历史**里（`docs/assets/imaging-basics/multi-case-*.png` 是脱敏后的论文插图，已跟踪）。若要彻底清除需 `git filter-repo` 重写历史 —— 这会改写全部 commit hash，需用户决定后再做。

## 5. 已知风险

- 8GB 显存：ViT-B 在 224 分辨率下 batch 超过 8 可能 OOM；当前 batch 8 峰值约 3.3GB。**AMP 尚未接入**（trainer 无 autocast/GradScaler），双输入若超限优先降 batch + 梯度累积。
- 数据集样本量小（1148 切片 / 641 例），从零训练容易欠拟合，基线指标可能偏低。
- 两子集设备与增益不同，需关注域差异带来的指标波动。
