# 数据集说明

## 1. 位置与规模

根目录：`F:\code\datasets\2D`（不在仓库内，不提交）

| 子集 | 病例目录数 | 图像数 |
|---|---|---|
| `PlaqueDataNii_SPARC_M` | 144 | 510 |
| `PlaqueDataNii_Zhongnan_M` | 497 | 638 |
| 合计 | 641 | 1148 |

## 2. 目录与配对规则

```
F:\code\datasets\2D\
  PlaqueDataNii_SPARC_M\
    image_Niilist.txt          # 相对路径：./PlaqueDataNii_SPARC_M/<case>/<name>.nii.gz
    label_Niilist.txt          # 与图像一一对应
    <case>\<name>.nii.gz
    <case>\<name>_label.nii.gz
  PlaqueDataNii_Zhongnan_M\
    ...（结构相同）
```

- 图像与标签同名，标签后缀 `_label.nii.gz`。
- 切片为 2D NIfTI（示例 `159×239`、`260×390`），float32，强度 0~255。
- 标签为二值 0/1，示例前景占比 6%~12%。
- 列表中的路径以 `2D` 根为起点。

## 3. 病例标识

- SPARC：目录名形如 `<patientId>_<date>`，取 `_` 前部分为 patientId。
- Zhongnan：目录名形如 `0001`、`0002`，目录名即 patientId。
- 为避免跨子集 ID 冲突，统一加前缀：`SPARC:<id>` / `ZN:<id>`。

## 4. 划分规则

- **两个子集完全分开**：SPARC 与 Zhongnan 各自训练、各自测试，不合并。
- **不设验证集**（最终评估标准由老师后续给出，届时按新标准重跑）。
- 每个子集内部按 patientId 分组，同一病人所有切片只出现在一个 split。
- 默认 8:2 病例级 train/test，seed 固定；划分文件：
  ```
  lists/lists_Carotid/SPARC/{train,test}.txt
  lists/lists_Carotid/Zhongnan/{train,test}.txt
  ```
- 当前划分结果（seed 1234）：

| 子集 | train | test |
|---|---|---|
| SPARC | 116 例 / 428 张 | 28 例 / 82 张 |
| Zhongnan | 398 例 / 518 张 | 99 例 / 120 张 |

- 由 `python datasets/dataset_carotid.py` 生成。

### 4.1 dev 快筛子集（`lists_Carotid_dev`，**已撤销，脚本保留**）

> 2026-10-24 23:50 撤销：全量 SPARC 实测只需 2.5 h/job，一晚上够跑 3 个完整方案，不需要降级到子集。
> 下列产物与脚本保留备用，但**当晚训练未使用**，`_dev` 中间结果已删除。

夜间用全量数据只能跑 6 个方案（每个 1.2–1.5 h），做单变量对比样本太少，因此曾另建一个**病例级** dev 子集：

- 生成：`python scripts\build_dev_subset.py --subset SPARC --ratio 0.4`
- **train**：按 patientId 抽 40% 病例（SPARC 46 例 / 175 张，原 116 例 / 428 张）；病例整体进出，不拆切片。
- **test**：原样复制官方 holdout（SPARC 28 例 / 82 张），因此 dev 与全量**共用同一测试集**，数字可比。
- 训练侧用 `--list_dir ./lists/lists_Carotid_dev` 切换；`train.py` 自动给 snapshot 目录加 `_dev` 后缀，
  快筛结果与全量结果物理隔离。
- 用途限定：只做**方案排序**，60 epoch 未收敛，绝对值不可与全量 120 epoch 比较。

## 5. 预处理（单通道原图基线与三通道早融合一致）

1. 读取 NIfTI → 取 2D 强度图。
2. 1%~99% 分位截断，按原图统计量做 z-score（单通道）。
3. 双线性缩放到 `224×224`，标签用最近邻。
4. 训练增强：90° 随机旋转、随机翻转、±20° 小角度旋转；验证/测试只用缩放。
5. 三通道早融合与三通道＋小波引导注意力的三通道版本：原图通道、去噪重建通道、高频能量通道，共用同一 mean/std。

## 6. 指标定义（**2026-10-26 起切换物理单位，旧数字为像素口径不可直接比**）

| 指标 | 定义 | 单位 |
|---|---|---|
| Dice | `2|P∩G| / (|P|+|G|)`，空预测且空真值记 1 | 无量纲 |
| ASSD | 平均对称表面距离（medpy `assd`，`voxelspacing`=NIfTI 头分辨率） | **mm** |
| HD95 | 95% 豪斯多夫距离（medpy `hd95`，同上） | **mm** |
| DeltaArea | `area(P) − area(G)`，正值=预测偏大，同时报绝对值均值 | **mm²** |

口径说明（2026-10-26 起）：

- 原图为非正方形（约 2:3）且面内分辨率约 0.065~0.068 mm/px；训练/预测在 224×224 正方形网格上进行，**距离指标不在 224 网格上算**，而是把预测用 `ndi.zoom` 最近邻重采样回原分辨率网格（[metrics_carotid.py](../metrics_carotid.py) `original_grid_pair`），与原分辨率 GT、NIfTI 头 spacing 一起交给 medpy 计算，assd/hd95 直接得 mm、DeltaArea 直接得 mm²。
- 验证（模拟预测对照实验，2026-10-26）：重采样路径相对原网格物理参考 assd 误差 ≤9.2%、hd95 ≤17.4%（小目标量化误差上界），Dice 与 224 网格差 <0.6%。
- **历史 metrics.csv 与 test 曲线均为像素口径**（assd/hd95 为 224 网格像素数、DeltaArea 为像素数），与本口径不可直接比，比较时只看 Dice 或注明口径。
- 一侧为空时距离指标取图像对角线物理长度作为最坏罚分（`_surface_fallback`）。

## 7. Carotid2D 新数据集与官方划分（2026-10-26 老师提供，**已接入代码**）

### 7.1 位置与划分文件

- 数据根目录：`F:\code\datasets\Carotid2D`（不在仓库内）。
- 划分文件（仓库内）：`datasets/split/`，路径均以 `./` 开头、以该根目录为起点，标签规则不变（同名 + `_label.nii.gz`）。

| 文件 | 内容 | 图像数 |
|---|---|---|
| `filelist_train101` | SPARC 训练列表 | 350 |
| `filelist_test43` | SPARC 测试列表 | 160 |
| `trainfile_zhongnan.txt` | Zhongnan 训练列表 | 448 |
| `testfile_zhongnan.txt` | Zhongnan 测试列表 | 187 |

- 文件名中的 101 / 43 **推断**为 train/test 的病例数（总数 144 例与旧划分一致，待核对）。
- SPARC 图像 350+160=510 张，与旧 `PlaqueDataNii_SPARC_M` 总数一致；**疑似同一批图像把目录后缀从 `_M` 换成 `_S` 后重新划分**，`_M`/`_S` 预处理是否不同待向老师确认。
- Zhongnan 448+187=635 张，旧口径为 638 张（差 3 张，待核对）；目录名即病例 ID（如 `1334`、`20750`），与旧规则兼容。

### 7.2 与旧口径对比（**新旧数字不可混排**）

| 项 | 旧（现役代码） | 新（老师提供） |
|---|---|---|
| 根目录 | `F:\code\datasets\2D` | `F:\code\datasets\Carotid2D` |
| 子集目录 | `PlaqueDataNii_SPARC_M` / `PlaqueDataNii_Zhongnan_M` | `PlaqueDataNii_SPARC_S` / `PlaqueDataNii_Zhongnan_S` |
| SPARC 划分 | 116 例/428 张训练，28 例/82 张测试（自划分，seed 1234） | **101 例/350 张训练，43 例/160 张测试（官方给定）** |
| 划分文件 | `lists/lists_Carotid/<Subset>/{train,test}.txt` | `datasets/split/` 下单个文件（两种命名，无扩展名） |
| 测试集功效 | 28 例 | 43 例：对 +1.0 Dice（约需 32 对）勉强够，对 +0.5（约需 128 对）仍不够 |

- **全部已存结果（单通道基线 0.9092 等）都在旧划分上取得，切换后必须在新划分上重跑基线。**

### 7.3 接入所需改动（**2026-10-26 已全部实施**）

现行代码把数据根与划分写死，新划分不能零改动接入：

1. `datasets/dataset_carotid.py` 顶部 `DATA_ROOT = F:\code\datasets\2D`、`SUBSETS` 指向 `_M` 目录（`build_splits` 只用于再生成旧划分，可不改）。
2. `train.py` 的 dataset_config **无条件覆盖** `--root_path`（只有 `--list_dir` 允许 CLI 覆盖），Carotid 的 `root_path` 需改为 `F:\code\datasets\Carotid2D`（或放开 CLI 覆盖）。**已实施**：L92-93 root_path/list_dir 已切到新根与新划分。
3. `trainer.py` 只认 `<list_dir>/<Subset>/train.txt|test.txt` 结构，需把 `datasets/split/` 的四个文件转换为 `lists/lists_Carotid2D/SPARC/{train,test}.txt` 与 `lists/lists_Carotid2D/Zhongnan/{train,test}.txt`（内容去掉行首 `./` 即可，`read_list` 本身兼容 `./`，转换只为路径拼装）。**已实施**：四个文件由用户创建，冒烟加载确认 160 例测试集。
4. `patient_key` 对新路径天然兼容（SPARC 目录名含 `SPARC`；Zhongnan 目录名即 ID），无需改。
5. `scripts/eval_carotid.py` L102-103 默认值已切 `F:\code\datasets\Carotid2D` / `./lists/lists_Carotid2D`。

冒烟验证（2026-10-26）：`--max_epochs 2` 全链路跑通，测试集 n=160（与官方 test43 的 160 张一致），快照目录自动带 `_c2d` 后缀（train.py L138-139）。
