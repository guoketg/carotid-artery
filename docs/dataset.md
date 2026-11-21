# 数据集说明

## 1. 旧数据集（`F:\code\datasets\2D`，不在仓库内，不提交）

| 子集 | 病例数 | 图像数 |
|---|---|---|
| `PlaqueDataNii_SPARC_M` | 144 | 510 |
| `PlaqueDataNii_Zhongnan_M` | 497 | 638 |

- 图像与标签同名，标签后缀 `_label.nii.gz`；2D NIfTI，float32，强度 0~255；标签二值 0/1。
- patientId：SPARC 取目录名 `_` 前段，Zhongnan 目录名即 ID；跨子集加前缀 `SPARC:` / `ZN:`。
- 划分（**已作废口径，仅存档**）：两子集完全分开，不设验证集；曾按 patientId 分组 8:2 train/test（seed 1234），文件在 `lists/lists_Carotid/<Subset>/{train,test}.txt`。2026-11-17 起划分一律改用老师提供的官方划分文件（见 §4 与 [CLAUDE.md](../CLAUDE.md) 第 1 条），不再自行按病例/切片重划。
  - SPARC：116 例/428 张 train，28 例/82 张 test；Zhongnan：398 例/518 张 train，99 例/120 张 test。
- dev 快筛子集（`lists_Carotid_dev`）已撤销，脚本保留备用。

## 2. 预处理

1. 1%~99% 分位截断，按原图统计量 z-score（单通道）。
2. 双线性缩放到 224×224，标签最近邻。
3. 训练增强：90° 随机旋转、随机翻转、±20° 小角度旋转；验证/测试只缩放。
4. 三通道版本（早融合 / 小波引导注意力）：原图、去噪重建、高频能量通道，共用同一 mean/std。

## 3. 指标（2026-10-26 起物理单位口径，旧像素口径数字不可直接比）

| 指标 | 定义 | 单位 |
|---|---|---|
| Dice | `2|P∩G| / (|P|+|G|)`，双空记 1 | 无量纲 |
| ASSD | medpy `assd`，NIfTI 头 spacing | mm |
| HD | medpy `hd`，最大表面距离，同上 | mm |
| DeltaArea | `area(P) − area(G)`，正值=预测偏大 | mm² |

- 预测经 `ndi.zoom` 最近邻重采样回原分辨率网格再计算（[metrics_carotid.py](../metrics_carotid.py) `original_grid_pair`）。
- 一侧为空时距离指标取图像对角线物理长度作最坏罚分。
- 历史 metrics.csv 与 test 曲线为像素口径，跨口径比较只看 Dice 或注明口径。

## 4. Carotid2D 数据集（2026-10-26 老师提供，现役）

- 根目录 `E:\mkl\code\datasets\Carotid2D`，子集 `PlaqueDataNii_SPARC_S` / `PlaqueDataNii_Zhongnan_S`，标签规则不变。
- **现役划分（2026-11-21 起）**：同组研究生提供的划分文件，已落库为 `lists/lists_official/<Subset>/{train,val,test}.txt`，用 `--list_dir ./lists/lists_official` 启用，`val` 承担 best-val 选模：
  - SPARC 280 / 70 / 160 张；Zhongnan 357 / 88 / 191 张（原始 txt 里 Zhongnan train/val 各多一行重复，按文件名去重后得到此数）。
  - 两子集内部 train/val/test 患者与切片互不重叠；我们与研究生在同一批 Zhongnan 切片上缺 `20428_3` 一张（我们 635、他 636）。
- **旧划分（历史臂口径，2026-11-20 及之前）**：SPARC 350 / 160、Zhongnan 448 / 187，文件在 `lists/lists_Carotid2D/<Subset>/{train,test}.txt`，不设验证集、终点 ep200 取末轮。
  - 旧文档与 results/experiments 里出现的「官方划分」「SPARC 官方划分 n=160」「448/187」「350/160」**一律指这一套**，只服务历史表格，不得用于新实验。
  - 两套划分不可比：同一 A0 s42 权重，旧 187 例 Dice 0.9210、研究生 191 例 0.8992，**+2.18 点纯来自划分差异**；两套 test 的切片交集只有 66 张，94 张落在他的 train、27 张在他的 val。
  - 旧划分内部已核对：train/test 患者交集 0、切片交集 0（无泄漏），是同一批切片的另一次随机切分，不是泄漏。
- 快照目录自动带 `_c2d` 后缀；`train.py`、`scripts/eval_carotid.py` 默认值已切到该根目录。
- 待确认：`_M`/`_S` 预处理是否相同。
- **全部 2026-11-20 及之前的已存结果都在旧划分上取得，与新配方/新划分的数字不可同表比较。**
