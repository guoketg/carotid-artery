# 数据集说明

## 1. 旧数据集（`F:\code\datasets\2D`，不在仓库内，不提交）

| 子集 | 病例数 | 图像数 |
|---|---|---|
| `PlaqueDataNii_SPARC_M` | 144 | 510 |
| `PlaqueDataNii_Zhongnan_M` | 497 | 638 |

- 图像与标签同名，标签后缀 `_label.nii.gz`；2D NIfTI，float32，强度 0~255；标签二值 0/1。
- patientId：SPARC 取目录名 `_` 前段，Zhongnan 目录名即 ID；跨子集加前缀 `SPARC:` / `ZN:`。
- 划分：两子集完全分开，不设验证集；按 patientId 分组 8:2 train/test（seed 1234），文件在 `lists/lists_Carotid/<Subset>/{train,test}.txt`，由 `python datasets/dataset_carotid.py` 生成。
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
| HD95 | medpy `hd95`，同上 | mm |
| DeltaArea | `area(P) − area(G)`，正值=预测偏大 | mm² |

- 预测经 `ndi.zoom` 最近邻重采样回原分辨率网格再计算（[metrics_carotid.py](../metrics_carotid.py) `original_grid_pair`）。
- 一侧为空时距离指标取图像对角线物理长度作最坏罚分。
- 历史 metrics.csv 与 test 曲线为像素口径，跨口径比较只看 Dice 或注明口径。

## 4. Carotid2D 新数据集（2026-10-26 老师提供，已接入，现役口径）

- 根目录 `F:\code\datasets\Carotid2D`，子集 `PlaqueDataNii_SPARC_S` / `PlaqueDataNii_Zhongnan_S`，标签规则不变。
- 官方划分文件 `datasets/split/`：SPARC train 350 张 / test 160 张，Zhongnan train 448 张 / test 187 张。
- 已转换为 `lists/lists_Carotid2D/<Subset>/{train,test}.txt`；`train.py`、`scripts/eval_carotid.py` 默认值均已切到新根与新划分；快照目录自动带 `_c2d` 后缀。
- SPARC 官方划分为 101 例 train / 43 例 test（文件名即病例数），与旧自划分 116/28 不同；43 例测试集对 +1.0 Dice 勉强够功效，对 +0.5 仍不够。
- 待确认：`_M`/`_S` 预处理是否相同；Zhongnan 新口径 635 张 vs 旧 638 张差 3 张。
- **全部已存结果（含单通道基线 90.92%）在旧划分上取得，新划分上须重跑基线。**
- 冒烟已验证：`--max_epochs 2` 全链路跑通，测试集 n=160。
