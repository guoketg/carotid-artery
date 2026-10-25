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

## 6. 指标定义

| 指标 | 定义 |
|---|---|
| Dice | `2|P∩G| / (|P|+|G|)`，空预测且空真值记 1 |
| ASSD | 平均对称表面距离（像素/体素单位） |
| HD95 | 95% 豪斯多夫距离 |
| DeltaArea | `area(P) − area(G)` 像素数，正值=预测偏大，同时报绝对值均值 |
