# 实验方案

## 1. 消融矩阵

| 编号 | 配置 | 验证目标 |
|---|---|---|
| A0 | 原图单通道 TransUNet | baseline |
| A1 | C1/C2/C3 三通道早融合，无注意力 | 预处理本身是否有效 |
| A2 | 原图复制 3 通道 + WGA（伪分支） | 融合结构本身是否有效 |
| A3 | 三通道 + 轻量小波分支 + WGA | 完整方法 |

附加实验：小波级数（1 vs 2）、软阈值开关、融合位置（stage2 / stage3 / 两处）、CLAHE 替代 C2、DoG 替代 C3。

## 2. 评估协议

- **两个子集分开**：SPARC 与 Zhongnan 各自训练、各自测试，不混合。
- 暂不设验证集，训练固定轮数后直接用测试集评估；最终标准待老师给出后按其定义重跑。
- 指标：Dice、ASSD、HD95、DeltaArea（预测面积 − 真值面积）。
- 显著性：paired t-test 或 Wilcoxon signed-rank，p < 0.05。
- 必须同时报告 Params、FLOPs、单张推理时延，排除"涨点来自更大模型"。
- 斑块为小目标：损失用 `Dice + BCE`（或 CE+Dice），重点看 HD95 与 DeltaArea 的偏差方向。
- 失败案例：低回声斑块、声影区、钙化伴声影、增益过曝。

## 3. 训练超参（沿用用户已调好的官方脚本配置）

训练入口统一使用官方 [train.py](file:///f:/code/MyPython/TransUNet/train.py) + [trainer.py](file:///f:/code/MyPython/TransUNet/trainer.py)，只把数据换成了颈动脉数据集。

| 项 | 值 |
|---|---|
| 输入 | 224×224，单通道（A0） |
| batch | **2** |
| epoch | **200** |
| 优化器 | **SGD**，lr 0.01，momentum 0.9，weight_decay 1e-4，poly 衰减（power 0.9） |
| 损失 | 0.5 × CE + 0.5 × Dice |
| 预训练 | **启用**：`R50+ViT-B_16.npz`（ImageNet21k） |
| 数据加载 | `num_workers=1`（Windows 内存限制） |
| AMP | 官方脚本未启用 |
| test 曲线 | 每 5 个 epoch 在测试集上算一次四项指标（`--eval_test_every 5`，0 关闭） |

### epoch 选择策略（用户指定）

1. 先用一轮 200 epoch 的训练，每 5 epoch 记录一次测试集指标，写入 `test_curve.csv`；
2. 从曲线上挑一个**整数 epoch**（Dice 高、HD95/ASSD 不过差的区间中点，避免贴住偶然峰值）；
3. 之后所有消融实验（A0/A1/A3、SPARC/Zhongnan）统一固定该 epoch 训练，不再在测试集上早停或调 epoch。


## 4. 运行命令

```powershell
# 生成病例级划分（每个子集独立，train/test）
.\venv\Scripts\python.exe datasets\dataset_carotid.py

# 夜间队列（顺序执行，一次只跑一个）
powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1 -DryRun   # 只打印计划
powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1            # 正式执行

# baseline（A0）：两个子集分别跑（官方 train.py，200 epoch / batch 2 / SGD / workers 1）
.\venv\Scripts\python.exe train.py --subset SPARC
.\venv\Scripts\python.exe train.py --subset Zhongnan
```

## 5. 日志与产物

- 输出目录：`../model/TU_Carotid224/TU_pretrain_R50-ViT-B_16_skip3_epo200_bs2_224_<subset>/`
  - `log.txt`：完整训练日志
  - `epoch_*.pth`：每 50 个 epoch 及最后一轮的权重
  - `metrics.csv`：测试集逐样本四项指标（trainer 末尾自动执行）
  - `log/`：tensorboardX 事件
- 禁止复用 baseline 目录名，避免权重互相覆盖。
