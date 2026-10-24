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

- Python 3.12.10，torch 2.14.1+cu132，RTX 5060 Laptop 8GB。
- 不新建 conda 环境，不使用 Docker。
- 新依赖只装进 `venv`，并同步更新 `requirements.txt`。
- **禁止** 用旧版 `torch==1.4.0` 覆盖现有环境。

## 3. 硬约束

1. 数据划分**必须按病例**（patient-level），禁止切片级随机划分。
2. 实验输出不得覆盖 baseline 权重，目录名带方法后缀。
3. 训练显存上限 8GB：`img_size=224`、`batch<=8`、开启 AMP；不够就用梯度累积。
4. 数据加载固定 `num_workers=1`，不要调大：本机内存紧张，WSL 同时开启时会冲到 95%，需要用户手动杀进程。
5. 指标口径固定：**Dice、ASSD、HD95、DeltaArea**（预测面积 − 真值面积）。
6. 冒烟测试样本 ≤ 50，避免无意义开销。
7. 关键决策、路径、超参变更必须回写 `docs/` 并在本文件索引。
8. epoch 选择：先用 `--eval_test_every 5` 跑出 test 曲线，挑一个整数 epoch 固定；后续所有消融复用该 epoch，不再反复在测试集上调。

## 4. 文档索引

| 文档 | 内容 |
|---|---|
| [docs/technical-design.md](docs/technical-design.md) | 小波预处理、三通道、分支结构、注意力融合的设计决策与理由 |
| [docs/dataset.md](docs/dataset.md) | 数据集结构、划分规则、预处理与归一化 |
| [docs/experiments.md](docs/experiments.md) | 消融矩阵、评估协议、统计检验、运行命令 |
| [docs/status.md](docs/status.md) | 环境与权重校验记录、进度清单、已知问题 |

## 5. 当前状态

- 环境、预训练权重（R50+ViT-B_16）、依赖、Windows `pjoin` 修复：已完成，见 `docs/status.md`。
- 进行中：接入颈动脉数据集，跑通 baseline 并给出四项指标。
