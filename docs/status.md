# 状态与环境校验记录

## 2026-10-23/24 夜间队列

- 已实现：`networks/wavelet.py`（Haar DWT/IDWT + 软阈值 + 三通道）、`networks/wavelet_model.py`（小波分支 + WGA 交叉注意力 / SE 对照）；`train.py` 新增 `--use_wavelet`、`--fusion {none,se,cross}`、`--heads`，默认 `batch_size=8`。
- 融合位置为 **ViT 输出瓶颈（14×14，196 token）**，不是原设计的 stage2/3 skip：实现简单、注意力开销低，baseline 完全不受影响。偏差已记入 `technical-design.md`。
- batch=8 实测：峰值显存 3.3GB，纯模型单步 0.36s，含数据约 0.5s，SPARC 约 27s/epoch。
- 夜间队列 6 个任务，统一 `--max_epochs 120 --eval_test_every 10`，顺序 a0_sparc → a3_sparc → a1_sparc → a0_zhongnan → a3_zhongnan → a1_zhongnan，预计 6 小时。
- 冒烟（1 epoch，仅验证链路）：A1 dice 0.745，A3 dice 0.756。

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

## 4. 进度

- [x] 确认代码基座与环境
- [x] 下载并校验预训练编码器
- [x] 确认数据集位置与格式（641 例 / 1148 张）
- [x] 补齐 venv 依赖
- [x] 重写 `requirements.txt` 并切换 origin
- [x] 修复 Windows 路径分隔符 bug
- [ ] 接入数据集与病例级划分
- [ ] baseline（A0）跑通并产出四项指标（SPARC / Zhongnan 分开）
- [x] 训练入口已切回官方 `train.py`（200 epoch / bs2 / SGD / workers1 / 预训练启用）
- [ ] 实现 `networks/wavelet.py`
- [ ] 实现 `networks/fusion.py`
- [ ] 完成 A0/A1/A2/A3 消融与统计检验

## 5. 已知风险

- 8GB 显存：ViT-B 在 224 分辨率下 batch 超过 8 可能 OOM，需保持 AMP 与梯度累积。
- 数据集样本量小（1148 切片 / 641 例），从零训练容易欠拟合，基线指标可能偏低。
- 两子集设备与增益不同，需关注域差异带来的指标波动。
