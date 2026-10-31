# 项目状态（精简版）

> 本文件只回答三件事：**现在在跑什么 / 已经得到了什么结论 / 下一步等什么**。
> **结果数字统一记在 [results.md](results.md)**（纯结果台账，本文件不放大表）；计划与判定规则见 [experiments.md](experiments.md)。
> 2026-10-23 ~ 10-27 的完整过程记录已归档到 [archive/status-history-2026-10.md](archive/status-history-2026-10.md)。
> 术语解释见 [study/edge-fusion-explained.md](study/edge-fusion-explained.md)；融合算子逐个讲解见 [study/fusion-operators-explained.md](study/fusion-operators-explained.md)。

最后更新：2026-10-30（判定口径统一为**终点 epoch 200**：`eval_batch_ep199.ps1` 16 job 全部重算完成，results.md 全表换 ep200 数字；新增 pitfalls.md 集中记录踩坑；GPU 当前空闲，无实验在跑）

---

## 1. 环境（只列会变的东西）

| 项 | 值 |
|---|---|
| 代码根 | `E:\mkl\code\python_code\carotid-artery` |
| 数据根 | `E:\mkl\code\datasets\Carotid2D`（`PlaqueDataNii_SPARC_S` 510 图 / `PlaqueDataNii_Zhongnan_S` 636 图） |
| 权重根 | `E:\mkl\code\python_code\model`（`vit_checkpoint\imagenet21k\R50+ViT-B_16.npz`、`TU_Carotid224` 快照） |
| Python | `venv\Scripts\python.exe`（3.11.5，numpy 2.4.6 / scipy 1.17.1） |
| GPU | RTX 3060 12GB |
| 划分 | 官方划分 `lists/lists_Carotid2D/{SPARC,Zhongnan}/{train,test}.txt`，病例级无泄漏 |

---

## 2. 队列状态（已全部跑完，GPU 空闲）

**2026-10-30 04:32 最后一队收口**：文献复现三网络（U-Net/XNet/GobletNet）跑完 exit 0×3，至此全部训练与评测队列收口，无失败 job（唯一异常是融合算子 AFF 首跑在 ep5 因 GroupNorm 崩溃，修复后于批次 0+1 重跑成功）。评测产物（终点 epoch 200 口径）：各快照目录 `metrics_ep199.csv` + `logs/queue/eval199_*.log`（16 快照，2026-10-30 18:10 收口）。当前 GPU 空闲（无 train.py 进程）。

| 队列 | job 数 | 内容 | 状态 CSV |
|---|---|---|---|
| `queue_c2d.ps1`（主队列） | 6 | 单骨干基线 seed2024/3407、双骨干跨支 QKV 注意力 seed2024/3407、输入级边缘图（基线+1 通道）、边缘图+双骨干 | `logs/queue/queue_c2d_status.csv` |
| `queue_c2d_fuse.ps1`（融合算子横评） | 6 | AFF（崩溃 1 次）、gate 门控、deform 可变形大核、DDBE 单级挂载、基线 seed1234/3407 | `logs/queue/queue_c2d_fuse_status.csv` |
| `queue_c2d_repro.ps1`（批次 0+1） | 4 | DDBE 四级全挂、DDBE×4+ACAM（BDFR 完整版）、AFF 重跑、主路吃 LL 低频图（XNet 便宜变体） | `logs/queue/queue_c2d_repro_status.csv` |
| `eval_batch_ep199.ps1`（评测） | 16 | 终点 epoch 200 重算 + 病例级 Wilcoxon，16 个全部 exit 0（含补记快照 edgedeform） | `logs/queue/eval_batch_ep199_status.csv` |
| `queue_repro_nets.ps1`（文献复现） | 3 | 纯卷积 U-Net / XNet / GobletNet（官方划分 SPARC、seed1234、200ep），全部 exit 0 | `logs/queue/queue_repro_nets_status.csv` |

**队列启动方式必须用 WMI**（否则训练会被终端回收，踩过坑）：

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -File "...\queue_c2d_repro.ps1"'}
```

---

## 3. 已经得到的结论

### 3.1 基线与横评（口径：官方划分 + bs4/ga2 + 200 epoch，判定点 **终点 epoch 200**，SPARC test n=160）

**全部结果数字（3-seed 基线/双骨干、单 seed 变体横评、复现三网络）统一见 [results.md](results.md) §1–§4**，此处只留读法：

- A0 三 seed 均值 **90.32%**、dual auto 三 seed 均值 **90.32%**（完全持平）→ 双骨干判负。
- 单 seed 横评最佳点：edgedual **90.74%** → ddbefull 90.65% → deform / edgedeform 90.60%——均为单 seed 方向证据，无一达到幅度门槛（|ΔDice| ≥ 0.5% 或 HD95 降 ≥ 0.5 mm）。
- 补记快照 `*_SPARC_edge_dual_L1234_deform_c2d`（边缘图 + 双骨干 deform）已纳入本轮 epoch_199 评测，90.60% / 0.1410 / 0.61 / 0.3769（见 results.md §2）。

### 3.2 边缘融合（输入级）：信息增量确认，正式训练未过门槛

代码位置：[networks/wavelet.py](../networks/wavelet.py) `build_hf_image()`；开关 `train.py --edge_in 1`。

做法：stem 吃 4 通道 = 原图 3 通道 + 1 张小波高频重建边缘图。

- **先行诊断**（小网络门槛实验）：SPARC +1.81% / Zhongnan +1.83%，两个子集一致 → 边缘通道确实携带原图没有的信息。
- **正式训练**（§1.8）：单骨干 edge ΔDice −0.01%（无差异）；双骨干 edgedual 90.74% 全场最佳但 ΔDice +0.30% → **线索保留，不作结论**。备选：高频能量图（改一行）。

术语与指标含义看不懂 → [study/edge-fusion-explained.md](study/edge-fusion-explained.md)。

### 3.3 踩过的坑（避免重犯）

→ 全部集中记录在 **[pitfalls.md](pitfalls.md)**（进程管理 / Windows 环境 / 网络实现 / 实验管理 / 数据口径 五类），新坑持续追加到该文件。

### 3.4 文献复现三网络（2026-10-30 队列跑完，exit 0×3）

论文底座复现线：XNet（ICCV 2023 小波低/高频双分支 + X 形融合，双输出各自监督）、GobletNet（TMI 2024 双编码器 + FAM 逐级融合）、以及与两者同积木同容量的纯卷积 U-Net 基线。三个网络各自独立训练（`--arch unet|xnet|gobletnet`），与 baseline/dual 线互不混用；官方划分 SPARC、seed1234、200 epoch、等效 batch 8、全部 from scratch（无预训练可载——自定义 block 架构与任何公开 checkpoint 不匹配，且两篇论文官方协议即从头训）。

ep200 指标（Dice / ASSD / HD / HD95、参数量、训练耗时）**完整表见 [results.md](results.md) §4**，读法：

1. **XNet vs 同底座 U-Net 基线：方向为正**（87.06% vs 86.55%，+0.5%）——低频/高频双分支 + X 形融合在这个数据上没有兑现论文声称的收益，方向证据保留、不作结论。
2. **GobletNet 更差**（86.28%，比 U-Net 低 0.3%，边界两项也更差），133M 参数但双编码器 + FAM 在小数据上更难训（ep5-10 曲线也落后）。
3. **纯卷积复现线整体比 TransUNet 基线（90.32%）低约 4 个点**：这是不同骨干家族的容量/归纳偏置差异，**不是**实验结论的直接比较对象——复现线内部（U-Net ↔ XNet ↔ GobletNet）同积木同口径互比才有效。
4. 口径注意：本线读数（含 2026-10-31 补算的 HD）取 `epoch_199.pth` 重推理，dice/assd/hd95 与 `test_curve.csv` ep200 行一致（终点 epoch 200，与全项目口径一致）；跨线比较只看方向不看小数位。

快照目录（权重根 `E:\mkl\code\python_code\model\TU_Carotid224\`）：`..._epo200_bs4_ga2_224_SPARC_unet_c2d`、`..._epo200_bs2_ga4_224_SPARC_xnet_c2d`、`..._epo200_bs4_ga2_224_SPARC_gobletnet_c2d`。队列脚本 [scripts/queue_repro_nets.ps1](../scripts/queue_repro_nets.ps1)，状态 `logs/queue/queue_repro_nets_status.csv`。网络实现：[networks/unet.py](../networks/unet.py)、[networks/xnet.py](../networks/xnet.py)、[networks/gobletnet.py](../networks/gobletnet.py)。

---

## 4. 下一步等什么

1. **批次 2 已实现未入队**（GobletNet 式：单主干 + 轻 HF 支 + FAM 逐层）：冒烟 5 项全过（恒等起步 max|Δlogits|=0、+1.206M 参数），入队命令 `--hf_branch 1 --hf_lambda 0.3`（快照 `*_SPARC_hfbr3_c2d`），批次 0+1 结论已落，可入队。实现细节见 [experiments.md](experiments.md) §1.9 执行记录②。
2. **批次 3（Syn-Net）改方案待定**：老师要求先分别做局部支/全局支、再两个骨干一起看是否有真涨点——方案讨论中，`--fuse_op dct` 暂缓实现（见 experiments.md §1.9 批次 3）。批次 4（WMREN WRDM）视结论排期。
3. 全部结论目前只在 SPARC 单子集成立；尚无变体达到幅度门槛（|ΔDice| ≥ 0.5% 或 HD95 降 ≥ 0.5 mm）——edgedual / ddbefull / deform / edgedeform 四个方向为正的点，择优去 Zhongnan 复现。
4. 病人图像仍在 git 历史里，彻底清除需 `git filter-repo` 重写历史（会改全部 commit hash，**待用户批准**）。

---

## 5. 已知风险

- 病人隐私：工作区与 git 历史仍有病人图像/标识文件（已从当前索引移除，历史未清）。
- 队列不自动续跑：机器重启或断电后需按 `logs/queue/queue_c2d_repro_status.csv` 手工 `-Skip` 重启。
- 旧口径数字：10-25 之前是「旧划分 + 像素口径」；2026-10-30 起判定口径从 ep150/ep149 换为**终点 epoch 200**（A0 90.16%→90.32%），更早文档里的 ep149/ep150 数字全部作废，以 [results.md](results.md) 为准。
