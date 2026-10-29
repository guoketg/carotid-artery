# 项目状态（精简版）

> 本文件只回答三件事：**现在在跑什么 / 已经得到了什么结论 / 下一步等什么**。
> 2026-10-23 ~ 10-27 的完整过程记录已归档到 [archive/status-history-2026-10.md](archive/status-history-2026-10.md)；10-28 三队列收口与文献复现计划见 [experiments.md](experiments.md) §1.8–§1.9。
> 术语解释见 [study/edge-fusion-explained.md](study/edge-fusion-explained.md)；融合算子逐个讲解见 [study/fusion-operators-explained.md](study/fusion-operators-explained.md)。

最后更新：2026-10-28 23:36（三个训练队列 + ep149 评测批次全部收口，4 个漏掉的 Wilcoxon compare 已补齐，无失败 job）

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

## 2. 队列状态（已全部跑完）

**2026-10-28 23:09 全部收口**：三个训练队列 + ep149 评测批次跑完，无失败 job（唯一异常是融合算子 AFF 首跑在 ep5 因 GroupNorm 崩溃，修复后于批次 0+1 重跑成功）。评测产物：各快照目录 `metrics_ep149.csv` + `logs/queue/eval149_*.log`。

| 队列 | job 数 | 内容 | 状态 CSV |
|---|---|---|---|
| `queue_c2d.ps1`（主队列） | 6 | 单骨干基线 seed2024/3407、双骨干跨支 QKV 注意力 seed2024/3407、输入级边缘图（基线+1 通道）、边缘图+双骨干 | `logs/queue/queue_c2d_status.csv` |
| `queue_c2d_fuse.ps1`（融合算子横评） | 6 | AFF（崩溃 1 次）、gate 门控、deform 可变形大核、DDBE 单级挂载、基线 seed1234/3407 | `logs/queue/queue_c2d_fuse_status.csv` |
| `queue_c2d_repro.ps1`（批次 0+1） | 4 | DDBE 四级全挂、DDBE×4+ACAM（BDFR 完整版）、AFF 重跑、主路吃 LL 低频图（XNet 便宜变体） | `logs/queue/queue_c2d_repro_status.csv` |
| `eval_batch_ep149.ps1`（评测） | 14 | ep149 重算 + 病例级 Wilcoxon，14 个全部完成（含补齐的 4 个 compare） | `logs/queue/eval_batch_ep149_status.csv` |

**队列启动方式必须用 WMI**（否则训练会被终端回收，踩过坑）：

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -File "...\queue_c2d_repro.ps1"'}
```

---

## 3. 已经得到的结论

### 3.1 基线与横评（口径：官方划分 + bs4/ga2 + 200 epoch，判定点 ep149 重算，SPARC test n=160）

ep150 逐切片汇总（Dice % / ASSD mm / HD95 mm），来自 §1.8（`epoch_149.pth` 重算，与 `test_curve.csv` 交叉验证吻合）：

| 配置 | s1234 | s2024 | s3407 | 3-seed 均值 |
|---|---|---|---|---|
| A0 单骨干基线 | 89.95% / 0.1498 / 0.3997 | 90.27% / 0.1498 / 0.4146 | 90.26% / 0.1508 / 0.4303 | **90.16%** |
| 双骨干 auto（L1234） | 90.40% / 0.1476 / 0.4064 | 90.11% / 0.1468 / 0.4036 | 89.85% / 0.1570 / 0.4461 | **90.12%** |

批次 0+1 收口补全（单 seed 1234，口径同上，精确值取自各快照 `test_curve.csv` 的 ep150 行）：

| 配置 | Dice | ASSD mm | HD95 mm | 一句话读法 |
|---|---|---|---|---|
| 双骨干逐像素门控 gate | 90.35% | 0.1434 | 0.4034 | 与 auto 同噪声带 |
| 双骨干可变形大核 deform | 90.61% | 0.1412 | 0.3788 | 唯一三项全优于基线，未过门槛 |
| 双骨干 AFF 两级注意力（重跑成功） | 90.17% | 0.1469 | 0.4011 | 五档垫底，p=0.454/0.663/0.375 |
| 输入级边缘图（单骨干） | 90.20% | 0.1476 | 0.3973 | 未过门槛 |
| 输入级边缘图 + 双骨干 auto | 90.72% | 0.1416 | 0.3895 | 全场最佳点，p=0.069 边缘 |
| DDBE 单级挂载（解码器 level3） | 90.08% | 0.1513 | 0.4404 | HD95 反 +0.041，判负 |
| DDBE 四级全挂（无 ACAM） | 90.33% | 0.1439 | 0.3936 | 比单级好，p=0.462/0.288/0.876 未过 |
| DDBE×4 + ACAM（BDFR 完整版） | 90.53% | 0.1441 | 0.3935 | 全场第三，p=0.334/0.419/0.731 未过 |
| 双骨干 auto + 主路吃 LL 低频重建图 | 90.43% | 0.1478 | 0.4130 | 与主路原图无差异，p=0.447/0.742/0.426 |

### 3.2 病例级 Wilcoxon 结论（43 例配对，详见 §1.8）

1. **融合算子不敏感**：aff/gate/deform 对 auto 全部未过门槛（HD95 无一降 ≥0.5 mm 且 p<0.05）→ 维持 auto。deform 是唯二方向一致为正的点（ΔDice +0.59%、HD95 −0.019 mm）但 p=0.113/0.154/0.276。AFF 重跑补齐最后一格：90.17%/0.1469/0.4011 五档垫底，p=0.454/0.663/0.375，Wilcoxon 也未过（已补）。
2. **双骨干线判负**：3-seed 均值 90.12% vs A0 90.16% 无增益；s3407 上还**显著变差**（Dice p=0.0036、ASSD p=0.0097）。增益若存在也掩埋在 seed 方差之下。
3. **ddbe3 单级挂载无效**：ΔDice −0.03%，HD95 反 +0.078 mm → 用户判断正确，别人的论文（BDFR-Net）是**四级全挂 + 每级 ACAM**。批次 0+1 复现已收口：四级全挂 90.33%、完整版（+ACAM）90.53%，Wilcoxon 已补齐——四级 p=0.462/0.288/0.876、完整版 p=0.334/0.419/0.731，**均未过门槛**；ACAM 净贡献 +0.20% Dice 但 p=0.334 → DDBE 线方向为正、不显著，不作结论。
4. **edgedual 全场最佳点**（90.72% / 0.1416 / 0.3895，p=0.069 边缘），deform 次之——两者都只是方向证据，不作结论。
5. **主路输入换 LL 低频重建图无差异**（XNet 便宜变体）：90.43%/0.1478/0.4130 vs 主路原图 90.40%/0.1476/0.4064——主路吃什么（原图 vs 低频近似）不是当前瓶颈。
   注：各比较均出现 `delta_area` p<0.001（A0 系统性低估面积 −0.90 mm²，变体向 0 靠拢）——这是**面积偏差校正**而非边界精度，不计入判定。

### 3.3 边缘融合（输入级）：信息增量确认，正式训练未过门槛

代码位置：[networks/wavelet.py](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py) `build_hf_image()`；开关 `train.py --edge_in 1`。

做法：stem 吃 4 通道 = 原图 3 通道 + 1 张小波高频重建边缘图。

- **先行诊断**（小网络门槛实验）：SPARC +1.81% / Zhongnan +1.83%，两个子集一致 → 边缘通道确实携带原图没有的信息。
- **正式训练**（§1.8）：单骨干 edge1234 ΔDice +0.27%（未过门槛）；双骨干 edgedual 90.72% 全场最佳但 p=0.069 边缘 → **线索保留，不作结论**。备选：高频能量图（改一行）。

术语与指标含义看不懂 → [study/edge-fusion-explained.md](study/edge-fusion-explained.md)。

### 3.4 踩过的坑（避免重犯）

| 坑 | 表现 | 结论 |
|---|---|---|
| 终端直接起后台进程 | 训练跑到 iteration 71 被无痕杀掉 | 必须 WMI 脱离进程树 |
| Windows 下 `DataLoader(num_workers>0)` 缺 `if __name__ == "__main__"` | spawn 死锁，进程卡死 | 所有探针脚本必须带 main 保护 |
| `.ps1` 存成 UTF-8 无 BOM | 中文注释导致 `ParserError` | 必须 UTF-8 **with BOM** |
| 融合分支两个零初始化相乘 | γ 和投影互相锁死，永远学不动 | 一条链上只能有一个零 |
| 快照目录名与队列脚本不一致 | 新旧权重混在同一目录 | 队列 `Snapshot` 字段必须与 train.py 拼装结果逐字对齐 |
| 融合支 GroupNorm 在 GAP 后 1×1 空间 | eval batch=1 校验崩溃（faff exit 1 @ ep5） | GAP 之后不能接空间归一化，改 SE 式 MLP |
| `trainer.py` 末尾 `metrics.csv` 是 ep199 | 与 ep150 判定点错位 | 用 `epoch_149.pth` + [eval_carotid.py](file:///e:/mkl/code/python_code/carotid-artery/scripts/eval_carotid.py) 重算（~0.3 min/快照） |
| watcher `-Only` 单跑 eval 跳过 Wilcoxon | `$snapOf` 基线查表在 `-Only` 过滤后才构建，查不到基线快照 | 基线查表必须在过滤前构建——已修并补齐 4 个 compare（2026-10-28） |

---

## 4. 下一步等什么

1. ~~补 4 个 Wilcoxon compare~~ **已完成（2026-10-28 23:36）**：[eval_batch_ep149.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/eval_batch_ep149.ps1) 的 `$snapOf` 构建已移到 `-Only`/`-Skip` 过滤之前（删掉原 L59-61 的重复构建），重跑 `faff / ddbe4 / ddbefull / mainll` 四条 `-Only`（各 0.3 min，exit=0）。结果：faff p=0.454/0.663/0.375、ddbe4 p=0.462/0.288/0.876、ddbefull p=0.334/0.419/0.731、mainll p=0.447/0.742/0.426——**四个全部未过门槛**，已回写 §3.1/§3.2。
2. **批次 2 已实现未入队**（GobletNet 式：单主干 + 轻 HF 支 + FAM 逐层）：冒烟 5 项全过（恒等起步 max|Δlogits|=0、+1.206M 参数），入队命令 `--hf_branch 1 --hf_lambda 0.3`（快照 `*_SPARC_hfbr3_c2d`），Wilcoxon 已补齐、批次 0+1 结论已落，可入队。实现细节见 [experiments.md](experiments.md) §1.9 执行记录②。
3. 批次 3（Syn-Net `--fuse_op dct`）、批次 4（WMREN WRDM）视批次 1–2 结果排期。
4. 全部结论目前只在 SPARC 单子集成立；Wilcoxon 已全部补齐，按门槛无一过线——deform / 边缘图+双骨干 / BDFR 完整版三个方向为正的点，择优去 Zhongnan 复现。
5. 病人图像仍在 git 历史里，彻底清除需 `git filter-repo` 重写历史（会改全部 commit hash，**待用户批准**）。

---

## 5. 已知风险

- 病人隐私：工作区与 git 历史仍有病人图像/标识文件（已从当前索引移除，历史未清）。
- 统计功效：测试集 43 例（SPARC）；检测 +1.0% Dice 勉强够，+0.5% 不够——deform/edgedual 这类 +0.6% 的点判不了性。
- 队列不自动续跑：机器重启或断电后需按 `logs/queue/queue_c2d_repro_status.csv` 手工 `-Skip` 重启。
- 旧口径数字：10-25 之前的实验结果是「旧划分 + 像素口径」，与现在的数字不可直接比较；ep149 重算后本文件 §3.1 与 10-28 01:05 版 status.md 的 90.49% 属**不同快照产物**，以本版为准。
