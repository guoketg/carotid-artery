# 项目状态（精简版）

> 本文件只回答三件事：**现在在跑什么 / 已经得到了什么结论 / 下一步等什么**。
> 2026-10-23 ~ 10-27 的完整过程记录已归档到 [archive/status-history-2026-10.md](archive/status-history-2026-10.md)；10-28 三队列收口与文献复现计划见 [experiments.md](experiments.md) §1.8–§1.9。
> 术语解释见 [study/edge-fusion-explained.md](study/edge-fusion-explained.md)；融合算子逐个讲解见 [study/fusion-operators-explained.md](study/fusion-operators-explained.md)。

最后更新：2026-10-28 17:47

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

## 2. 现在在跑什么（队列）

**批次 0+1 合并队列 `scripts/queue_c2d_repro.ps1`（2026-10-28 17:16 WMI 启动，pid 26772），4 个 job，全部 200 epoch / bs4 + ga2 / 每 5 epoch 评估，取 ep150。预计全程约 12 h。**

| # | job | 做什么 | 快照名 |
|---|---|---|---|
| 1 | `ddbe4` | 解码器**四级全挂** DDBE（BDFR-Net 完整版消融：只加模块） | `..._SPARC_ddbe4_c2d` |
| 2 | `ddbefull` | DDBE×4 + ACAM skip 跨级语义对齐（对齐论文原文） | `..._SPARC_ddbe4_acam_c2d` |
| 3 | `faff` | dual 融合算子横评 AFF 重跑（GroupNorm 崩溃已修，§1.8） | `..._SPARC_dual_L1234_aff_c2d` |
| 4 | `mainll` | 搭车：XNet 便宜变体 `--main_input ll`（主路吃 LL 低频图，零新代码） | `..._SPARC_dual_L1234_auto_mainLL_c2d` |

复现口径与判定规则见 [experiments.md](experiments.md) §1.9（批次 0–4，用户已批准）。status CSV：`logs/queue/queue_c2d_repro_status.csv`。

**队列启动方式必须用 WMI**（否则训练会被终端回收，踩过坑）：

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -File "...\queue_c2d_repro.ps1"'}
```

---

## 3. 已经得到的结论

### 3.1 基线与横评（口径：官方划分 + bs4/ga2 + 200 epoch，判定点 ep149 重算，SPARC test n=160）

ep150 逐切片汇总（Dice / ASSD mm / HD95 mm），来自 §1.8（`epoch_149.pth` 重算，与 `test_curve.csv` 交叉验证吻合）：

| 配置 | s1234 | s2024 | s3407 | 3-seed 均值 |
|---|---|---|---|---|
| A0 单骨干基线 | 0.8995 / 0.1498 / 0.3997 | 0.9027 / 0.1498 / 0.4146 | 0.9026 / 0.1508 / 0.4303 | **0.9016** |
| 双骨干 auto（L1234） | 0.9040 / 0.1476 / 0.4064 | 0.9011 / 0.1468 / 0.4036 | 0.8985 / 0.1570 / 0.4461 | **0.9012** |

### 3.2 病例级 Wilcoxon 四条结论（43 例配对，详见 §1.8）

1. **融合算子不敏感**：aff/gate/deform 对 auto 全部未过门槛（HD95 无一降 ≥0.5 mm 且 p<0.05）→ 维持 auto。deform 是唯二方向一致为正的点（ΔDice +0.0059、HD95 −0.019 mm）但 p=0.113。
2. **双骨干线判负**：3-seed 均值 0.9012 vs A0 0.9016 无增益；s3407 上还**显著变差**（Dice p=0.0036、ASSD p=0.0097）。增益若存在也掩埋在 seed 方差之下。
3. **ddbe3 单级挂载无效**：ΔDice −0.0003，HD95 反 +0.078 mm → 用户判断正确，别人的论文（BDFR-Net）是**四级全挂 + 每级 ACAM**，转批次 1 全量复现（正在跑）。
4. **edgedual 全场最佳点**（0.9072 / 0.1416 / 0.3895，p=0.069 边缘），deform 次之——两者都只是方向证据，不作结论。
   注：各比较均出现 `delta_area` p<0.001（A0 系统性低估面积 −0.90 mm²，变体向 0 靠拢）——这是**面积偏差校正**而非边界精度，不计入判定。

### 3.3 边缘融合（输入级）：信息增量确认，正式训练未过门槛

代码位置：[networks/wavelet.py](file:///e:/mkl/code/python_code/carotid-artery/networks/wavelet.py) `build_hf_image()`；开关 `train.py --edge_in 1`。

做法：stem 吃 4 通道 = 原图 3 通道 + 1 张小波高频重建边缘图。

- **先行诊断**（小网络门槛实验）：SPARC +0.0181 / Zhongnan +0.0183，两个子集一致 → 边缘通道确实携带原图没有的信息。
- **正式训练**（§1.8）：单骨干 edge1234 ΔDice +0.0027（未过门槛）；双骨干 edgedual 0.9072 全场最佳但 p=0.069 边缘 → **线索保留，不作结论**。备选：高频能量图（改一行）。

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

---

## 4. 下一步等什么

1. **批次 1 结果**（正在跑）：队列第 1 位 `ddbe4` 预计 ~19:45 跑完（实测 ~120 iter/min，比 ddbe3 慢一倍）→ **评测已自动挂起**：[watch_eval_repro.ps1](file:///e:/mkl/code/python_code/carotid-artery/scripts/watch_eval_repro.ps1)（WMI pid 16200）轮询队列 CSV，job 完成即自动跑 ep149 eval + 病例级 Wilcoxon（对照 §1.9 判定规则：|ΔDice|≥0.005 或 HD95 降 ≥0.5 mm 且 p<0.05），结果在 `logs/queue/watch_eval_<job>.log`。ddbe4 与 ddbefull 的差 = ACAM 的净贡献。
2. 队列第 3 位 `faff` 补上融合算子横评最后一格；第 4 位 `mainll` 是 XNet 便宜变体的免费一枪。
3. **批次 2 已实现**（GobletNet 式：单主干 + 轻 HF 支 + FAM 逐层）：冒烟 5 项全过（恒等起步 max|Δlogits|=0、+1.206M 参数），入队命令 `--hf_branch 1 --hf_lambda 0.3`（快照 `*_SPARC_hfbr3_c2d`），等批次 1 结论后入队。实现细节见 [experiments.md](experiments.md) §1.9 执行记录②。
4. 批次 3（Syn-Net `--fuse_op dct`）、批次 4（WMREN WRDM）视批次 1–2 结果排期。
5. 病人图像仍在 git 历史里，彻底清除需 `git filter-repo` 重写历史（会改全部 commit hash，**待用户批准**）。

---

## 5. 已知风险

- 病人隐私：工作区与 git 历史仍有病人图像/标识文件（已从当前索引移除，历史未清）。
- 统计功效：测试集 43 例（SPARC）；检测 +1.0 Dice 勉强够，+0.5 不够——deform/edgedual 这类 +0.006 的点判不了性。
- 队列不自动续跑：机器重启或断电后需按 `logs/queue/queue_c2d_repro_status.csv` 手工 `-Skip` 重启。
- 旧口径数字：10-25 之前的实验结果是「旧划分 + 像素口径」，与现在的数字不可直接比较；ep149 重算后本文件 §3.1 与 10-28 01:05 版 status.md 的 0.9049 属**不同快照产物**，以本版为准。
