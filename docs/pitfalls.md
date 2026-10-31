# 踩坑记录（pitfalls）

> 本文件只记录**踩过的坑**：现象 → 根因 → 正确做法。新坑随手往表里加一行，避免重犯。
> 相关文档：结果数字 → [results.md](results.md)；当前状态 → [status.md](status.md)；计划/判定规则 → [experiments.md](experiments.md)。

## 1. 训练与进程管理

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 终端直接起后台进程 | 训练跑到 iteration 71 被无痕杀掉 | 终端关闭/超时会回收子进程树 | **必须用 WMI 脱离进程树启动**：`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -File "...\queue_xxx.ps1"'}` |
| 队列不自动续跑 | 机器重启/断电后队列停在中途 | 队列脚本无断点续跑逻辑 | 按 `logs/queue/*_status.csv` 手工 `-Skip` 已完成项重启 |
| watcher `-Only` 单跑 eval 跳过 Wilcoxon | 单独重跑某个变体时没有 p 值 | `$snapOf` 基线查表在 `-Only` 过滤**之后**才构建，查不到基线快照 | 基线查表必须在 `-Only`/`-Skip` 过滤**之前**构建（已修，2026-10-28） |

## 2. Windows / PowerShell 环境

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| `.ps1` 存成 UTF-8 无 BOM | 中文注释导致 `ParserError` | PowerShell 5 按 ANSI 解析无 BOM 的 UTF-8 | 所有 `.ps1` 必须存 **UTF-8 with BOM** |
| Windows 下 `DataLoader(num_workers>0)` | 探针脚本卡死无输出 | spawn 方式需要入口保护 | 所有探针脚本必须带 `if __name__ == "__main__":` |
| PS5.1 `Format-Hex -Count` | 报参数不存在错误 | PS5 的 Format-Hex 无 `-Count` 参数（PS7 才有） | 用 `Format-Hex \| Select-Object -First N` 或直接不验字节 |

## 3. 网络实现

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 融合分支两个零初始化相乘 | γ 和投影互相锁死，loss 永远不动 | 一条链上两处 ×0，梯度无法流出 | 一条链上**只能有一个零初始化** |
| GAP 之后接 GroupNorm | faff 在 ep5 eval batch=1 崩溃（`input size [1, 8, 1, 1]` 校验失败） | GAP 后特征是 1×1，GN 无空间维可归一化 | GAP 之后改 SE 式 MLP（带 bias），不接空间归一化 |
| DDBE 挂载时机太早 | 同 seed 基础权重与不挂载版本对不上 | 提前建模块会扰动 RNG 消耗顺序 | **全部基础权重初始化之后**再建挂载模块（保证同 seed 基础权重逐位不变） |

## 4. 实验管理

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 快照目录名与队列脚本不一致 | 新旧权重混在同一目录 | 队列 `Snapshot` 字段没和 train.py 拼装规则对齐 | 队列 Snapshot 字段必须与 train.py（L110-153 拼接规则）**逐字对齐** |
| `metrics.csv` 是 ep199 与判定点错位 | 读到的数字和判定点对不上 | trainer.py 末尾 metrics.csv 是最后一轮的产物，而判定点可能在中途 | 用判定点的存档（如 `epoch_199.pth`）+ [eval_carotid.py](../scripts/eval_carotid.py) 重算（~0.3 min/快照），批量用 [scripts/eval_batch_ep199.ps1](../scripts/eval_batch_ep199.ps1) |
| 补训快照漏登记 | `edge_dual_L1234_deform` 手动训练后没进评测批次 | 手动起的训练没同步进评测脚本 `$jobs` 列表 | 手动训练的快照**当场**补进评测脚本的 job 表 |
| 在测试集上调 epoch | 曲线取峰值 epoch 当判定点 = 变相调参 | 测试集参与了模型选择 | 判定点固定（现口径：**终点 epoch 200**），不取曲线峰值 |

## 5. 数据与口径

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 旧划分/像素口径数字混入 | 10-25 前的 90.49% 和现在的数字对不上 | 划分方式和指标单位都变了 | 旧口径结果不进 results.md；以 [results.md](results.md) 为唯一结果来源 |
| 病人图像进 git 历史 | 工作区已删但 git 历史仍可 checkout | `git rm` 只清当前索引 | 彻底清除需 `git filter-repo` 重写历史（会改全部 commit hash，**待用户批准**后执行） |
| 病例级 vs 逐切片口径混读 | Wilcoxon 表的 Dice（90.56%）和横评表（90.32%）对不上 | 前者 43 病例先均值再平均，后者 160 切片直接平均 | 两种口径都对，比较时**只在同口径内比**，表头已标注 |
