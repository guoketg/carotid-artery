# 踩坑记录（pitfalls）

> 本文件只记录**踩过的坑**：现象 → 根因 → 正确做法。新坑随手往表里加一行，避免重犯。
> 相关文档：结果数字 → [results.md](results.md)；当前状态 → [status.md](status.md)；计划/判定规则 → [experiments.md](experiments.md)。

## 1. 训练与进程管理

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 终端直接起后台进程 | 训练跑到 iteration 71 被无痕杀掉 | 终端关闭/超时会回收子进程树 | **必须用 WMI 脱离进程树启动**：`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -File "...\queue_xxx.ps1"'}` |
| 队列不自动续跑 | 机器重启/断电后队列停在中途 | 队列脚本无断点续跑逻辑 | 按 `logs/queue/*_status.csv` 手工 `-Skip` 已完成项重启 |
| 误判"重复训练进程" | 杀掉一个"重复"的 python 后，正在跑的任务一起死（2026-11-04 CCFR A1 死于 stage2 ep11） | `venv\Scripts\python.exe` 只是启动器，会以**子进程**再拉起 `C:\Python\Python311\python.exe` 执行同一命令；`Get-Process python` 里两个同名进程属于**同一个 job** | 判重只看 `ParentProcessId` 与 `CreationDate`：子进程的父是 venv 那个即是同一任务；真正的重复任务表现为两个**互不为父子**的 train.py 根进程 |
| WMI `-File` 形式后带 `*>> log` | 队列脚本启动后立刻退出：无 python 进程、无 job 日志，且日志路径被 `Join-Path` 建成了**目录**（2026-11-09） | WMI `Win32_Process.Create` **不经 shell 解析**：`-File script *>> log` 里的 `*>>` 与路径被当位置参数绑进脚本 `param`（$Python="*>>"、$Root=日志路径），脚本空转；Join-Path 照着跑把错误路径建成目录 | 要捕获队列脚本自身输出，必须用 `-Command "& 'script.ps1' *>> 'log'"` 包装形式；`-File` 形式只裸跑脚本、不带任何重定向参数 |

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
| TF32 下用 eye 初始化做"恒等 conv" | DRDAC 恒等起步冒烟 Δlogits=5e-2（应为 0），基础权重逐位一致、fuse=[eye\|0] 单测也对（2026-11-09） | TF32 下 cuDNN conv 会把**输入激活**舍入到 10-bit 尾数：`1.0×TF32(x) ≠ x`，即使权重是精确的 eye；残差/编码链多级复累后误差放大到 O(1)（四级后 logits 差 1.48） | 恒等起步必须走 **`x + branch` 残差 + fuse 零初始化**（0×任何值=0，TF32 下也精确），与 DDBE 同型；不要让 x 穿过任何 conv 后直接输出 |
| 独立探针建 R50 网络报 ZeroDivision | `Embeddings` 里 `h // patch_size[0]` 除零 | 手工用 `CONFIGS_ViT_seg['R50-ViT-B_16']` 时没设 `patches.grid` | R50 混合架构必须设 `cfg.patches.grid = (14, 14)`（train.py L625 的逻辑），冒烟/探针脚本别漏 |

## 4. 实验管理

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 快照目录名与队列脚本不一致 | 新旧权重混在同一目录 | 队列 `Snapshot` 字段没和 train.py 拼装规则对齐 | 队列 Snapshot 字段必须与 train.py（L110-153 拼接规则）**逐字对齐** |
| `metrics.csv` 是 ep199 与判定点错位 | 读到的数字和判定点对不上 | trainer.py 末尾 metrics.csv 是最后一轮的产物，而判定点可能在中途 | 用判定点的存档（如 `epoch_199.pth`）+ [eval_carotid.py](../scripts/eval_carotid.py) 重算（~0.3 min/快照），批量用 [scripts/eval_batch_ep199.ps1](../scripts/eval_batch_ep199.ps1) |
| 补训快照漏登记 | `edge_dual_L1234_deform` 手动训练后没进评测批次 | 手动起的训练没同步进评测脚本 `$jobs` 列表 | 手动训练的快照**当场**补进评测脚本的 job 表 |
| 在测试集上调 epoch | 曲线取峰值 epoch 当判定点 = 变相调参 | 测试集参与了模型选择 | 判定点固定（现口径：**终点 epoch 200**），不取曲线峰值 |
| NaN 训练静默通过（2026-11-16） | `unet_sdfaux_cons_ds`（U-Net 三项全叠）iter 3136 起间歇 `loss : nan`、ep75 起 `test curve` 恒为 dice=0.00% / assd=16.91 / hd=16.91，**脚本仍 exit 0 并落盘 `epoch_199.pth`**，队列按 exit_code 记 success | 训练循环不检查损失是否有限；队列只认退出码，不看 `test summary` | 两道守卫：① 训练侧检测到非有限损失即跳过该步（并记计数），连续 N 次则中止并写失败标记；② 队列侧解析 `test summary`，`dice < 1%` 直接判失败、单独记进状态 CSV。**在守卫落地前，人工读队列时必须扫一遍 `test summary`** |
| 队列在跑时改被 worker 重新导入的文件 | `fgre` ep5 评测崩 `AttributeError: 'CarotidDataset' object has no attribute 'weights'` | 评测新建 DataLoader worker 重新导入新版类，而主进程仍持旧实例 | 队列在跑期间不改 `datasets/`、`networks/`、`trainer.py` 等会被子进程重新导入的文件；要改就等收口 |
| 杀队列脚本想「保住正在跑的那个 job」（2026-11-16） | `Stop-Process` 掉 `queue_c2d_stack.ps1` 的 powershell（PID 640）后，臂 1 的 train.py 又写了 4 行日志就停住，GPU 显存 5380→1027 MiB，`iter 4852/8800` 处静默退出，快照目录只有 `log.txt` / `test_curve.csv`、**没有 `.pth`** | 队列脚本里的 `& $Python @argsList *> $jobLog` 中 `*>` 由 **PowerShell 建管道读子进程输出再写文件**；PowerShell 一死，读端关闭，子进程写 stdout 失败随之退出。`&` 同步等待并不构成进程组 | 要停某个队列的**后续** job，不能杀队列脚本。正确做法：① 让当前 job 自然跑完（队列会在两个 job 之间重新读盘/受 `Get-TrainingProcess` 阻断）；② 或直接杀当前训练进程 + 队列脚本，明确接受该 job 作废；③ 或提前用 `-Only` 只跑需要的臂。**绝不要指望「杀父进程、留子进程」** |
| 队列互锁第 3 次：优先级有「两个来源」（2026-11-17） | `queue_c2d_resnet_loss` 首臂 17:38 收口后与 `queue_c2d_stack`（`-Only` 重挂）互相等待，**GPU 空转 20.8 h**（11-16 17:38 → 11-17 14:22），期间无 python 进程、GPU 4%，两个启动日志各刷 1247 / 1297 行 `busy, waiting 60s...` | 互斥是「严格全序」，但**自身优先级由脚本手写**（`Set-QueuePriority 3`）、**对方优先级查表**（表里登记 8）：resnet(self=3) 认为 stack(表=5) 更高、stack(self=5) 认为 resnet(表=8) 更高 → 成环。总序证明只在「两边同源」时成立 | 优先级**只能有一个来源**：现在本进程的优先级也由「本进程 CommandLine 查表」得到（`Get-SelfQueuePriority`，缓存一次），`Set-QueuePriority` 降级为一致性自检（不一致只告警、以表为准）；等待日志追加 `Get-BlockReason`，直接打印「在等谁（PID / 脚本名 / 优先级）」。**新写队列脚本时，手写数值必须与表一致，或干脆不写** |

## 5. 数据与口径

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| 旧划分/像素口径数字混入 | 10-25 前的 90.49% 和现在的数字对不上 | 划分方式和指标单位都变了 | 旧口径结果不进 results.md；以 [results.md](results.md) 为唯一结果来源 |
| 病人图像进 git 历史 | 工作区已删但 git 历史仍可 checkout | `git rm` 只清当前索引 | 彻底清除需 `git filter-repo` 重写历史（会改全部 commit hash，**待用户批准**后执行） |

## 6. 事故复盘：队列互锁空转 20.8 h（2026-11-17）

**一句话**：互斥规则本身没写错，错在**优先级有两个来源**——「自身值」由脚本手写、「对方值」查表；两者一旦不相等，两个队列就会各自认为对方更高，成环。

### 6.1 时间线

| 时间 | 事件 |
|---|---|
| 11-16 16:31 | `queue_c2d_resnet_loss.ps1` 起跑，脚本内 `Set-QueuePriority 3`，而优先级表登记 **8**；首臂 `RES-cvar` 开跑 |
| 11-16 16:42 | `queue_c2d_stack.ps1 -Only U-FGREnc-RHD` 重挂，`Set-QueuePriority 5`，表里也是 5 |
| 11-16 17:38 | `RES-cvar` 收口 exit 0（89.53 / 0.162 / HD 0.83，见 [results.md](results.md) §13）；resnet 进入第 2 臂 `RES-rhd6` 的等待循环 |
| 11-16 17:38 → 11-17 14:22 | 双方互等：resnet 见 stack 表值 5 > 自身 3 → 等；stack 见 resnet 表值 8 > 自身 5 → 等。**GPU 全空、无任何 python 进程**，两个启动日志各刷 1247 / 1297 行 `busy, waiting 60s...` |
| 11-17 14:22 | 人工巡检发现：GPU 1161 MiB / 4%、无 python、两个 powershell 空转 |
| 11-17 14:25 | 终止两个队列进程（PID 42700 / 3348），`-Only` 重挂 resnet 剩余 11 臂 + stack 单臂，实测恢复训练（GPU 99%） |

### 6.2 损失

- **20.8 h GPU 空转**；resnet 批 ETA 从 11-17 04:30 推到 11-18 ≈ 03:00，stack 单臂推到 11-18 ≈ 05:00。
- 期间无任何 job 产出，因此**不存在**污染读数：没有半截权重被当成结果。

### 6.3 根因链

1. **直接原因**：`queue_c2d_resnet_loss.ps1` 手写 `Set-QueuePriority 3`，`queue_priority.ps1` 表里登记 8，两者不一致。
2. **机制原因**：`Test-QueuePreempts` 拿「自身值（手写）」比「对方值（查表）」。11-15 引入的「严格全序」证明只在**两边同源**时成立，而代码没有强制同源。
3. **流程原因**：启动前没有可执行的一致性检查；等待日志只打 `busy, waiting`、不打「在等谁」，人工只能靠翻进程表倒推。
4. **放大原因**：无人值守过夜，发现靠次日巡检，没有超时告警。

### 6.4 修复（已落地）

| 项 | 内容 |
|---|---|
| 单一来源 | `queue_priority.ps1` 新增 `Get-SelfQueuePriority`：本进程优先级由「本进程 CommandLine 查表」得到并缓存一次；`Set-QueuePriority` 降级为一致性自检（不一致只告警、以表为准）。等待关系由 `Test-QueueOrderBefore` 唯一定义 = (优先级, PID) 严格全序 |
| 可观测 | 新增 `Get-BlockReason`：队列等待日志改为 `busy, waiting 60s... blocked by: PID / 脚本名 / 优先级`；实例见 `logs/queue/launcher_stack_resume2.log` |
| 可自检 | 新增 [check_queue_priority.ps1](../scripts/check_queue_priority.ps1)：静态比对「每个参与互斥的脚本，手写值 == 表登记值」，并打印全序与当前存活队列的推断等待关系，有 error 时 **exit 1** |
| 自检不干扰 | `Get-PreemptingQueue` 排除 `check_queue_priority.ps1` 自身——它的文件名也含 `queue_`，否则跑一次自检就会让队列误判「有更高优先级的队列在等」 |
| 记录 | 本节 + [status.md](status.md) 首段 + [results.md](results.md) §13 + 本文 §4 表格行 |

### 6.5 防复发清单（每次挂队列前逐条过）

1. 新队列脚本先登记进 `scripts/queue_priority.ps1` 的表，**手写 `Set-QueuePriority` 必须与表一致**（或者干脆不写）。
2. 挂载前跑 `powershell -File scripts\check_queue_priority.ps1`，**必须 exit 0** 才允许启动。
3. 挂载后 5 分钟内看一次启动日志：正常应出现 `[job] train.py ...`；若出现 `busy, waiting`，直接看 `blocked by:` 后面的 PID / 脚本名 / 优先级，判断是「真有人在训练」还是「在空等」。
4. 隔夜队列次日第一条检查必须是 **GPU 利用率 + `blocked by` 日志**；不看等于没挂。
5. 一旦发现 A 等 B、B 又在等 A：**先杀优先级低（表值小）的那个**，不要指望它们自己解开；解开后按第 2 条自检再重挂。

## 7. 事故复盘：TransUNet 线已封存却仍被挂载（2026-11-17，两次）

**一句话**：骨干口径 11-16 17:00 已切换并写进 [CLAUDE.md](../CLAUDE.md) 第 18 条，但「文档写了」不等于「执行拦住了」——两次新队列仍把 `--arch baseline`（= R50-ViT-B_16 + DecoderCup，TransUNet 线）当宿主，于是升级为**代码级拦截 + 挂载前审计**。

### 7.1 两次事故

| 时间 | 事件 | 后果 |
|---|---|---|
| 11-16 08:55–11:30 | `queue_c2d_cvar_lambda_r50.ps1` 的 5 臂 cvartail 权重（λ/q）消融，全部跑在 `--arch baseline --vit_layers 0` 上 | 2.5 h GPU 产出**不能对外使用**，只能留作内部对照（当时口径尚未切换，属命名沿用的历史遗留，但结论已与新口径脱钩） |
| 11-17 15:24 | `queue_c2d_cluster_weight_v2.ps1` 挂载，4 臂里有 2 臂（`CLW-echo-R-SPARC` / `CLW-echo-R-ZN`）写死 `vitL0` | 队列刚挂上就**反锁了正在跑的损失批**（见 7.3），16:30 起 GPU 空转 15 min；已终止该队列（其设计另属 CLAUDE.md 第 19 条禁止项，本就判负） |

### 7.2 根因

1. **口径以「R50」这个简称流传**：`--arch baseline --vit_layers 0` 在 11-16 之前一直被叫「R50」，脚本名、快照前缀、文档标题都写 `r50`。切换对外口径后旧脚本没有改名/加封存标记，新队列照抄旧脚本就自动跑回 TransUNet。
2. **没有可执行的拦截**：验收依赖人工读 CLAUDE.md；`check_queue_priority.ps1` 只查优先级，不查骨干。
3. **两次都是「照抄旧脚本」**：不是有人故意用 TransUNet，而是复用模板时未核对 arch 一行。

### 7.3 顺带暴露的第二个互锁（15 min，同一根因族）

杀掉 `cluster_weight_v2` 之前，`queue_c2d_resnet_loss.ps1`（p=8）与它（表登记 p=7）互相等待，GPU 从 16:30（`RES-rhd6` 收口）空到 16:45（杀进程）：

- **缺陷 a：优先级表是进程启动时读进内存的。** resnet 队列 14:25 启动时表里还没有 `cluster_weight_v2`（15:24 才新增 p=7），故它把对方当「未登记」；
- **缺陷 b：`UnknownQueuePriority = 10` 在「数字大者先跑」的比较器下是最高优先级**，注释却写「按最低处理」。于是 cluster(self=7) 等 resnet(8)、resnet 等 cluster(未登记=10)→ 成环。
- 修法：`Get-QueuePriorityTable` 按文件 mtime **每次重读**（后补的表项对已启动队列也可见）；`UnknownQueuePriority` 降到 **1**（低于表中最小登记值 5，未登记队列只等不抢）。

### 7.4 修复（已落地）

| 项 | 内容 |
|---|---|
| 代码级拦截 | [train.py](../train.py) 在 `parse_args()` 后直接拒绝 `--arch baseline`（默认值也是 baseline，故少写 `--arch` 的老命令同样被拦），报错里给出两条合法宿主；确需复现历史读数才加 `--allow_legacy_tu 1` |
| 挂载前审计 | 新增 [check_backbone_policy.ps1](../scripts/check_backbone_policy.ps1)：静态列出仍写死 TransUNet 的脚本，运行态扫存活 `python` 进程里的 `--arch baseline`；有存活事故进程即 **exit 1**。`-Scripts <file>` 可对拟挂载脚本点名判错 |
| 文档口径 | [CLAUDE.md](../CLAUDE.md) 第 18 条开头加禁令块（含两次事故记录）；`queue_c2d_cvar_lambda_r50.ps1`、`queue_c2d_cluster_weight_v2.ps1` 头部加「已封存/已作废，禁止再挂载」 |
| 优先级修复 | 见 7.3：表按 mtime 重读、未登记降到 1 |

### 7.5 防复发清单（新增 3 条，与 §6.5 并列）

1. **挂队列前必须同时过两道自检**：`check_queue_priority.ps1` **和** `check_backbone_policy.ps1`，两者都 exit 0 才允许启动。
2. **复用旧队列脚本时，逐字核对 `--arch` 一行**：合法值只有 `--arch wmren --wmren_ablate unet_dec --wmren_enc r50`（纯 ResNet50）与 `--arch unet`（弱骨干）。看到 `baseline`/`vitL0` 一律改掉再挂。
3. **新队列挂载后 5 分钟内，除看 `train.py` 是否启动，还要看启动命令行里没有 `baseline`**；`train.py` 会以 `--arch baseline 是 TransUNet 线...` 立刻退出，日志里出现该行即为事故。

