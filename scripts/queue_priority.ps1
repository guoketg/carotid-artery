# 队列互斥与优先级（2026-11-15 新增；2026-11-17 修掉互锁，被各 queue_*.ps1 dot-source）
#
# 背景（两次事故）：
#   ① 2026-11-15：旧 Get-OtherQueue 语义是「只要还有别的队列进程活着就等」，
#      queue_c2d_distloss_unet.ps1 与 queue_c2d_fgr_cvartail_3seed.ps1 在首 job 收口后互相等待，
#      00:24 起空转 8 h 20 m，GPU 全空。
#   ② 2026-11-17：改成「严格全序」后仍互锁一次，原因不是排序规则，而是**优先级有两个来源**：
#      「自身优先级」由脚本手写 Set-QueuePriority 给，「对方优先级」由下面这张表给。
#      queue_c2d_resnet_loss.ps1 手写 3、表里登记 8，于是
#        resnet(self=3) 看 stack(表=5) → 认为对方更高 → 等；
#        stack (self=5) 看 resnet(表=8) → 认为对方更高 → 等。
#      结果 2026-11-16 17:38 起空转 20.8 h，ResNet 损失批的 11 臂一条都没开跑。
#
# 现在的规则（唯一优先级来源）：
#   **本进程自己的优先级也从这张表查**——拿本进程 CommandLine 去匹配表里的脚本名，
#   而不是信任脚本手写的数字。这样「我眼中的我」与「别人眼中的我」永远是同一个值，
#   等待关系是 (优先级, PID) 上的严格全序 ⇒ 不可能成环。
#   未登记的队列一律按 UnknownQueuePriority 处理，同样对所有人一致。
#
#   ③ 2026-11-17 16:30 又互锁一次（15 min，靠杀队列解开），两个叠加缺陷：
#      a) **表是进程启动时读进内存的**：resnet 队列 14:25 启动时表里还没有
#         cluster_weight_v2（15:24 才新增 p=7），它于是把对方当成「未登记」；
#      b) **UnknownQueuePriority=10 数值上高于 8 和 5**，注释写「最低」但比较器是「数字大者先跑」，
#         未登记队列实际是**最高**优先级。于是 cluster(self=7) 等 resnet(8)、resnet(自己 8) 等
#         cluster(未登记=10)，再次成环。
#      修法：Get-QueuePriorityTable 每次按文件 mtime 重读表（后补的表项对已启动队列也可见）；
#             UnknownQueuePriority 降到 1（低于表中最小登记值 5），未登记队列永远只等、不抢。
#
# 用法（队列脚本里）：
#   . (Join-Path $PSScriptRoot "queue_priority.ps1")
#   while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
#       Write-Host "[job] busy, waiting 60s... blocked by: $(Get-BlockReason)"
#       Start-Sleep -Seconds 60
#   }
#   Set-QueuePriority <n> 仍可调用，但只做一致性自检（数值以本表为准）。

$script:QueuePriorityTable = @(
    @{ pat = "queue_a0_wrmm.ps1";                p = 32 },  # A0 配方 + 新划分 + best-val，跑解码器 WRMM 八臂（用户 2026-11-21 直接指派，最高）
    @{ pat = "queue_repro_a0.ps1";               p = 31 },  # 复现同组研究生 A0（主表 U-Net 行；老师要求一致）
    @{ pat = "queue_align_unet.ps1";             p = 30 },  # UNet 复现口径对齐（同组研究生配置；老师要求结果一致，用户 2026-11-20 直接指派）
    @{ pat = "queue_a0_matrix.ps1";              p = 29 },  # A0 配方 + 研究生划分 + best-val：2x2 消融重跑 16 臂（DDBE+lap / RHD / 叠加 / FGR），基线由 a0_wrmm 提供（用户 2026-11-21 指派）
    @{ pat = "queue_c2d_ddbe_ref_fill.ps1";      p = 23 },  # DDBE+lap+RHD 2×2 矩阵的两条缺失参照（U-Net×ZN 结构单挂、R50×ZN RHD I2 单挂）；用户 2026-11-20 指派「很关键」
    @{ pat = "queue_c2d_dec_wrmm.ps1";           p = 22 },  # 解码器 WRMM 四臂（Zhongnan/SPARC × R50+DecoderCup / U-Net）+ 重跑 R50 FGR+RHD I6（用户 2026-11-20 直接指派，最高）
    @{ pat = "queue_c2d_edge_bigate.ps1";        p = 21 },  # 双向逐级门控边缘支单臂（用户 2026-11-20 直接指派，最高；待当前训练收口后立刻起跑）
    @{ pat = "queue_c2d_wrmm_dec.ps1";           p = 19 },  # 解码器 WRMM+lap（R50 编码器 + U-Net 解码器，SPARC/ZN 各一臂；用户 2026-11-20 直接指派，先跑）
    @{ pat = "queue_c2d_fgr_rhd_r50.ps1";        p = 19 },  # R50 + DecoderCup 上 FGR(解码器) + RHD sym 融合两臂 I6/I2（用户 2026-11-20 直接指派；与 wrmm_dec 同级，按 PID 全序排在它之后）
    @{ pat = "queue_c2d_rhd.ps1";                p = 50 },  # #82 主判：R50 上 RHD I6 / I2
    @{ pat = "queue_c2d_cvar_lambda_r50.ps1";    p = 48 },  # cvartail 剂量消融（用户 2026-11-15 直接指派）
    @{ pat = "queue_c2d_lossv2_unet.ps1";        p = 45 },  # 损失设计 v2 第一段：U-Net 三个新角度 + 逐级叠加
    @{ pat = "queue_c2d_lossv2_r50.ps1";         p = 40 },  # 损失设计 v2 第二段：R50 迁移验证
    @{ pat = "queue_c2d_fgr_cvartail_3seed.ps1"; p = 30 },  # 用户指定的 3-seed 集合 42/317/1029
    @{ pat = "queue_c2d_brp_r50.ps1";            p = 26 },  # R50 编码器分带重投影位置消融（用户 2026-11-18 直接指派，先排上队）
    @{ pat = "queue_c2d_ddbe_lap_hosts.ps1";     p = 18 },  # DDBE+拉普拉斯结构 / RHD 融合：U-Net 单挂 + U-Net 融合(I6) + R50 融合(I2)（用户 2026-11-19 直接指派）
    @{ pat = "queue_c2d_wrmm_lap_rhd.ps1";       p = 17 },  # WRMM 编码器 + lap + RHD（U-Net 侧 I6 / ResNet 侧 I2，SPARC 与 Zhongnan 各两臂）（用户 2026-11-19 直接指派）
    @{ pat = "queue_c2d_ddbe_lap_rhd_zn.ps1";    p = 15 },  # DDBE+lap+RHD 的 Zhongnan 训练版两臂（用户 2026-11-19 直接指派）
    @{ pat = "queue_c2d_tu_lap_rhd.ps1";         p = 14 },  # 完整 TransUNet 上的 DDBE+lap(+RHD I2)（用户 2026-11-19 显式授权解封）
    @{ pat = "queue_c2d_cross_center_matrix.ps1"; p = 2 },  # 跨中心矩阵（纯推理）；刻意最低，永不阻塞训练
    @{ pat = "queue_c2d_rhd_r50_term.ps1";       p = 16 },  # RHD term 补齐：h / ah 在默认 ResNet50（R50+DecoderCup）上（用户 2026-11-19 直接指派）
    @{ pat = "queue_c2d_distloss_unet.ps1";      p = 20 },  # U-Net 距离类损失机制筛选（剩余两臂）
    @{ pat = "queue_c2d_cvartail_r50_3seed_zn.ps1"; p = 15 },  # R50 cvartail 补 seed / 中南（最低一档，排链末）
    @{ pat = "queue_c2d_edge_label_ops.ps1";      p = 12 },  # 标签侧边缘算子四臂：sobel_orig + morph k=7（用户 2026-11-18 直接指派，先 R50 后 U-Net）
    @{ pat = "queue_c2d_resnet_loss.ps1";        p = 8 },  # 纯 ResNet50 骨干（R50 直连 unet_dec）上的损失横评 12 臂
    @{ pat = "queue_c2d_cluster_weight_v2.ps1";  p = 7 },  # 聚类样本加权在 U-Net / 纯 R50 上的验证 4 臂（老师要求补跑）
    @{ pat = "queue_c2d_cvar_unet_w02.ps1";      p = 6 },  # U-Net 上 cvartail λ=0.2 单臂（用户 2026-11-18 直接指派，等聚类批之后）
    @{ pat = "queue_c2d_rhd_zn.ps1";             p = 6 },  # RHD sym I6 补 Zhongnan 两格（用户 2026-11-18 直接指派，等聚类批之后）
    @{ pat = "queue_c2d_stack.ps1";              p = 5 }   # 叠加批（结构+损失）：等当日全部队列收口后再跑
)
# 未登记的队列按最低优先级处理：它自己等所有登记队列，登记队列只等比自己高的。
# 注意：比较器是「数字大者先跑」，所以这里必须比表中**最小**登记值还小（当前最小 5，取 1）；
# 曾经写成 10（看似「低」实则高于 8 和 5），导致 2026-11-17 16:30 的第二次互锁。
$script:UnknownQueuePriority = 1
$script:SelfPriorityCache = $null
$script:QueuePriorityPath = Join-Path $PSScriptRoot "queue_priority.ps1"
$script:TableCache = $null
$script:TableCacheStamp = $null

# 优先级表按文件 mtime 重读：队列进程先启动、表项后补时（2026-11-17 16:30 事故），
# 已启动的进程也必须能看到新增条目，否则「我眼中的别人」与「别人眼中的自己」不一致 → 成环。
function Get-QueuePriorityTable {
    if (-not (Test-Path -LiteralPath $script:QueuePriorityPath)) { return $script:QueuePriorityTable }
    $stamp = (Get-Item -LiteralPath $script:QueuePriorityPath).LastWriteTimeUtc.Ticks
    if ($null -ne $script:TableCache -and $script:TableCacheStamp -eq $stamp) { return $script:TableCache }
    $txt = Get-Content -LiteralPath $script:QueuePriorityPath -Raw -Encoding UTF8
    $ms = [regex]::Matches($txt, '@\{\s*pat\s*=\s*"([^"]+)"\s*,\s*p\s*=\s*(\d+)\s*\}')
    if ($ms.Count -eq 0) { return $script:QueuePriorityTable }
    $script:TableCache = @($ms | ForEach-Object { @{ pat = $_.Groups[1].Value; p = [int]$_.Groups[2].Value } })
    $script:TableCacheStamp = $stamp
    return $script:TableCache
}

function Get-QueuePriorityOf {
    param([string]$CommandLine)
    if (-not $CommandLine) { return [int]$script:UnknownQueuePriority }
    foreach ($e in (Get-QueuePriorityTable)) {
        if ($CommandLine -like "*$($e.pat)*") { return [int]$e.p }
    }
    return [int]$script:UnknownQueuePriority
}

# 本进程的优先级：只从「本进程 CommandLine 查表」得到，缓存一次。
# WMI 偶发读不到（2026-11-17 干跑时出现过一次，返回了未登记值 1）：重试 3 次；
# 若始终拿不到 CommandLine，则**不写缓存**，让后续调用还能纠正，避免把错误值钉死。
function Get-SelfQueuePriority {
    if ($null -ne $script:SelfPriorityCache) { return [int]$script:SelfPriorityCache }
    $cl = $null
    for ($i = 0; $i -lt 3 -and -not $cl; $i++) {
        try {
            $me = Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction Stop
            if ($me) { $cl = $me.CommandLine }
        } catch { $cl = $null }
        if (-not $cl) { Start-Sleep -Milliseconds 200 }
    }
    if (-not $cl) { $cl = [Environment]::CommandLine }
    $val = [int](Get-QueuePriorityOf $cl)
    if ($cl) { $script:SelfPriorityCache = $val }
    return $val
}

# 兼容旧脚本：不再参与比较，只做一致性自检（不匹配时提示，实际仍以表为准）。
function Set-QueuePriority {
    param([int]$Priority)
    $real = Get-SelfQueuePriority
    if ($Priority -ne $real) {
        Write-Host "[queue] 警告：脚本手写优先级 $Priority 与优先级表推出的 $real 不一致；实际以表为准（$real）。"
    }
}

# 全序比较的唯一定义：先比优先级（大者优先），同级比 PID（小者优先）。
# A 优先于 B 时返回 $true。所有判断（含自检脚本）都走这里，避免各处各写一套。
function Test-QueueOrderBefore {
    param([int]$PriorityA, [int]$PidA, [int]$PriorityB, [int]$PidB)
    if ($PriorityA -ne $PriorityB) { return ($PriorityA -gt $PriorityB) }
    return ($PidA -lt $PidB)
}

function Test-QueuePreempts {
    param([int]$OtherPriority, [int]$OtherPid)
    return (Test-QueueOrderBefore -PriorityA $OtherPriority -PidA $OtherPid `
                                  -PriorityB (Get-SelfQueuePriority) -PidB $PID)
}

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# 返回「严格优先于本队列」的其它队列进程（默认实现；需要额外开关的脚本可自行包装本函数）
function Get-PreemptingQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            (Test-IsQueueInvocation $_.CommandLine) -and
            # 自检脚本名里也带 queue_，必须排除，否则跑自检会让队列误以为有更高优先级的队列在等
            ($_.CommandLine -notlike "*check_queue_priority.ps1*") -and
            (Test-QueuePreempts -OtherPriority (Get-QueuePriorityOf $_.CommandLine) -OtherPid $_.ProcessId)
        }
}

# 「是不是真在跑队列脚本」的判据（2026-11-19 收紧）：
# 旧写法只要求命令行里出现 queue_*.ps1 字样，于是 `Select-String ... queue_xxx.ps1`、
# `Get-Content queue_xxx.ps1`、编辑器/日志查看器这类只是「提到」文件名的进程也会被当成队列，
# 优先级还会被算成 Unregistered=1 —— 一旦这种进程长期挂着，所有队列都会永远等它（假死锁）。
# 新判据要求命令行既出现 queue_*.ps1、又出现真正的调用形式（-File 或 & 调用）。
function Test-IsQueueInvocation {
    param([string]$CommandLine)
    if ($CommandLine -notmatch 'queue_[^\s''"]*\.ps1') { return $false }
    if ($CommandLine -match '-File\s') { return $true }
    if ($CommandLine -match '&\s') { return $true }
    return $false
}

# 全局启动锁（2026-11-19 新增）：同优先级多实例（例如同一脚本按 -Only 挂了三次）靠 PID 全序排队，
# 判断与启动之间仍存在毫秒级竞态窗口 —— 两个实例可能同时看到「无 trainer、无更高优先级队列」而双开，
# 造成两张卡同时被占 / OOM。用一次性文件创建（CreateNew 原子）做互斥，只有拿到锁的实例才允许起 python。
# 锁只在「一次训练真正在跑」期间持有；进程被强杀会留下残锁，超时后带警告放行，不会永久锁死。
function Enter-LaunchLock {
    param([int]$TimeoutSec = 180)
    $lockPath = Join-Path $env:TEMP 'carotid_gpu_launch.lock'
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $fs = [System.IO.File]::Open($lockPath, 'CreateNew', 'Write', 'None')
            $fs.Close()
            return $lockPath
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    Write-Host "[lock] 警告：等锁超过 ${TimeoutSec}s，按残锁处理并放行（请人工确认没有第二个 trainer）"
    return $null
}

function Exit-LaunchLock {
    param([string]$LockPath)
    if ($LockPath -and (Test-Path $LockPath)) {
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-OtherQueue { Get-PreemptingQueue }

# 诊断用：把「到底在等谁」打成一行，写进队列日志，避免下次只能靠猜。
function Get-BlockReason {
    $parts = @()
    $tp = @(Get-TrainingProcess)
    if ($tp.Count -gt 0) {
        $parts += "python train.py PID " + (($tp | ForEach-Object { $_.ProcessId }) -join ',')
    }
    $pq = @(Get-PreemptingQueue)
    if ($pq.Count -gt 0) {
        $items = $pq | ForEach-Object {
            $name = ""
            foreach ($e in (Get-QueuePriorityTable)) {
                if ($_.CommandLine -like "*$($e.pat)*") { $name = $e.pat; break }
            }
            if (-not $name) { $name = "(未登记队列)" }
            "PID $($_.ProcessId) $name p=$(Get-QueuePriorityOf $_.CommandLine)"
        }
        $parts += "queue " + ($items -join ' ; ')
    }
    if ($parts.Count -eq 0) { return "none" }
    return ($parts -join ' | ')
}
