# 2026-11-19 队列健康检查（只读，不改状态、不杀进程）
#
# 目的：在挂新队列前/后回答三个问题
#   1) 现在到底有几个 trainer 在跑（父/子 re-exec 不算两个任务）；
#   2) 有没有「僵尸 trainer」（父进程已死 / CPU 不再推进）——它会让所有队列永远等下去，表现为假死锁；
#   3) 队列之间的等待关系是否还能收敛（优先级 + PID 是全序，理论上无环；这里做实测复核）。
#
# 退出码：0 = 无异常；1 = 有 error 级异常（僵尸 trainer / 多个顶层 trainer / 残锁超期）。
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts\check_queue_health.ps1
#       -CpuProbeSec 8 采样窗口；-Quiet 只输出结论
param(
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [int]$CpuProbeSec = 8,
    [switch]$Quiet
)

$ErrorActionPreference = "Continue"
Push-Location $Root
. (Join-Path $Root "scripts\queue_priority.ps1")
Pop-Location

$errors = @()
$warns = @()

# ---- 1) trainer 的真实数量（去掉父/子同一任务的重复计数）----
$allPy = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" |
           Where-Object { $_.CommandLine -like "*train.py*" })
$roots = @($allPy | Where-Object {
    $ppid = $_.ParentProcessId
    -not ($allPy | Where-Object { $_.ProcessId -eq $ppid })
})
if (-not $Quiet) {
    Write-Output ("[trainer] 命中 train.py 的 python 进程 = {0}，去掉父子重复后的顶层任务 = {1}" -f $allPy.Count, $roots.Count)
    foreach ($p in $allPy) {
        Write-Output ("  pid {0} ppid {1} start {2}" -f $p.ProcessId, $p.ParentProcessId, $p.CreationDate)
    }
}
if ($roots.Count -gt 1) { $errors += "顶层 trainer 有 $($roots.Count) 个（应 <= 1，检查是否双开）" }

# ---- 2) 僵尸 trainer：父进程已死 / CPU 不推进 ----
foreach ($r in $roots) {
    $pp = Get-CimInstance Win32_Process -Filter "ProcessId=$($r.ParentProcessId)" -ErrorAction SilentlyContinue
    if (-not $pp) {
        $errors += "trainer pid $($r.ProcessId) 的父进程 $($r.ParentProcessId) 已不存在（孤儿进程，会永久阻塞所有队列）"
    } elseif (-not (Test-IsQueueInvocation $pp.CommandLine)) {
        $warns += "trainer pid $($r.ProcessId) 的父进程 $($r.ParentProcessId) 不是队列调用形式（可能是手工启动）"
    }
}
if ($roots.Count -ge 1) {
    # 注意：train.py 会以同一份命令行再起一个子进程（venv python → 系统 python 3.11），
    # 真正在算的是子进程，父进程只是等待。所以 CPU 探针必须按「进程树」求和，
    # 只看父进程会误判为卡死。
    function Get-TreeCpu([int]$rootPid) {
        $all = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name)
        $tree = @($rootPid); $frontier = @($rootPid)
        while ($frontier.Count -gt 0) {
            $next = @($all | Where-Object { $frontier -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId })
            $next = @($next | Where-Object { $tree -notcontains $_ })
            $tree += $next; $frontier = $next
        }
        $cpu = 0.0
        foreach ($tp in $tree) { $cpu += (Get-Process -Id $tp -ErrorAction SilentlyContinue).CPU }
        return [math]::Round($cpu, 2)
    }
    $s1 = @{}; foreach ($r in $roots) { $s1[$r.ProcessId] = Get-TreeCpu $r.ProcessId }
    Start-Sleep -Seconds $CpuProbeSec
    foreach ($r in $roots) {
        $cpu2 = Get-TreeCpu $r.ProcessId
        $d = [math]::Round(($cpu2 - $s1[$r.ProcessId]), 2)
        if (-not $Quiet) { Write-Output ("  pid {0} 进程树在 {1}s 内累计 CPU 增量 = {2}s" -f $r.ProcessId, $CpuProbeSec, $d) }
        if ($d -lt 0.5) { $warns += "trainer pid $($r.ProcessId) 的进程树在 ${CpuProbeSec}s 内 CPU 几乎不涨（疑似卡死或长等待）" }
    }
}

# ---- 3) 队列进程与等待链 ----
$qs = @(Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -and (Test-IsQueueInvocation $_.CommandLine) -and
                       ($_.CommandLine -notlike "*check_queue_priority.ps1*") } |
        ForEach-Object {
            $nm = "(未登记)"
            foreach ($e in (Get-QueuePriorityTable)) { if ($_.CommandLine -like "*$($e.pat)*") { $nm = $e.pat; break } }
            [pscustomobject]@{ pid = $_.ProcessId; p = (Get-QueuePriorityOf $_.CommandLine); name = $nm }
        })
if (-not $Quiet) {
    Write-Output "[queue] 队列实例 $($qs.Count) 个（按真实优先级降序、同级 PID 升序 = 实际执行顺序）"
    foreach ($q in ($qs | Sort-Object @{e='p';Descending=$true}, @{e='pid';Descending=$false})) {
        $mark = if ($q.p -eq 1) { "  <- 未登记=1，会被所有登记队列永久压住" } else { "" }
        Write-Output ("  p={0,-3} pid {1,-6} {2}{3}" -f $q.p, $q.pid, $q.name, $mark)
    }
}
# 全序自检：任取两实例，必须能判出先后（不存在「互相等」）
$pairs = 0; $bad = @()
for ($i = 0; $i -lt $qs.Count; $i++) {
    for ($j = $i + 1; $j -lt $qs.Count; $j++) {
        $a = $qs[$i]; $b = $qs[$j]; $pairs++
        $aFirst = Test-QueueOrderBefore -PriorityA $a.p -PidA $a.pid -PriorityB $b.p -PidB $b.pid
        $bFirst = Test-QueueOrderBefore -PriorityA $b.p -PidA $b.pid -PriorityB $a.p -PidB $a.pid
        if (-not ($aFirst -xor $bFirst)) { $bad += "pid $($a.pid) 与 $($b.pid) 判不出先后" }
    }
}
if ($bad.Count -gt 0) { $errors += $bad }
if (-not $Quiet) { Write-Output ("[order] 两两比较 {0} 对，判不出先后的 {1} 对（>0 才可能死锁）" -f $pairs, $bad.Count) }

# ---- 4) 残锁 ----
$lockPath = Join-Path $env:TEMP 'carotid_gpu_launch.lock'
if (Test-Path $lockPath) {
    $age = [math]::Round(((Get-Date) - (Get-Item $lockPath).LastWriteTime).TotalMinutes, 1)
    if ($age -gt 30) { $errors += "启动锁残留 $age 分钟（$lockPath），疑似队列进程被强杀" }
    else { $warns += "启动锁存在，已 ${age} 分钟（正常：有训练正在跑）" }
}

# ---- 5) 假阳性来源：只是「提到」队列脚本名、并非在跑的进程 ----
$mentions = @(Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match 'queue_[^\s''"]*\.ps1') -and
                   -not (Test-IsQueueInvocation $_.CommandLine) -and $_.ProcessId -ne $PID })
if ($mentions.Count -gt 0) {
    $warns += "有 $($mentions.Count) 个进程只是在命令行里提到队列脚本名、并非在跑队列（旧判据会把它们当队列）"
    if (-not $Quiet) { foreach ($m in $mentions) { Write-Output ("  [mention] pid {0}: {1}" -f $m.ProcessId, $m.CommandLine.Substring(0, [math]::Min(140, $m.CommandLine.Length))) } }
}

Write-Output "==== 结论 ===="
if ($errors.Count -eq 0) { Write-Output "死锁风险：未发现 error 级异常" } else { foreach ($e in $errors) { Write-Output ("ERROR: " + $e) } }
foreach ($w in $warns) { Write-Output ("WARN: " + $w) }
if ($errors.Count -gt 0) { exit 1 } else { exit 0 }
