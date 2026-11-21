# 队列优先级自检（2026-11-17 队列互锁 20.8 h 事故后新增）
#
# 为什么需要它：
#   2026-11-17 事故的根因不是排序规则，而是「优先级有两个来源」——
#   脚本手写的 Set-QueuePriority 3 与优先级表登记的 8 不一致，
#   两个队列于是各自认为对方更高，互相等待 20.8 h（见 docs/pitfalls.md §6）。
#   本脚本把「脚本手写值 == 表登记值」做成可执行的静态检查，
#   并把当前存活队列的等待关系打出来。
#
# 用法（重启 / 新挂队列前必跑一次，退出码 0 才允许启动）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_queue_priority.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_queue_priority.ps1 -Static   # 只查静态一致性
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_queue_priority.ps1 -Live     # 只查当前存活队列
#
# 退出码：0 = 无 error（可有 warning）；1 = 有 error，禁止启动新队列。

param(
    [switch]$Static,
    [switch]$Live
)

$ErrorActionPreference = "Continue"
if (-not $Static -and -not $Live) { $Static = $true; $Live = $true }

. (Join-Path $PSScriptRoot "queue_priority.ps1")

$errors = @()
$warns = @()

function Add-Err { param([string]$m) $script:errors += $m }
function Add-Warn { param([string]$m) $script:warns += $m }

if ($Static) {
    Write-Host "=== 1) 静态检查：参与互斥的脚本，手写值 vs 优先级表 ==="
    $rows = @()
    $idle = 0
    $all = @(Get-ChildItem $PSScriptRoot -Filter "queue_*.ps1" |
             Where-Object { $_.Name -ne "queue_priority.ps1" } | Sort-Object Name)

    foreach ($f in $all) {
        $text = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
        $joins = ($text -match 'queue_priority\.ps1')
        if (-not $joins) { $idle++; continue }   # 历史脚本（已收口、不参与全序），只统计不逐条告警

        $m = [regex]::Match($text, '(?m)^\s*Set-QueuePriority\s+(\d+)')
        $declared = if ($m.Success) { [int]$m.Groups[1].Value } else { $null }
        $table = [int](Get-QueuePriorityOf $f.Name)
        $registered = ($table -ne [int]$script:UnknownQueuePriority)

        $rows += [pscustomobject]@{
            script   = $f.Name
            declared = if ($null -eq $declared) { "—" } else { $declared }
            table    = if ($registered) { $table } else { "未登记" }
            ok       = if (-not $registered) { "ERR" } elseif ($null -eq $declared) { "ERR" } elseif ($declared -eq $table) { "OK" } else { "ERR" }
        }

        if (-not $registered) {
            Add-Err "$($f.Name)：参与了互斥但没登记在 queue_priority.ps1 的优先级表里（会被当成 p=10）"
            continue
        }
        if ($null -eq $declared) {
            Add-Err "$($f.Name)：没有 Set-QueuePriority（请写上并与表一致，便于自检）"
            continue
        }
        if ($declared -ne $table) {
            Add-Err "$($f.Name)：手写 Set-QueuePriority $declared ≠ 表登记 $table —— 这正是 2026-11-17 互锁的成因"
        }
    }

    $rows | Format-Table -AutoSize | Out-String | Write-Host
    Write-Host "（另有 $idle 个历史队列脚本未接入 queue_priority.ps1，视为 p=1 最低档：只等别人、不抢卡，不在本检查范围）"
    Write-Host ""

    # 并列优先级：同级靠 PID 分先后，不会成环，但顺序不确定，提示一下
    $dup = $rows | Where-Object { $_.table -ne "未登记" } | Group-Object { $_.table } | Where-Object { $_.Count -gt 1 }
    foreach ($g in $dup) {
        Add-Warn "优先级 $($g.Name) 被多个队列共用（$(($g.Group | ForEach-Object { $_.script }) -join ', ')）：同级按 PID 排序，先后不确定"
    }

    Write-Host "=== 2) 参与互斥的队列全序（数字大者先跑）==="
    $rows | ForEach-Object { [pscustomobject]@{ p = [int]$_.table; script = $_.script } } |
        Sort-Object -Property p -Descending | Format-Table -AutoSize | Out-String | Write-Host
}

if ($Live) {
    Write-Host "=== 3) 运行态检查：当前存活队列进程与推断的等待关系 ==="
    $alive = @(Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
               Where-Object {
                   $_.ProcessId -ne $PID -and
                   $_.CommandLine -and
                   ($_.CommandLine -like "*queue_*.ps1*") -and
                   ($_.CommandLine -notlike "*check_queue_priority.ps1*")
               })
    if ($alive.Count -eq 0) {
        Write-Host "当前无存活队列进程。"
    } else {
        $liveRows = @($alive | ForEach-Object {
            $p = [int](Get-QueuePriorityOf $_.CommandLine)
            $name = "(未登记)"
            foreach ($e in (Get-QueuePriorityTable)) {
                if ($_.CommandLine -like "*$($e.pat)*") { $name = $e.pat; break }
            }
            [pscustomobject]@{ pid = $_.ProcessId; script = $name; priority = $p; since = $_.CreationDate }
        } | Sort-Object -Property priority -Descending)
        $liveRows | Format-Table -AutoSize | Out-String | Write-Host

        # 全序下只有「低优先级等高优先级」这一种边；打出来即可确认不是环
        foreach ($r in $liveRows) {
            $pre = @($liveRows | Where-Object {
                $_.pid -ne $r.pid -and
                (Test-QueueOrderBefore -PriorityA $_.priority -PidA $_.pid -PriorityB $r.priority -PidB $r.pid)
            })
            $on = if ($pre.Count -gt 0) { (($pre | ForEach-Object { "PID $($_.pid) $($_.script) p=$($_.priority)" }) -join ' ; ') } else { "无（可开跑）" }
            Write-Host ("  PID {0,-6} p={1,-3} {2,-34} 等待: {3}" -f $r.pid, $r.priority, $r.script, $on)
        }

        $tp = @(Get-TrainingProcess)
        if ($tp.Count -gt 0) {
            Write-Host ("注意：当前有 train.py 在跑（PID {0}），此刻挂新队列只会空转。" -f (($tp | ForEach-Object { $_.ProcessId }) -join ','))
        }
    }
}

Write-Host ""
if ($warns.Count -gt 0) { Write-Host "WARNING:"; $warns | ForEach-Object { Write-Host "  - $_" } }
if ($errors.Count -gt 0) {
    Write-Host "ERROR:"; $errors | ForEach-Object { Write-Host "  - $_" }
    Write-Host ""
    Write-Host "自检未通过：先把手写值与 queue_priority.ps1 的表格对齐，再启动队列。"
    exit 1
}
Write-Host "自检通过（无 error）。"
exit 0
