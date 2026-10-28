# Watcher for queue_c2d_repro: as soon as each job row (exit_code=0) lands in
# logs/queue/queue_c2d_repro_status.csv, run its ep149 eval + case-level Wilcoxon
# via eval_batch_ep149.ps1 -Only <job>. Launched via WMI (detached from terminal).
# Idempotent: rows already processed are never re-run; failed jobs (exit!=0) are
# logged and skipped. Exits after all 4 targets are processed.
# Usage (WMI):
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
#     CommandLine='powershell -ExecutionPolicy Bypass -NoProfile -File "...\watch_eval_repro.ps1"' }
param(
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [int]$PollSec = 300
)

$ErrorActionPreference = "Continue"
$queueCsv = Join-Path $Root "logs\queue\queue_c2d_repro_status.csv"
$watchLog = Join-Path $Root "logs\queue\watch_eval_repro.log"
$targets = @("ddbe4", "ddbefull", "faff", "mainll")
$pending = @{}
foreach ($t in $targets) { $pending[$t] = $true }

function Write-WatchLog([string]$msg) {
    "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg |
        Out-File -FilePath $watchLog -Append -Encoding utf8
}

Write-WatchLog "watcher start, targets: $($targets -join ', ')"

while ($true) {
    if (Test-Path $queueCsv) {
        $rows = @(Import-Csv -Path $queueCsv)
        foreach ($r in $rows) {
            $name = $r.job
            if (-not $pending.ContainsKey($name)) { continue }
            if ($r.exit_code -eq "0") {
                Write-WatchLog "job $name finished (exit 0, $($r.minutes) min) -> run ep149 eval"
                $evalLog = Join-Path $Root "logs\queue\watch_eval_$name.log"
                powershell -ExecutionPolicy Bypass -NoProfile -File `
                    (Join-Path $Root "scripts\eval_batch_ep149.ps1") -Only $name *> $evalLog
                Write-WatchLog "eval $name done, log: $evalLog"
            }
            else {
                Write-WatchLog "job $name FAILED (exit $($r.exit_code)), skip eval"
            }
            $pending.Remove($name) | Out-Null
        }
    }
    if ($pending.Count -eq 0) {
        Write-WatchLog "all targets processed, watcher exit"
        break
    }
    Start-Sleep -Seconds $PollSec
}
