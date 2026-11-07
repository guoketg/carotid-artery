# WMREN 组件留一归因（docs/experiments.md §1.9 补测），单 seed(1234) 串行队列。
#
# 宿主固定为 K2（--wmren_ablate unet_dec）：§1.9 已把 WRMM 定为增益载体，
# 因此剩下的「WRMM 内部还有哪块在起作用」必须在 K2 上问，否则会被 REM 的负贡献污染。
#
# 臂表（对照 K2 = 87.77 / 0.1923 / HD 0.979）：
#   M1  K2 + no_mlkm  → _wmren_abunetdec_nm  多尺度大核单元整体值多少
#   M2  K2 + no_gate  → _wmren_abunetdec_ng  WRDM Eq.(4)(5) 软加权 vs 固定 0.5
#
# 已有臂（勿重复跑）：
#   K2  无消融            87.77
#   W-L2 K2 + no_wave     88.11  ← 小波支整体为负（去掉更好）
#
# 判定（§0）：|ΔDice| ≥ 0.5 或 HD 降 ≥ 0.1mm 才算有效；单 seed 噪声带 ±0.3。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_leaveout.ps1
#   ... -Only "M1"   只跑指定任务
#   ... -DryRun      只打印命令
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

# name, extra args, log tag
$jobs = @(
    @{ n = "M1"; tag = "wmren_k2_nomlkm"; a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_no_mlkm", "1") },
    @{ n = "M2"; tag = "wmren_k2_nogate"; a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_no_gate", "1") }
)

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n.ToUpper() }
    if (-not $jobs) { Write-Host "[leaveout] -Only '$Only' matched nothing"; exit 1 }
}

$statusPath = Join-Path $logDir "queue_c2d_wmren_leaveout_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    if (-not $DryRun) {
        while (Get-TrainingProcess) {
            Write-Host "[$($job.n)] another train.py is running, wait 60s..."
            Start-Sleep -Seconds 60
        }
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
    $argsList = @("train.py", "--subset", "SPARC", "--seed", "1234") + $common + $job.a

    Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[$($job.n)] log: $jobLog"
    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @argsList *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.n)] end exit=$code ${minutes}min"
}
Write-Host "[leaveout] queue finished"
