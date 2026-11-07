# 模块内 Haar → 拉普拉斯替换（docs/experiments.md §1.12），单 seed(1234) 串行队列。
#
# 宿主选择依据 §1.9 归因：WRMM 是增益载体（K2−U-Net=+1.22）、REM 负贡献
# （W2−K2=−0.63），故 WMREN 侧一律在 K2 宿主（--wmren_ablate unet_dec）上做替换。
#
# 臂表：
#   W-L1  K2 + lap         → _wmren_abunetdec_wolap   对照 K2(87.77)
#   W-L2  K2 + no_wave     → _wmren_abunetdec_nw      控制臂：小波支整体值多少
#   B-L1  DDBE4 + lap      → _ddbe4_ddhflap           对照 ddbe4
#   B-L2  DDBE4 + hf=none  → _ddbe4_ddhfnone         控制臂：HF 支整体值多少
#   B-L3  DDBE4 + hf=pyr   → _ddbe4_ddhfpyr          正统拉普拉斯金字塔（B-L1 的公平版）
#   W-L4  K2 + pyr         → _wmren_abunetdec_wopyr   正统拉普拉斯金字塔（3 个真实尺度、保留符号）
#
# 判定（§1.12）：|ΔDice| ≥ 0.5 或 HD 降 ≥ 0.1mm 才算有效；单 seed 噪声带 ±0.3。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lap_swap.ps1
#   ... -Only "W-L1,B-L1"   只跑指定任务
#   ... -DryRun             只打印命令
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
    @{ n = "W-L1"; tag = "wlap_k2"; a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_wave_op", "lap") },
    @{ n = "W-L2"; tag = "wnowave_k2"; a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_no_wave", "1") },
    @{ n = "B-L1"; tag = "ddbe4_hflap"; a = @("--arch", "baseline", "--ddbe", "4", "--ddbe_hf_op", "lap") },
    @{ n = "B-L2"; tag = "ddbe4_hfnone"; a = @("--arch", "baseline", "--ddbe", "4", "--ddbe_hf_op", "none") },
    @{ n = "B-L3"; tag = "ddbe4_hfpyr"; a = @("--arch", "baseline", "--ddbe", "4", "--ddbe_hf_op", "pyr") },
    @{ n = "W-L4"; tag = "wpyr_k2"; a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_wave_op", "pyr") }
)

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n.ToUpper() }
    if (-not $jobs) { Write-Host "[lapswap] -Only '$Only' matched nothing"; exit 1 }
}

$statusPath = Join-Path $logDir "queue_c2d_lap_swap_status.csv"
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
Write-Host "[lapswap] queue finished"
