# 骨干归因（§1.13）：单 seed(1234) 串行队列。
#
# 目的：把「R50 结构」与「ImageNet21k 预训练」两个因素拆开。
#   参考（已有）：
#     U-Net      from scratch  86.55  ← 纯卷积、无预训练
#     A          R50 + 预训练  90.49  ← --vit_layers 0
#     A0         R50+ViT + 预训练 90.32
#     B          ViT only + 预训练 89.19
#   本队列：
#     V-N1       R50 + 随机初始化  = ?  → Δ_pretrain = A − V-N1，Δ_R50struct = V-N1 − U-Net
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_backbone_attrib.ps1
#   ... -DryRun            只打印命令
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
    @{ n = "V-N1"; tag = "vitL0_nopre"; a = @("--arch", "baseline", "--vit_layers", "0", "--no_pretrain", "1") }
)

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n }
    if (-not $jobs) { Write-Host "[bonet] -Only '$Only' matched nothing"; exit 1 }
}

$statusPath = Join-Path $logDir "queue_c2d_backbone_attrib_status.csv"
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
Write-Host "[bonet] queue finished"
