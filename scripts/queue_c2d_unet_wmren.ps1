# 2026-11-01 接力队列：U-Net 骨架整体替换版 WMREN（IJCAI 2025）。
# 与 queue_c2d_unet_ddbe.ps1 串行：等 DDBE 队列进程与所有 train.py 退出后再启动，
# 避免两个队列同时抢 GPU。
#
#   W1  --arch wmren   WRDM 下采样(前三级) + REM 解码融合(2×CBAM + 2×SAFM/CREM)
#
# 参数 60.3M → 70.6M（+10.3M），对照 U-Net 基线 86.55% / HD95 0.77。
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-DdbeQueue {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*queue_c2d_unet_ddbe.ps1*" }
}

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$snapshot = Join-Path $modelDir "${pre}_SPARC_wmren_c2d"

if (-not $DryRun) {
    while ((Get-DdbeQueue) -or (Get-TrainingProcess)) {
        Write-Host "[wmren] ddbe queue or train.py still running, wait 60s..."
        Start-Sleep -Seconds 60
    }
}

if ((Test-Path $snapshot) -and -not $DryRun) {
    $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Move-Item -LiteralPath $snapshot -Destination $bak
    Write-Host "[wmren] archived previous run to $bak"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$jobLog = Join-Path $logDir "W1_unet_wmren_$ts.log"
$statusPath = Join-Path $logDir "queue_c2d_unet_wmren_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$argsList = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "wmren") + $common
Write-Host "[wmren] command: $Python $($argsList -join ' ')"
Write-Host "[wmren] log: $jobLog"

if ($DryRun) { exit 0 }

$start = Get-Date
Push-Location $Root
& $Python @argsList *> $jobLog
$code = $LASTEXITCODE
Pop-Location
$end = Get-Date
$minutes = [math]::Round(($end - $start).TotalMinutes, 1)
"W1_unet_wmren,$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
Write-Host "[wmren] end exit=$code ${minutes}min"
