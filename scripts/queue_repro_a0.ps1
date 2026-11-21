# 2026-11-20 队列：复现同组研究生 A0 U-Net（老师要求 U-Net 行结果一致）
#
# 关键事实（读他们的包得到，与之前那张超参表不是同一件事）：
#   他们包的 A0（= 论文主对比表的 U-Net 行，macro_dice 0.8846 / 0.9042）是
#     PlainUNet base_channel=32，7,239,745 参数，双线性上采样 + 1×1 降维 + 双卷积；
#     AdamW lr1e-4 + wd 0.01 + cosine 到 1e-6，batch 16，100 ep，clip 1.0，
#     BCE+Dice（各自权重 1.0，相加），best-val（验证集逐样本 Dice 均值）选模。
#   数据链：先 scipy zoom(order=1) 双线性 resize 到 224，再 1%~99% 分位 min-max；
#     mask 最近邻 + 阈值；增强只有水平翻转 0.5 / 垂直 0.2 / ±10°仿射 0.4 / gamma 0.5 /
#     轻对比度 0.25。README 明确写"不要再额外加 CLAHE"。
#   而之前那张表（±30°、CLAHE、偏置场、wd0、恒定 lr、bs4、cv2 三次插值）是他们包里
#     README §6 的 compare 管线 UNet（7,762,465 参数，ConvTranspose 上采样，种子 123/1029/3407），
#     不是 U-Net 行，两者按 README 要求不能混。
#
# 判定：直接跑他们的代码 + 他们的清单，逐位复现 0.8846 / 0.9042 才算"一致"；
#   若对不上（> 0.5 个点），说明数据或环境有差异，再逐项排查。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_repro_a0.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_repro_a0.ps1 -Only sparc
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Package = "C:\Users\ranzh\Downloads\U-Net\U-Net",
    [string]$DataRoot = "E:\mkl\code\datasets\Carotid2D",
    [string]$OutRoot = "E:\mkl\code\python_code\carotid-artery\logs\repro_a0",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 31

$logDir = Join-Path $PSScriptRoot "..\logs\queue"
$logDir = (Resolve-Path $logDir).Path
New-Item -ItemType Directory -Force -Path $logDir, $OutRoot | Out-Null
$statusPath = Join-Path $logDir "queue_repro_a0_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$man = Join-Path $Package "manifests"

$jobs = @()
foreach ($seed in 42, 317, 1029) {
    $jobs += @{ name = "A0-SPARC-s$seed"; args = @(
        "train.py", "--root", $DataRoot,
        "--train-list", (Join-Path $man "train.csv"),
        "--val-list", (Join-Path $man "val.csv"),
        "--test-list", (Join-Path $man "test.csv"),
        "--seed", "$seed", "--out", (Join-Path $OutRoot "sparc_seed$seed"), "--workers", "2");
        tag = "sparc" }
    $jobs += @{ name = "A0-ZN-s$seed"; args = @(
        "train.py", "--root", $DataRoot,
        "--train-list", (Join-Path $man "zhongnan_train.csv"),
        "--val-list", (Join-Path $man "zhongnan_val.csv"),
        "--test-list", (Join-Path $man "external.csv"),
        "--seed", "$seed", "--out", (Join-Path $OutRoot "zn_seed$seed"), "--workers", "2");
        tag = "zn" }
}

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.name -or $keep -contains $_.tag })
    Write-Host "[a0] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.name }) -join ', ')"
}

foreach ($job in $jobs) {
    $lock = $null
    while (-not $DryRun) {
        $lock = Enter-LaunchLock -TimeoutSec 20
        if ((Get-TrainingProcess) -or (Get-OtherQueue)) {
            Exit-LaunchLock $lock; $lock = $null
            Write-Host "[$($job.name)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
            Start-Sleep -Seconds 60
            continue
        }
        break
    }
    Write-Host "[$($job.name)] $Python $($job.args -join ' ')"
    if ($DryRun) { continue }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $safe = ($job.name -replace '-', '_').ToLower()
    $jobLog = Join-Path $logDir "${safe}_$ts.log"
    $start = Get-Date
    Push-Location $Package
    & $Python @($job.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    Exit-LaunchLock $lock
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.name),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.name)] end exit=$code ${minutes}min"
}
Write-Host "[a0] queue finished"
