# DDBE + 拉普拉斯在第二个数据集（Zhongnan）上的复现验证。
#
# 背景：SPARC 上「R50 + BDFR-Net DDBE×4、ACAM 关、HF/LF 换单尺度拉普拉斯」
#       = 90.9960 / 0.13577 / HD 0.6581，相对 R50 基线 90.4903 为 ΔDice +0.506、
#       ΔHD −0.048（单 seed 刚过 Dice 线）。用户要求换数据集复核，防止是 SPARC
#       特有的波动。
#
# 口径与 SPARC 完全一致（§0）：200ep / SGD 0.01 / momentum 0.9 / wd 1e-4 /
#       bs4 + ga2 / 每 5 epoch 评估 / seed 1234 / loss = 0.5·CE + 0.5·Dice。
# 两个 job 必须成对：Z0 是同一数据集上的 R50 基线，否则无从算 Δ。
#
# Zhongnan：448 张训练 / 187 张测试（lists\lists_Carotid2D\Zhongnan）。
#
# 前置：等 F1（queue_c2d_r50_fem.ps1，R50+FEM）结束，不抢卡。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_zn_ddbe_lap.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_zn_ddbe_lap.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prevCsv = Join-Path $logDir "queue_c2d_r50_fem_status.csv"
$statusPath = Join-Path $logDir "queue_c2d_zn_ddbe_lap_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "Zhongnan",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "Z0"; tag = "zn_base"
       a = @("--arch", "baseline", "--vit_layers", "0")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_c2d" },
    @{ n = "Z1"; tag = "zn_ddbe_lap"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--bdfr_v2", "1", "--bdfr_acam", "0", "--bdfr_wave_op", "lap")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_bdfr2_na_bwlap_c2d" }
)

foreach ($job in $jobs) {
    $argsList = @("train.py") + $common + $job.a
    Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[$($job.n)] snapshot: $(Join-Path $modelDir $job.snap)"
}
if ($DryRun) { exit 0 }

while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
    $rows = 0
    if (Test-Path -LiteralPath $prevCsv) {
        $rows = @(Get-Content -LiteralPath $prevCsv | Select-Object -Skip 1 |
                  Where-Object { $_ -ne "" }).Count
    }
    if (-not $busy -and $rows -ge 1) { break }
    Start-Sleep -Seconds 60
}

foreach ($job in $jobs) {
    $snapshot = Join-Path $modelDir $job.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[$($job.n)] archived previous run to $bak"
    }
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
    $argsList = @("train.py") + $common + $job.a
    Write-Host "[$($job.n)] log: $jobLog"

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
Write-Host "[zn] queue finished"
