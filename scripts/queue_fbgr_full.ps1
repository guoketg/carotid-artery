# FBGR 三臂接力队列（F-B）：等 queue_zn_unet_base 收口后自动跑。
#
# 背景：R50 上 FGR 与 DDBE-lap 同点串联互覆（90.632 < 单臂 90.795/90.996），
#       诊断是 single-core 改写把两带搅在同一 2C→C 投影里（算子同型互抢）。
#       FBGR（--fgr_mode band）拆双零核并联分带改写，docs/module-suite-design.md §3。
#       第三臂原定 HFE-UNet，但同口径结果已存在（87.42，+0.87 已过线，
#       归档 A.4 C 组），改跑 full 档（HFE+FBGR 同开）补消融阶梯最后一行。
#
# 三臂（全部 SPARC seed 1234 bs4 ga2 ep200，smoke_fbgr_band.py 全绿后入队）：
#   FBGR-U    --arch unet --fgr 1 --fgr_mode band
#             snap: TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_fgrb_unet_c2d
#   FBGR-R50  --arch baseline --vit_layers 0 --fgr 1 --fgr_mode band
#             snap: TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fgrb_c2d
#   FULL-U    --arch unet --edge_in 1 --fgr 1 --fgr_mode band
#             snap: TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_edge_fgrb_unet_eowavelet_c2d
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fbgr_full.ps1 -DryRun
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prevCsv = Join-Path $logDir "queue_zn_unet_base_status.csv"
$statusPath = Join-Path $logDir "queue_fbgr_full_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$jobs = @(
    @{ n = "FBGR-U"; tag = "unet_fgrb"
       a = @("train.py","--dataset","Carotid","--subset","SPARC","--seed","1234",
             "--batch_size","4","--grad_accum","2","--max_epochs","200","--eval_test_every","5",
             "--arch","unet","--fgr","1","--fgr_mode","band")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_fgrb_unet_c2d" },
    @{ n = "FBGR-R50"; tag = "r50_fgrb"
       a = @("train.py","--dataset","Carotid","--subset","SPARC","--seed","1234",
             "--batch_size","4","--grad_accum","2","--max_epochs","200","--eval_test_every","5",
             "--arch","baseline","--vit_layers","0","--fgr","1","--fgr_mode","band")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fgrb_c2d" },
    @{ n = "FULL-U"; tag = "unet_hfe_fgrb"
       a = @("train.py","--dataset","Carotid","--subset","SPARC","--seed","1234",
             "--batch_size","4","--grad_accum","2","--max_epochs","200","--eval_test_every","5",
             "--arch","unet","--edge_in","1","--fgr","1","--fgr_mode","band")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_edge_fgrb_unet_eowavelet_c2d" }
)

Write-Host "[F-B] waiting for prevCsv: $prevCsv"
foreach ($j in $jobs) {
    Write-Host "[F-B] $($j.n) $Python $($j.a -join ' ')"
    Write-Host "[F-B]   snapshot: $(Join-Path $modelDir $j.snap)"
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
    Start-Sleep -Seconds 120
}

foreach ($j in $jobs) {
    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[$($j.n)] archived previous run to $bak"
    }
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.tag)_$ts.log"
    Write-Host "[$($j.n)] log: $jobLog"

    $start = Get-Date
    $jobArgs = $j.a
    Push-Location $Root
    & $Python @jobArgs *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.n)_$($j.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($j.n)] end exit=$code ${minutes}min"
}
Write-Host "[fbgr_full] queue finished"
