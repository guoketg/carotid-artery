# Zhongnan unet 成对基线（Z-UB）：等 queue_fgr_multiseed_zn 收口后自动跑。
#
# 背景：ZUF（unet+fgr Zhongnan seed1234）在 queue_fgr_multiseed_zn 里，
#       但 Zhongnan 上没有 unet 基线快照（只有 R50 Z0），无基线则算不出 Δ。
#       本脚本等前一队列 3 行状态全部落盘且 GPU 空闲后，跑一个 job：
#       --arch unet --subset Zhongnan --seed 1234（口径与 experiments.md §0 统一口径逐字一致）。
#
# 快照：TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_unet_c2d
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_zn_unet_base.ps1 -DryRun
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prevCsv = Join-Path $logDir "queue_fgr_multiseed_zn_status.csv"
$statusPath = Join-Path $logDir "queue_zn_unet_base_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$job = @{ n = "ZUB"; tag = "zn_unet_base"
          a = @("train.py","--dataset","Carotid","--subset","Zhongnan","--seed","1234",
                "--batch_size","4","--grad_accum","2","--max_epochs","200","--eval_test_every","5",
                "--arch","unet")
          snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_unet_c2d" }

Write-Host "[ZUB] $Python $($job.a -join ' ')"
Write-Host "[ZUB] snapshot: $(Join-Path $modelDir $job.snap)"
if ($DryRun) { exit 0 }

while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
    $rows = 0
    if (Test-Path -LiteralPath $prevCsv) {
        $rows = @(Get-Content -LiteralPath $prevCsv | Select-Object -Skip 1 |
                  Where-Object { $_ -ne "" }).Count
    }
    if (-not $busy -and $rows -ge 3) { break }
    Start-Sleep -Seconds 120
}

$snapshot = Join-Path $modelDir $job.snap
if (Test-Path $snapshot) {
    $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Move-Item -LiteralPath $snapshot -Destination $bak
    Write-Host "[ZUB] archived previous run to $bak"
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
Write-Host "[ZUB] log: $jobLog"

$start = Get-Date
$jobArgs = $job.a
Push-Location $Root
& $Python @jobArgs *> $jobLog
$code = $LASTEXITCODE
Pop-Location
$end = Get-Date
$minutes = [math]::Round(($end - $start).TotalMinutes, 1)
"ZUB_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
    Out-File -FilePath $statusPath -Append -Encoding utf8
Write-Host "[ZUB] end exit=$code ${minutes}min"
Write-Host "[zn_unet_base] queue finished"
