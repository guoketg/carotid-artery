# R-GEF 去门控对照（单 job）：边缘注入保留，可靠性门控换成常数 r≡0.5。
#
# 目的：
#   G1（--edge_fusion rel，gamma=0）     = 90.4039，Δ −0.086
#   G2（--edge_fusion rel，gamma=0.1）   = 90.6230，Δ +0.133（仍不过 0.5）
#   本臂（--edge_fusion relconst，gamma=0.1）= 参数与 G2 完全一致，唯一差别是 r 恒为 0.5。
#   若本臂明显高于 G2 → 逐像素门控本身有害，边缘注入有用；
#   若本臂与 G2 持平或更低 → 边缘注入这一路本身没信息，与门控无关。
#
# 骨干：ResNet-50（ImageNet21k 预训练，vit_layers=0）+ DecoderCup
# 协议：SPARC 官方划分，200ep / bs4+ga2 / SGD 0.01 / seed 1234（单 seed）
#
# 前置：等 queue_c2d_wmren_r50_extend（7 job）跑完，不抢卡。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_rgef_relconst.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_rgef_relconst.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$extendCsv = Join-Path $logDir "queue_c2d_wmren_r50_extend_status.csv"
$statusPath = Join-Path $logDir "queue_c2d_rgef_relconst_status.csv"
$snapshot = Join-Path $modelDir "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_egrelconst_gm10_c2d"

if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$argsList = @(
    "train.py", "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--vit_layers", "0", "--arch", "baseline",
    "--edge_fusion", "relconst", "--edge_gamma", "0.1"
)

function Test-ExtendDone {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $rows = @(Get-Content -LiteralPath $Path | Select-Object -Skip 1 | Where-Object { $_ -ne "" })
    return ($rows.Count -ge 7)
}

while (-not $DryRun) {
    $busy = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
    if (-not $busy -and (Test-ExtendDone -Path $extendCsv)) { break }
    Start-Sleep -Seconds 60
}

Write-Host "[relconst] $Python $($argsList -join ' ')"
Write-Host "[relconst] snapshot: $snapshot"
if ($DryRun) { exit 0 }

if (Test-Path $snapshot) {
    $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Move-Item -LiteralPath $snapshot -Destination $bak
    Write-Host "[relconst] archived previous run to $bak"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$jobLog = Join-Path $logDir "rgef_relconst_$ts.log"
Write-Host "[relconst] log: $jobLog"

$start = Get-Date
Push-Location $Root
& $Python @argsList *> $jobLog
$code = $LASTEXITCODE
Pop-Location
$end = Get-Date
$minutes = [math]::Round(($end - $start).TotalMinutes, 1)
"relconst,$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
Write-Host "[relconst] end exit=$code ${minutes}min"
