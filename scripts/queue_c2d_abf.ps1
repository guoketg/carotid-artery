# 自研 ABF（Asymmetric Band Fusion，跨路非对称频带搬运）四臂验证。
#
# 背景：R50 强骨干上已测的融合/边缘模块全部无效或仅方向证据（results.md §5：
#       FreqFusion +0.083 / DDBE-Haar +0.435 / DDBE-lap +0.506 / R-GEF +0.133 /
#       MREF -0.115 / FEM -0.098），唯一反复出现的正信号是显式频带解耦。
#       ABF 设计（证据驱动，见设计讨论）：跨路（skip 与上采样特征）非对称频带
#       搬运——低频支取 u_L-s_L（补语义）、高频支取 s_H-u_H（补细节），纯残差
#       回注不改写 skip；alpha 通道级可学，tanh 分歧调幅（非 sigmoid 门控），
#       gamma_0=0.1 可学。
#
# 四臂（全部 SPARC / seed 1234 / bs4+ga2 / 200ep / 每 5 epoch 评估）：
#   A1  R50(vitL0)+DecoderCup + abf      对照基线 90.4903 / 0.14473 / HD95 0.41634
#   A2  R50(vitL0)+DecoderCup + abftied  归因对照（alpha_H = -alpha_L，参数完全匹配）：
#                                        若 tied 与 untied 同涨，增益来自残差回注通路
#                                        本身而非非对称频带分离
#   A3  unet 解码器 + abf                弱骨干空间更大（U-Net 基线 86.55 / 0.2250 / HD95 0.77）
#   A4  unet 解码器 + abftied            归因对照（参数完全匹配）
#
# 前置：等 Zhongnan 队列（queue_c2d_zn_ddbe_lap.ps1，Z0+Z1 两行）结束，不抢卡。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_abf.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_abf.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prevCsv = Join-Path $logDir "queue_c2d_zn_ddbe_lap_status.csv"
$statusPath = Join-Path $logDir "queue_c2d_abf_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "A1"; tag = "abf"
       a = @("--arch", "baseline", "--vit_layers", "0", "--edge_fusion", "abf", "--edge_gamma", "0.1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_egabf_gm10_c2d" },
    @{ n = "A2"; tag = "abftied"
       a = @("--arch", "baseline", "--vit_layers", "0", "--edge_fusion", "abftied", "--edge_gamma", "0.1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_egabftied_gm10_c2d" },
    @{ n = "A3"; tag = "unet_abf"
       a = @("--arch", "unet", "--edge_fusion", "abf", "--edge_gamma", "0.1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_unet_egabf_gm10_c2d" },
    @{ n = "A4"; tag = "unet_abftied"
       a = @("--arch", "unet", "--edge_fusion", "abftied", "--edge_gamma", "0.1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_unet_egabftied_gm10_c2d" }
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
    if (-not $busy -and $rows -ge 2) { break }
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
Write-Host "[abf] queue finished"
