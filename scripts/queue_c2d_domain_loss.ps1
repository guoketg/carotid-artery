# 2026-11-01 接力队列：颈动脉超声领域适配损失（不依赖任何图像细节假设）。
#
# 损失口径（--domain_loss 1）：
#   0.5 * 边界带加权 CE（GT 膨胀腐蚀差 ±2px 带，λ=3）
# + 0.5 * Dice
# + 0.05 * 面积一致性 |Σp − Σgt| / Σgt
# + 0.1 * 远处假阳性惩罚（GT 膨胀 10px 之外的预测前景质量占比）
#
# 已否决的 clDice：实测 SPARC GT 是半宽约 19px 的厚块，
# 官方软骨架在这类实心区域上恒为空集，损失恒等于 1、无梯度，故不采用。
#
# 两个 job 串行，等 ddbe / wmren 两个队列与所有 train.py 退出后再启动。
#   D1  --arch baseline --domain_loss 1   对照单通道基线 90.16% / HD95 0.44
#   D2  --arch unet     --domain_loss 1   对照纯卷积 U-Net 86.55% / HD95 0.77
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-OtherQueue {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*queue_c2d_unet_ddbe.ps1*" -or
                       $_.CommandLine -like "*queue_c2d_unet_wmren.ps1*" }
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

$jobs = @(
    @{ Name = "D1_baseline_dl"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "baseline", "--domain_loss", "1") + $common;
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dl_c2d" },
    @{ Name = "D2_unet_dl"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--domain_loss", "1") + $common;
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dl_unet_c2d" }
)

if (-not $DryRun) {
    while ((Get-OtherQueue) -or (Get-TrainingProcess)) {
        Write-Host "[domain-loss] ddbe/wmren queue or train.py still running, wait 60s..."
        Start-Sleep -Seconds 60
    }
}

foreach ($job in $jobs) {
    Write-Host "[domain-loss] $($job.Name) snapshot: $($job.Snapshot)"
}
if ($DryRun) { exit 0 }

foreach ($job in $jobs) {
    if (Test-Path $job.Snapshot) {
        $bak = "$($job.Snapshot)_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $job.Snapshot -Destination $bak
        Write-Host "[domain-loss] archived previous run to $bak"
    }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$statusPath = Join-Path $logDir "queue_c2d_domain_loss_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[domain-loss] command: $Python $($job.Args -join ' ')"
    Write-Host "[domain-loss] log: $jobLog"
    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[domain-loss] $($job.Name) end exit=$code ${minutes}min"
}
