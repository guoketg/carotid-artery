# 2026-11-04 队列：BDFR-Net（3612 Submission）忠实复现的 4 组消融。
# 骨干/数据集沿用本项目：TransUNet（R50-ViT-B/16，--arch baseline）+ SPARC 颈动脉；
# 除数据集与骨干外，按论文 Eq.(1)–(10) 重写解码器（networks/bdfr_v2.py + DecoderCupBDFR）。
#
#   B0  --arch baseline                          纯 TransUNet（论文口径重跑，作为同一协议下的基线）
#   B1  --arch baseline --bdfr_v2 1              完整论文实现（DDBE×4 + ACAM 替代 concat）
#   B2  --arch baseline --bdfr_v2 1 --bdfr_ddbe 0  完整去掉 DDBE（连同其 ECA），只留 ACAM
#   B3  --arch baseline --bdfr_v2 1 --bdfr_acam 0  去掉 ACAM（回退 concat），只留 DDBE×4
#
# 论文口径（§3.1）：AdamW lr 1e-4 / wd 1e-4、Dice+CE、rotation+gray-scale+gamma 增强、
# batch 16、100 epoch。硬件限制（12GB 单卡）用 bs8 × ga2 = 有效 bs16 等价替代；
# 输入 224²（本骨干的位置编码与 R50 网格按 224 构造，论文 256² 会引入骨干侧差异）。
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_bdfr2.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_bdfr2.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilBdfr2" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC", "--seed", "1234",
    "--img_size", "224",
    "--batch_size", "8", "--grad_accum", "2",
    "--max_epochs", "100", "--eval_test_every", "5",
    "--optim", "adamw", "--base_lr", "0.0001", "--gamma_aug", "1", "--loss", "ce_dice"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo100_bs8_ga2_lr0.0001_224"

$jobs = @(
    # 显存最重的 B1 先跑：若 12GB 放不下会立刻失败，不至于白等一整夜
    @{ Name = "B1_bdfr2_full"; Args = @("train.py", "--arch", "baseline", "--bdfr_v2", "1") + $common;
       Snapshot = "${pre}_SPARC_bdfr2_c2d" },
    @{ Name = "B0_baseline"; Args = @("train.py", "--arch", "baseline") + $common;
       Snapshot = "${pre}_SPARC_c2d" },
    @{ Name = "B3_bdfr2_no_acam"; Args = @("train.py", "--arch", "baseline", "--bdfr_v2", "1", "--bdfr_acam", "0") + $common;
       Snapshot = "${pre}_SPARC_bdfr2_na_c2d" },
    @{ Name = "B2_bdfr2_no_ddbe"; Args = @("train.py", "--arch", "baseline", "--bdfr_v2", "1", "--bdfr_ddbe", "0") + $common;
       Snapshot = "${pre}_SPARC_bdfr2_nod_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_bdfr2_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[bdfr2] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[bdfr2] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[bdfr2] ===== start $($job.Name) ====="
    Write-Host "[bdfr2] command: $Python $($job.Args -join ' ')"
    Write-Host "[bdfr2] snapshot: $($job.Snapshot)"
    Write-Host "[bdfr2] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[bdfr2] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[bdfr2] all jobs finished"
