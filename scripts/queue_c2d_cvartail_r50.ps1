# 2026-11-14 队列：cvartail 迁移到 R50（1 job，单变量）。
#
# 依据：results.md §10 —— cvartail 在 U-Net 上是目前唯一双门槛全过的损失臂：
#   UNet 86.55 / 0.225 / 1.14  ->  87.37 / 0.200 / 0.96（Dice +0.82、HD −0.18）
#   配置：--loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05（q=0.1 的 CVaR 尾部距离项）
# 本臂只换骨干，损失参数逐字不变（复现，不是调参）：
#   R50-TransUNet（--arch baseline --vit_layers 0）配对 A0 90.4903 / 0.14473 / HD 0.7062
# 判定：ΔHD <= -0.1 mm（主判）且 ΔDice >= -0.3（护栏）；过线后 3 seed（42/317/1029）复核。
# 快照：TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_cvar_c2d
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvartail_r50.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvartail_r50.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerCvar -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerCvar]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            $_.CommandLine -like "*queue_*.ps1*" -and
            $_.CommandLine -notlike "*queue_c2d_cvartail_r50*"
        }
}

$statusPath = Join-Path $logDir "queue_c2d_cvartail_r50_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "R50CVAR"; tag = "r50_cvar"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--loss", "cvartail", "--tl_w", "0.1", "--tl_q", "0.1", "--tl_area_w", "0.05")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_cvar_c2d" }
)

foreach ($job in $jobs) {
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.snap
    $argsList = @("train.py") + $common + $job.a
    Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[$($job.n)] snapshot: $snapshot"
    if ($DryRun) { continue }

    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[$($job.n)] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
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
Write-Host "[cvartail/r50] queue finished"
