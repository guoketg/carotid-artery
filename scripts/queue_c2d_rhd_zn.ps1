# 2026-11-18 queue: RHD (regional Hausdorff distance loss, #82) on Zhongnan, 2 jobs, single variable.
#
# Why this queue: the 2x2 acceptance grid (CLAUDE.md rule 21) had zero Zhongnan coverage for RHD.
# Only SPARC had readings: U-Net x SPARC = +1.66 / -0.253 (passes), ResNet50 x SPARC = +0.15 / -0.033
# and +0.33 / -0.052 (fails). Zhongnan both cells were empty, so this queue fills the two missing
# cells with the single strongest config (sym + I6) at seed 1234, direction check only.
#
# Jobs (paired baselines already on disk, same seed 1234):
#   ZN-U-RHD6   vs U-Net x Zhongnan baseline  90.0347 / 0.15421 / HD 0.71573 (n=187)
#   ZN-R50-RHD6 vs R50+DecoderCup x Zhongnan baseline 92.0041 / 0.12266 / HD 0.56511 (n=187)
# Judgement (registered before the run): delta Dice >= +0.5 AND delta HD <= -0.1 mm.
# Single seed only; 3-seed (42/317/1029) re-run needs explicit approval (rule 21), not scheduled here.
#
# Snapshots:
#   ..._224_Zhongnan_lrhd_TsymW0.5I6_unet_c2d
#   ..._vitL0_..._224_Zhongnan_lrhd_TsymW0.5I6_c2d
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\queue_c2d_rhd_zn.ps1 -DryRun
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\queue_c2d_rhd_zn.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerRhdZn -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerRhdZn]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Strict total-order queue mutex; priority comes from queue_priority.ps1 (single source of truth).
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 6

$statusPath = Join-Path $logDir "queue_c2d_rhd_zn_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "Zhongnan",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "ZN-U-RHD6"; tag = "zn_unet_rhd_i6"
       a = @("--arch", "unet",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_lrhd_TsymW0.5I6_unet_c2d" },
    @{ n = "ZN-R50-RHD6"; tag = "zn_r50_rhd_i6"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lrhd_TsymW0.5I6_c2d" }
)

foreach ($job in $jobs) {
    # Never share the GPU with another run: wait for every training process and every
    # higher-priority queue to clear first.
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
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
Write-Host "[rhd_zn] queue finished"
