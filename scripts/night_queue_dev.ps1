# Fast single-variable screening on the case-level dev subset (SPARC, 175 slices / 46 cases).
# Runs AFTER any in-flight train.py drains, then STOPS and leaves the GPU free
# (full-data queues are launched manually the next morning, after reading the ranking).
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dev.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dev.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dev.ps1 -Only d05_deform
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [string]$Skip = "",
    [int]$IdleMinutes = 3,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilDev" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# Dev subset: 60 epochs is enough for RANKING only (never for paper numbers).
$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--list_dir", "./lists/lists_Carotid_dev",
    "--batch_size", "4", "--max_epochs", "60", "--eval_test_every", "20"
)
$d = @("--arch", "dual", "--levels", "1,2,3,4")

# One factor changes per row; reference point = d01 (auto / L1234 / w8 / dk2 / bidir / pretrained side).
$jobs = @(
    @{ Name = "d00_base";   Args = @("train.py", "--arch", "baseline", "--tag", "base") + $common },
    @{ Name = "d01_auto";   Args = @("train.py") + $d + @("--fuse_op", "auto") + $common },
    @{ Name = "d02_gate";   Args = @("train.py") + $d + @("--fuse_op", "gate") + $common },
    @{ Name = "d03_concat"; Args = @("train.py") + $d + @("--fuse_op", "concat") + $common },
    @{ Name = "d04_uni";    Args = @("train.py") + $d + @("--fuse_op", "auto", "--bidir", "0") + $common },
    @{ Name = "d05_deform"; Args = @("train.py") + $d + @("--fuse_op", "deform") + $common },
    @{ Name = "d06_aff";    Args = @("train.py") + $d + @("--fuse_op", "aff") + $common },
    @{ Name = "d07_cs";     Args = @("train.py") + $d + @("--fuse_op", "auto", "--cross_scale", "1") + $common },
    @{ Name = "d08_L3";     Args = @("train.py", "--arch", "dual", "--levels", "3", "--fuse_op", "auto") + $common },
    @{ Name = "d09_L34";    Args = @("train.py", "--arch", "dual", "--levels", "3,4", "--fuse_op", "auto") + $common },
    @{ Name = "d10_L123";   Args = @("train.py", "--arch", "dual", "--levels", "1,2,3", "--fuse_op", "auto") + $common },
    @{ Name = "d11_w4";     Args = @("train.py") + $d + @("--fuse_op", "auto", "--window", "4", "--tag", "w4") + $common },
    @{ Name = "d12_w16";    Args = @("train.py") + $d + @("--fuse_op", "auto", "--window", "16", "--tag", "w16") + $common },
    @{ Name = "d13_dk1";    Args = @("train.py") + $d + @("--fuse_op", "auto", "--dk_ratio", "1", "--tag", "dk1") + $common },
    @{ Name = "d14_dk4";    Args = @("train.py") + $d + @("--fuse_op", "auto", "--dk_ratio", "4", "--tag", "dk4") + $common },
    @{ Name = "d15_sp0";    Args = @("train.py") + $d + @("--fuse_op", "auto", "--side_pretrained", "0") + $common }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}
if ($Skip -ne "") {
    $skipNames = @($Skip -split "," | ForEach-Object { $_.Trim() })
    $jobs = @($jobs | Where-Object { $skipNames -notcontains $_.Name })
    Write-Host "[dev] skipping: $($skipNames -join ', ')"
}

$statusPath = Join-Path $logDir "queue_dev_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# Wait for the in-flight full run (r1_sparc) to drain.
$idle = 0
while (-not $DryRun) {
    if (Get-TrainingProcess) {
        $idle = 0
        Start-Sleep -Seconds 60
    } else {
        $idle++
        if ($idle -ge $IdleMinutes) { break }
        Write-Host "[dev] GPU idle $idle/$IdleMinutes min, waiting for the full run to finish..."
        Start-Sleep -Seconds 60
    }
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[dev] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    # snapshot suffix is auto-built by train.py (adds _dev for lists_Carotid_dev);
    # names are resolved here only for archiving.
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[dev] ===== start $($job.Name) ====="
    Write-Host "[dev] command: $Python $($job.Args -join ' ')"
    Write-Host "[dev] log: $jobLog"
    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[dev] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

# Phase 2 removed on purpose: screening finishes early, GPU stays free for the
# morning decision (which winners to promote to the full-data queues).

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[dev] all jobs finished; GPU left free for morning triage"
