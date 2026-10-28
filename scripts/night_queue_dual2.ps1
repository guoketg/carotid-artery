# Overnight queue 2: strong-fusion operators (D11 r4/r5/r6).
# Queued AFTER night_queue_dual.ps1: waits until the GPU has been idle for 10 min,
# so the two queues never interleave and never share the 8 GB card.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual2.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual2.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual2.ps1 -Only r4_sparc
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [string]$Skip = "",
    [int]$IdleMinutes = 10,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilDual2" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid", "--arch", "dual", "--levels", "1,2,3,4",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "120", "--eval_test_every", "10"
)
$prefix = "TU_pretrain_R50-ViT-B_16_skip3_epo120_bs4_ga2_224"

# Order == priority: SPARC (main benchmark) for all three operators first,
# then the Zhongnan cross-dataset repeats.
$jobs = @(
    @{ Name = "r4_sparc";    Args = @("train.py", "--subset", "SPARC",    "--fuse_op", "deform") + $common; Snapshot = "${prefix}_SPARC_dual_L1234_deform_c2d" },
    @{ Name = "r5_sparc";    Args = @("train.py", "--subset", "SPARC",    "--fuse_op", "aff") + $common; Snapshot = "${prefix}_SPARC_dual_L1234_aff_c2d" },
    @{ Name = "r6_sparc";    Args = @("train.py", "--subset", "SPARC",    "--fuse_op", "auto", "--cross_scale", "1") + $common; Snapshot = "${prefix}_SPARC_dual_L1234_auto_cs_c2d" },
    @{ Name = "r4_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--fuse_op", "deform") + $common; Snapshot = "${prefix}_Zhongnan_dual_L1234_deform_c2d" },
    @{ Name = "r5_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--fuse_op", "aff") + $common; Snapshot = "${prefix}_Zhongnan_dual_L1234_aff_c2d" },
    @{ Name = "r6_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--fuse_op", "auto", "--cross_scale", "1") + $common; Snapshot = "${prefix}_Zhongnan_dual_L1234_auto_cs_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

if ($Skip -ne "") {
    $skipNames = @($Skip -split "," | ForEach-Object { $_.Trim() })
    $jobs = @($jobs | Where-Object { $skipNames -notcontains $_.Name })
    Write-Host "[queue2] skipping: $($skipNames -join ', ')"
}

$statusPath = Join-Path $logDir "queue_dual2_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# Wait for queue 1 to drain: 10 consecutive idle minutes means the previous
# queue is done (its inter-job gaps are only seconds).
$idle = 0
while (-not $DryRun) {
    if (Get-TrainingProcess) {
        $idle = 0
        Start-Sleep -Seconds 60
    } else {
        $idle++
        if ($idle -ge $IdleMinutes) { break }
        Write-Host "[queue2] GPU idle $idle/$IdleMinutes min, waiting for queue 1 to finish..."
        Start-Sleep -Seconds 60
    }
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[queue2] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[queue2] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[queue2] ===== start $($job.Name) ====="
    Write-Host "[queue2] command: $Python $($job.Args -join ' ')"
    Write-Host "[queue2] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[queue2] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[queue2] all jobs finished"
