# Overnight queue: dual-encoder interleaved + 4-level bidirectional fusion (D11).
# Same mechanics as night_queue.ps1: one job at a time, archive old snapshots, keep machine awake.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue_dual.ps1 -Skip r1_sparc
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [string]$Skip = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Prevent system sleep while training.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilDual" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# batch 4 + grad_accum 2 == effective batch 8.
# Measured: batch 8 peaks at 5.0 GB and falls into the WDDM paging zone -> 2.6x slower per step.
$common = @(
    "--dataset", "Carotid", "--arch", "dual", "--levels", "1,2,3,4",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "120", "--eval_test_every", "10"
)
$prefix = "TU_pretrain_R50-ViT-B_16_skip3_epo120_bs4_ga2_224"

# Order == priority: main method on both subsets first, then gate / concat / unidirectional controls.
$jobs = @(
    @{ Name = "r1_sparc";    Args = @("train.py", "--subset", "SPARC",    "--fuse_op", "auto") + $common; Snapshot = "${prefix}_SPARC_dual_L1234_auto_c2d" },
    @{ Name = "r1_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--fuse_op", "auto") + $common; Snapshot = "${prefix}_Zhongnan_dual_L1234_auto_c2d" },
    @{ Name = "r2_sparc";    Args = @("train.py", "--subset", "SPARC",    "--fuse_op", "gate") + $common; Snapshot = "${prefix}_SPARC_dual_L1234_gate_c2d" },
    @{ Name = "r2_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--fuse_op", "gate") + $common; Snapshot = "${prefix}_Zhongnan_dual_L1234_gate_c2d" },
    @{ Name = "r3_sparc";    Args = @("train.py", "--subset", "SPARC",    "--fuse_op", "concat") + $common; Snapshot = "${prefix}_SPARC_dual_L1234_concat_c2d" },
    @{ Name = "r3_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--fuse_op", "concat") + $common; Snapshot = "${prefix}_Zhongnan_dual_L1234_concat_c2d" },
    @{ Name = "r7_sparc";    Args = @("train.py", "--subset", "SPARC",    "--fuse_op", "auto", "--bidir", "0") + $common; Snapshot = "${prefix}_SPARC_dual_L1234_auto_uni_c2d" },
    @{ Name = "r7_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--fuse_op", "auto", "--bidir", "0") + $common; Snapshot = "${prefix}_Zhongnan_dual_L1234_auto_uni_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

if ($Skip -ne "") {
    $skipNames = @($Skip -split "," | ForEach-Object { $_.Trim() })
    $jobs = @($jobs | Where-Object { $skipNames -notcontains $_.Name })
    Write-Host "[queue] skipping: $($skipNames -join ', ')"
}

$statusPath = Join-Path $logDir "queue_dual_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[queue] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[queue] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[queue] ===== start $($job.Name) ====="
    Write-Host "[queue] command: $Python $($job.Args -join ' ')"
    Write-Host "[queue] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[queue] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[queue] all jobs finished"
