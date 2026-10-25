# Sequential overnight training queue.
# Runs one job at a time, waits for the previous one, never runs two in parallel.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1            # run all jobs
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1 -DryRun   # print plan only
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1 -Only a0_sparc
#   powershell -ExecutionPolicy Bypass -File scripts\night_queue.ps1 -Skip a0_sparc,a1_sparc   # resume
param(
    [string]$Python = "F:\code\MyPython\TransUNet\venv\Scripts\python.exe",
    [string]$Root = "F:\code\MyPython\TransUNet",
    [string]$Only = "",
    [string]$Skip = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "F:\code\MyPython\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Keep the machine awake while the queue runs.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtil" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# Jobs run strictly in this order. Each entry: name + python args + snapshot folder.
# Order puts the SPARC baseline/attention comparison first so that even a partial night
# still yields the core result.
$common = @("--batch_size", "8", "--max_epochs", "120", "--eval_test_every", "10")
$prefix = "TU_pretrain_R50-ViT-B_16_skip3_epo120_bs8_224"
$jobs = @(
    @{ Name = "a0_sparc"; Args = @("train.py", "--subset", "SPARC") + $common; Snapshot = "${prefix}_SPARC" },
    @{ Name = "a3_sparc"; Args = @("train.py", "--subset", "SPARC", "--use_wavelet", "1", "--fusion", "cross") + $common; Snapshot = "${prefix}_SPARC_w3_cross" },
    @{ Name = "a1_sparc"; Args = @("train.py", "--subset", "SPARC", "--use_wavelet", "1") + $common; Snapshot = "${prefix}_SPARC_w3" },
    @{ Name = "a0_zhongnan"; Args = @("train.py", "--subset", "Zhongnan") + $common; Snapshot = "${prefix}_Zhongnan" },
    @{ Name = "a3_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--use_wavelet", "1", "--fusion", "cross") + $common; Snapshot = "${prefix}_Zhongnan_w3_cross" },
    @{ Name = "a1_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--use_wavelet", "1") + $common; Snapshot = "${prefix}_Zhongnan_w3" }
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

$statusPath = Join-Path $logDir "queue_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    # Enforce one training process at a time; wait for anything already running.
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
