# 批次 0+1 复现队列（experiments.md §1.9）：faff 重跑 + ddbe4（四级全挂 DDBE）+ ddbefull（DDBE×4+ACAM）+ mainLL（XNet 搭车）。
# 口径与 §1.7 一致：SPARC / bs4+ga2 / 200 epoch（poly 绑死）/ eval_test_every 5，判定点 ep150（epoch_149.pth）。
# 判定规则（§1.9，跑之前写死）：|ΔDice| ≥ 0.005（病例级）或 HD95 下降 ≥ 0.5 mm 且 p < 0.05。
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_repro.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_repro.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_repro.ps1 -Skip faff
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilRepro" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# 统一超参：与 queue_c2d*.ps1 完全一致（poly 调度绑死 max_epochs=200）。
$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

# 顺序 == 优先级：批次 1 核心（ddbe4 → ddbefull，同 seed 配对消融「挂几层 + ACAM 增量」）
# → 批次 0 收尾（faff 修复后重跑）→ 搭车（mainLL，XNet 式低频主路）。
$jobs = @(
    @{ Name = "ddbe4";     Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--ddbe", "4") + $common;               Snapshot = "${pre}_SPARC_ddbe4_c2d" },
    @{ Name = "ddbefull";  Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--ddbe", "4", "--acam", "1") + $common; Snapshot = "${pre}_SPARC_ddbe4_acam_c2d" },
    @{ Name = "faff";      Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "aff") + $common; Snapshot = "${pre}_SPARC_dual_L1234_aff_c2d" },
    @{ Name = "mainll";    Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto", "--main_input", "ll") + $common; Snapshot = "${pre}_SPARC_dual_L1234_auto_mainLL_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

if ($Skip -ne "") {
    $skipNames = @($Skip -split "," | ForEach-Object { $_.Trim() })
    $jobs = @($jobs | Where-Object { $skipNames -notcontains $_.Name })
    Write-Host "[repro] skipping: $($skipNames -join ', ')"
}

$statusPath = Join-Path $logDir "queue_c2d_repro_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[repro] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[repro] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[repro] ===== start $($job.Name) ====="
    Write-Host "[repro] command: $Python $($job.Args -join ' ')"
    Write-Host "[repro] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[repro] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[repro] all jobs finished"
