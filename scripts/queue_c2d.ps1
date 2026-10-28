# 新口径（官方划分 Carotid2D + mm 物理单位 + bs4/grad_accum2 + max_epochs 200，取 epoch_149.pth）补齐队列。
# 目的：消掉 advisor-report-2026-10-26 §6 里两个未消掉的混杂（基线 batch 口径不一致、单 seed），
#       并补上老师指示的跨子集复现。
# 关键约定：
#   * 所有 job 必须 --max_epochs 200 --eval_test_every 5（poly 衰减总长绑死 max_epochs，
#     只有同调度下取 epoch 150 才可比；历史 seed1234 基线就是这个口径）。
#   * 所有 job 必须 --batch_size 4 --grad_accum 2（等效 batch 8，与历史基线同口径）。
#   * 快照目录名由 train.py 自动拼装，这里的 Snapshot 字段用于归档上一轮产物，必须与之一致。
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d.ps1 -Only b2024
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d.ps1 -Skip b2024,b3407
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilC2D" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# 统一超参：新口径。--max_epochs 200 是 poly 调度的一部分，不能改成 120。
$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$dual = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto")
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

# 顺序 == 优先级：
#   1) 双骨干 SPARC seed1234 —— 与已跑完的基线 seed1234 直接配对，最快拿到主结论
#   2) 3) 基线 seed2024/3407 —— 估计方差
#   4) 5) 双骨干 seed2024/3407 —— 3 seed × 2 配置的完整方差估计
#   6) 7) Zhongnan 复现 —— 最低优先级，只在前 5 个跑完后才占用 GPU
$jobs = @(
    @{ Name = "d1234_sparc";    Args = @("train.py", "--subset", "SPARC",    "--seed", "1234") + $dual + $common; Snapshot = "${pre}_SPARC_dual_L1234_auto_c2d" },
    @{ Name = "b2024_sparc";    Args = @("train.py", "--subset", "SPARC",    "--seed", "2024") + $common;            Snapshot = "${pre}_s2024_SPARC_c2d" },
    @{ Name = "b3407_sparc";    Args = @("train.py", "--subset", "SPARC",    "--seed", "3407") + $common;            Snapshot = "${pre}_s3407_SPARC_c2d" },
    @{ Name = "d2024_sparc";    Args = @("train.py", "--subset", "SPARC",    "--seed", "2024") + $dual + $common; Snapshot = "${pre}_s2024_SPARC_dual_L1234_auto_c2d" },
    @{ Name = "d3407_sparc";    Args = @("train.py", "--subset", "SPARC",    "--seed", "3407") + $dual + $common; Snapshot = "${pre}_s3407_SPARC_dual_L1234_auto_c2d" },
    @{ Name = "b1234_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--seed", "1234") + $common;           Snapshot = "${pre}_Zhongnan_c2d" },
    @{ Name = "d1234_zhongnan"; Args = @("train.py", "--subset", "Zhongnan", "--seed", "1234") + $dual + $common;  Snapshot = "${pre}_Zhongnan_dual_L1234_auto_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

if ($Skip -ne "") {
    $skipNames = @($Skip -split "," | ForEach-Object { $_.Trim() })
    $jobs = @($jobs | Where-Object { $skipNames -notcontains $_.Name })
    Write-Host "[c2d] skipping: $($skipNames -join ', ')"
}

$statusPath = Join-Path $logDir "queue_c2d_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[c2d] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[c2d] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[c2d] ===== start $($job.Name) ====="
    Write-Host "[c2d] command: $Python $($job.Args -join ' ')"
    Write-Host "[c2d] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[c2d] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[c2d] all jobs finished"
