# 2026-11-01 队列：Syn-Net(JBHI 2025) 复现线，对应 docs/experiments.md §1.6 的 F1 / S1 两行。
# 统一口径：SPARC / seed 1234 / bs4 + ga2（等效 bs8）/ 200 epoch / --eval_test_every 5。
#
#   F1        --arch dual --fuse_op frc --fds 1
#             四级 FrCFusion 不变，解码器 14²/28²/56² 加 FDS 辅助监督（0.2/0.3/0.5），
#             主损失仍是 0.5CE+0.5Dice
#   S1        --arch split --fuse_op frc_s --fds 1 --loss bce_wiou
#             R50 局部支 + ViT-B/16 全局支两条独立支路、都吃原图；FrCFusion 严格版；
#             FDS；损失 mean(BCE+wIoU)
#
# 快照名（train.py L217-245 拼接规则）：
#   F1_smoke → ${pre}_SPARC_dual_L1234_frc_fds_smoke_c2d
#   F1       → ${pre}_SPARC_dual_L1234_frc_fds_c2d
#   S1       → ${pre}_SPARC_split_frc_s_fds_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_syn_fds.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_syn_fds.ps1
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

# Prevent system sleep while training.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilSynFds" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

$jobs = @(
    @{ Name = "F1_smoke"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234",
                                   "--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "frc",
                                   "--fds", "1", "--tag", "smoke",
                                   "--batch_size", "4", "--grad_accum", "2",
                                   "--max_epochs", "2", "--eval_test_every", "1");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dual_L1234_frc_fds_smoke_c2d" },
    @{ Name = "F1_dual_frc_fds"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234",
                                          "--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "frc",
                                          "--fds", "1",
                                          "--batch_size", "4", "--grad_accum", "2",
                                          "--max_epochs", "200", "--eval_test_every", "5");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dual_L1234_frc_fds_c2d" },
    @{ Name = "S1_split_frcs_fds"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234",
                                            "--arch", "split", "--fuse_op", "frc_s",
                                            "--fds", "1", "--loss", "bce_wiou",
                                            "--batch_size", "4", "--grad_accum", "2",
                                            "--max_epochs", "200", "--eval_test_every", "5");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_split_frc_s_fds_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_syn_fds_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[syn-fds] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    if ((Test-Path $job.Snapshot) -and -not $DryRun) {
        $bak = "$($job.Snapshot)_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $job.Snapshot -Destination $bak
        Write-Host "[syn-fds] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[syn-fds] ===== start $($job.Name) ====="
    Write-Host "[syn-fds] command: $Python $($job.Args -join ' ')"
    Write-Host "[syn-fds] snapshot: $($job.Snapshot)"
    Write-Host "[syn-fds] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[syn-fds] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[syn-fds] all jobs finished"
