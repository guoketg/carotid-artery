# 2026-11-01 接力队列：两篇论文的补全复现（等 Syn-Net 队列 queue_c2d_syn_fds.ps1 与所有 train.py 退出）。
#
#   W2  --arch wmren
#       WMREN (IJCAI 2025) 完整复现：4×WRMM（WRDM 双支小波+残差 + MLKM）
#       + 3×REM（CBAM + CREM + SAFM）。替换掉 W1 那份缺 MLKM、右支只有单层
#       3×3 s2、REM 只两级带 CBAM 的简化版（W1 目录会被归档成 _bak_*）。
#   V1  --arch unet --asr 1
#       WeaveSeg (ICCV 2025) 的 ASR：解码器 28²/56²/112² 三级上采样换成
#       1×1 通道对齐 + ASR(F_high, F_low)，其余与 U-Net 基线逐键一致。
#       注意 ASR 版比基线少 2.93M 参数（57.37M vs 60.30M），容量不等，
#       结论只能按"同预算更好"读，不能单独算 ASR 净增益。
#
# 统一口径：SPARC / seed 1234 / bs4 + ga2（等效 bs8）/ 200 epoch / --eval_test_every 5。
# 快照：W2 → ${pre}_SPARC_wmren_c2d；V1 → ${pre}_SPARC_asr_unet_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_asr.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_asr.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilWmrenAsr" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-SynQueue {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*queue_c2d_syn_fds.ps1*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

$jobs = @(
    @{ Name = "W2_wmren_full"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234",
                                        "--arch", "wmren",
                                        "--batch_size", "4", "--grad_accum", "2",
                                        "--max_epochs", "200", "--eval_test_every", "5");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_wmren_c2d" },
    @{ Name = "V1_unet_asr"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234",
                                      "--arch", "unet", "--asr", "1",
                                      "--batch_size", "4", "--grad_accum", "2",
                                      "--max_epochs", "200", "--eval_test_every", "5");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_asr_unet_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_wmren_asr_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while (-not $DryRun -and ((Get-SynQueue) -or (Get-TrainingProcess))) {
        Write-Host "[wmren-asr] syn-fds queue or train.py still running, wait 60s..."
        Start-Sleep -Seconds 60
    }

    if ((Test-Path $job.Snapshot) -and -not $DryRun) {
        $bak = "$($job.Snapshot)_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $job.Snapshot -Destination $bak
        Write-Host "[wmren-asr] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[wmren-asr] ===== start $($job.Name) ====="
    Write-Host "[wmren-asr] command: $Python $($job.Args -join ' ')"
    Write-Host "[wmren-asr] snapshot: $($job.Snapshot)"
    Write-Host "[wmren-asr] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[wmren-asr] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[wmren-asr] all jobs finished"
