# 2026-11-02 接力队列：ASR v2（逐通道自适应核 + tanh 软偏移）
#
# 为什么接力而不是并列：本机 12GB 显存只够跑一个 train.py，且两个队列脚本
# 若同时盯着 GPU，会在前一个 job 结束的间隙里同时抢卡。这里显式等待
# queue_c2d_dl_sweep.ps1 进程退出后再启动，保证严格串行。
#
# 对照口径（§0）：SPARC / bs4 + ga2 / 200 epoch / seed 1234 / ep200 读数。
#   U-Net 基线    86.55（已有）
#   ASR v1        81.30（V1，判为实现不等价）
#   ASR v2        本脚本 —— 只改模块实现，其他一切不变，单变量可比
#
# 快照名：${pre}_SPARC_asr2_unet_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_asr2.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_asr2.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilAsr2" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-PreviousQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_c2d_dl_sweep.ps1*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$jobName = "ASR2"
$jobArgs = @("train.py", "--subset", "SPARC", "--batch_size", "4", "--grad_accum", "2",
             "--max_epochs", "200", "--eval_test_every", "5", "--seed", "1234",
             "--arch", "unet", "--asr", "1")
$snapshot = Join-Path $modelDir "${pre}_SPARC_asr2_unet_c2d"

$statusPath = Join-Path $logDir "queue_c2d_asr2_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

Write-Host "[asr2] command: $Python $($jobArgs -join ' ')"
Write-Host "[asr2] snapshot: $snapshot"
if ($DryRun) {
    Write-Host "[asr2] dry-run only, waiting for previous queue: $((Get-PreviousQueue | Measure-Object).Count) proc"
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "[asr2] queue finished"
    exit 0
}

# 等前一个队列脚本退出（它内部已经保证 job 之间不并行）
while (Get-PreviousQueue) {
    Write-Host "[asr2] waiting for queue_c2d_dl_sweep to finish (60s)..."
    Start-Sleep -Seconds 60
}
while (Get-TrainingProcess) {
    Write-Host "[asr2] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

if (Test-Path $snapshot) {
    $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Move-Item -LiteralPath $snapshot -Destination $bak
    Write-Host "[asr2] archived previous run to $bak"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$jobLog = Join-Path $logDir "$($jobName)_$ts.log"
Write-Host "[asr2] ===== start $jobName ====="
Write-Host "[asr2] log: $jobLog"

$start = Get-Date
Push-Location $Root
& $Python @jobArgs *> $jobLog
$code = $LASTEXITCODE
Pop-Location
$end = Get-Date
$mins = [math]::Round(($end - $start).TotalMinutes, 1)
"$jobName,$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
    Out-File -FilePath $statusPath -Append -Encoding utf8
Write-Host "[asr2] ===== done $jobName exit=$code ($mins min) ====="

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[asr2] queue finished"
