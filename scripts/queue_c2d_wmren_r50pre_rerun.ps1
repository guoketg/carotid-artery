# 补跑 2026-11-07 被中断的 R50-WRMM 臂（原 queue_c2d_wmren_r50pre.ps1 的 F1）。
#
# 中断原因：15:11 误杀系统 Python 子进程（实为 venv 启动器的正常子进程），44.5 min 处退出 -1。
#
# 对照：K2（WRMM 编码 + unet_dec，从头训，87.77 / 0.1923 / HD 0.979）
#       A0 90.32 只作跨骨干参照，不判门槛。
# 判定（§0）：|ΔDice| ≥ 0.5 或 HD 降 ≥ 0.1 mm。
#
# 串行策略：等 queue_c2d_r50_fusion（F1–F3）与 queue_c2d_r50_egf（G1）都跑完再开始。
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_r50pre_rerun.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_r50pre_rerun.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilR50Rerun" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Test-QueueDone([string]$file, [int]$rows) {
    $p = Join-Path $logDir $file
    if (-not (Test-Path $p)) { return $false }
    $n = @(Get-Content $p | Select-Object -Skip 1 | Where-Object { $_ -ne "" }).Count
    return ($n -ge $rows)
}

$argsList = @(
    "train.py", "--subset", "SPARC", "--seed", "1234",
    "--dataset", "Carotid", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_enc", "r50wrmm"
)
$snapshot = Join-Path $modelDir "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_wmren_abunetdec_encr50wrmm_c2d"

while (-not $DryRun) {
    if ((Test-QueueDone "queue_c2d_r50_fusion_status.csv" 3) -and
        (Test-QueueDone "queue_c2d_r50_egf_status.csv" 1) -and
        -not (Get-TrainingProcess)) { break }
    Write-Host "[r50rerun] waiting: fusion/egf queues not finished or GPU busy, 60s..."
    Start-Sleep -Seconds 60
}

if ((Test-Path $snapshot) -and -not $DryRun) {
    $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Move-Item -LiteralPath $snapshot -Destination $bak
    Write-Host "[r50rerun] archived interrupted run to $bak"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$jobLog = Join-Path $logDir "wmren_encr50wrmm_rerun_$ts.log"
Write-Host "[r50rerun] $Python $($argsList -join ' ')"
Write-Host "[r50rerun] log: $jobLog"
if ($DryRun) { $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null; exit 0 }

$statusPath = Join-Path $logDir "queue_c2d_wmren_r50pre_rerun_status.csv"
if (-not (Test-Path $statusPath)) { "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8 }

$start = Get-Date
Push-Location $Root
& $Python @argsList *> $jobLog
$code = $LASTEXITCODE
Pop-Location
$end = Get-Date
$minutes = [math]::Round(($end - $start).TotalMinutes, 1)
"r50wrmm_rerun,$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[r50rerun] finished exit=$code ${minutes}min"
