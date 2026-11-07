# FDFUNet 复现主对比队列（docs/study/fdfunet-reproduction.md §7）。
# 依次跑「论文同构基线 UNet*」→「完整 FDFUNet」，两者除 --fdfunet_variant 外命令完全一致，
# 保证 ΔDice 只归因于 DRDAC / MFDF / ASCA。
# 口径：SPARC / seed 1234 / 224² / bs4+ga2 / 200 epoch / eval_test_every 5 / Adam 1e-4 cosine→1e-5 /
#       loss 0.5BCE+0.5IoU+Dice（论文 Eq.6）。判定点 ep150（epoch_149.pth）。
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fdfunet.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fdfunet.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 训练期间阻止系统休眠（与 queue_c2d_repro.ps1 同写法）
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilFdfunet" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "train.py",
    "--dataset", "Carotid", "--subset", "SPARC", "--seed", "1234",
    "--img_size", "224", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--optim", "adam", "--base_lr", "1e-4",
    "--loss", "bce_iou_dice", "--num_workers", "1",
    "--arch", "fdfunet", "--tag", "fdf"
)

# 顺序：先基线后完整版，基线早出数也便于中途核对
$jobs = @(
    @{ Name = "unetstar"; Args = $common + @("--fdfunet_variant", "unetstar") },
    @{ Name = "fdfunet";  Args = $common }
)

$statusPath = Join-Path $logDir "queue_fdfunet_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[fdf] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[fdf] ===== start $($job.Name) ====="
    Write-Host "[fdf] command: $Python $($job.Args -join ' ')"
    Write-Host "[fdf] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[fdf] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[fdf] all jobs finished"
