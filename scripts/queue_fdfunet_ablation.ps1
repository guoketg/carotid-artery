# FDFUNet 单模块减法消融队列（Neural Networks 2024，Table 7 的减法版）。
# 三条 job 都在「完整 FDFUNet」上只关一个模块，其余与 queue_fdfunet.ps1 的完整版完全一致：
#   no_drdac = 编码器 DRDAC 换普通 3×3 卷积
#   no_mfdf  = FEM 只保留 ASCA
#   no_asca  = FEM 只保留 MFDF
# 对照：完整 FDFUNet / UNet* 已有快照（queue_fdfunet.ps1，2026-11-05 00:39 收口），不重跑。
# 口径：SPARC / seed 1234 / 224² / bs4+ga2 / 200 epoch / eval_test_every 5 /
#       Adam 1e-4 cosine→1e-5 / loss 0.5BCE+0.5IoU+Dice（论文 Eq.6）。判定点 ep200。
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fdfunet_ablation.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fdfunet_ablation.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 训练期间阻止系统休眠（与 queue_fdfunet.ps1 同写法）
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilFdfAbl" -Namespace Win32 -PassThru
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

$jobs = @(
    @{ Name = "no_drdac"; Args = $common + @("--fdfunet_ablate", "no_drdac") },
    @{ Name = "no_mfdf";  Args = $common + @("--fdfunet_ablate", "no_mfdf") },
    @{ Name = "no_asca";  Args = $common + @("--fdfunet_ablate", "no_asca") }
)

$statusPath = Join-Path $logDir "queue_fdfunet_ablation_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[fdfab] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "fdfab_$($job.Name)_$ts.log"
    Write-Host "[fdfab] ===== start $($job.Name) ====="
    Write-Host "[fdfab] command: $Python $($job.Args -join ' ')"
    Write-Host "[fdfab] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[fdfab] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[fdfab] all jobs finished"
