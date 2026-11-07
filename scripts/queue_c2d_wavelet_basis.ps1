# 2026-11-05 队列：小波基选择实验（C 组口径，5 个 job 串行）。
# 对应 docs/experiments.md §1.11。宿主固定为 C 组的 U-Net + 边缘通道
# (--arch unet --edge_in 1 --edge_op wavelet)，Haar 那一行复用已有结果
# ${pre}_SPARC_edge_unet_eowavelet_c2d（Dice 87.42），本队列只补非 Haar 基。
#
# 只换滤波器组，其余全部锁死：log1p 域、1 级分解、LL 置零、IDWT 重建、
# 逐样本 z-score、U-Net 结构/参数量(60.298 M)/超参/seed。统一口径 $common：
#   SPARC / seed 1234 / bs4 + ga2（等效 bs8）/ 200 epoch / --eval_test_every 5
#
# 快照名（train.py 拼接规则：_edge → _wb<basis> → _unet → _eowavelet → _c2d）：
#   ${pre}_SPARC_edge_wbdb2_unet_eowavelet_c2d
#   ${pre}_SPARC_edge_wbdb4_unet_eowavelet_c2d
#   ${pre}_SPARC_edge_wbsym4_unet_eowavelet_c2d
#   ${pre}_SPARC_edge_wbcoif1_unet_eowavelet_c2d
#   ${pre}_SPARC_edge_wbbior1.5_unet_eowavelet_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wavelet_basis.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wavelet_basis.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilWB" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

$jobs = @(
    @{ Name = "WB1_db2";    Basis = "db2";
       Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet",
                "--edge_in", "1", "--edge_op", "wavelet", "--wavelet_basis", "db2") + $common },
    @{ Name = "WB2_db4";    Basis = "db4";
       Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet",
                "--edge_in", "1", "--edge_op", "wavelet", "--wavelet_basis", "db4") + $common },
    @{ Name = "WB3_sym4";   Basis = "sym4";
       Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet",
                "--edge_in", "1", "--edge_op", "wavelet", "--wavelet_basis", "sym4") + $common },
    @{ Name = "WB4_coif1";  Basis = "coif1";
       Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet",
                "--edge_in", "1", "--edge_op", "wavelet", "--wavelet_basis", "coif1") + $common },
    @{ Name = "WB5_bior1.5"; Basis = "bior1.5";
       Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet",
                "--edge_in", "1", "--edge_op", "wavelet", "--wavelet_basis", "bior1.5") + $common }
)
foreach ($job in $jobs) {
    $job.Snapshot = "${pre}_SPARC_edge_wb$($job.Basis)_unet_eowavelet_c2d"
}

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_wavelet_basis_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[wbasis] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[wbasis] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[wbasis] ===== start $($job.Name) ($($job.Basis)) ====="
    Write-Host "[wbasis] command: $Python $($job.Args -join ' ')"
    Write-Host "[wbasis] snapshot: $($job.Snapshot)"
    Write-Host "[wbasis] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[wbasis] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[wbasis] all jobs finished"
