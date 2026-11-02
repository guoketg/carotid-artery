# 2026-11-02 队列（第二轮）：
#   ① W2_s2024  WMREN 完整版复跑 seed 2024 —— 确认 +0.59 / HD −0.21 是否可复现
#   ② DL 扫参   A0 底座上对领域适配损失的三项权重做单因子 + 全加强，冲 §0 门槛（|ΔDice|≥0.5%）
#
# 统一口径：SPARC / bs4 + ga2（等效 bs8）/ 200 epoch / --eval_test_every 5；除 W2 外 seed 1234。
#
# 快照名（train.py 拼接规则；domain_loss 非默认超参会自动带 dl_tag）：
#   W2_s2024 → ${pre}_s2024_SPARC_wmren_c2d
#   DL_lam6  → ${pre}_SPARC_dlL6.0R2A0.05F0.1M10_c2d
#   DL_area  → ${pre}_SPARC_dlL3.0R2A0.15F0.1M10_c2d
#   DL_far   → ${pre}_SPARC_dlL3.0R2A0.05F0.3M10_c2d
#   DL_all   → ${pre}_SPARC_dlL6.0R2A0.15F0.3M10_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_dl_sweep.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_dl_sweep.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilDlSweep" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$common = @("train.py", "--subset", "SPARC", "--batch_size", "4", "--grad_accum", "2",
            "--max_epochs", "200", "--eval_test_every", "5")

$jobs = @(
    @{ Name = "W2_s2024"; Args = $common + @("--seed", "2024", "--arch", "wmren");
       Snapshot = Join-Path $modelDir "${pre}_s2024_SPARC_wmren_c2d" },
    @{ Name = "DL_lam6"; Args = $common + @("--seed", "1234", "--arch", "baseline",
                                            "--domain_loss", "1", "--dl_lambda", "6");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dlL6.0R2A0.05F0.1M10_c2d" },
    @{ Name = "DL_area"; Args = $common + @("--seed", "1234", "--arch", "baseline",
                                            "--domain_loss", "1", "--dl_area_w", "0.15");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dlL3.0R2A0.15F0.1M10_c2d" },
    @{ Name = "DL_far"; Args = $common + @("--seed", "1234", "--arch", "baseline",
                                           "--domain_loss", "1", "--dl_far_w", "0.3");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dlL3.0R2A0.05F0.3M10_c2d" },
    @{ Name = "DL_all"; Args = $common + @("--seed", "1234", "--arch", "baseline",
                                           "--domain_loss", "1", "--dl_lambda", "6",
                                           "--dl_area_w", "0.15", "--dl_far_w", "0.3");
       Snapshot = Join-Path $modelDir "${pre}_SPARC_dlL6.0R2A0.15F0.3M10_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_dl_sweep_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[dl-sweep] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    if ((Test-Path $job.Snapshot) -and -not $DryRun) {
        $bak = "$($job.Snapshot)_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $job.Snapshot -Destination $bak
        Write-Host "[dl-sweep] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[dl-sweep] ===== start $($job.Name) ====="
    Write-Host "[dl-sweep] command: $Python $($job.Args -join ' ')"
    Write-Host "[dl-sweep] snapshot: $($job.Snapshot)"
    Write-Host "[dl-sweep] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[dl-sweep] ===== done $($job.Name) exit=$code ($mins min) ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[dl-sweep] queue finished"
