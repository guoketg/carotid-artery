# 2026-11-07 队列：R50 单编码器协议下的融合算子对照（3 个 job 串行）。
# 宿主统一为新协议 A：--vit_layers 0（去掉 12 层 ViT，只留 R50 + DecoderCup，23.5 M），
# 统一口径：SPARC / seed 1234 / bs4 + ga2 / 200 epoch / 每 5 epoch 测。
# A 基线已跑过（90.49 / ASSD 0.142 / HD 0.71），此队列不重跑基线。
#
#   F1  --ffusion 1                            FreqFusion（NeurIPS 2024，Eq.6–13）
#                                              插在 up+skip 融合点，+0.413 M
#   F2  --bdfr_v2 1 --bdfr_acam 0              DDBE 块（DDBE+ECA）只挂解码特征，skip 仍 concat
#                                              即 B3 配置搬到 R50 宿主，约 +30 M
#   F3  同 F2 + --bdfr_wave_op lap             DDBE 块内 Haar → 单尺度拉普拉斯（零参数变化）
#
# 判定（对 A = 90.49 / HD 0.71）：
#   过门槛 = Dice ≥ 90.99 或 HD ≤ 0.61；两条都不过则记"不显著"。
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_fusion.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_fusion.ps1
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

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilR50Fusion" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC", "--seed", "1234",
    "--vit_layers", "0",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224"

$jobs = @(
    # 显存增量最小的先跑，便于早发现 OOM
    @{ Name = "F1_r50_ffusion"; Args = @("train.py", "--arch", "baseline", "--ffusion", "1") + $common;
       Snapshot = "${pre}_SPARC_c2d_ff" },
    @{ Name = "F2_r50_ddbe_only"; Args = @("train.py", "--arch", "baseline", "--bdfr_v2", "1", "--bdfr_acam", "0") + $common;
       Snapshot = "${pre}_SPARC_c2d_bdfr2_na" },
    @{ Name = "F3_r50_ddbe_lap"; Args = @("train.py", "--arch", "baseline", "--bdfr_v2", "1", "--bdfr_acam", "0", "--bdfr_wave_op", "lap") + $common;
       Snapshot = "${pre}_SPARC_c2d_bdfr2_na_bwlap" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_r50_fusion_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[r50f] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[r50f] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[r50f] ===== start $($job.Name) ====="
    Write-Host "[r50f] command: $Python $($job.Args -join ' ')"
    Write-Host "[r50f] snapshot: $($job.Snapshot)"
    Write-Host "[r50f] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[r50f] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[r50f] all jobs finished"
