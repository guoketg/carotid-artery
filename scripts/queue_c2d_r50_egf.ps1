# 2026-11-07 队列：R-GEF（可靠性门控边缘融合）在 R50 协议上的第一枪。
#
# 设计：解码器 skip 融合点，残差注入 skip' = skip + γ·(r ⊙ E)
#   E = 1×1(ReLU(BN(Laplacian(Gaussian(skip))))   固定核，不学习
#   r = σ(1×1([E, 上采样后的解码特征]))            逐像素可靠性
#   γ 初始化 0 → 训练第 0 步与基线逐位等价（已实测 max|Δ|=0.0）
#
# 宿主 A：--vit_layers 0（R50 + DecoderCup，23.5M），基线 90.49 / ASSD 0.142 / HD 0.71
#
# 只有一个 job：先看完整模块有没有增益。
#   过门槛 = Dice ≥ 90.99 或 HD95 ≤ 0.61；不过则记"不显著"，不再做拆分消融。
#
# 依赖：必须等 queue_c2d_r50_fusion（F1–F3）全部跑完再开始，避免抢卡。
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_egf.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_egf.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilR50Egf" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

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
    @{ Name = "G1_r50_rgef"; Args = @("train.py", "--arch", "baseline", "--edge_fusion", "rel") + $common;
       Snapshot = "${pre}_SPARC_c2d_egrel" }
)

$statusPath = Join-Path $logDir "queue_c2d_r50_egf_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# 前置队列（F1–F3）的完成标记：其 status.csv 至少 3 条数据行
$preQueueStatus = Join-Path $logDir "queue_c2d_r50_fusion_status.csv"
function Test-PreQueueDone {
    if (-not (Test-Path $preQueueStatus)) { return $false }
    $rows = @(Get-Content $preQueueStatus | Select-Object -Skip 1 | Where-Object { $_ -ne "" })
    return ($rows.Count -ge 3)
}

foreach ($job in $jobs) {
    while (-not $DryRun) {
        $busy = Get-TrainingProcess
        if (-not $busy -and (Test-PreQueueDone)) { break }
        if ($busy) { Write-Host "[rgef] another train.py is running, waiting 60s..." }
        else { Write-Host "[rgef] waiting for queue_c2d_r50_fusion to finish (F1-F3)..." }
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[rgef] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[rgef] ===== start $($job.Name) ====="
    Write-Host "[rgef] command: $Python $($job.Args -join ' ')"
    Write-Host "[rgef] snapshot: $($job.Snapshot)"
    Write-Host "[rgef] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[rgef] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[rgef] all jobs finished"
