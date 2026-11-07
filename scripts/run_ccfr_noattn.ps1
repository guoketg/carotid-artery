# 2026-11-05 CCFR Stage 1 换成 U-Net 底座（去掉每级空间注意力）的对照实验
#
# 背景：CCFR 的 Stage 1 是 EA-Net = U-Net + 每级空间注意力 + 多尺度边缘监督。
# 今天已用严格口径跑出 EA-Net 版两阶段：SPARC 81.55 / Zhongnan 83.25（tag=eanet）。
# 本脚本只关空间注意力（--ccfr_noattn 1），边缘支、损失、超参、Stage 2 flow 全部不动，
# 用来单变量回答两件事：
#   1) Stage 1 上 U-Net 底座 vs EA-Net 底座，谁高；
#   2) flow 精修在 U-Net 底座上还有没有增益（对照 SPARC 上 EA-Net 的 90.85→90.83）。
#
# 口径完全沿用 eanet 那两条：stage1_epochs 100 + val 10% 选 best，stage2 20 ep，
# seed 1234，bs4 + ga2，ccfr_base 56，ccfr_pretrain 0（从零）。
# 参数量：ea 24.09M（有注意力 24.36M），总 33.55M（33.82M），差异只来自注意力本身。
#
# 文件名故意不带 queue_ 前缀：weaveseg 队列会把任何 queue_*.ps1 当"其他队列"等待，
# 若本脚本叫 queue_*.ps1，两边会互相等待形成死锁。本脚本反向等待所有 queue_*.ps1，
# 因此实际执行顺序是 wavelet(5 项) → weaveseg(4 项) → 本脚本(2 项)。
#
# 快照名：${pre}_${subset}_ccfr_noattn_unet_s1
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_ccfr_noattn.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\run_ccfr_noattn.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilCcfrNoattn" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# 只认 -File 启动的队列脚本（排除 Trae/终端的 -Command 包装进程，否则会永远等到包装进程退出）
function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_*.ps1*" -and
                       $_.CommandLine -like "*-File*" -and
                       $_.CommandLine -notlike "*run_ccfr_noattn.ps1*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$statusPath = Join-Path $logDir "queue_ccfr_noattn_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$jobs = @(
    @{ Name = "SPARC";    Desc = "SPARC (315 train / 35 val)" },
    @{ Name = "Zhongnan"; Desc = "Zhongnan (404 train / 44 val)" }
)

function New-JobArgs($job) {
    @("train.py", "--dataset", "Carotid", "--subset", $job.Name,
      "--list_dir", "./lists/lists_Carotid2D", "--num_workers", "4",
      "--max_epochs", "200", "--batch_size", "4", "--grad_accum", "2",
      "--seed", "1234", "--arch", "ccfr", "--ccfr_stage", "both",
      "--ccfr_pretrain", "0", "--ccfr_base", "56", "--ccfr_noattn", "1",
      "--stage1_epochs", "100", "--stage1_pick", "best",
      "--tag", "unet_s1")
}

if ($DryRun) {
    foreach ($job in $jobs) {
        $a = New-JobArgs $job
        Write-Host "[ccfr-noattn] $($job.Name) $($job.Desc): $Python $($a -join ' ')"
        Write-Host "[ccfr-noattn]   snapshot: $(Join-Path $modelDir "${pre}_$($job.Name)_ccfr_noattn_unet_s1")"
    }
    Write-Host "[ccfr-noattn] dry-run only, other queue procs: $((Get-OtherQueue | Measure-Object).Count), train.py procs: $((Get-TrainingProcess | Measure-Object).Count)"
    Get-OtherQueue | ForEach-Object { Write-Host "[ccfr-noattn]   waiting on: $($_.ProcessId) $($_.CommandLine.Substring(0,[Math]::Min(90,$_.CommandLine.Length)))" }
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    exit 0
}

# 严格串行：先等别的队列脚本退出，再等任何 train.py 退出，绝不抢卡
while (Get-OtherQueue) {
    Write-Host "[ccfr-noattn] waiting for another queue script to finish (60s)..."
    Start-Sleep -Seconds 60
}
while (Get-TrainingProcess) {
    Write-Host "[ccfr-noattn] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

Push-Location $Root
foreach ($job in $jobs) {
    $snapshot = Join-Path $modelDir "${pre}_$($job.Name)_ccfr_noattn_unet_s1"

    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[ccfr-noattn] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "ccfr_noattn_$($job.Name)_$ts.log"
    Write-Host "[ccfr-noattn] ===== start $($job.Name) ($($job.Desc)) ====="
    Write-Host "[ccfr-noattn] log: $jobLog"

    $start = Get-Date
    & $Python (New-JobArgs $job) *> $jobLog
    $code = $LASTEXITCODE
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[ccfr-noattn] ===== done $($job.Name) exit=$code ($mins min) ====="
}
Pop-Location

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[ccfr-noattn] queue finished"
