# 2026-11-03 第七队列：CCFR 的论文保真版（预训练 R50 编码器）
#
# 为什么单开一队：论文 EA-Net 把灰度复制成 RGB 喂 ImageNet 预训练 ResNet（原文
# "This setting adapts the input requirement of pretrained ResNet backbones"），
# 而第六队列（queue_ccfr.ps1）为了与 §0 口径一致，EA-Net 是从零 1 通道训练。
# 本项目自己的读数已量化这个缺口：预训练 R50+ViT 的 A0 = 90.32，纯卷积从零的
# U-Net = 86.55 / WMREN = 87.14，所以预训练确实值 1~3 点 Dice。
#
# C5 = 只把 Stage 1 的编码器换成 ImageNet21k 预训练 R50（--ccfr_pretrain 1），
# 解码器、边缘支、priors 顺序、Stage 2 flow 头与 C1 完全一致，因此
#   C5 vs C1 可以直接读作「预训练编码器 + 3 通道输入」这一处改动的贡献。
# 注意：C5 与 C1 的 EA-Net 架构不同（R50 编码器 vs 从零 U-Net），
# 两边参数量也不同（C5 22.45 M / C1 33.82 M），别当成只差权重的严格对照。
#
# 口径其余同 §0：bs4 + ga2 / 200 ep / ep200 读数 / SPARC 350 训练、160 测试
# （论文数据 Carotid25 不可得，这是已知偏差）。
#
# 快照名（train.py 自动拼）：
#   C5  ${pre}_SPARC_ccfr_pr50_c2d
#
# 本队会先等 queue_ccfr.ps1 整个队列退出（不只是等当前 train.py），
# 避免两个队列在 job 间隙抢 GPU。
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr_pretrain.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr_pretrain.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilCCFRP" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$queueStatus = Join-Path $logDir "queue_ccfr_status.csv"
$queueTotal = 4

function Get-CCFRQueueDone {
    if (-not (Test-Path $queueStatus)) { return -1 }
    return ((Get-Content $queueStatus | Where-Object { $_.Trim() -ne '' }).Count - 1)
}

function Test-CCFRQueueBusy {
    # 双保险：进程名匹配可能因启动方式失效，队列自己的状态表一定可信
    $proc = Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*scripts\queue_ccfr.ps1*" }
    if ($proc) { return $true }
    $done = Get-CCFRQueueDone
    if ($done -ge 0 -and $done -lt $queueTotal) { return $true }
    return $false
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$common = @("train.py", "--arch", "ccfr", "--subset", "SPARC", "--dataset", "Carotid",
            "--batch_size", "4", "--grad_accum", "2", "--max_epochs", "200",
            "--eval_test_every", "5", "--list_dir", "./lists/lists_Carotid2D",
            "--num_workers", "4", "--img_size", "224", "--ccfr_pretrain", "1")

$jobs = @(
    @{ name = "CCFR_pr50"; args = $common + @("--ccfr_stage", "both", "--seed", "1234",
                                              "--stage1_epochs", "200", "--stage1_pick", "200",
                                              "--stage2_epochs", "20", "--flow_steps", "5");
       snap = "${pre}_SPARC_ccfr_pr50_c2d" }
)

$statusPath = Join-Path $logDir "queue_ccfr_pretrain_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

Write-Host "[ccfr-pr] $($jobs.Count) jobs, serial"
foreach ($j in $jobs) { Write-Host ("  {0,-12} -> {1}" -f $j.name, $j.snap) }
if ($DryRun) {
    Write-Host ("[ccfr-pr] dry-run only; queue busy: {0}, queue done rows: {1}/{2}, train.py running: {3} proc" -f `
        (Test-CCFRQueueBusy), (Get-CCFRQueueDone), $queueTotal, (Get-TrainingProcess | Measure-Object).Count)
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "[ccfr-pr] queue finished"
    exit 0
}

while (Test-CCFRQueueBusy) {
    Write-Host "[ccfr-pr] queue_ccfr.ps1 still running ($(Get-CCFRQueueDone)/$queueTotal done), waiting 120s..."
    Start-Sleep -Seconds 120
}
while (Get-TrainingProcess) {
    Write-Host "[ccfr-pr] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

foreach ($j in $jobs) {
    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[ccfr-pr] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.name)_$ts.log"
    Write-Host "[ccfr-pr] ===== start $($j.name) ====="
    Write-Host "[ccfr-pr] log: $jobLog"
    Write-Host "[ccfr-pr] cmd: $Python $($j.args -join ' ')"

    $start = Get-Date
    Push-Location $Root
    & $Python @($j.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[ccfr-pr] ===== done $($j.name) exit=$code ($mins min) ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[ccfr-pr] queue finished"
