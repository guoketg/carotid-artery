# 2026-11-13 队列：边缘监督分支（共享编码器 + 双解码器），单 seed 两臂。
#
# 用户要求：把原标签经过边缘算子得到边缘 GT → 编码器共享 → 两个解码器
#   （一个主分割，一个边缘分支）。先在 UNet 与 R50 预训练骨干上分别验证，单 seed。
#
# 做法（每个臂只变"有没有边缘监督分支"这一个因素）：
#   - 边缘 GT：原标签（224² long）二值化 → k×k 形态学梯度 dilate−erode（k=3 →
#     1 px 内环 + 1 px 外环），与 CCFR Stage1 的 _edge_target 同口径；
#   - 编码器完全共享；边缘解码器与主解码器**逐层对等**（UNet：完整第二解码器
#     +22.6M；R50-TransUNet：第二个 DecoderCup + 2 类头 +7.4M），不是轻量头；
#   - 损失：边缘项与主分割**同形式同权重**（同一个 seg_loss = 0.5CE+0.5Dice，
#     --edge_branch_w 1.0），主分割输出/损失/评估口径与基线逐字一致；
#   - 用户先验：边缘分支在 TransUNet 骨干上应有约 +1.0 Dice；若明显没到，
#     优先当作实现/口径错误排查（见 edge_metrics.csv 的 edgeDice 与主掩膜派生对照）。
#
# 注意（容量混杂）：边缘臂比基座多一个完整解码器（UNet +22.6M、R50 +7.4M），
# 若增益只在边缘臂出现，后续需补"同容量但目标退化为常数/随机"的对照臂才能
# 把收益归因到边缘监督本身。本批先按用户指定的结构跑。
#
# 对照（同 seed 1234、同协议，不重跑）：
#   UNet 基线 86.55 / 0.2250 / HD 1.14（..._SPARC_unet_c2d）
#   R50 预训练基线 90.4903 / 0.14473 / medpy HD 0.7062（..._vitL0_..._SPARC_c2d）
# 判定（§0）：|ΔDice| >= 0.5 且 HD 同向，终点 epoch 200，单 seed 不作为"稳定"结论。
#
# 冒烟：venv\Scripts\python.exe scripts\smoke_edge_branch.py
#   （形状、同 seed 基础权重逐位对齐、参数增量、边缘/主头梯度链路、边缘算子几何、
#     边缘指标边界行为与拼图落盘）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_branch.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_branch.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.Power]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    # 任何其它 queue_*.ps1（排除本脚本自身），保证接力、不抢卡
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            $_.CommandLine -like "*queue_*.ps1*" -and
            $_.CommandLine -notlike "*queue_c2d_edge_branch*"
        }
}

$statusPath = Join-Path $logDir "queue_c2d_edge_branch_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "EB-U"; tag = "unet_edge_branch"
       a = @("--arch", "unet", "--edge_branch", "1", "--edge_branch_w", "1.0",
             "--edge_kernel", "3")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_unet_c2d_eb" },
    @{ n = "EB-R50"; tag = "r50_edge_branch"
       a = @("--arch", "baseline", "--vit_layers", "0", "--edge_branch", "1",
             "--edge_branch_w", "1.0", "--edge_kernel", "3")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d_eb" }
)

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.snap
    $argsList = @("train.py") + $common + $job.a
    Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[$($job.n)] snapshot: $snapshot"
    if ($DryRun) { continue }

    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[$($job.n)] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
    Write-Host "[$($job.n)] log: $jobLog"

    $start = Get-Date
    Push-Location $Root
    & $Python @argsList *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.n)] end exit=$code ${minutes}min"
}
Write-Host "[edge_branch] queue finished"
