# 2026-11-19 队列：区域 Hausdorff 距离损失 term 补齐（2 job，单变量，默认 ResNet50 宿主）
#
# 宿主口径（CLAUDE.md 第 18 / 20 条）：--arch baseline --vit_layers 0
#   = ImageNet21k 预训练 ResNet50 四级编码器 + DecoderCup，不含 ViT（vit_layers=0 时 Transformer 段一行不建）
#   配对基线 A0 = 90.4903 / 0.14473 / HD 0.7062（SPARC 官方划分 n=160，seed 1234，bs4+ga2，200 ep）
#   本节不再出现其它 ResNet 宿主写法；读数只与本表基线配对。
#
# 背景：RHD 在该宿主上只跑过 sym 两档（I6 90.64 / +0.15、I2 90.82 / +0.33，均判负，results.md §10.2a），
#   term 的另两档 h（soft-max 压最坏一点）与 ah（两侧取最大）从未在该宿主上训练过——
#   它们只在 U-Net 与另一套解码器的对照臂上有读数，按第 18 条 Δ 不可跨表继承，因此这里补齐。
#
# 两臂（只差 --rhd_term，其余逐字相同：rhd_w 0.5、rhd_iters 6、seed 1234、bs4+ga2、ep200）：
#   RHD-H  = --rhd_term h  （两侧先 alpha-LogSumExp 再取最大，alpha=150；机制上等同「直接优化 HD」一族，
#                            Karimi & Salcudean, IEEE TMI 2020。源码已在项内减掉常数 alpha*log(N)，
#                            减后上界 = iters，可与 0.5CE+0.5Dice 等权相加）
#   RHD-AH = --rhd_term ah （miss/extra 两侧各算距离能量后取最大，不做 LogSumExp 平滑）
#
# 判定（跑前登记，同 §10.2a 口径）：ΔHD <= -0.1 mm（主判）且 ΔDice >= -0.3（护栏），对 A0 单 seed 配对。
#   参考：同宿主 sym I6 为 +0.15 / -0.036、sym I2 为 +0.33 / -0.056。
# 快照（train.py 自动拼 _lrhd_T<term>W<w>I<iters>）：
#   ..._vitL0_epo200_bs4_ga2_224_SPARC_lrhd_ThW0.5I6_c2d
#   ..._vitL0_epo200_bs4_ga2_224_SPARC_lrhd_TahW0.5I6_c2d
#
# 用法（先 -DryRun 核对命令行与快照名）：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_rhd_r50_term.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_rhd_r50_term.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerRhdTerm -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerRhdTerm]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：严格全序优先级（本队列 = 16，见 queue_priority.ps1；唯一优先级来源）
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 16

$statusPath = Join-Path $logDir "queue_c2d_rhd_r50_term_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "RHD-H"; tag = "r50_rhd_h_i6"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--loss", "rhd", "--rhd_term", "h", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lrhd_ThW0.5I6_c2d" },
    @{ n = "RHD-AH"; tag = "r50_rhd_ah_i6"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--loss", "rhd", "--rhd_term", "ah", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lrhd_TahW0.5I6_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[rhdterm] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
}

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡（CLAUDE.md 第 16 条）
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
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
Write-Host "[rhdterm] queue finished"
