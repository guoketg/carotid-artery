# 2026-11-18 队列：U-Net 上 cvartail 的 λ=0.2 单臂补齐（用户直接指派）。
#
# 为什么要这一臂：R50 侧已用份额探针反解出「对齐点 λ≈0.2」（同 λ 下 R50 的 CVaR 梯度份额只有 U-Net 的
# 约一半，19.2% vs 34.9%），但那条只验证了 R50。U-Net 侧历史上只跑过 λ=0.1（+0.82 / HD −0.18）。
# 本臂回答：弱骨干上 λ=0.2 是否把「+0.82」再抬一档，还是像 R50 那样在 λ≥0.4 才开始用 Dice 换。
#
# 【2026-11-18 口径更正，勿删】上面「对齐点」的说法只对 R50 成立，搬到 U-Net 不成立：
# R50 的 ρ(A0)=1.92 而 U-Net 的 ρ(A0)=3.49，所以 U-Net 的 λ=0.1 就已经是 R50 的 λ≈0.18 那份份额，
# 本臂的 0.2 是「2× 上沿剂量」，不是「对齐点」。按整段积分剂量（λ·∫e）反解，与 R50 λ0.2 同剂量的
# U-Net λ 是 0.15；按末端均衡份额反解是 0.12。三种口径给出 0.10 / 0.15 / 0.20，本臂取上沿。
# 预注册预测见 docs/results.md §10.3(d)：末端 e→0.070~0.074、ΔHD 只能 −0.02~−0.03（−0.1 门大概率
# 不可达）、ΔDice +0.7~+0.9 与 λ0.1 不可分辨。判词若成立应写「λ 在弱骨干上也无剂量响应」。
#
# 口径与 λ=0.1 臂完全一致（单变量）：0.5CE + 0.5Dice + 0.05·面积一致 + λ·CVaR_10%(距离加权不一致)，
# d0=10 px、cap=3；SPARC 官方划分、seed 1234、bs4+ga2、200 epoch、eval_test_every 5。
# 对照臂（同 seed 1234，medpy HD 口径）：
#   U-Net 基线        86.5455 / 0.22500 / HD 1.13618
#   U-Net cvartail λ0.1 87.3700 / 0.20000 / HD 0.96000（快照 ..._lcvartail_W0.1Q0.1A0.05_unet_c2d）
# 判定：ΔDice >= +0.5 且 ΔHD <= −0.1 mm（对基线）；与 λ0.1 臂比只看四个指标同向。
#
# 用法：
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\queue_c2d_cvar_unet_w02.ps1 -DryRun
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\queue_c2d_cvar_unet_w02.ps1
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

. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 6

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$statusPath = Join-Path $logDir "queue_c2d_cvar_unet_w02_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "unet", "--loss", "cvartail", "--tl_w", "0.2", "--tl_q", "0.1", "--tl_area_w", "0.05"
)

$jobs = @(
    @{ n = "UNet-CVaR-w0.2-SPARC"; tag = "cvarw02_unet"
       a = @()
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_lcvartail_W0.2Q0.1A0.05_unet_c2d" }
)

foreach ($job in $jobs) {
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
Write-Host "[c2d cvar unet w0.2] queue finished"
