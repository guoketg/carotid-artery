# 2026-11-12 队列：DDBE / FGR 的「R50 随机初始化」归因批（拆「ResNet 结构」与「ImageNet 预训练」）。
#
# 动机（用户决策）：§5(1)(3)(16) 的 DDBE +0.435 / FGR +0.305 都是在 ImageNet21k 预训练
# R50 上测的增量，无法回答「增益来自 ResNet 结构本身，还是来自预训练特征」。本轮把
# 骨干换成随机初始化（--no_pretrain 1），只变这一个因素，模块配置与预训练臂逐字一致。
#
# 三臂成对比较（全部 SPARC seed 1234 bs4 ga2 ep200，SGD 0.01，ce_dice，vitL0 = R50-TransUNet）：
#   NP-CTL  --arch baseline --vit_layers 0 --no_pretrain 1
#           snap: TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d（2026-11-07 已有，直接复用）
#           读数：89.59% / ASSD 0.16520 / medpy HD 0.79 / HD95 0.51（同快照 metrics.csv 逐切片均值）
#   NP-DDBE --arch baseline --vit_layers 0 --no_pretrain 1 --ddbe 4      （Haar，ACAM 关，与 §5(3) 同配置）
#           snap: ..._SPARC_ddbe4_c2d
#   NP-FGR  --arch baseline --vit_layers 0 --no_pretrain 1 --fgr 1        （whole 档，与 §5(16) 同配置）
#           snap: ..._SPARC_fgr_c2d
#
# 复用的理由（不重跑 CTL）：该快照与预训练臂同协议同 seed，指标代码在本批（11-12）之前
# 未再改动（`hd`/`hd95` 两列自 11-02 起就是 medpy 口径）；DDBE/FGR 模板一律在基础权重
# 初始化之后才创建（RNG 对齐纪律，见 vit_seg_modeling.py），同 seed 下基础权重与 NP-CTL
# 逐位对齐，因此三臂仍是成对比较，Δ 只来自模块本身。
#
# 预训练侧对照（§5，同 seed 1234 同协议，HD 均为 medpy 口径）：
#   A0   90.4903 / HD 0.7062
#   DDBE 90.9251 / HD 0.6292（ΔDice +0.435 / ΔHD −0.077）
#   FGR  90.7955 / HD 0.69732（ΔDice +0.305 / ΔHD −0.009）
#
# 判定（§0 统一协议）：ΔDice ≥ +0.5 且 HD 同向为主结论；其余为方向证据。
# 归因读法：
#   Δ_nopre ≈ Δ_pre   → 增益来自「ResNet 结构 + 解码器接法」，预训练只是整体抬升
#   Δ_nopre ≪ Δ_pre   → 增益依赖 ImageNet 预训练特征，属「预训练放大器」
#   Δ_nopre ≤ 0       → 模块只对预训练特征有效，不能作为独立贡献
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_nopre_ddbe_fgr.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_nopre_ddbe_fgr.ps1
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

$ctlName = "TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d"
$ctlSnap = Join-Path $modelDir $ctlName
if (-not (Test-Path (Join-Path $ctlSnap "epoch_199.pth"))) {
    Write-Warning "[NP] control snapshot missing/incomplete: $ctlSnap (需要先补跑随机初始化基线)"
}

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$statusPath = Join-Path $logDir "queue_nopre_ddbe_fgr_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "baseline", "--vit_layers", "0", "--no_pretrain", "1"
)

$jobs = @(
    @{ n = "NP-DDBE"; tag = "r50_nopre_ddbe4"
       a = @("--ddbe", "4")
       snap = "TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_ddbe4_c2d" },
    @{ n = "NP-FGR"; tag = "r50_nopre_fgr"
       a = @("--fgr", "1")
       snap = "TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fgr_c2d" }
)

foreach ($job in $jobs) {
    # 等 GPU 空出来再开跑，绝不与其它训练抢卡
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[$($job.n)] another train.py is running, waiting 60s..."
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
Write-Host "[nopre ddbe/fgr] queue finished"
