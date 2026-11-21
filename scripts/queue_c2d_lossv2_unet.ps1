# 损失设计 v2 第一段：弱骨干 U-Net 上的三个新角度（单臂 → 逐级叠加），全部单 seed 1234。
#
# 三个项（自然语言 + 旗标，互不同族，都不碰「距离剖面聚合」那一族）：
#   几何场回归   --sdf aux --sdf_w 1.0 --sdf_tau 12
#                GT 带符号距离场的窄带 L1 + eikonal，掩膜仍由分类头给出；
#                与 cvartail/RHD 的区别是「场回归」而不是「误差×距离的聚合」。
#   斑点扰动一致性 --cons_w 0.2 --cons_sigma 0.08 --cons_gain 0.1 --cons_warmup 20
#                同一图像两种斑点实现下前景预测的 1−Dice，两视各自照旧算 0.5CE+0.5Dice；
#                改的是「跨扰动方差」，不接触 GT 距离图。
#   多尺度深监督  --ds 1
#                解码 112²/56² 两级各挂 1×1 头（权重 0.3/0.15），推理只取主输出。
#
# 逐级叠加（用户指定顺序 1 → 2 → 3），每臂都在同宿主同 seed 上与同一基线配对：
#   U-sdf → U-cons → U-ds → U-sdf-cons → U-sdf-cons-ds
# 后两臂回答「可加性」：sdf+cons 是否优于两个单臂；三个全叠是否还有增量。
#
# 参照读数（同宿主 SPARC seed 1234）：UNet 基线 86.5455 / ASSD 0.3138 / HD 1.13618。
# 判定（docs/experiments.md §0，跑前登记，不按结果挑臂重报）：
#   主判 ΔHD ≤ −0.1 mm；护栏 ΔDice ≥ −0.3（不许用 Dice 换 HD）；另记 ASSD 方向需一致。
#   单 seed 噪声带 ±0.3，单 seed 读数不得被表述为"稳定"或"已验证"。
#
# 控制臂（-WithControl，默认关）：--cons_off 1 与 U-cons 逐字同参，但第二视角不扰动、
#   一致性项恒 0，只保留「多一次前向 + BN 统计被更新一次」的效应。
#   若 U-cons 过线，**必须**补这个控制臂才能把增益归给一致性本身；不补则只能报"加了就涨"。
#
# 成本（历史实测 U-Net 单臂约 92 min）：
#   sdf +5%（距离变换每 batch 4 ms 级）| ds +0%（两个 1×1 头）| cons ≈ 1.6×（约 145 min）
#   五臂合计约 9.5 h；带控制臂约 12 h。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_unet.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_unet.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_unet.ps1 -Only U-cons
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_unet.ps1 -WithControl
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string[]]$Only = @(),
    [switch]$WithControl,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerLossv2 -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerLossv2]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：优先级 45（低于 queue_c2d_rhd 50，高于 3-seed 30 与叠加批 5），见 queue_priority.ps1
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 45

$statusPath = Join-Path $logDir "queue_c2d_lossv2_unet_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5", "--arch", "unet"
)
$sdf = @("--sdf", "aux", "--sdf_w", "1.0", "--sdf_tau", "12")
$cons = @("--cons_w", "0.2", "--cons_sigma", "0.08", "--cons_gain", "0.1", "--cons_warmup", "20")
$ds = @("--ds", "1")

$baseUnet = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_unet_c2d"

$jobs = @(
    @{ n = "U-sdf"; tag = "unet_sdfaux"
       a = $common + $sdf
       snap = "${baseUnet}_sdfaux" },

    @{ n = "U-cons"; tag = "unet_cons"
       a = $common + $cons
       snap = "${baseUnet}_cons_w0.2_s0.08" },

    @{ n = "U-ds"; tag = "unet_ds"
       a = $common + $ds
       snap = "${baseUnet}_ds" },

    @{ n = "U-sdf-cons"; tag = "unet_sdfaux_cons"
       a = $common + $sdf + $cons
       snap = "${baseUnet}_sdfaux_cons_w0.2_s0.08" },

    @{ n = "U-sdf-cons-ds"; tag = "unet_sdfaux_cons_ds"
       a = $common + $sdf + $cons + $ds
       snap = "${baseUnet}_sdfaux_ds_cons_w0.2_s0.08" }
)

if ($WithControl) {
    $jobs += @{ n = "U-cons-ctrl"; tag = "unet_cons_ctrl"
                a = $common + $cons + @("--cons_off", "1")
                snap = "${baseUnet}_cons_w0.2_s0.08_ctrl" }
}

# -Only <臂名列表>：只跑指定臂（接续/单人复核用；臂名见上面 n 字段）
if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($s in $Only) { $sel += @($s -split '[,;\s]+' | Where-Object { $_ }) }
    $jobs = @($jobs | Where-Object { $sel -contains $_.n })
    if ($jobs.Count -eq 0) {
        Write-Host "[lossv2-unet] -Only 未匹配任何臂：$($sel -join ',')"
        exit 1
    }
    Write-Host "[lossv2-unet] resume mode, jobs: $(($jobs | ForEach-Object { $_.n }) -join ',')"
}

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.snap
    $argsList = @("train.py") + $job.a
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
Write-Host "[lossv2-unet] queue finished"
