# 损失设计 v2 第二段：强骨干 ResNet-50（ImageNet21k 预训练，vit_layers=0）+ DecoderCup 上验证。
#
# 与第一段（queue_c2d_lossv2_unet.ps1）逐字同参，只换宿主，回答两个问题：
#   ① 三项在强骨干上是否还成立（A0 90.4903 / 0.14473 / HD 0.7062 同 seed 配对）；
#   ② 三个全叠在强骨干上是否与弱骨干同向 —— 历史上 cvartail、像素重加权在 R50 上全部归零
#      （docs/results.md §9.3(d)），所以这一段是"迁移验证"而不是"再筛一遍"。
#
# 四臂（全部单 seed 1234）：
#   R50-sdf        几何场回归（窄带 L1 + eikonal，--sdf aux）
#   R50-cons       斑点扰动一致性（1−Dice(p, p')，第二视角各自照旧算 0.5CE+0.5Dice）
#   R50-ds         多尺度深监督（112²/56² 两级 1×1 头，权重 0.3/0.15，推理零开销）
#   R50-all        三项全叠
# 单臂保留的理由：若全叠归零，没有单臂就无法区分"某一项在强骨干上就是负的"与
#   "三项互相抵消"；只有全叠一臂会让结论无法解释。
#
# 判定（docs/experiments.md §0，跑前登记）：
#   主判 ΔHD ≤ −0.1 mm；护栏 ΔDice ≥ −0.3；ASSD 需与 HD 同向。
#   单 seed 噪声带 ±0.3，过线后如要 3 seed（42/317/1029）须先经用户同意（CLAUDE.md §15）。
#
# 成本：R50 单臂约 45–50 min，cons 臂约 70 min（+1 前向），四臂合计约 4 h。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_r50.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_r50.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_r50.ps1 -Only R50-all
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_lossv2_r50.ps1 -WithControl
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string[]]$Only = @(),
    [switch]$WithControl,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerLossv2R -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerLossv2R]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：优先级 40（排在 queue_c2d_lossv2_unet 45 之后、3-seed 30 之前）
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 40

$statusPath = Join-Path $logDir "queue_c2d_lossv2_r50_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "baseline", "--vit_layers", "0"
)
$sdf = @("--sdf", "aux", "--sdf_w", "1.0", "--sdf_tau", "12")
$cons = @("--cons_w", "0.2", "--cons_sigma", "0.08", "--cons_gain", "0.1", "--cons_warmup", "20")
$ds = @("--ds", "1")

$baseR50 = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d"

$jobs = @(
    @{ n = "R50-sdf"; tag = "r50_sdfaux"
       a = $common + $sdf
       snap = "${baseR50}_sdfaux" },

    @{ n = "R50-cons"; tag = "r50_cons"
       a = $common + $cons
       snap = "${baseR50}_cons_w0.2_s0.08" },

    @{ n = "R50-ds"; tag = "r50_ds"
       a = $common + $ds
       snap = "${baseR50}_ds" },

    @{ n = "R50-all"; tag = "r50_sdfaux_cons_ds"
       a = $common + $sdf + $cons + $ds
       snap = "${baseR50}_sdfaux_ds_cons_w0.2_s0.08" }
)

if ($WithControl) {
    $jobs += @{ n = "R50-cons-ctrl"; tag = "r50_cons_ctrl"
                a = $common + $cons + @("--cons_off", "1")
                snap = "${baseR50}_cons_w0.2_s0.08_ctrl" }
}

# -Only <臂名列表>：只跑指定臂（接续/单人复核用；臂名见上面 n 字段）
if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($s in $Only) { $sel += @($s -split '[,;\s]+' | Where-Object { $_ }) }
    $jobs = @($jobs | Where-Object { $sel -contains $_.n })
    if ($jobs.Count -eq 0) {
        Write-Host "[lossv2-r50] -Only 未匹配任何臂：$($sel -join ',')"
        exit 1
    }
    Write-Host "[lossv2-r50] resume mode, jobs: $(($jobs | ForEach-Object { $_.n }) -join ',')"
}

foreach ($job in $jobs) {
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
Write-Host "[lossv2-r50] queue finished"
