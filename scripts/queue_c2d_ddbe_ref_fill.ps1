# 2026-11-20 队列：补 DDBE+lap+RHD 的 2×2 矩阵最后两个参照臂（单 seed 1234）
#
# 用户 2026-11-20 指派：「缺少的全部补上，这很关键」。矩阵里融合臂四格已齐，
# 缺的是两条「单挂」参照：
#   臂 1  U-Net × Zhongnan 的 DDBE+lap 结构单挂（融合臂 90.95 目前拆不开结构/损失）
#   臂 2  R50+DecoderCup × Zhongnan 的 RHD sym I2 单挂（融合臂 92.62 的 I2 配对参照缺失）
#
# 配置逐字对齐既有格子，只去掉另一个变量：
#   臂 1  --arch unet --ddbe 4 --ddbe_hf_op lap            （无 RHD；对照 ZN U-Net 基线）
#   臂 2  --arch baseline --vit_layers 0 + RHD sym w0.5 iters 2 （无 DDBE/BDFR；对照 Z0 与 Z1）
#
# Zhongnan 域内配对参照（同划分 n=187，seed 1234，bs4 ga2，200 ep）：
#   U-Net ZN 基线          90.0347 / 0.15421 / HD 0.71573
#   U-Net ZN DDBE+lap+RHD  90.95 / 0.139 / 0.63（本脚本臂 1 是它的结构参照）
#   R50   ZN 基线 Z0       92.0041 / 0.12266 / HD 0.56511
#   R50   ZN DDBE+lap Z1   92.5592 / 0.11408 / HD 0.52187
#   R50   ZN DDBE+lap+RHD  92.62 / 0.113 / 0.49（本脚本臂 2 是它的损失参照）
#
# 判定（跑前登记）：
#   臂 1 对 U-Net ZN 基线，ΔDice >= +0.5 且 ΔHD <= -0.1 mm；并与融合臂做可加性（结构+损失是否叠加）。
#   臂 2 对 Z1 判可加性（ΔDice >= +0.5 且 ΔHD <= -0.1 mm），对 Z0 报绝对增量；同时给出 I2 与 I6 的档位差。
#
# 快照名（train.py 自动拼）
#   臂 1 ..._224_Zhongnan_ddbe4_ddhflap_unet_c2d
#   臂 2 ..._skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lrhd_TsymW0.5I2_c2d
#
# 用法：先 -DryRun 核对；正式跑可 -Only 逐臂挂
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerDdbeRef -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerDdbeRef]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 23

$statusPath = Join-Path $logDir "queue_c2d_ddbe_ref_fill_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "Zhongnan",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "UNet-DDBElap-ZN"; tag = "unet_ddbe4_lap_zn"
       a = @("--arch", "unet", "--ddbe", "4", "--ddbe_hf_op", "lap")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_ddbe4_ddhflap_unet_c2d" },
    @{ n = "R50-RHD-I2-ZN"; tag = "r50_rhd_i2_zn"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "2")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lrhd_TsymW0.5I2_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[ddberef] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
}

foreach ($job in $jobs) {
    $lock = $null
    while (-not $DryRun) {
        $lock = Enter-LaunchLock -TimeoutSec 20
        if ((Get-TrainingProcess) -or (Get-OtherQueue)) {
            Exit-LaunchLock $lock; $lock = $null
            Write-Host "[$($job.n)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
            Start-Sleep -Seconds 60
            continue
        }
        break
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
    Exit-LaunchLock $lock
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.n)] end exit=$code ${minutes}min"
}
Write-Host "[ddberef] queue finished"
