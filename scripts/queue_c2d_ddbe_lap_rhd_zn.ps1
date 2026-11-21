# 2026-11-19 队列：DDBE+拉普拉斯+RHD 两臂的 Zhongnan 训练版（单 seed 1234）
#
# 用户 11-19 指派：「两个数据集 Zhongnan 以及 SPARC 都要做」。SPARC 侧两臂已由
# scripts/queue_c2d_ddbe_lap_hosts.ps1 挂载（p=18），本脚本只补 Zhongnan 侧的同配置两臂。
#
# 配置逐字继承 SPARC 侧（只换 --subset），保证同宿主同协议可比：
#   臂 1 U-Net 侧：--arch unet --ddbe 4 --ddbe_hf_op lap + RHD sym w0.5 iters 6
#   臂 2 ResNet 侧：--arch baseline --vit_layers 0 --bdfr_v2 1 --bdfr_wave_op lap
#                  + RHD sym w0.5 iters 2
# iters 按宿主分（U-Net 6 / ResNet 2），与 SPARC 侧一致。
#
# Zhongnan 域内参照（同划分 n=187）：
#   U-Net ZN 基线 90.0347 / 0.15421 / HD 0.71573
#   R50   ZN 基线 92.0041 / 0.12266 / HD 0.56511
# Zhongnan 上这两个宿主都没有「DDBE+lap」结构单挂，故本批只报绝对读数 + 方向，
# 主判对各自 ZN 基线（ΔDice >= +0.5 且 ΔHD <= -0.1 mm），结构可加性不在 ZN 侧判。
#
# 快照名（train.py 自动拼）
#   臂 1 ..._224_Zhongnan_ddbe4_ddhflap_lrhd_TsymW0.5I6_unet_c2d
#   臂 2 ..._skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d
#
# 用法：先 -DryRun 核对；正式跑用 -Only 逐臂挂
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerDdbeZn -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerDdbeZn]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 15

$statusPath = Join-Path $logDir "queue_c2d_ddbe_lap_rhd_zn_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "Zhongnan",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "UNet-DDBElap-RHD-I6-ZN"; tag = "unet_ddbe4_lap_rhd_i6_zn"
       a = @("--arch", "unet", "--ddbe", "4", "--ddbe_hf_op", "lap",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_ddbe4_ddhflap_lrhd_TsymW0.5I6_unet_c2d" },
    @{ n = "R50-DDBElap-RHD-I2-ZN"; tag = "r50_bdfr2lap_rhd_i2_zn"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--bdfr_v2", "1", "--bdfr_wave_op", "lap",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "2")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[ddbezn] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
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
Write-Host "[ddbezn] queue finished"
