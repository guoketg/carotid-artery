# 2026-11-19 队列：完整 TransUNet（--vit_layers 12）上的 DDBE+拉普拉斯(+RHD)
#
# 用户 11-19 指派「TransUNet 的也补上」= 对 CLAUDE.md 第 18 条封存令的显式授权。
# 因此本脚本每一行命令都写明 --allow_legacy_tu 1（train.py 的代码级闸门只认这个开关），
# check_backbone_policy.ps1 也据此把该行按「已授权豁免」列出而不是判 error（绝不静默豁免）。
#
# 宿主口径（第 18 条：Δ 只对同宿主同 seed 同协议）
#   TU 基线（--arch baseline --vit_layers 12，R50-ViT-B_16 + DecoderCup，2026-10-28 读数）
#   = 90.3154 / 0.14566 / HD 0.64312 / HD95 0.39235（SPARC 160，seed 1234，bs4 ga2，200 ep）
#
# 为什么先补「结构单挂」再补「融合」：TU 上只有 v1 实现的 DDBE+lap 读数
#   （--ddbe 4 --ddbe_hf_op lap，90.3193 / 0.14465 / 0.63974，对基线 +0.004 / −0.003 = 噪声），
#   而 ResNet50 侧那条 +0.506 是 v2 实现（--bdfr_v2，非残差改写）。两者不是同一台机器，
#   所以 TU 上必须先用 v2 形态把结构项单独量一次，融合臂才有归因对象；否则分不清
#   「结构无效」与「结构有效但被 RHD 吃掉」。
#
# RHD 参数：按用户 11-19 的口径「TransUNet 用 ResNet 的配置」= sym / w0.5 / iters 2
#   （论文默认 iters=2，也是 R50 侧落地的那一档；TU 上 RHD 至今零 run）。
#
# 快照名（vit_layers=12 时目录名不带 _vitL 段）
#   臂 1 ..._224_SPARC_bdfr2_na_bwlap_c2d
#   臂 2 ..._224_SPARC_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d
#   臂 3 ..._224_Zhongnan_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d
#
# 判定（跑前登记）
#   臂 1 对 TU 基线 90.3154 / HD 0.64312：ΔDice >= +0.5 且 ΔHD <= -0.1 mm。
#   臂 2 主判对臂 1（可加性）；次判对 TU 基线。臂 1 若判负，臂 2 只记「不可加」。
#   臂 3（Zhongnan 训练）只报绝对读数与方向一致性，不判门槛（ZN 侧无同宿主结构臂参照）。
#
# 用法：先 -DryRun 核对；正式跑可 -Only 逐臂挂
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerTuLap -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerTuLap]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 14

$statusPath = Join-Path $logDir "queue_c2d_tu_lap_rhd_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "TU-DDBElap2-SPARC"; tag = "tu_bdfr2lap_sparc"; sub = "SPARC"
       a = @("--arch","baseline","--vit_layers","12","--bdfr_v2","1","--bdfr_wave_op","lap","--allow_legacy_tu","1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_bdfr2_na_bwlap_c2d" },
    @{ n = "TU-DDBElap2-RHD-I2-SPARC"; tag = "tu_bdfr2lap_rhd_i2_sparc"; sub = "SPARC"
       a = @("--arch","baseline","--vit_layers","12","--bdfr_v2","1","--bdfr_wave_op","lap","--allow_legacy_tu","1","--loss","rhd","--rhd_term","sym","--rhd_w","0.5","--rhd_iters","2")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d" },
    @{ n = "TU-DDBElap2-RHD-I2-ZN"; tag = "tu_bdfr2lap_rhd_i2_zn"; sub = "Zhongnan"
       a = @("--arch","baseline","--vit_layers","12","--bdfr_v2","1","--bdfr_wave_op","lap","--allow_legacy_tu","1","--loss","rhd","--rhd_term","sym","--rhd_w","0.5","--rhd_iters","2")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[tulap] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
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
    $argsList = @("train.py") + $common + @("--subset", $job.sub) + $job.a
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
Write-Host "[tulap] queue finished"
