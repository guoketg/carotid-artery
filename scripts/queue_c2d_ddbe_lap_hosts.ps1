# 2026-11-19 队列：DDBE+拉普拉斯（结构）与 RHD(sym,I2)（损失）的融合，两臂（单 seed 1234）
#
# 宿主口径（CLAUDE.md 第 18 / 20 条）：
#   臂 1 U-Net：--arch unet，from scratch，配对基线 86.5455 / 0.22500 / HD 1.13618（SPARC 官方划分 n=160）
#   臂 2 默认 ResNet50：--arch baseline --vit_layers 0（ImageNet21k 预训练 R50 + DecoderCup，不含 ViT），
#         配对基线 A0 90.4903 / 0.14473 / HD 0.7062
#
# 为什么先跑臂 1（用户 2026-11-19 指派「先去验证一下」）：
#   U-Net 上的 DDBE 只跑过 Haar 版（--ddbe 4 → 86.34 / HD 1.089，对基线 −0.21），
#   拉普拉斯版从未在该宿主上训练过；而 ResNet50 上的 +0.506（90.9960 / HD 0.6581，results.md §5(4)）
#   用的是 --bdfr_v2 那套实现（非残差改写），U-Net 分支没有这套实现，只能走 --ddbe 4 --ddbe_hf_op lap
#   （零初始化残差回注、恒等起步）。所以「DDBE+拉普拉斯在 U-Net 上到底正不正」是一个独立未知，
#   必须先有这条读数，融合臂（臂 1 + RHD）才有归因对象。
#   归因纪律：U-Net 的 DDBE 是 v1 实现，与 ResNet50 的 v2 实现不是同一台机器，读数不跨表比较。
#
# 臂 2（U-Net 融合臂，2026-11-19 追加）：在臂 1 之上叠加同一 RHD 项，
#   --loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6
#   为什么用 I6 而不是 I2：该宿主上唯一有配对 Δ 且双门槛过线的 RHD 档就是 sym I6
#   （88.2057 / 0.18146 / HD 0.88286，+1.660 / −0.253），融合臂要与它做可加性判断；
#   用 I2 的话该宿主没有 I2 单挂参照，还得再补一条单挂臂才能归因。
#   主判：对 U-Net 基线 86.5455 / 1.13618；次判（可加性）对 RHD 单挂 88.2057 / 0.88286，
#   要求 >= +0.3 Dice 或 >= +0.05 HD，过不了记「不可加」。
#
# 臂 3（融合臂，用户在 2026-11-19 明确指定 ResNet 侧 RHD 用 sym I2）：
#   --arch baseline --vit_layers 0 --bdfr_v2 1 --bdfr_wave_op lap
#   --loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 2
#   结构单挂参照 = 同宿主 bdfr2_na_bwlap 90.9960 / 0.13577 / HD 0.6581（§5(4)，同 seed 同协议）
#   损失单挂参照 = 同宿主 rhd sym I2 90.82 / 0.139 / HD 0.65（§10.2a，论文默认 iters=2）
#   融合臂的 RHD 参数不调，逐字继承损失单挂：sym / w0.5 / iters2 / alpha150（alpha 只对 term=h 生效）
#
# 判定（跑前登记）：
#   臂 1 对 U-Net 基线：主判 ΔDice >= +0.5 且 ΔHD <= -0.1 mm。判负则融合臂（RHD 之上挂一个负模块）预期打折。
#   臂 2 主判对 A0：ΔDice >= +0.5 且 ΔHD <= -0.1 mm；
#         次判（可加性）对更强的那个单挂臂（bdfr2_na_bwlap 90.9960）：要求 >= +0.3 Dice 或 >= +0.05 HD，
#         过不了次判记「不可加」，不记「融合有效」。
#   单 seed 按 CLAUDE.md 第 21 条只作筛选，不进主表、不称「稳」。
#
# 快照（train.py 自动拼，顺序为 ddbe/bdfr 段在前、损失段在后）：
#   臂 1 ..._epo200_bs4_ga2_224_SPARC_ddbe4_ddhflap_unet_c2d
#   臂 2 ..._epo200_bs4_ga2_224_SPARC_ddbe4_ddhflap_lrhd_TsymW0.5I6_unet_c2d
#   臂 3 ..._skip3_vitL0_epo200_bs4_ga2_224_SPARC_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d
#
# 用法（先 -DryRun 核对命令行与快照名）：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_ddbe_lap_hosts.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_ddbe_lap_hosts.ps1 -Only UNet-DDBElap
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerDdbeLap -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerDdbeLap]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：严格全序优先级（本队列 = 18，见 queue_priority.ps1；唯一优先级来源）
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 18

$statusPath = Join-Path $logDir "queue_c2d_ddbe_lap_hosts_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "UNet-DDBElap"; tag = "unet_ddbe4_lap"
       a = @("--arch", "unet", "--ddbe", "4", "--ddbe_hf_op", "lap")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_ddbe4_ddhflap_unet_c2d" },
    @{ n = "UNet-DDBElap-RHD"; tag = "unet_ddbe4_lap_rhd_i6"
       a = @("--arch", "unet", "--ddbe", "4", "--ddbe_hf_op", "lap",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_ddbe4_ddhflap_lrhd_TsymW0.5I6_unet_c2d" },
    @{ n = "R50-DDBElap-RHDI2"; tag = "r50_bdfr2lap_rhd_i2"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--bdfr_v2", "1", "--bdfr_wave_op", "lap",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "2")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[ddbelap] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
}

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡（CLAUDE.md 第 16 条）；
    # 2026-11-19 追加一次性启动锁，堵判断与启动之间的竞态（见 queue_priority.ps1 Enter-LaunchLock）。
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
Write-Host "[ddbelap] queue finished"
