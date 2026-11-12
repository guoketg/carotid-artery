# 2026-11-12 队列：骨干归因的 3-seed 补跑（seed 42 / 317 / 1029，SPARC，12 job 串行）。
#
# 目的：把 docs/experiments.md §3.1 的四条单 seed 骨干归因数字（U-Net / R50 随机初始化 /
# R50 预训练 / TransUNet 预训练）钉成均值±sd，用于回答两个问题：
#   Δ_pretrain    = R50-预训练 − R50-随机初始化     （ImageNet21k 预训练值多少）
#   Δ_R50struct   = R50-随机初始化 − U-Net          （ResNet 结构本身值多少）
# 每个 seed 内四臂成对比较，Δ 按 seed 配对后取均值±sd（CLAUDE.md 第 13 条第 6 款）。
# seed 三件套 42/317/1029 是课题组统一集合（用户指定，便于与课题组其它工作横向对比），
# 已写入 docs/experiments.md §0 与 CLAUDE.md §15；后续 3-seed 补跑沿用同一组。
#
# 四个臂（全部 SPARC、bs4+ga2、ep200、SGD 0.01、ce_dice、终点 ep200，单变量只换骨干）：
#   U-Net         --arch unet                                    （from scratch，60.3M）
#   R50-nopre     --arch baseline --vit_layers 0 --no_pretrain 1 （R50 结构 + 随机初始化，20.2M）
#   R50-pre       --arch baseline --vit_layers 0                 （R50 + ImageNet21k，20.2M）
#   TransUNet-pre --arch baseline                                （R50+ViT-B_16 12 层 + ImageNet21k）
#
# seed 1234 的现成读数（不进本批 3-seed 集合，只作参照；HD 为 medpy 口径）：
#   U-Net         86.55  / 0.2250  / 1.14
#   R50-nopre     89.59  / 0.16520 / 0.79   （HD95 0.51）
#   R50-pre       90.4903 / 0.14473 / 0.7062
#   TransUNet-pre 90.32  / 0.1457  / 0.64
#
# 快照命名（train.py 自动生成，本脚本只做存在性检查/归档，不手写目录名）：
#   U-Net         TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_s{seed}_SPARC_unet_c2d
#   R50-nopre     TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_s{seed}_SPARC_c2d
#   R50-pre       TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_s{seed}_SPARC_c2d
#   TransUNet-pre TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_s{seed}_SPARC_c2d
#
# 已完成的 job 直接跳过（epoch_199.pth 存在即视为完成，对应「有现成的话就直接用」）；
# 目录存在但未完成则归档为 _bak_<时间戳> 后重跑。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_backbone_3seed.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_backbone_3seed.ps1
#   ... -Only UNet_s42,R50-nopre_s42        只跑指定 job
#   ... -IgnoreOtherQueues                  不等待其它 queue_*.ps1（默认等待，避免抢卡）
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [switch]$IgnoreOtherQueues,
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

$selfName = "queue_c2d_backbone_3seed.ps1"

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    if ($IgnoreOtherQueues) { return @() }
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.CommandLine -and
            ($_.CommandLine -like "*\scripts\queue_*.ps1*") -and
            ($_.CommandLine -notlike "*$selfName*")
        }
}

$statusPath = Join-Path $logDir "queue_c2d_backbone_3seed_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$seeds = @(42, 317, 1029)

$arms = @(
    @{ n = "UNet"; tag = "unet"
       a = @("--arch", "unet")
       base = "TU_pretrain_R50-ViT-B_16_skip3"
       suf = "_unet" },
    @{ n = "R50-nopre"; tag = "r50_nopre"
       a = @("--arch", "baseline", "--vit_layers", "0", "--no_pretrain", "1")
       base = "TU_R50-ViT-B_16_skip3_vitL0"
       suf = "" },
    @{ n = "R50-pre"; tag = "r50_pre"
       a = @("--arch", "baseline", "--vit_layers", "0")
       base = "TU_pretrain_R50-ViT-B_16_skip3_vitL0"
       suf = "" },
    @{ n = "TransUNet-pre"; tag = "transunet_pre"
       a = @("--arch", "baseline")
       base = "TU_pretrain_R50-ViT-B_16_skip3"
       suf = "" }
)

$jobs = @()
foreach ($seed in $seeds) {
    foreach ($arm in $arms) {
        $snap = "{0}_epo200_bs4_ga2_224_s{1}_SPARC{2}_c2d" -f $arm.base, $seed, $arm.suf
        $jobs += @{ n = "$($arm.n)_s$seed"; seed = $seed; tag = $arm.tag; a = $arm.a; snap = $snap }
    }
}

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n.ToUpper() }
    if (-not $jobs) { Write-Host "[bb3] -Only '$Only' matched nothing"; exit 1 }
}

foreach ($job in $jobs) {
    $snapshot = Join-Path $modelDir $job.snap
    $done = Test-Path (Join-Path $snapshot "epoch_199.pth")

    if ($done -and -not $DryRun) {
        Write-Host "[bb3][$($job.n)] already complete, reuse: $snapshot"
        continue
    }

    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
    while (-not $DryRun) {
        $train = Get-TrainingProcess
        $other = @(Get-OtherQueue)
        if (-not $train -and $other.Count -eq 0) { break }
        $why = if ($train) { "train.py running" } else { "other queue waiting/running" }
        Write-Host "[bb3][$($job.n)] $why, wait 60s..."
        Start-Sleep -Seconds 60
    }

    $argsList = @("train.py", "--seed", "$($job.seed)") + $common + $job.a
    Write-Host "[bb3][$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[bb3][$($job.n)] snapshot: $snapshot"
    if ($DryRun) { continue }

    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[bb3][$($job.n)] archived incomplete run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "backbone_$($job.tag)_s$($job.seed)_$ts.log"
    Write-Host "[bb3][$($job.n)] log: $jobLog"

    $start = Get-Date
    Push-Location $Root
    & $Python @argsList *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[bb3][$($job.n)] end exit=$code ${minutes}min"
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[bb3] queue finished"
