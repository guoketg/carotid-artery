# 排队：FGR(编码器位置) + cvartail 损失联合训练（弱骨干 U-Net，SPARC，3 seed）
#
# 用户要求（2026-11-14）：把 FGR 与自研的 CVaR 尾部损失 cvartail 一起训，看结果，3 seed，用 U-Net；
# 位置指定为**编码器**（`--fgr_pos enc`，改写主链后继续下采样并作 skip；U-Net 上单挂 88.62 / HD 0.93，
# 比解码器位置 88.2484 / HD 0.96818 略高，同为单 seed、差 0.37 在噪声带内）。
#
# 为什么这两条能叠：两者正交且都已单变量过线/接近过线——
#   FGR（`--fgr 1 --fgr_pos enc`，编码器四级频带门控改写，改写后继续下采样并作 skip，+2.79M）：
#       U-Net 上 88.62 / 0.176 / HD 0.93（解码器位置同模块为 88.2484 / HD 0.96818，两者差 0.37 在噪声带内）
#   cvartail（`--loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05`）：U-Net 上
#       87.37 / 0.200 / HD 0.96（ΔDice +0.82、ΔHD −0.18，唯一双门槛全过的损失臂）
#   一个是特征侧改写（模块），一个是优化目标侧对距离尾部加权（损失），作用面不重叠，
#   不存在 DDBE+FGR 那种「两个改写机器抢同一份频率内容」的互覆风险。
#
# 臂（本队列 = 联合臂；对照臂不重跑，用历史同协议读数，见 docs/results.md §7(4)(5)、§10.1）：
#   FGREnc+cvar : --arch unet --fgr 1 --fgr_pos enc --loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05
#                 seed 42 / 317 / 1029（experiments.md §0 指定的 3-seed 集合）
# 快照：TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_s<seed>_SPARC_fgre_lcvartail_W0.1Q0.1A0.05_unet_c2d
#
# -WithControl：额外补 3 个「仅 FGREnc、seed 42/317/1029」的成对对照（每臂约 110 min）。
#   不补的话只能把联合臂的 3-seed 均值与 FGREnc 单 seed 88.62 作非成对比较（噪声带 ±0.3），
#   报不出成对 Δ；要写进论文的均值±sd 表就得有成对基线，故留此开关，默认关。
#
# 判定（docs/experiments.md §0）：多 seed 合并成一行（均值 ± sd），
#   主看 ΔDice ≥ +0.5 或 ΔHD ≤ −0.1 mm（相对 UNet 基线 86.5455 / HD 1.13618 与 FGR 单挂）；
#   HD 为 medpy 口径；报数只用整个 SPARC 测试集 160 张切片的均值，不按病例聚合。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fgr_cvartail_3seed.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fgr_cvartail_3seed.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fgr_cvartail_3seed.ps1 -WithControl
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$WithControl,
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

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_*.ps1*" -and
                       $_.CommandLine -notlike "*queue_c2d_fgr_cvartail_3seed.ps1*" }
}

$statusPath = Join-Path $logDir "queue_c2d_fgr_cvartail_3seed_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5", "--arch", "unet"
)
$cvartail = @("--loss", "cvartail", "--tl_w", "0.1", "--tl_q", "0.1", "--tl_area_w", "0.05")

$jobs = @()
foreach ($seed in @("42", "317", "1029")) {
    $jobs += @{ n = "FGREncCvar-s$seed"; tag = "fgre_cvartail_s$seed"
                a = @("--seed", $seed, "--fgr", "1", "--fgr_pos", "enc") + $cvartail
                snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_s${seed}_SPARC_fgre_lcvartail_W0.1Q0.1A0.05_unet_c2d" }
}
if ($WithControl) {
    foreach ($seed in @("42", "317", "1029")) {
        $jobs += @{ n = "FGREncOnly-s$seed"; tag = "fgre_only_s$seed"
                    a = @("--seed", $seed, "--fgr", "1", "--fgr_pos", "enc")
                    snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_s${seed}_SPARC_fgre_unet_c2d" }
    }
}

foreach ($job in $jobs) {
    # 等 GPU 空出来再开跑，绝不与其它训练/队列抢卡（硬约束 16）
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] another train.py / queue is running, waiting 60s..."
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
Write-Host "[fgr + cvartail 3seed] queue finished"
