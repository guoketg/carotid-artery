# 2026-11-13 队列：FGR 挂载位置研究（编码器 / skip / 解码器）+ 编码器融合模块 EFF。
#
# 背景：FGR 历史上固定挂在解码器四级输出（§7(4) 弱骨干 +1.703 / −0.168）。位置
# 从未做过同变量对照；编码器侧此前的注入式模块（MFDF/FEM/ABF）全部判负，但都是
# 注入式而非改写式。本队列在弱骨干 U-Net（基线 86.5455 / 0.22500 / HD 1.13618）
# 上只变「挂在哪」，回答三件事：
#   FGREnc  --arch unet --fgr 1 --fgr_pos enc  : 改写编码器主干四级输出（改写后的
#              x_i 继续下采样，同时作 skip）——频带内容进入编码主链。
#   FGRSkip --arch unet --fgr 1 --fgr_pos skip : 只改写送进解码器的 skip 副本，
#              编码器主干保持原样——区分「改 skip 内容」与「改编码主链」。
#   EFF     --arch unet --eff 1               : 编码器跨尺度频率融合（自顶向下级联：
#              hf(x_i) + lf(深层上下文)，通道轴频率门控改写，fuse 零初始化）。
#
# 对照：解码器位置已有 unet+fgr = 88.2484 / 0.19369 / 0.96818（§7(4)），不重跑。
# 四臂参数增量都是 +2,788,160（EFF 的 proj 与通道翻倍正好抵消），位置对比不是容量对比。
#
# 冒烟（scripts/smoke_fgr_pos_eff.py）：四臂 IDENTITY（同 seed 基础权重逐位一致）、
# 参数量精确相等、EFF 梯度链路（fuse→gate→proj 逐级解开）、旗标互斥全部通过。
#
# 判定（§0 统一协议，HD 用 medpy 口径）：|ΔDice| ≥ 0.5 或 HD 降 ≥ 0.1 mm，终点 ep200。
# 读法：若 enc/skip > dec，说明增益来自「编码器侧频带内容」而不只是解码器改写；
# 若都低于 dec，说明解码器输出点是最优接点（与「深层语义饱和前补细节」一致）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_pos_eff.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_pos_eff.ps1
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

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$statusPath = Join-Path $logDir "queue_fgr_pos_eff_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5", "--arch", "unet"
)

$jobs = @(
    @{ n = "FGREnc"; tag = "fgre"
       a = @("--fgr", "1", "--fgr_pos", "enc")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_fgre_unet_c2d" },
    @{ n = "FGRSkip"; tag = "fgrs"
       a = @("--fgr", "1", "--fgr_pos", "skip")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_fgrs_unet_c2d" },
    @{ n = "EFF"; tag = "eff"
       a = @("--eff", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_eff_unet_c2d" }
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
Write-Host "[fgr pos/eff] queue finished"
