# 2026-11-10 队列：FGR+DDBE 组合臂（R50 强骨干）+ UNet 弱骨干 FGR / FGR+DDBE 两臂。
#
# 背景（用户决策）：§5(16) FGR +0.305 三项同向、§5(3) DDBE-Haar +0.435，两者都是
# 「解码器四级输出改写」且归因互补（ddbeCTL 坐实 DDBE 增益来自频率内容，FGR 是
# 纯频率内容+轻门控的极简实现）。本轮验证：
#   R50combo --ddbe 4 --fgr 1 : R50 解码器四级同点串联 DDBE→FGR（两个改写模块叠用
#                               是否互补/冗余；对照 A0 90.4903、DDBE 90.9251、FGR 90.7955）
#   UNetFGR --arch unet --fgr 1 : 弱骨干上仅 FGR（对照 UNet 86.55）
#   UNetCombo --arch unet --ddbe 4 --fgr 1 : 弱骨干上 FGR+DDBE（对照 UNet 86.55）
# 弱骨干臂补齐 §7 的骨干强弱矩阵：DDBE 系在弱骨干上是否仍有正信号。
#
# 冒烟（scripts/smoke_fgr_ddbe_combo.py）：R50 ddbe4+fgr +9.95M、UNet fgr +2.79M、
# UNet ddbe4+fgr +40.14M，全部 IDENTITY（同 seed 基础权重逐位一致，恒等起步成立）。
# UNet 侧 DDBE 挂载移到 init_kaiming 之后（pitfalls 3 RNG 对齐纪律）。
#
# 判定（§0 统一协议，HD 用 medpy 口径）：|ΔDice| ≥ 0.5 或 HD 降 ≥ 0.1 mm，终点 ep200。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_ddbe_combo.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_ddbe_combo.ps1
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

$statusPath = Join-Path $logDir "queue_fgr_ddbe_combo_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "R50combo"; tag = "r50ddbe4fgr"
       a = @("--arch", "baseline", "--vit_layers", "0", "--ddbe", "4", "--fgr", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_ddbe4_fgr_c2d" },
    @{ n = "UNetFGR"; tag = "unetfgr"
       a = @("--arch", "unet", "--fgr", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_fgr_unet_c2d" },
    @{ n = "UNetCombo"; tag = "unetddbe4fgr"
       a = @("--arch", "unet", "--ddbe", "4", "--fgr", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_ddbe4_fgr_unet_c2d" }
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
Write-Host "[fgr/ddbe combo] queue finished"
