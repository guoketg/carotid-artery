# 2026-11-14 队列：组合领域损失的「跨骨干 / 跨模块」迁移复验（2 job）。
#
# 组合损失 D2（UNet 弱骨干上唯一过线的那条）：
#   seg_loss = 0.5*ce_bw(lam=3, radius=2) + 0.5*Dice + 0.05*area + 0.1*far_fp(margin=10)
#   UNet 86.55 / 0.225 / 1.14  ->  87.01 / 0.202 / 0.99（Dice +0.46 未过 0.5，HD −0.15 过线）
#   单项拆开跑全是噪声：LFA −0.28、LBC +0.16、AB-noFar −0.05、DL-far03 +0.36；
#   剂量 far_w 0/0.1/0.3 = 86.50/87.01/86.91、HD 1.05/0.99/1.01，0.1 在峰上。
#
# 本队列两臂，损失口径与 D2 逐字一致（--domain_loss 1，L3.0/R2/A0.05/F0.1/M10）：
#   LR50 = R50-TransUNet（--arch baseline --vit_layers 0）+ 组合损失
#          对照：R50 vitL0 基线 90.4903 / 0.14473 / HD 0.7062（results.md §5）
#          参考：同一损失在 12 层 A0 上 D1 = 90.73（对 3-seed 90.32 是 +0.41，对同 seed 是 +0.24）
#   UFG  = U-Net + FGREnc（--arch unet --fgr 1 --fgr_pos enc）+ 组合损失
#          对照：UNet+FGREnc 88.62（results.md §7(4)），UNet 基线 86.55 / 0.225 / 1.14
#          注意容量混杂：FGREnc 比 UNet 多 +2,788,160 参数，所以这一臂只能回答
#          「FGR 改写 + 组合监督是否叠加」，不能单独归因给损失。
# 判定：|ΔDice| >= 0.5 且 HD 同向；SPARC seed 1234 bs4 ga2 ep200，单变量。
# 快照：..._SPARC_dl_c2d / ..._SPARC_fgre_dl_unet_c2d
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_loss_transfer.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_loss_transfer.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerXfer -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerXfer]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            $_.CommandLine -like "*queue_*.ps1*" -and
            $_.CommandLine -notlike "*queue_c2d_loss_transfer*"
        }
}

$statusPath = Join-Path $logDir "queue_c2d_loss_transfer_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "LR50"; tag = "r50_dl"
       a = @("--arch", "baseline", "--vit_layers", "0", "--domain_loss", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_dl_c2d" },
    @{ n = "UFG"; tag = "unet_fgre_dl"
       a = @("--arch", "unet", "--fgr", "1", "--fgr_pos", "enc", "--domain_loss", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_fgre_dl_unet_c2d" }
)

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
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
Write-Host "[loss/transfer] queue finished"
