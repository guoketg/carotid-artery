# 2026-11-14 队列：U-Net 弱骨干上的三条「降距离指标」损失臂（各 1 臂单变量）。
#
# 目的：把「别人怎么降 ASSD/HD」的三条代表写法在同一个弱骨干、同一协议下横评，
#       只回答机制问题，不外推到 R50（弱骨干结论不外推，见 docs/pitfalls.md）。
#
#   A  区域 HD 平均项（Guzzi et al., MLMI 2025 主用式 L_AH_sym）
#      --loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6
#      两侧距离剖面按各自质量归一后取均值：miss 侧与 extra 侧都被压。
#
#   B  区域 HD 最大项（同一篇的 L_H，机制上等同 Karimi & Salcudean TMI 2020 的
#      「直接优化 Hausdorff」一族：用 soft-max 压最坏一点，而不是压平均）
#      --loss rhd --rhd_term h --rhd_w 0.5 --rhd_iters 6
#      注意：官方 logsum 式含常数 alpha*log(N)（224² 约 1624），已减掉该常数
#      得真正的 soft-max（上界 = iters = 6），否则无法与 0.5CE+0.5Dice 等权相加。
#
#   C  边界图加权项（Kervadec et al., MIDL 2019 Boundary Loss）
#      --loss boundary --bl_w 5.0
#      用 GT 有符号距离图给预测概率加权，把监督落在「边界位置」而不是区域面积。
#      论文用 alpha 由 0 递增的调度；本项目无调度，默认 --bl_w 5.0 是按初始量级
#      标定的（离线 L_B≈0.07 vs 0.5CE+0.5Dice≈0.7），固定不扫参。
#      说明：这一臂替代 cbDice（MICCAI 2024）——cbDice 需要可微骨架 + 欧氏距离
#      变换 + 骨架半径加权，本机 venv 无 skimage/cuCIM，且公式细节未核；它真正
#      对距离指标起作用的是「边界 Dice」部分，本臂就是那一族的无中心线版本。
#
# 判定（§0 统一协议，SPARC 全部 160 张切片均值，终点 ep200，单 seed 1234）：
#   主判 HD：ΔHD <= -0.1 mm；护栏 Dice：ΔDice >= -0.3。
#   配对对照 U-Net 基线 86.55 / ASSD 0.225 / HD 1.14（..._SPARC_unet_c2d）。
#   同批已跑过的参照：LFA 86.27/1.09、LBC 86.71/1.11、cvartail 87.37/0.96。
#
# 冒烟：venv\Scripts\python.exe scripts\smoke_distloss_unet.py
#   （U-Net 前向、三条损失对参数的非零有限梯度、h 项量级上界、boundary 项量级、空前景）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_distloss_unet.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_distloss_unet.ps1
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

function Get-OtherQueue {
    $self = $PID
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.ProcessId -ne $self -and
            $_.CommandLine -and
            $_.CommandLine -like "*queue_*.ps1*"
        }
}

$statusPath = Join-Path $logDir "queue_c2d_distloss_unet_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC", "--arch", "unet",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$base = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC"

$jobs = @(
    @{ n = "RHDSYM"; tag = "unet_rhd_sym"
       a = @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "${base}_lrhd_TsymW0.5I6_unet_c2d" },
    @{ n = "RHDH"; tag = "unet_rhd_h"
       a = @("--loss", "rhd", "--rhd_term", "h", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "${base}_lrhd_ThW0.5I6_unet_c2d" },
    @{ n = "BND"; tag = "unet_boundary"
       a = @("--loss", "boundary", "--bl_w", "5.0")
       snap = "${base}_lboundary_unet_c2d" }
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
Write-Host "[distloss/unet] queue finished"
