# 2026-11-20 队列：UNet 复现口径对齐（同组研究生的实验配置）
#
# !!! 2026-11-21 作废，不得再挂载 !!!
# 本脚本对他配置的解读（wd 0 / 恒定 lr / clip 10 / bce_dice 0.5+0.5）是拿到 A0 包之前
# 的猜测；按 logs/repro_a0/*/run_config.json，他实际是 wd 0.01 / cosine→1e-6 / clip 1.0
# / BCE+Dice 各 1.0 / base_channel 32 / bs 16。现行对齐锚改用 --recipe a0：
# bs 8 + grad_accum 1 + --arch unet --unet_style plain --unet_base 64，
# 见 docs/experiments.md §0.1.1 与 scripts/queue_a0_wrmm.ps1 / queue_a0_matrix.ps1。

# 目的（老师要求「UNet 的结果一致」）：把训练侧全部换成他那套，看能否落在他的表内数字
#   SPARC  UNet  macro_dice 0.8846±0.0034  IoU 0.7968±0.0050  hd95 0.5829±0.0173
#                  ASSD 0.1818±0.0057  recall 0.9028±0.0100  precision 0.8768±0.0044
#   Zhongnan UNet macro_dice 0.9042±0.0014  IoU 0.8279±0.0021  hd95 0.4498±0.0052
#                  ASSD 0.1520±0.0017  recall 0.9146±0.0053  precision 0.8998±0.0058
#
# 与本仓库既有口径的差异（逐条对应他那张表）：
#   优化器   SGD 0.01 + momentum0.9 + wd1e-4 + poly0.9   ->  AdamW 1e-4 + wd0 + 恒定 lr
#   epoch    200                                          ->  100
#   batch    bs4 + grad_accum2（BN 看 4）                  ->  bs8 + ga1（真 batch 8）
#   裁剪     无                                            ->  max_norm 10
#   损失     ce_dice 0.5/0.5（softmax）                     ->  bce_dice 0.5/0.5（BCEWithLogits+sigmoid Dice）
#   预处理   z-score 后 scipy zoom(order=3)                 ->  分位 min-max 后 cv2.INTER_CUBIC（mask 最近邻）
#   增强     翻转 + ±15° 旋转                              ->  +±30°/缩放/平移/亮度对比度/gamma/CLAHE/模糊锐化/噪声/偏置场
#   seed     1234 单跑                                      ->  42 / 317 / 1029 三 seed
#
# 未对齐（本仓库没有验证集，split 只有 train/test，见 dataset_carotid.build_splits）：
#   他按「验证集 Dice 最高」选模，我们仍取末轮（epoch 99）读数，曲线每 5 epoch 落 test_curve.csv。
#   一旦他把 val 的病例清单给我们，再补 best-val 选模；否则用 test 选模就是拿测试集挑权重，不能报。
#
# 判定（跑前登记）：与他的数字比，|Δmacro_dice| <= 0.5 且 hd95 差距 <= 0.1 mm 才算「一致」。
#   macro_dice 的口径要先问清（是否含背景类平均）；我们这边 record 的是前景 Dice，
#   若他是 (Dice_fg + Dice_bg)/2，同一个模型会高 6~7 个点，直接比会误判。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_align_unet.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_align_unet.ps1 -Only 1,2   # 先跑两臂试点
#   powershell -ExecutionPolicy Bypass -File scripts\queue_align_unet.ps1             # 六臂全跑
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 30

$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "queue_align_unet_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# 对齐口径的公共部分（逐条对应他表里的设置）
$common = @(
    "train.py", "--dataset", "Carotid",
    "--batch_size", "8", "--grad_accum", "1",
    "--max_epochs", "100", "--eval_test_every", "5",
    "--arch", "unet",
    "--optim", "adamw", "--base_lr", "1e-4", "--wd", "0", "--clip_grad", "10",
    "--loss", "bce_dice", "--align_proto", "1"
)

$jobs = @(
    @{ n = "1"; name = "ALIGN-SPARC-s42";   args = $common + @("--subset", "SPARC",    "--seed", "42") },
    @{ n = "2"; name = "ALIGN-ZN-s42";      args = $common + @("--subset", "Zhongnan", "--seed", "42") },
    @{ n = "3"; name = "ALIGN-SPARC-s317";  args = $common + @("--subset", "SPARC",    "--seed", "317") },
    @{ n = "4"; name = "ALIGN-SPARC-s1029"; args = $common + @("--subset", "SPARC",    "--seed", "1029") },
    @{ n = "5"; name = "ALIGN-ZN-s317";     args = $common + @("--subset", "Zhongnan", "--seed", "317") },
    @{ n = "6"; name = "ALIGN-ZN-s1029";    args = $common + @("--subset", "Zhongnan", "--seed", "1029") }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.name })
    Write-Host "[align] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.name }) -join ', ')"
}

foreach ($job in $jobs) {
    $lock = $null
    while (-not $DryRun) {
        $lock = Enter-LaunchLock -TimeoutSec 20
        if ((Get-TrainingProcess) -or (Get-OtherQueue)) {
            Exit-LaunchLock $lock; $lock = $null
            Write-Host "[$($job.name)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
            Start-Sleep -Seconds 60
            continue
        }
        break
    }
    Write-Host "[$($job.name)] $Python $($job.args -join ' ')"
    if ($DryRun) { continue }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $safe = ($job.name -replace '-', '_').ToLower()
    $jobLog = Join-Path $logDir "${safe}_$ts.log"
    Write-Host "[$($job.name)] log: $jobLog"

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    Exit-LaunchLock $lock
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.name),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.name)] end exit=$code ${minutes}min"
}
Write-Host "[align] queue finished"
