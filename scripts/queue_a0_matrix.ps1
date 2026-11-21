# A0 配方 + 研究生划分（lists_official）+ best-val 的 2x2 消融重跑队列（2026-11-21）。
#
# 口径：--recipe a0（AdamW 1e-4 / wd 0.01 / cosine->1e-6 / clip 1.0 / BCE+Dice 各 1.0
#       / 100 ep / a0_prep / select_val 1）+ --list_dir ./lists/lists_official
#       + seed 42 单 seed；bs 8 + grad_accum 1（真批大小、不累积）。
#       2026-11-21 14:12 定案：bs8 + ga1 是唯一协议（真批 8、一次 step、不累积）。
#       此前 12:10 的 bs4+ga4（等效 16）与更早的 bs8+ga2 一律作废，读数不得入表。
#       dec_WRMM 臂 bs8 实测 8886 MiB、base 臂 3216~5830 MiB，均未 OOM。
#       基线不重复跑：由 scripts\queue_a0_wrmm.ps1 的 arm 1/4/7/10 提供
#       （UNET-SPARC-base / UNET-ZN-base / R50DC-SPARC-base / R50DC-ZN-base）。
#
# 每个格子（宿主 x 数据集）四条臂，单变量：
#   DDBE+lap      U-Net: --ddbe 4 --ddbe_hf_op lap
#                 R50  : --bdfr_v2 1 --bdfr_wave_op lap（DecoderCup 解码器四级，v2 接法）
#   RHD           --loss rhd --rhd_base bce_dice_sum --rhd_term sym --rhd_w 0.5
#                 （U-Net 用 --rhd_iters 6，R50 用 2，沿用历史分档）
#   DDBE+lap+RHD  上面两者叠加
#   FGR(解码器)   --fgr 1（四级输出，fgr_pos=dec 默认）
#
# RHD 的基础项走 --rhd_base bce_dice_sum，与 A0 基线只差 rhd_w*L_rhd 一个变量；
# 冒烟 Temp\smoke_bds_rhd.py 已验（bce_dice_sum_parts 与 A0 式逐位一致、w=0 恒等）。
#
# 判定（跑前登记，单 seed 只作筛选）：ΔDice >= +0.5 且 ΔHD <= -0.1 mm；不过线即该臂无效。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_a0_matrix.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_a0_matrix.ps1 -Only 3
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 29

$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "queue_punet_bs8_matrix_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @("train.py", "--dataset", "Carotid",
            "--list_dir", "./lists/lists_official",
            "--recipe", "a0",
            "--seed", "42", "--batch_size", "8", "--grad_accum", "1",
            "--max_epochs", "100", "--eval_test_every", "0")
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$unet = @("--arch", "unet", "--unet_style", "plain", "--unet_base", "64")
$r50  = @("--arch", "baseline", "--vit_layers", "0")

$ddbeU = @("--ddbe", "4", "--ddbe_hf_op", "lap")
$ddbeR = @("--bdfr_v2", "1", "--bdfr_wave_op", "lap")

function RhdArgs([int]$iters) {
    @("--loss", "rhd", "--rhd_base", "bce_dice_sum",
      "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "$iters")
}
$fgr = @("--fgr", "1")

$jobs = @(
    @{ n = "1";  name = "A0-PUNET-SPARC-ddbeLap";     args = $common + $unet + @("--subset", "SPARC") + $ddbeU },
    @{ n = "2";  name = "A0-PUNET-SPARC-rhdI6";       args = $common + $unet + @("--subset", "SPARC") + (RhdArgs 6) },
    @{ n = "3";  name = "A0-PUNET-SPARC-ddbeLap-rhdI6"; args = $common + $unet + @("--subset", "SPARC") + $ddbeU + (RhdArgs 6) },
    @{ n = "4";  name = "A0-PUNET-SPARC-fgr";         args = $common + $unet + @("--subset", "SPARC") + $fgr },

    @{ n = "5";  name = "A0-R50DC-SPARC-ddbeLap";    args = $common + $r50 + @("--subset", "SPARC") + $ddbeR },
    @{ n = "6";  name = "A0-R50DC-SPARC-rhdI2";      args = $common + $r50 + @("--subset", "SPARC") + (RhdArgs 2) },
    @{ n = "7";  name = "A0-R50DC-SPARC-ddbeLap-rhdI2"; args = $common + $r50 + @("--subset", "SPARC") + $ddbeR + (RhdArgs 2) },
    @{ n = "8";  name = "A0-R50DC-SPARC-fgr";        args = $common + $r50 + @("--subset", "SPARC") + $fgr },

    @{ n = "9";  name = "A0-PUNET-ZN-ddbeLap";        args = $common + $unet + @("--subset", "Zhongnan") + $ddbeU },
    @{ n = "10"; name = "A0-PUNET-ZN-rhdI6";          args = $common + $unet + @("--subset", "Zhongnan") + (RhdArgs 6) },
    @{ n = "11"; name = "A0-PUNET-ZN-ddbeLap-rhdI6";  args = $common + $unet + @("--subset", "Zhongnan") + $ddbeU + (RhdArgs 6) },
    @{ n = "12"; name = "A0-PUNET-ZN-fgr";            args = $common + $unet + @("--subset", "Zhongnan") + $fgr },

    @{ n = "13"; name = "A0-R50DC-ZN-ddbeLap";       args = $common + $r50 + @("--subset", "Zhongnan") + $ddbeR },
    @{ n = "14"; name = "A0-R50DC-ZN-rhdI2";         args = $common + $r50 + @("--subset", "Zhongnan") + (RhdArgs 2) },
    @{ n = "15"; name = "A0-R50DC-ZN-ddbeLap-rhdI2"; args = $common + $r50 + @("--subset", "Zhongnan") + $ddbeR + (RhdArgs 2) },
    @{ n = "16"; name = "A0-R50DC-ZN-fgr";           args = $common + $r50 + @("--subset", "Zhongnan") + $fgr }
)

foreach ($job in $jobs) {
    if ($Only -and ($Only -split '[,;\s]+') -notcontains $job.n) { continue }
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.name)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
        Start-Sleep -Seconds 60
    }
    Write-Host "[$($job.name)] $Python $($job.args -join ' ')"
    if ($DryRun) { continue }

    $lock = Enter-LaunchLock
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $safe = ($job.name -replace '-', '_').toLower()
    $jobLog = Join-Path $logDir "${safe}_$ts.log"
    $start = Get-Date
    Push-Location $Root
    & $Python @($job.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    Exit-LaunchLock -LockPath $lock
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.name),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.name)] end exit=$code ${minutes}min"
}
Write-Host "[a0_matrix] queue finished"

