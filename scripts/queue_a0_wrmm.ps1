# 2026-11-21 队列：A0 配方 + 新划分（lists_official）+ best-val 选模，跑解码器 WRMM 八臂
#
# 配方（--recipe a0 一条命令点亮，等价于同组研究生 A0）：
#   AdamW lr1e-4 / wd 0.01 / cosine -> 1e-6 / clip 1.0 / BCE+Dice 各 1.0
#   / 100 ep / a0_prep（先 scipy zoom(order=1) resize 224，再 1%~99% 分位 min-max）
#   / --select_val 1（val 逐样本 Dice 最高那轮存 best_model.pth，测试前载回）
# 划分：lists/lists_official/{SPARC,Zhongnan}/{train,val,test}.txt
#   SPARC 280/70/160、Zhongnan 357/88/191（研究生给的原划分，已去重落库）
#
# 宿主与配对（Δ 只对同宿主、同 seed、同协议、同配方）：
#   U-Net       = --arch unet        （60.30M；--dec_wrmm 1 -> 80.16M）
#   R50+DecoderCup = --arch baseline --vit_layers 0（20.22M；--dec_wrmm 1 -> 25.15M）
#
# batch：bs 8 + grad_accum 1（真批大小、不做累积，与复现线的做法一致）。bs 16 在 12 GB 卡上装不下（base 臂 bs8 已 reserved 5830 MiB）。
# 注意：U-Net + dec_WRMM/MLKM 这类多分支解码器在 bs8 下实测 reserved 11434 MiB（93%），有 OOM 风险，臂级失败由队列记 exit code，不静默降批。
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 32

$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "queue_punet_bs8_wrmm_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @("train.py", "--dataset", "Carotid",
            "--list_dir", "./lists/lists_official",
            "--recipe", "a0",
            "--seed", "42", "--batch_size", "8", "--grad_accum", "1",
            "--max_epochs", "100", "--eval_test_every", "0")
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# 2026-11-21 12:40 起 UNet 宿主换成标准 U-Net（PlainUNet，原版 64→1024，28.95 M）；
# 残差版（60.30 M）只留 --unet_style res 复现历史。
$unet = @("--arch", "unet", "--unet_style", "plain", "--unet_base", "64")
$r50  = @("--arch", "baseline", "--vit_layers", "0")

$jobs = @(
    @{ n = "1"; name = "A0-PUNET-SPARC-base";     args = $common + $unet + @("--subset", "SPARC") },
    @{ n = "2"; name = "A0-PUNET-SPARC-wrdmlap";  args = $common + $unet + @("--subset", "SPARC", "--dec_wrmm", "1", "--dec_wrmm_op", "lap") },
    @{ n = "3"; name = "A0-PUNET-SPARC-wrmmlap";  args = $common + $unet + @("--subset", "SPARC", "--dec_wrmm", "1", "--dec_wrmm_op", "lap", "--dec_wrmm_mlkm", "1") },
    @{ n = "4"; name = "A0-PUNET-ZN-base";        args = $common + $unet + @("--subset", "Zhongnan") },
    @{ n = "5"; name = "A0-PUNET-ZN-wrdmlap";     args = $common + $unet + @("--subset", "Zhongnan", "--dec_wrmm", "1", "--dec_wrmm_op", "lap") },
    @{ n = "6"; name = "A0-PUNET-ZN-wrmmlap";     args = $common + $unet + @("--subset", "Zhongnan", "--dec_wrmm", "1", "--dec_wrmm_op", "lap", "--dec_wrmm_mlkm", "1") },
    @{ n = "7"; name = "A0-R50DC-SPARC-base";    args = $common + $r50 + @("--subset", "SPARC") },
    @{ n = "8"; name = "A0-R50DC-SPARC-wrdmlap"; args = $common + $r50 + @("--subset", "SPARC", "--dec_wrmm", "1", "--dec_wrmm_op", "lap") },
    @{ n = "9"; name = "A0-R50DC-SPARC-wrmmlap"; args = $common + $r50 + @("--subset", "SPARC", "--dec_wrmm", "1", "--dec_wrmm_op", "lap", "--dec_wrmm_mlkm", "1") },
    @{ n = "10"; name = "A0-R50DC-ZN-base";      args = $common + $r50 + @("--subset", "Zhongnan") },
    @{ n = "11"; name = "A0-R50DC-ZN-wrdmlap";   args = $common + $r50 + @("--subset", "Zhongnan", "--dec_wrmm", "1", "--dec_wrmm_op", "lap") },
    @{ n = "12"; name = "A0-R50DC-ZN-wrmmlap";   args = $common + $r50 + @("--subset", "Zhongnan", "--dec_wrmm", "1", "--dec_wrmm_op", "lap", "--dec_wrmm_mlkm", "1") }
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
Write-Host "[a0_wrmm] queue finished"

