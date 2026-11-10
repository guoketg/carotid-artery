# 两个补充实验（串行，单 seed）：
#
#   S1  R50 与 WRMM 串行（enc=r50seq）
#       与 r50wrmm 的差别只在接线：r50wrmm 是 R50 四级并联、WRMM 只精炼本级 skip，
#       WRMM 输出不进 R50；r50seq 是 WRMM 插进 R50 主链，WRMM1 的 112² 输出既作 skip
#       又喂 block1，WRMM2/3/4 输出同理既作 skip 又喂下一级 block，跳连也来自 WRMM。
#       通道对齐：WRMM1 做成 64→64（block1 入口是 64），112² 的 skip 再 1×1 升到 128。
#       对照：r50wrmm 89.7390（同级并联版）与 R50U（R50 直连基线）。
#
#   S2  GobletNet λ=0
#       去掉 HF 图的低频回灌（H = HL+LH+HH+λ·LL 中的 λ），看 84.84%（λ=0.3）
#       与 U-Net 86.33% 的差距里有多少来自这个 λ。
#
# 前置：等 extend（7 job）与 relconst（1 job）结束，不抢卡。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50seq_goblet0.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50seq_goblet0.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$relconCsv = Join-Path $logDir "queue_c2d_rgef_relconst_status.csv"
$statusPath = Join-Path $logDir "queue_c2d_r50seq_goblet0_status.csv"

if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "S1"; tag = "r50seq"
       a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_enc", "r50seq")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_wmren_abunetdec_encr50seq_c2d" },
    @{ n = "S2"; tag = "goblet_lam0"
       a = @("--arch", "gobletnet", "--hf_lambda", "0")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_gobletnet_lam0_c2d" }
)

function Test-Rows {
    param([string]$Path, [int]$Min)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $rows = @(Get-Content -LiteralPath $Path | Select-Object -Skip 1 | Where-Object { $_ -ne "" })
    return ($rows.Count -ge $Min)
}

foreach ($job in $jobs) {
    while (-not $DryRun) {
        $busy = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
            Where-Object { $_.CommandLine -like "*train.py*" }
        if (-not $busy -and (Test-Rows -Path $relconCsv -Min 1)) { break }
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
Write-Host "[r50seq_goblet0] queue finished"
