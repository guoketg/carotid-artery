# MREF（多因子可靠性边缘融合）在 R50+DecoderCup 主宿主上的单 seed 实验。
#
# 结构：每个 skip 汇合点上串联 4 个 MREF 块（DecoderCup 3 个汇合点 → 全模型 12 个块），
#       不是只在一个点上加一个块。单块容量有限（R-GEF 1.02M 实测无增益），级联后模块
#       12.16M，模型 20.22M → 32.38M。
# 每个块内部：多尺度拉普拉斯（σ=1,2）+ Sobel 幅值三路边缘、斑点噪声图（局部 std/均值）、
#       跨支不一致度 u=|x↑−skip|，双头门控（边缘注入 r_e / 校正式改写 r_u），γ1/γ2 初值 0.1。
#
# 对照：基线 R50+DecoderCup 90.4903 / 0.14473 / HD95 0.41634（seed 1234）
# 判定（§0）：ΔDice ≥ 0.5 或 HD95 降 ≥ 0.1 mm。
#
# 前置：等 queue_c2d_r50seq_goblet0.ps1 两个 job 结束（其状态表 2 行），不抢卡。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_mref.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_mref.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prevCsv = Join-Path $logDir "queue_c2d_r50seq_goblet0_status.csv"
$statusPath = Join-Path $logDir "queue_c2d_mref_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$job = @{ n = "N1"; tag = "egmref4"
          a = @("--arch", "baseline", "--vit_layers", "0", "--edge_fusion", "mref",
                "--edge_depth", "4", "--edge_gamma", "0.1")
          snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_egmref4_gm10_c2d" }

while (-not $DryRun) {
    $busy = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
    $rows = 0
    if (Test-Path -LiteralPath $prevCsv) {
        $rows = @(Get-Content -LiteralPath $prevCsv | Select-Object -Skip 1 | Where-Object { $_ -ne "" }).Count
    }
    if (-not $busy -and $rows -ge 2) { break }
    Start-Sleep -Seconds 60
}

$snapshot = Join-Path $modelDir $job.snap
$argsList = @("train.py") + $common + $job.a
Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
Write-Host "[$($job.n)] snapshot: $snapshot"
if ($DryRun) { exit 0 }

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
Write-Host "[mref] queue finished"
