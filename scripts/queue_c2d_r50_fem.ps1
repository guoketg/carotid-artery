# FDFUNet（Neural Networks 2024）的 FEM 整块搬到 R50 + DecoderCup 主骨干上。
#
# 依据用户要求「先全部挂上去，不要单独挂」：FEM = MFDF（多尺度频域滤波，LN → 通道
# 对半切 → DASConv / FDF 交错）+ ASCA（非对称空间-通道注意力），两个子块一起上，
# 不在这一轮做拆分消融。
#
# 挂载：DecoderCup 的 3 个 skip 汇合点（28²/512、56²/256、112²/64）各一个 FEM，
#       skip' = skip + γ·FEM(skip)，γ 初值 0.1（0 会让模块等于关闭，见 G1 教训）。
#       注意 DRDAC 不属于 FEM，它是 FDFUNet 主干 DownBlock 的卷积体，本轮不搬。
#
# 对照：R50 单骨干基线（--vit_layers 0）90.4903 / ASSD 0.14473 / HD 0.7062（seed 1234）
# 判定（§0）：ΔDice ≥ 0.5 或 HD 降 ≥ 0.1 mm。
# 若不过线，下一轮按 MFDF / ASCA 留一拆分（用户第 3 点）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_fem.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_fem.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$statusPath = Join-Path $logDir "queue_c2d_r50_fem_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$job = @{ n = "F1"; tag = "egfem"
          a = @("--arch", "baseline", "--vit_layers", "0",
                "--edge_fusion", "fem", "--edge_gamma", "0.1")
          snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_egfem_gm10_c2d" }

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
Write-Host "[fem] queue finished"
