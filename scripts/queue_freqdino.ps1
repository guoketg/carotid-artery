# FreqDINO 移植批（arXiv 2512.11335，超声分割）三臂接力队列：等 queue_fbgr_full 收口后自动跑。
#
# 口径：不换数据集、不换骨干，只把论文的频域模块四级全挂 R50-TransUNet 解码器
#       （28²/256 → 56²/128 → 112²/64 → 224²/16，与 DDBE 四级全挂同口径），
#       其余一切照抄本仓 §0 配置。
#       论文消融基线是「DINOv3 冻结 + 轻 adapter」= 82.35 Dice 的弱头，不是 U-Net；
#       因此这批判定只看我们自己的增量，不引用论文的 +2.21 / +0.96。
#
# 三臂（全部 SPARC seed 1234 bs4 ga2 ep200，smoke_freqdino.py 全绿后入队）：
#   FD-MFEA  --arch baseline --vit_layers 0 --fd_mfea 1
#            snap: ..._SPARC_fdmfea_c2d
#   FD-FGBR  --arch baseline --vit_layers 0 --fd_fgbr 1
#            snap: ..._SPARC_fdfgbr_c2d
#   FD-BOTH  --arch baseline --vit_layers 0 --fd_mfea 1 --fd_fgbr 1（论文原序 MFEA→FGBR）
#            snap: ..._SPARC_fdmfea_fdfgbr_c2d
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_freqdino.ps1 -DryRun
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prevCsv = Join-Path $logDir "queue_fbgr_full_status.csv"
$statusPath = Join-Path $logDir "queue_freqdino_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$jobs = @(
    @{ n = "FD-MFEA"; tag = "r50_fdmfea"
       a = @("train.py","--dataset","Carotid","--subset","SPARC","--seed","1234",
             "--batch_size","4","--grad_accum","2","--max_epochs","200","--eval_test_every","5",
             "--arch","baseline","--vit_layers","0","--fd_mfea","1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fdmfea_c2d" },
    @{ n = "FD-FGBR"; tag = "r50_fdfgbr"
       a = @("train.py","--dataset","Carotid","--subset","SPARC","--seed","1234",
             "--batch_size","4","--grad_accum","2","--max_epochs","200","--eval_test_every","5",
             "--arch","baseline","--vit_layers","0","--fd_fgbr","1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fdfgbr_c2d" },
    @{ n = "FD-BOTH"; tag = "r50_fd_both"
       a = @("train.py","--dataset","Carotid","--subset","SPARC","--seed","1234",
             "--batch_size","4","--grad_accum","2","--max_epochs","200","--eval_test_every","5",
             "--arch","baseline","--vit_layers","0","--fd_mfea","1","--fd_fgbr","1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fdmfea_fdfgbr_c2d" }
)

Write-Host "[FD] waiting for prevCsv: $prevCsv"
foreach ($j in $jobs) {
    Write-Host "[FD] $($j.n) $Python $($j.a -join ' ')"
    Write-Host "[FD]   snapshot: $(Join-Path $modelDir $j.snap)"
}
if ($DryRun) { exit 0 }

while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
    $rows = 0
    if (Test-Path -LiteralPath $prevCsv) {
        $rows = @(Get-Content -LiteralPath $prevCsv | Select-Object -Skip 1 |
                  Where-Object { $_ -ne "" }).Count
    }
    if (-not $busy -and $rows -ge 3) { break }
    Start-Sleep -Seconds 120
}

foreach ($j in $jobs) {
    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[$($j.n)] archived previous run to $bak"
    }
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.tag)_$ts.log"
    Write-Host "[$($j.n)] log: $jobLog"

    $start = Get-Date
    $jobArgs = $j.a
    Push-Location $Root
    & $Python @jobArgs *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.n)_$($j.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($j.n)] end exit=$code ${minutes}min"
}
Write-Host "[freqdino] queue finished"
