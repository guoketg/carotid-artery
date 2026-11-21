# 2026-11-20 单臂：双向逐级门控边缘支（用户直接指派，最高优先级）
#
# 目的：回答「两个解码器必须交互」这一族到底有没有用。此前所有边缘臂（c2d_eb 四臂、
# R-GEF/MREF/FEM/ABF）都没有让边缘解码器与主解码器在前向上互相传递特征，本臂是第一次。
#
# 结构（networks/unet.py --edge_gate bi）：
#   每一级：主解码特征 → 1×1 零初始化投影 → 加进边缘解码特征（语义上下文去噪）；
#          边缘特征 → 1×1 头 → sigmoid → 门控主解码特征（边界定位回灌）。
#   四个 α 与四个投影全零初始化，第 0 步与 --edge_branch 1 的既有臂逐位等价。
#   参数量：+0.350M（82.940M → 83.290M）。
#
# 对照（同 seed 1234、同协议，不重跑）：
#   U-Net 基线                          86.55 / 0.22500 / 1.136
#   U-Net + 边缘支（morph k=3，无交互）  86.65 / 0.21599 / 1.076
# 判定：ΔDice >= +0.5 且 ΔHD <= -0.1 mm 才过线；不过线则整条边缘交互线收口。

param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 21

$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "queue_c2d_edge_bigate_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$argsList = @(
    "train.py",
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "unet",
    "--edge_branch", "1", "--edge_branch_w", "1.0",
    "--edge_gate", "bi"
)

while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
    Write-Host "[EB-BIGATE-U] busy, waiting 60s... blocked by: $(Get-BlockReason)"
    Start-Sleep -Seconds 60
}

Write-Host "[EB-BIGATE-U] $Python $($argsList -join ' ')"
if ($DryRun) { exit 0 }

$lock = Enter-LaunchLock
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$jobLog = Join-Path $logDir "unet_edge_bigate_$ts.log"
$start = Get-Date
Push-Location $Root
& $Python @argsList *> $jobLog
$code = $LASTEXITCODE
Pop-Location
$end = Get-Date
Exit-LaunchLock -LockPath $lock
$minutes = [math]::Round(($end - $start).TotalMinutes, 1)
"EB-BIGATE-U,$start,$end,$minutes,$code,$jobLog" |
    Out-File -FilePath $statusPath -Append -Encoding utf8
Write-Host "[EB-BIGATE-U] end exit=$code ${minutes}min"
