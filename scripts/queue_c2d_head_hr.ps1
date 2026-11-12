# 2026-11-12 队列：224 全分辨率证据支路（--head_hr 1）单臂 SPARC。
#
# 动机：docs/low-score-samples.md §6。实测解码器末级 224²/16ch 既无 skip、也不接触原图，
# 最细图像证据只到 112²；低分样本恰是短轴 1.1 mm 的薄结构，误差是 1 原始像素宽的边界带。
# 本臂在该级补 4 个原图派生通道（I / 5×5 均值 / 21×21 均值 / 9×9 局部标准差），
# 两层 3×3 后输出 logits 残差，末层零初始化 → 起点与基线逐位等价（已 CPU 冒烟：
# 参数 5282、基础权重逐位一致、初始 logits 差 0.0）。
#
# 对照臂（同 seed 1234 同协议，HD 为 medpy 口径）：
#   A0  TU_..._SPARC_c2d  epoch_199.pth  Dice 90.4903 / ASSD 0.145 / HD 0.7062
#
# 本臂快照：TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d_hr
# 判定（§0 统一协议）：ΔDice ≥ +0.5 且 ASSD / HD 同向不变差才算通过；
# 未过线则按 docs/low-score-samples.md §5 收口，不再追加边界模块。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_head_hr.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_head_hr.ps1
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

$statusPath = Join-Path $logDir "queue_c2d_head_hr_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "baseline", "--vit_layers", "0", "--n_skip", "3", "--img_size", "224",
    "--head_hr", "1"
)

$jobs = @(
    @{ n = "HR-SPARC"; tag = "r50_head_hr_sparc"
       a = @()
       snap = "TU_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d_hr" }
)

foreach ($job in $jobs) {
    # 等 GPU 空出来再开跑，绝不与其它训练抢卡
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[$($job.n)] another train.py is running, waiting 60s..."
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
Write-Host "[c2d head_hr] queue finished"
