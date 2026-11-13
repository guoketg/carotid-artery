# 2026-11-13 队列：距离尾部损失 CVaR-DSL 单臂筛选（UNet 弱骨干）。
#
# 损失口径：0.5CE + 0.5Dice + 0.05*面积一致性 + 0.1 * CVaR_10%( |p-gt| * clamp(D_gt_boundary/10px, max=3) )
#   D 为到 GT 边界的像素距离（|有符号距离图|），每张切片取最差 10% 像素的均值。
# 依据：阶段 0 离线验证（scripts/analyze_hd_tail_surrogate.py，logs/analysis/hd_tail_surrogate_*.csv）
#   均值代理 rho(HD)=0.745/0.686，top10% 代理 rho(HD)=0.895/0.859，
#   控制均值后 top10% 与 HD 的偏相关仍有 0.767/0.779。
# 对照臂（同 seed 1234 同协议，HD 为 medpy 口径）：
#   UNet 基线 86.5455 / 0.22500 / 1.13618
#   D2 完整领域损失 87.0053 / 0.20174 / 0.98995
# 判定：ΔDice >= +0.5 且 ASSD / HD 同向不变差。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvartail.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvartail.ps1
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

$statusPath = Join-Path $logDir "queue_c2d_cvartail_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "unet", "--loss", "cvartail", "--tl_w", "0.1", "--tl_q", "0.1", "--tl_area_w", "0.05"
)

$jobs = @(
    @{ n = "UNet-CVaR-area-SPARC"; tag = "unet_cvartail"
       a = @()
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_lcvartail_W0.1Q0.1A0.05_unet_c2d" }
)

foreach ($job in $jobs) {
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
Write-Host "[c2d cvartail] queue finished"
