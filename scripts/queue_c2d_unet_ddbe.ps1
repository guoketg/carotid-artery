# 2026-11-01 队列：U-Net 上挂 BDFR-Net 解码器改造（DDBE / ACAM）。
# 动机：WMREN(IJCAI-25) 与 BDFR 都把解码器整体换掉；此前 DDBE/ACAM 只在
# TransUNet(baseline) 上跑过（无增益），U-Net 上从未验证，而 U-Net 对边缘
# 输入更敏感（+1.21% vs TransUNet 0）。两个 job 单变量对照 U-Net 基线 86.55%。
#
#   U1  --arch unet --ddbe 4          四级解码器全挂 DDBE（零初始化恒等起步）
#   U2  --arch unet --ddbe 4 --acam 1 再加每级 skip 的 ACAM 跨级对齐
#
# 统一口径与 c2d 队列一致：SPARC / seed 1234 / bs4+ga2 / 200 epoch / 每 5 ep 评估。
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_unet_ddbe.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_unet_ddbe.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilUnetDdbe" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

$jobs = @(
    @{ Name = "U1_unet_ddbe4"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--ddbe", "4") + $common;
       Snapshot = "${pre}_SPARC_ddbe4_unet_c2d" },
    @{ Name = "U2_unet_ddbe4_acam"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--ddbe", "4", "--acam", "1") + $common;
       Snapshot = "${pre}_SPARC_ddbe4_acam_unet_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_unet_ddbe_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[unet-ddbe] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[unet-ddbe] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[unet-ddbe] ===== start $($job.Name) ====="
    Write-Host "[unet-ddbe] command: $Python $($job.Args -join ' ')"
    Write-Host "[unet-ddbe] snapshot: $($job.Snapshot)"
    Write-Host "[unet-ddbe] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[unet-ddbe] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[unet-ddbe] all jobs finished"
