# 2026-11-14 队列：连续界面表示（SDF / 距离场）两臂。
#
# 与 §12 的"边缘/高频"路线不同：这里不改输入、不做推理期后处理，改的是**输出表示**——
#   解码器末端多一个 SDF 头，回归到 GT 的带符号距离场（内部为正，像素单位，裁剪 ±tau），
#   零水平集即分割边界；距离场把"边界在哪"变成稠密回归问题，监督信号覆盖整幅图而不只是
#   1 px 边缘带，梯度不受前景/背景像素数失衡影响。
#
# 两臂（同 seed 1234、同协议，各 200 epoch）：
#   SDF-out：logits = [-s·sdf, +s·sdf]，**替换**分类头输出（表示层面的改动，主实验）
#   SDF-aux：分类头输出不变，SDF 头只作辅助监督（只加监督信号，不动推理路径，对照组）
# 两臂都额外加窄带 L1 + eikonal 正则（|∇d|=1）。
#
# 对照（同 seed 1234、同协议，不重跑）：
#   完整 TransUNet 基线 90.32 / HD 0.64（..._SPARC_c2d）
#   EB-TU（只加边缘辅助监督） 90.48 / HD 0.63（..._SPARC_c2d_eb）
#
# 串行约定：启动前等待所有 train.py 与其它 queue_*.ps1 退出，全程只占一个 python 进程。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_sdf.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_sdf.ps1
#   ... -Only SDF-out            只跑指定 job
#   ... -IgnoreOtherQueues       不等待其它 queue_*.ps1（仍等待 train.py）
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [switch]$IgnoreOtherQueues,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerSdf -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
[Win32.PowerSdf]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$selfName = "queue_c2d_sdf.ps1"

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    if ($IgnoreOtherQueues) { return @() }
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.CommandLine -and
            ($_.CommandLine -like "*\scripts\queue_*.ps1*") -and
            ($_.CommandLine -notlike "*$selfName*")
        }
}

$statusPath = Join-Path $logDir "queue_c2d_sdf_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "SDF-out"; tag = "transunet_sdf_out"
       a = @("--arch", "baseline", "--sdf", "out", "--sdf_w", "1.0", "--sdf_tau", "12")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_c2d_sdfout" },
    @{ n = "SDF-aux"; tag = "transunet_sdf_aux"
       a = @("--arch", "baseline", "--sdf", "aux", "--sdf_w", "1.0", "--sdf_tau", "12")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_c2d_sdfaux" }
)

foreach ($job in $jobs) {
    if ($Only -and $Only -ne $job.n) { continue }

    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.snap
    $argsList = @("train.py") + $common + $job.a
    Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[$($job.n)] snapshot: $snapshot"
    if ($DryRun) { continue }

    if (Test-Path (Join-Path $snapshot "epoch_199.pth")) {
        Write-Host "[$($job.n)] already finished, skip"
        continue
    }

    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[$($job.n)] archived previous partial run to $bak"
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
Write-Host "[sdf] queue finished"
