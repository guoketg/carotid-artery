# 2026-11-14 队列：完整 TransUNet（12 层 ViT）+ 边缘分支 + 边缘概率注入主路。
#
# 与 queue_c2d_edge_tu.ps1 的唯一差别是 --edge_fuse add：
#   γ 零初始化标量，把边缘概率（第 1 类）双线性插值到主解码器最浅层特征上做残差相加；
#   γ=0 起步，因此与"不注入"逐位一致，梯度会自己决定要不要用边缘。
#
# 对照（同 seed 1234、同协议，不重跑）：
#   完整 TransUNet 基线 90.32 / HD 0.64（..._SPARC_c2d）
#   EB-TU（只辅助监督）  90.48 / HD 0.63（..._SPARC_c2d_eb）
# 快照：TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_c2d_eb_efuse
#
# 串行约定：启动前等待所有 train.py 与其它 queue_*.ps1 退出，全程只占一个 python 进程。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_fuse.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_fuse.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerEf -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
[Win32.PowerEf]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            $_.CommandLine -like "*queue_*.ps1*" -and
            $_.CommandLine -notlike "*queue_c2d_edge_fuse*"
        }
}

$statusPath = Join-Path $logDir "queue_c2d_edge_fuse_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "EB-TU-fuse"; tag = "transunet_edge_fuse"
       a = @("--arch", "baseline", "--edge_branch", "1", "--edge_branch_w", "1.0",
             "--edge_kernel", "3", "--edge_fuse", "add")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_c2d_eb_efuse" }
)

foreach ($job in $jobs) {
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
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
Write-Host "[edge_fuse] queue finished"
