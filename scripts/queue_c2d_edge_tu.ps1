# 2026-11-14 队列：完整 TransUNet（R50-ViT-B/16，默认 12 层 ViT）+ 边缘监督分支。
#
# 背景：此前 EB-R50 实际传了 --vit_layers 0（纯 R50 局部支），主任务 -0.08、edgeDice 0.321，
# 与用户一年前在完整 TransUNet 上观察到的约 +1.0 Dice 不一致。本队列只补这一臂，
# 用于判断差异是否来自“去掉 ViT”这一配置，而不是边缘方法本身。
#
# 配置与 EB-R50 逐字一致，唯一差别是不传 --vit_layers（默认 12 层）：
#   --arch baseline --edge_branch 1 --edge_branch_w 1.0 --edge_kernel 3
# 对照（同 seed 1234、同协议，不重跑）：
#   完整 TransUNet 基线 90.32 / HD 0.64（TU_pretrain_R50-ViT-B_16_skip3_..._SPARC_c2d）
#   EB-R50（vitL0）     90.41 / HD 0.69（..._vitL0_..._SPARC_c2d_eb）
# 快照：TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_c2d_eb
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_tu.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_tu.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerTu -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerTu]::SetThreadExecutionState($keepAwake) | Out-Null

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
            $_.CommandLine -notlike "*queue_c2d_edge_tu*"
        }
}

$statusPath = Join-Path $logDir "queue_c2d_edge_tu_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "EB-TU"; tag = "transunet_edge_branch"
       a = @("--arch", "baseline", "--edge_branch", "1", "--edge_branch_w", "1.0",
             "--edge_kernel", "3")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_c2d_eb" }
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
Write-Host "[edge_tu] queue finished"
