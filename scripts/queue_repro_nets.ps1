# 文献复现网络队列：纯卷积 U-Net 基线 / XNet（小波低高频双分支+X 形融合）/ GobletNet（双编码器+FAM）。
#
# 三个网络各自独立训练，与 baseline/dual 互不混用（train.py --arch unet|xnet|gobletnet）。
# 口径与主队列一致：官方划分 SPARC 子集、200 epoch、每 5 epoch 评一次测试集、seed 1234。
# 等效 batch 统一为 8：unet/gobletnet 用 bs4+ga2；xnet 双分支最吃显存（bs2 峰值 3.7GB），
# 用 bs2+ga4。12GB 的 RTX 3060 下三者都在预算内，但别把显存吃到顶（WDDM 换页惩罚）。
#
# 快照目录名与 train.py 拼出的一致（硬约束 2）：
#   TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_unet_c2d
#   TU_pretrain_R50-ViT-B_16_skip3_epo200_bs2_ga4_224_SPARC_xnet_c2d
#   TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_gobletnet_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_repro_nets.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_repro_nets.ps1
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

# Prevent system sleep while training.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilRepro" -Namespace Win32 -PassThru
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
    "--subset", "SPARC",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--seed", "1234"
)
$pre4 = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$pre2 = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs2_ga4_224"

# 顺序即优先级：先 U-Net 基线（复现对照的锚点），再 XNet，最后 GobletNet。
$jobs = @(
    @{ Name = "unet_sparc";     Args = @("train.py", "--arch", "unet",     "--batch_size", "4", "--grad_accum", "2") + $common; Snapshot = "${pre4}_SPARC_unet_c2d" },
    @{ Name = "xnet_sparc";     Args = @("train.py", "--arch", "xnet",     "--batch_size", "2", "--grad_accum", "4") + $common; Snapshot = "${pre2}_SPARC_xnet_c2d" },
    @{ Name = "gobletnet_sparc"; Args = @("train.py", "--arch", "gobletnet", "--batch_size", "4", "--grad_accum", "2") + $common; Snapshot = "${pre4}_SPARC_gobletnet_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_repro_nets_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    # 等 GPU 空出来再开跑，绝不与其它训练抢卡
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[repro] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[repro] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[repro] ===== start $($job.Name) ====="
    Write-Host "[repro] command: $Python $($job.Args -join ' ')"
    Write-Host "[repro] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[repro] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[repro] all jobs finished"
