# 后续队列（2026-10-28 夜）：融合算子横评 + DDBE 门槛验证 + 基线补齐。
#
# 为什么要单独一个接力脚本：queue_c2d.ps1 / queue_c2d_edge.ps1 都在跑，
# PowerShell 启动时就把整个脚本读进内存，事后改文件对运行中的实例无效，
# 所以新建第三个接力脚本：等前两个队列全部退出且 GPU 空闲后再开跑。
#
# 6 个 job（全部 SPARC，统一口径 $common = bs4/ga2/200ep/eval5，取 ep150）：
#   1) faff1234_sparc  --fuse_op aff    双骨干 AFF MS-CAM 横评（对照 d1234 auto，零实现成本）
#   2) fgate1234_sparc --fuse_op gate   逐元素门控横评（R6 对照，同上）
#   3) fdeform1234_sparc --fuse_op deform D-LKA 可变形大核横评（同上）
#   4) ddbe3_1234_sparc --ddbe 3        BDFR-Net 式 DDBE 单级门槛验证（level3=112²/64ch，
#                                       零初始化恒等起步，冒烟已过：同 seed 基础权重逐位相等、
#                                       eval 恒等 max|Δ|=0），对照 A0 基线 seed1234
#   5) b1234_sparc     A0 单通道原图基线 seed1234 重建 —— 原产物（test_curve.csv/metrics.csv）
#                      在本机已缺失，ddbe3 的病例级 Wilcoxon 配对必须有它
#   6) b3407_sparc     基线 seed3407 补跑 —— 上一轮在 epoch≈50 被中途杀掉，无 metrics.csv
#
# 快照目录名必须与 train.py 实际拼出来的一致（硬约束 2：不得撞名覆盖）：
#   dual + aff/gate/deform → ..._224_SPARC_dual_L1234_<op>_c2d
#   baseline + ddbe3       → ..._224_SPARC_ddbe3_c2d
#   baseline seed1234      → ..._224_SPARC_c2d
#   baseline seed3407      → ..._224_s3407_SPARC_c2d（旧的残缺目录会被归档为 _bak_<ts>）
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fuse.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fuse.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Prevent system sleep while training.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilFuse" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

# 只要还有别的 queue_c2d* 队列脚本活着，就不能抢 GPU（排除自己）。
function Get-OtherQueueAlive {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_c2d*" } |
        Where-Object { $_.ProcessId -ne $PID }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$dual = @("--arch", "dual", "--levels", "1,2,3,4")
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

# 顺序 == 优先级：先零成本横评（aff/gate/deform），再 DDBE 门槛，最后补基线。
$jobs = @(
    @{ Name = "faff1234_sparc";    Args = @("train.py", "--subset", "SPARC", "--seed", "1234") + $dual + @("--fuse_op", "aff")    + $common; Snapshot = "${pre}_SPARC_dual_L1234_aff_c2d" },
    @{ Name = "fgate1234_sparc";   Args = @("train.py", "--subset", "SPARC", "--seed", "1234") + $dual + @("--fuse_op", "gate")   + $common; Snapshot = "${pre}_SPARC_dual_L1234_gate_c2d" },
    @{ Name = "fdeform1234_sparc"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234") + $dual + @("--fuse_op", "deform") + $common; Snapshot = "${pre}_SPARC_dual_L1234_deform_c2d" },
    @{ Name = "ddbe3_1234_sparc";  Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--ddbe", "3") + $common;                Snapshot = "${pre}_SPARC_ddbe3_c2d" },
    @{ Name = "b1234_sparc";       Args = @("train.py", "--subset", "SPARC", "--seed", "1234") + $common;                                Snapshot = "${pre}_SPARC_c2d" },
    @{ Name = "b3407_sparc";       Args = @("train.py", "--subset", "SPARC", "--seed", "3407") + $common;                                Snapshot = "${pre}_s3407_SPARC_c2d" }
)

$statusPath = Join-Path $logDir "queue_c2d_fuse_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# ---- 接力：等前两个队列（queue_c2d.ps1 / queue_c2d_edge.ps1）全部退出 --------
if (-not $DryRun) {
    Write-Host "[fuse] waiting for the other queues to release the GPU..."
    while ($true) {
        $busy = @(Get-TrainingProcess).Count
        $others = @(Get-OtherQueueAlive).Count
        if ($busy -eq 0 -and $others -eq 0) {
            Write-Host "[fuse] GPU is free (busy=$busy, otherQueues=$others)"
            break
        }
        Write-Host "[fuse] busy=$busy otherQueues=$others ; sleep 60s"
        Start-Sleep -Seconds 60
    }
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[fuse] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[fuse] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[fuse] ===== start $($job.Name) ====="
    Write-Host "[fuse] command: $Python $($job.Args -join ' ')"
    Write-Host "[fuse] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[fuse] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[fuse] all jobs finished"
