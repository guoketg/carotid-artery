# 后续队列：边缘特征融合·输入级（stem 3→4 通道，第 4 通道 = 小波高频 IDWT 重建图，LL 置零）。
#
# 为什么要单独一个脚本：queue_c2d.ps1 已经在跑（pid 见 status.md 4.0.6），
# PowerShell 启动时就把整个脚本读进内存了，事后改文件对运行中的实例无效，
# 所以这里新建一个「接力」脚本：等主队列的 GPU 空下来（或主队列已退出）再开跑。
# 实验室 24 小时开机，GPU 一空就该有活干 —— 这就是本脚本存在的理由。
#
# 对照口径（technical-design.md 频率注入位置一节）：
#   * edge 变体一律与「同 seed、同子集、同超参」的单通道原图基线配对比较；
#   * 不重新跑基线 —— 主队列里的 b2024/b3407 与历史已完成的 seed1234 基线就是对照。
#
# 快照目录名必须与 train.py 实际拼出来的一致（硬约束 2：不得撞名覆盖）：
#   baseline + edge → ..._224_SPARC_edge_c2d
#   dual     + edge → ..._224_SPARC_edge_dual_L1234_auto_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [int]$WaitRows = 7,
    [string]$Only = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Prevent system sleep while training.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilEdge" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-MainQueueAlive {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_c2d.ps1*" } |
        Where-Object { $_.ProcessId -ne $PID }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$dual = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto")
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

# 顺序 == 优先级（2026-10-27 用户指令：只排 SPARC，中南全部撤掉，一套数据集够了）：
#   1) 单骨干 + edge（SPARC seed1234）—— 与已完成的单通道原图基线直接配对，
#      是「边缘信息到底有没有用」的最干净判据（1 个 job 就出结论）
#   2) 双骨干 + edge（SPARC seed1234）—— 与主队列的 d1234_sparc 配对，
#      看 edge 与双骨干叠加是否还有增量
$jobs = @(
    @{ Name = "e1234_sparc";     Args = @("train.py", "--subset", "SPARC",    "--seed", "1234", "--edge_in", "1") + $common;            Snapshot = "${pre}_SPARC_edge_c2d" },
    @{ Name = "de1234_sparc";    Args = @("train.py", "--subset", "SPARC",    "--seed", "1234", "--edge_in", "1") + $dual + $common;  Snapshot = "${pre}_SPARC_edge_dual_L1234_auto_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# ---- 接力：等主队列把 GPU 让出来 -----------------------------------------
if (-not $DryRun) {
    Write-Host "[edge] waiting for the main queue to release the GPU..."
    while ($true) {
        $rows = 0
        if (Test-Path $statusPath) { $rows = @(Import-Csv -Path $statusPath).Count }
        $busy = @(Get-TrainingProcess).Count
        $mainAlive = @(Get-MainQueueAlive).Count
        if ($busy -eq 0 -and ($rows -ge $WaitRows -or $mainAlive -eq 0)) {
            Write-Host "[edge] main queue released the GPU (rows=$rows, mainAlive=$mainAlive)"
            break
        }
        Write-Host "[edge] busy=$busy rows=$rows/$WaitRows mainAlive=$mainAlive ; sleep 60s"
        Start-Sleep -Seconds 60
    }
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[edge] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[edge] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[edge] ===== start $($job.Name) ====="
    Write-Host "[edge] command: $Python $($job.Args -join ' ')"
    Write-Host "[edge] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[edge] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[edge] all jobs finished"
