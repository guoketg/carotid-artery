# 2026-11-05 WeaveSeg（ICCV 2025）四行消融复现
#
# 论文 Table 1（细胞核实例分割，CoNSeP 等）：
#   Baseline         F1 0.725 / Dice 0.828 / HausD 27.381 / AJI 0.515
#   w/o ICW          F1 0.738 / Dice 0.835 / HausD 26.933 / AJI 0.529
#   w/o ASR          F1 0.746 / Dice 0.844 / HausD 26.354 / AJI 0.547
#   Full WeaveSeg    F1 0.759 / Dice 0.852 / HausD 25.835 / AJI 0.563
# 本脚本按同样的四行顺序跑，但任务/数据/指标换成项目的颈动脉语义分割。
#
# 唯一变量：
#   Baseline      --weave_asr 0 --weave_icw 0
#   w/o ICW       --weave_asr 1 --weave_icw 0
#   w/o ASR       --weave_asr 0 --weave_icw 1
#   Full WeaveSeg --weave_asr 1 --weave_icw 1
# 四行共用：Eq.(12) 五项损失（--loss weave5）、同一底座（U-Net 编码器+解码器骨架）、
# 同一组上下文头（sf/sb/su），所以差异只来自 ASR / ICW 本身。
#
# 口径（§0）：SPARC / bs4 + ga2 / 200 epoch / seed 1234 / ep200 读数。
#   U-Net 基线 86.55（已有），可直接对 Baseline 行做二次校验。
# 显存兜底：若某行 OOM，脚本自动用 bs2 + ga4 重跑同一行（快照名加 _bs2），
# 并在状态 CSV 里留下两次记录；不静默跳过。
#
# 快照名：${pre}_SPARC_lweave5_weave_{base,noicw,noasr,full}_c2d
# （train.py 对非 ce_dice 损失自动加 _lweave5，再按变体加 _weave_xxx）
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_weaveseg.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_weaveseg.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilWeaveSeg" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_*.ps1*" -and
                       $_.CommandLine -notlike "*queue_c2d_weaveseg.ps1*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$statusPath = Join-Path $logDir "queue_c2d_weaveseg_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$jobs = @(
    @{ Name = "weave_base";  Asr = 0; Icw = 0; Desc = "Baseline (no ASR, no ICW)" },
    @{ Name = "weave_noicw"; Asr = 1; Icw = 0; Desc = "w/o ICW (ASR only)" },
    @{ Name = "weave_noasr"; Asr = 0; Icw = 1; Desc = "w/o ASR (ICW only)" },
    @{ Name = "weave_full";  Asr = 1; Icw = 1; Desc = "Full WeaveSeg (ASR + ICW)" }
)

function New-JobArgs($job, $bs, $ga) {
    @("train.py", "--subset", "SPARC",
      "--batch_size", "$bs", "--grad_accum", "$ga", "--num_workers", "1",
      "--max_epochs", "200", "--eval_test_every", "5", "--seed", "1234",
      "--arch", "weaveseg", "--loss", "weave5",
      "--weave_asr", "$($job.Asr)", "--weave_icw", "$($job.Icw)",
      "--weave_iters", "3")
}

if ($DryRun) {
    foreach ($job in $jobs) {
        $a = New-JobArgs $job 4 2
        Write-Host "[weaveseg] $($job.Name) $($job.Desc): $Python $($a -join ' ')"
        Write-Host "[weaveseg]   snapshot: $(Join-Path $modelDir "${pre}_SPARC_lweave5_$($job.Name)_c2d")"
    }
    Write-Host "[weaveseg] dry-run only, other queue procs: $((Get-OtherQueue | Measure-Object).Count), train.py procs: $((Get-TrainingProcess | Measure-Object).Count)"
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    exit 0
}

# 严格串行：先等别的队列脚本退出，再等任何 train.py 退出，绝不抢卡
while (Get-OtherQueue) {
    Write-Host "[weaveseg] waiting for another queue script to finish (60s)..."
    Start-Sleep -Seconds 60
}
while (Get-TrainingProcess) {
    Write-Host "[weaveseg] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

Push-Location $Root
foreach ($job in $jobs) {
    $runs = @(@{ Bs = 4; Ga = 2; Suffix = "" })
    foreach ($run in $runs) {
        $snapshot = Join-Path $modelDir "${pre}_SPARC_lweave5_$($job.Name)$($run.Suffix)_c2d"
        $jobArgs = New-JobArgs $job $run.Bs $run.Ga
        $tag = "$($job.Name)$($run.Suffix)"

        if (Test-Path $snapshot) {
            $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
            Move-Item -LiteralPath $snapshot -Destination $bak
            Write-Host "[weaveseg] archived previous run to $bak"
        }

        $ts = Get-Date -Format "yyyyMMdd_HHmmss"
        $jobLog = Join-Path $logDir "${tag}_$ts.log"
        Write-Host "[weaveseg] ===== start $tag ($($job.Desc)) bs=$($run.Bs) ga=$($run.Ga) ====="
        Write-Host "[weaveseg] log: $jobLog"

        $start = Get-Date
        & $Python @jobArgs *> $jobLog
        $code = $LASTEXITCODE
        $end = Get-Date
        $mins = [math]::Round(($end - $start).TotalMinutes, 1)
        "$tag,$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
            Out-File -FilePath $statusPath -Append -Encoding utf8
        Write-Host "[weaveseg] ===== done $tag exit=$code ($mins min) ====="

        if ($code -ne 0 -and (Select-String -Path $jobLog -Pattern "OutOfMemoryError" -Quiet) -and $run.Suffix -eq "") {
            # 显存兜底：同一条命令行降到 bs2 + ga4（等效 batch 仍为 8）重跑一次
            Write-Host "[weaveseg] OOM detected, retry $($job.Name) with bs2 + ga4"
            $snapshot2 = Join-Path $modelDir "${pre}_SPARC_lweave5_$($job.Name)_bs2_c2d"
            if (Test-Path $snapshot2) {
                Move-Item -LiteralPath $snapshot2 -Destination ("${snapshot2}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
            }
            $jobArgs2 = New-JobArgs $job 2 4
            $ts2 = Get-Date -Format "yyyyMMdd_HHmmss"
            $jobLog2 = Join-Path $logDir "$($job.Name)_bs2_$ts2.log"
            $start2 = Get-Date
            & $Python @jobArgs2 *> $jobLog2
            $code2 = $LASTEXITCODE
            $end2 = Get-Date
            $mins2 = [math]::Round(($end2 - $start2).TotalMinutes, 1)
            "$($job.Name)_bs2,$($start2.ToString('yyyy-MM-dd HH:mm:ss')),$($end2.ToString('yyyy-MM-dd HH:mm:ss')),$mins2,$code2,$jobLog2" |
                Out-File -FilePath $statusPath -Append -Encoding utf8
            Write-Host "[weaveseg] ===== done $($job.Name)_bs2 exit=$code2 ($mins2 min) ====="
        }
    }
}
Pop-Location

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[weaveseg] queue finished"
