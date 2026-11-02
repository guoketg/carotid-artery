# 2026-11-02 第三队列：论文 Table 2/3 —— 损失基线对比 + 领域损失消融 + 主配置多 seed
#
# 位置：接力在 queue_c2d_asr2.ps1 之后（严格串行，12GB 显存只允许一个 train.py）。
#
# 目的（全部为论文表格服务，口径一律 §0）：
#   Phase 1  4 个损失基线，A0 骨干，其他一切不变 —— 回答「本文损失是不是只是换了个常见损失」
#              focal     = 0.5·Focal(γ=2,α=0.25) + 0.5·Dice
#              tversky   = Tversky(α=0.3,β=0.7)
#              boundary  = 0.5·CE + 0.5·Dice + 0.5·Boundary（Kervadec MIDL 2019 距离图加权）
#              bce_wiou  = mean(BCE + wIoU)（Syn-Net JBHI 2025）
#   Phase 2  3 个消融，从完整领域损失里各去掉一项 —— 回答「四项里哪项在起作用」
#              noBand    = λ=0（边界带加权退化为普通 CE）
#              noArea    = area_w=0
#              noFar     = far_w=0
#   Phase 3  主配置补 seed 2024 / 3407 —— 跨 seed 稳定性（§1.1 seed 噪声带 ±0.3%）
#
# 待定：$dlBestArgs 要在 04:50 的 sweep 结果出来后改。默认先用 D1 原始配置（L3.0 R2 A0.05 F0.1 M10）。
#
# 快照名（train.py 自动拼）：
#   Phase1  ${pre}_SPARC_l<loss>_c2d
#   Phase2  ${pre}_SPARC_dlL...R...A...F...M..._c2d
#   Phase3  ${pre}_s<seed>_SPARC_dl..._c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_loss_baselines.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_loss_baselines.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilLossBase" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-PreviousQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_c2d_asr2.ps1*" -or $_.CommandLine -like "*queue_c2d_union_follow.ps1*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$common = @("train.py", "--subset", "SPARC", "--batch_size", "4", "--grad_accum", "2",
            "--max_epochs", "200", "--eval_test_every", "5")

# sweep 出结果后改这一行即可（Phase 2/3 全部跟随）
$dlBestArgs = @("--domain_loss", "1")

$bp = @("--dl_lambda", "0", "--dl_area_w", "0.05", "--dl_far_w", "0.1")
$ap = @("--dl_lambda", "3.0", "--dl_area_w", "0", "--dl_far_w", "0.1")
$fp = @("--dl_lambda", "3.0", "--dl_area_w", "0.05", "--dl_far_w", "0")

$jobs = @(
    # ---- Phase 1：损失基线（A0 骨干） ----
    @{ name = "LB_focal";    args = $common + @("--arch", "baseline", "--loss", "focal");
       snap = "${pre}_SPARC_lfocal_c2d" },
    @{ name = "LB_tversky";  args = $common + @("--arch", "baseline", "--loss", "tversky");
       snap = "${pre}_SPARC_ltversky_c2d" },
    @{ name = "LB_boundary"; args = $common + @("--arch", "baseline", "--loss", "boundary");
       snap = "${pre}_SPARC_lboundary_c2d" },
    @{ name = "LB_wiou";     args = $common + @("--arch", "baseline", "--loss", "bce_wiou");
       snap = "${pre}_SPARC_lbcewiou_c2d" },
    # ---- Phase 2：领域损失消融（逐项去掉） ----
    @{ name = "AB_noBand";   args = $common + @("--arch", "baseline") + $dlBestArgs + $bp;
       snap = "${pre}_SPARC_dlL0R2A0.05F0.1M10_c2d" },
    @{ name = "AB_noArea";   args = $common + @("--arch", "baseline") + $dlBestArgs + $ap;
       snap = "${pre}_SPARC_dlL3.0R2A0F0.1M10_c2d" },
    @{ name = "AB_noFar";    args = $common + @("--arch", "baseline") + $dlBestArgs + $fp;
       snap = "${pre}_SPARC_dlL3.0R2A0.05F0M10_c2d" },
    # ---- Phase 3：主配置补 seed ----
    @{ name = "DL_s2024";    args = $common + @("--arch", "baseline", "--seed", "2024") + $dlBestArgs;
       snap = "${pre}_s2024_SPARC_dl_c2d" },
    @{ name = "DL_s3407";    args = $common + @("--arch", "baseline", "--seed", "3407") + $dlBestArgs;
       snap = "${pre}_s3407_SPARC_dl_c2d" }
)

$statusPath = Join-Path $logDir "queue_c2d_loss_baselines_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

Write-Host "[lossbase] $($jobs.Count) jobs, serial"
foreach ($j in $jobs) { Write-Host ("  {0,-12} -> {1}" -f $j.name, $j.snap) }
if ($DryRun) {
    Write-Host "[lossbase] dry-run only, waiting for queue_c2d_asr2: $((Get-PreviousQueue | Measure-Object).Count) proc"
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "[lossbase] queue finished"
    exit 0
}

# 等前一个队列脚本退出（它内部已保证 job 之间不并行）
while (Get-PreviousQueue) {
    Write-Host "[lossbase] waiting for queue_c2d_asr2 to finish (60s)..."
    Start-Sleep -Seconds 60
}
while (Get-TrainingProcess) {
    Write-Host "[lossbase] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

foreach ($j in $jobs) {
    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[lossbase] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.name)_$ts.log"
    Write-Host "[lossbase] ===== start $($j.name) ====="
    Write-Host "[lossbase] log: $jobLog"
    Write-Host "[lossbase] cmd: $Python $($j.args -join ' ')"

    $start = Get-Date
    Push-Location $Root
    & $Python @($j.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[lossbase] ===== done $($j.name) exit=$code ($mins min) ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[lossbase] queue finished"
