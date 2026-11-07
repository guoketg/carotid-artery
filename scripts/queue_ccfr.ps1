# 2026-11-03 第六队列：复现 ESWA-D-25-34301（CCFR：EA-Net + rectified flow 精修）
#
# 论文口径（Carotid25 自建 558 张 / Ar-PlaqSegm1 541 张）：
#   Stage1 EA-Net    Adam 1e-3 / wd 0.01 / 100 ep / Dice+BCE+0.4·Edge
#   Stage2 flow 头   AdamW 1e-4 / wd 0.01 / bs8 / 20 ep / K=5 / 损失权重 5:1:5
#   Carotid25: CCFR 87.06 Dice（EA-Net 单独 85.11，+1.95）
#
# 本项目口径（论文数据不可得，改在本项目 Carotid2D 上复现）：
#   train = SPARC train 350，test = SPARC test 160；§0 训练口径 bs4 + ga2 / 200 ep；
#   EA-Net 取 ep200 权重冻结，flow 头跑 20 ep（bs4 + ga2 ≈ 论文 bs8）。
#   "w/o flow refinement" = 同一 Job1 的 stage1_curve.csv ep200 读数，不另跑。
#
# 快照名（train.py 自动拼）：
#   C1  ${pre}_SPARC_ccfr_c2d              full, seed 1234
#   C2  ${pre}_SPARC_ccfr_s2_nocond_c2d    w/o condition（复用 C1 的 stage1_ep200.pth）
#   C3  ${pre}_SPARC_ccfr_s2_nounc_c2d     w/o uncertainty guidance（同上）
#   C4  ${pre}_s2024_SPARC_ccfr_c2d        full, seed 2024（噪声带）
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilCCFR" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$common = @("train.py", "--arch", "ccfr", "--subset", "SPARC", "--dataset", "Carotid",
            "--batch_size", "4", "--grad_accum", "2", "--max_epochs", "200",
            "--eval_test_every", "5", "--list_dir", "./lists/lists_Carotid2D",
            "--num_workers", "4", "--img_size", "224")
$stage1Ckpt = Join-Path $modelDir "${pre}_SPARC_ccfr_c2d\stage1_ep200.pth"

$jobs = @(
    @{ name = "CCFR_s1234"; args = $common + @("--ccfr_stage", "both", "--seed", "1234",
                                                "--stage1_epochs", "200", "--stage1_pick", "200",
                                                "--stage2_epochs", "20", "--flow_steps", "5");
       snap = "${pre}_SPARC_ccfr_c2d"; needs = $null },
    @{ name = "CCFR_nocond"; args = $common + @("--ccfr_stage", "2", "--seed", "1234",
                                                 "--ccfr_nocond", "1", "--ccfr_stage1_ckpt", $stage1Ckpt,
                                                 "--stage2_epochs", "20", "--flow_steps", "5");
       snap = "${pre}_SPARC_ccfr_s2_nocond_c2d"; needs = $stage1Ckpt },
    @{ name = "CCFR_nounc"; args = $common + @("--ccfr_stage", "2", "--seed", "1234",
                                                "--ccfr_nounc", "1", "--ccfr_stage1_ckpt", $stage1Ckpt,
                                                "--stage2_epochs", "20", "--flow_steps", "5");
       snap = "${pre}_SPARC_ccfr_s2_nounc_c2d"; needs = $stage1Ckpt },
    @{ name = "CCFR_s2024"; args = $common + @("--ccfr_stage", "both", "--seed", "2024",
                                                "--stage1_epochs", "200", "--stage1_pick", "200",
                                                "--stage2_epochs", "20", "--flow_steps", "5");
       snap = "${pre}_s2024_SPARC_ccfr_c2d"; needs = $null }
)

$statusPath = Join-Path $logDir "queue_ccfr_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

Write-Host "[ccfr] $($jobs.Count) jobs, serial"
foreach ($j in $jobs) { Write-Host ("  {0,-14} -> {1}" -f $j.name, $j.snap) }
if ($DryRun) {
    Write-Host "[ccfr] dry-run only; train.py running: $((Get-TrainingProcess | Measure-Object).Count) proc"
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "[ccfr] queue finished"
    exit 0
}

while (Get-TrainingProcess) {
    Write-Host "[ccfr] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

foreach ($j in $jobs) {
    if ($j.needs -and -not (Test-Path $j.needs)) {
        Write-Host "[ccfr] SKIP $($j.name): 缺少 $($j.needs)"
        "$($j.name),,,,2,MISSING_STAGE1" | Out-File -FilePath $statusPath -Append -Encoding utf8
        continue
    }

    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[ccfr] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.name)_$ts.log"
    Write-Host "[ccfr] ===== start $($j.name) ====="
    Write-Host "[ccfr] log: $jobLog"
    Write-Host "[ccfr] cmd: $Python $($j.args -join ' ')"

    $start = Get-Date
    Push-Location $Root
    & $Python @($j.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[ccfr] ===== done $($j.name) exit=$code ($mins min) ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[ccfr] queue finished"
