# 2026-11-14 队列：区域 Hausdorff 距离损失 #82（2 job，单变量）。
#
# 文献：Guzzi et al., MLMI 2025, "Regional Hausdorff Distance Losses"（DOI 10.1007/978-3-032-09513-8_7）
# 官方实现：lisaGUZZI/HD-Loss 的 lahsym.py / lh.py / compound_losses.py
#   软腐蚀 d+(p) = -erode(1-p, I)，3D 26 连通 → 本实现 2D 3x3 乘积；tan=False、norm=0、weight_ce=1
#   主损失 = CE + 1.0*LAH_sym；本实现沿用项目口径 0.5CE+0.5Dice，故 λ=0.5 与论文 CE:HD 等权对齐
#
# 两臂（都配对 A0 = R50 vitL0 seed1234：90.4903 / 0.14473 / HD 0.7062）：
#   I6 = 主臂 --rhd_iters 6：本数据 HD 尾部中位约 5.4 px，I 是可感知的最大距离，I=2 会截断尾部
#   I2 = 论文默认 --rhd_iters 2：原配参对照，回答「截断是否反而有利」
# 判定（跑前登记）：ΔHD <= -0.1 mm（主判）且 ΔDice >= -0.3（护栏，防止用 Dice 换 HD）；
#   过线后 3 seed（42/317/1029）复核；5-seed 集成不参与判定。
# 快照：..._SPARC_lrhd_TsymW0.5I6_c2d / ..._SPARC_lrhd_TsymW0.5I2_c2d
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_rhd.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_rhd.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerRhd -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerRhd]::SetThreadExecutionState($keepAwake) | Out-Null

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
            $_.CommandLine -notlike "*queue_c2d_rhd*"
        }
}

$statusPath = Join-Path $logDir "queue_c2d_rhd_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "RHD-I6"; tag = "r50_rhd_i6"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lrhd_TsymW0.5I6_c2d" },
    @{ n = "RHD-I2"; tag = "r50_rhd_i2"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "2")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lrhd_TsymW0.5I2_c2d" }
)

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
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
Write-Host "[rhd] queue finished"
