# 2026-11-14 队列：R50 cvartail 的 3 seed 复核（SPARC 补 2 run）+ 中南 3 run。
#
# 依据：results.md §10 —— cvartail 是唯一双门槛全过的损失臂（U-Net +0.82 / HD −0.18）；
#   现迁移到 R50（queue_c2d_cvartail_r50.ps1，seed 1234 在跑），本队列补齐它的多 seed
#   与跨中心证据。损失参数逐字沿用，不调参：
#     --loss cvartail --tl_w 0.1 --tl_q 0.1 --tl_area_w 0.05
#
# 5 个 job（串行，SPARC 2 个 + 中南 3 个）：
#   R50CVAR_s2024 / s3407   SPARC 3 seed 的另两个（主 seed 1234 由 queue_c2d_cvartail_r50 跑）
#   ZN-CVAR s1234/2024/3407 中南三 seed
#
# 判定：
#   SPARC —— 三 seed 合并为一行报均值±sd，主判 ΔHD（≤ −0.1 mm），护栏 ΔDice（≥ −0.3）。
#            配对对照为 R50-pre 同 seed 基线（seed 1234 = 90.4903 / 0.14473 / 0.7062）。
#   中南  —— 只有 seed 1234 有配对基线 Z0（TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_
#            bs4_ga2_224_Zhongnan_c2d）；s2024 / s3407 目前只能看方向，若要报均值±sd 的 Δ，
#            需再补 2 个中南基线 run（本队列不含，用户未批）。
#
# 已完成的 job 直接跳过（epoch_199.pth 存在即视为完成）；目录存在但未完成则归档后重跑。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvartail_r50_3seed_zn.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvartail_r50_3seed_zn.ps1
#   ... -Only R50CVAR_s2024                    只跑指定 job
#   ... -IgnoreOtherQueues                     不等待其它 queue_*.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [switch]$IgnoreOtherQueues,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerCvar3 -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerCvar3]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$selfName = "queue_c2d_cvartail_r50_3seed_zn.ps1"

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

$statusPath = Join-Path $logDir "queue_c2d_cvartail_r50_3seed_zn_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$base = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224"

$jobs = @(
    @{ n = "R50CVAR_s2024"; tag = "r50_cvar_s2024"; subset = "SPARC"; seed = "2024"
       snap = "${base}_s2024_SPARC_lcvartail_W0.1Q0.1A0.05_c2d" },
    @{ n = "R50CVAR_s3407"; tag = "r50_cvar_s3407"; subset = "SPARC"; seed = "3407"
       snap = "${base}_s3407_SPARC_lcvartail_W0.1Q0.1A0.05_c2d" },
    @{ n = "ZN-CVAR_s1234"; tag = "zn_cvar_s1234"; subset = "Zhongnan"; seed = "1234"
       snap = "${base}_Zhongnan_lcvartail_W0.1Q0.1A0.05_c2d" },
    @{ n = "ZN-CVAR_s2024"; tag = "zn_cvar_s2024"; subset = "Zhongnan"; seed = "2024"
       snap = "${base}_s2024_Zhongnan_lcvartail_W0.1Q0.1A0.05_c2d" },
    @{ n = "ZN-CVAR_s3407"; tag = "zn_cvar_s3407"; subset = "Zhongnan"; seed = "3407"
       snap = "${base}_s3407_Zhongnan_lcvartail_W0.1Q0.1A0.05_c2d" }
)

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n.ToUpper() }
    if (-not $jobs) { Write-Host "[cvar3zn] -Only '$Only' matched nothing"; exit 1 }
}

foreach ($job in $jobs) {
    $snapshot = Join-Path $modelDir $job.snap
    $done = Test-Path (Join-Path $snapshot "epoch_199.pth")

    if ($done -and -not $DryRun) {
        Write-Host "[cvar3zn][$($job.n)] already complete, reuse: $snapshot"
        continue
    }

    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
    while (-not $DryRun) {
        $train = Get-TrainingProcess
        $other = @(Get-OtherQueue)
        if (-not $train -and $other.Count -eq 0) { break }
        $why = if ($train) { "train.py running" } else { "other queue waiting/running" }
        Write-Host "[cvar3zn][$($job.n)] $why, wait 60s..."
        Start-Sleep -Seconds 60
    }

    $argsList = @("train.py", "--dataset", "Carotid", "--subset", $job.subset,
                  "--seed", $job.seed, "--batch_size", "4", "--grad_accum", "2",
                  "--max_epochs", "200", "--eval_test_every", "5",
                  "--arch", "baseline", "--vit_layers", "0",
                  "--loss", "cvartail", "--tl_w", "0.1", "--tl_q", "0.1", "--tl_area_w", "0.05")
    Write-Host "[cvar3zn][$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[cvar3zn][$($job.n)] snapshot: $snapshot"
    if ($DryRun) { continue }

    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[cvar3zn][$($job.n)] archived incomplete run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
    Write-Host "[cvar3zn][$($job.n)] log: $jobLog"

    $start = Get-Date
    Push-Location $Root
    & $Python @argsList *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[cvar3zn][$($job.n)] end exit=$code ${minutes}min"
}
[Win32.PowerCvar3]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[cvar3zn] queue finished"
