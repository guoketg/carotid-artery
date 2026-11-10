# 2026-11-10 队列：把头号信号 FGR 做实 —— unet+fgr 多 seed 复核 + Zhongnan 跨数据集单跑。
#
# 背景（用户决策）：§7(4) unet+fgr ΔDice +1.703 / ΔHD −0.168 双门槛全过（全项目最强
# 单模块正信号），但单 seed 1234；同批 unet+ddbe4+fgr +1.841 创新高。R50 上 DDBE+FGR
# 组合判负（同点互覆坐实），FGR 定为头号候选，本轮做实证据：
#   UFs2024 --arch unet --fgr 1 --seed 2024 : SPARC 多 seed（对照 s1234 88.2484）
#   UFs3407 --arch unet --fgr 1 --seed 3407
#   ZUF     --arch unet --fgr 1 --subset Zhongnan（seed 1234）: 跨数据集方向验证
#           （对照 Zhongnan U-Net 基线 results.md §8；DDBE-lap 在 Zhongnan 已有方向一致先例 +0.506）
# 判定：多 seed 行合并为均值±标准差一行输出（不得逐 seed 分行）；跨数据集只看方向。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_multiseed_zn.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_multiseed_zn.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.Power]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$statusPath = Join-Path $logDir "queue_fgr_multiseed_zn_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$jobs = @(
    @{ n = "UFs2024"; tag = "unetfgr_s2024"
       a = @("train.py", "--dataset", "Carotid", "--subset", "SPARC", "--seed", "2024",
             "--batch_size", "4", "--grad_accum", "2", "--max_epochs", "200", "--eval_test_every", "5",
             "--arch", "unet", "--fgr", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_s2024_SPARC_fgr_unet_c2d" },
    @{ n = "UFs3407"; tag = "unetfgr_s3407"
       a = @("train.py", "--dataset", "Carotid", "--subset", "SPARC", "--seed", "3407",
             "--batch_size", "4", "--grad_accum", "2", "--max_epochs", "200", "--eval_test_every", "5",
             "--arch", "unet", "--fgr", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_s3407_SPARC_fgr_unet_c2d" },
    @{ n = "ZUF"; tag = "zn_unetfgr"
       a = @("train.py", "--dataset", "Carotid", "--subset", "Zhongnan", "--seed", "1234",
             "--batch_size", "4", "--grad_accum", "2", "--max_epochs", "200", "--eval_test_every", "5",
             "--arch", "unet", "--fgr", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_fgr_unet_c2d" }
)

foreach ($job in $jobs) {
    # 等 GPU 空出来再开跑，绝不与其它训练抢卡
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[$($job.n)] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.snap
    $argsList = $job.a
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
Write-Host "[fgr multiseed/zn] queue finished"
