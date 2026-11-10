# 2026-11-07 队列（第三批）：WRMM 归因向 R50 编码器外推 + 拉普拉斯/去门控组合臂。
#
# 背景（seed 1234，SPARC 官方划分，200ep / bs4+ga2）：
#   K2 宿主 = WRMM 编码器（从零训）+ unet_dec：87.7742 / HD95 0.6060
#     MLKM 值 +1.006（去 MLKM 掉 1.01）
#     门控值 −0.452（去掉门控反而涨 0.45）
#     小波支值 +0.336（去掉掉 0.34，不显著）
#     拉普拉斯替换小波：尚未在 K2 上跑（本队列补）
#   R50 全段 + WRMM 精炼 + unet_dec：89.7390 / HD95 0.5051
#     但缺「R50 直连 unet_dec」基线，无法算 WRMM 精炼本身的值 → 本队列第 1 个 job 补上。
#
# 本队列 7 个 job（串行）：
#   R50U    R50（ImageNet21k）直连 unet_dec            = 表 5 缺失的基线
#   RW_nm   R50wrmm 宿主去 MLKM
#   RW_ng   R50wrmm 宿主去门控
#   RW_nw   R50wrmm 宿主去小波支
#   RW_lap  R50wrmm 宿主小波支换拉普拉斯
#   K2_cmb  WRMM 宿主：拉普拉斯 + 去门控（组合）
#   RW_cmb  R50wrmm 宿主：拉普拉斯 + 去门控（组合）
#
# 全部固定 unet_dec 解码器，保证与 K2 / R50-WRMM 已有读数同解码器、单变量可比。
# 预训练权重：只有 R50 家族（r50 / r50wrmm）能加载 ImageNet21k 的 conv_root+block1~3；
# WRMM 是论文自定义结构，公开 checkpoint 形状不匹配，只能从零训（不可「都预训练」）。
#
# 组合臂说明：「不用小波」= 小波支换成拉普拉斯（wave_op lap），
# 不是 no_wave（那会把左支整条去掉、与 lap 互斥，train.py 已禁止）。
#
# 判定（§0）：|ΔDice| ≥ 0.5 或 HD95 降 ≥ 0.1 mm；单 seed 噪声带 ±0.3。
#
# 前置：等 queue_c2d_r50_seedcheck（5 job）全部完成，避免抢卡。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_r50_extend.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_r50_extend.ps1
#   ... -Only RW_ng,K2_cmb
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

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilR50Ext" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$base = @(
    "train.py", "--dataset", "Carotid", "--subset", "SPARC", "--seed", "1234",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_wmren_abunetdec"
$u = @("--arch", "wmren", "--wmren_ablate", "unet_dec")

$jobs = @(
    @{ n = "R50U";   tag = "r50_unetdec_base";
       a = $base + $u + @("--wmren_enc", "r50");
       s = "${pre}_encr50_c2d" },
    @{ n = "RW_nm";  tag = "r50wrmm_no_mlkm";
       a = $base + $u + @("--wmren_enc", "r50wrmm", "--wmren_no_mlkm", "1");
       s = "${pre}_encr50wrmm_nm_c2d" },
    @{ n = "RW_ng";  tag = "r50wrmm_no_gate";
       a = $base + $u + @("--wmren_enc", "r50wrmm", "--wmren_no_gate", "1");
       s = "${pre}_encr50wrmm_ng_c2d" },
    @{ n = "RW_nw";  tag = "r50wrmm_no_wave";
       a = $base + $u + @("--wmren_enc", "r50wrmm", "--wmren_no_wave", "1");
       s = "${pre}_encr50wrmm_nw_c2d" },
    @{ n = "RW_lap"; tag = "r50wrmm_wave_lap";
       a = $base + $u + @("--wmren_enc", "r50wrmm", "--wmren_wave_op", "lap");
       s = "${pre}_encr50wrmm_wolap_c2d" },
    @{ n = "K2_cmb"; tag = "wrmm_lap_nogate";
       a = $base + $u + @("--wmren_wave_op", "lap", "--wmren_no_gate", "1");
       s = "${pre}_wolap_ng_c2d" },
    @{ n = "RW_cmb"; tag = "r50wrmm_lap_nogate";
       a = $base + $u + @("--wmren_enc", "r50wrmm", "--wmren_wave_op", "lap", "--wmren_no_gate", "1");
       s = "${pre}_encr50wrmm_wolap_ng_c2d" }
)

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n.ToUpper() }
    if (-not $jobs) { Write-Host "[ext] -Only '$Only' matched nothing"; exit 1 }
}

$statusPath = Join-Path $logDir "queue_c2d_wmren_r50_extend_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# 前置队列（seedcheck 5 job）完成标记
$seedStatusCsv = Join-Path $logDir "queue_c2d_r50_seedcheck_status.csv"
$runLog = Join-Path $logDir "queue_c2d_wmren_r50_extend_run.log"
Add-Content -Path $runLog -Value "$(Get-Date -Format 'HH:mm:ss') init: script=$PSCommandPath logDir=$logDir seedCsv=$seedStatusCsv"
function Test-PreQueueDone {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $rows = @(Get-Content -LiteralPath $Path | Select-Object -Skip 1 | Where-Object { $_ -ne "" })
    return ($rows.Count -ge 5)
}

foreach ($job in $jobs) {
    while (-not $DryRun) {
        $busy = Get-TrainingProcess
        if (-not $busy -and (Test-PreQueueDone -Path $seedStatusCsv)) { break }
        if ($busy) {
            Add-Content -Path $runLog -Value "$(Get-Date -Format 'HH:mm:ss') [$($job.n)] busy: train.py running, wait 60s"
        } else {
            $n = @(Get-Content -LiteralPath $seedStatusCsv | Select-Object -Skip 1 | Where-Object { $_ -ne "" }).Count
            Add-Content -Path $runLog -Value "$(Get-Date -Format 'HH:mm:ss') [$($job.n)] wait seedcheck: rows=$n/5, path=$seedStatusCsv"
        }
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.s
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[ext][$($job.n)] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
    Write-Host "[ext][$($job.n)] $Python $($job.a -join ' ')"
    Write-Host "[ext][$($job.n)] snapshot: $($job.s)"
    Write-Host "[ext][$($job.n)] log: $jobLog"
    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.a) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[ext][$($job.n)] end exit=$code ${minutes}min"
}
$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[ext] queue finished"
