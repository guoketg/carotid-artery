# 2026-11-07 队列（第二批）：R-GEF γ 修正 + F3 的 seed 复核。
#
# 背景读数（R50 宿主，--vit_layers 0，seed 1234）：
#   基线 A        90.4903 / ASSD 0.14473 / HD95 0.41634
#   F1 FreqFusion 90.5729（+0.083，不显著）
#   F2 DDBE only  90.9251（+0.435，未过 0.5）
#   F3 DDBE+lap   90.9960（+0.506，刚过线，单 seed 边界）
#   G1 R-GEF      90.4039（−0.086，不显著）
#
# G1 无效的直接原因（已查权重）：decoder.egf.*.gamma = −0.0044 / +0.0013 / −0.0005，
# 200 epoch 后 γ 几乎没离开 0 → skip' ≈ skip，模块实际处于关闭状态。
# 因此本队列第一个 job 只改 γ 初始化（0 → 0.1），其余口径完全不动。
#
# 后面 4 个 job 是 F3 的 seed 复核：为了 Δ 可比，基线 A 与 F3 都用 seed 2024/3407 各跑一遍。
#   （单 seed 噪声带 ±0.3，Δ=+0.506 落在边界，必须换 seed 看是否稳定。）
#
# 判定（§0）：|ΔDice| ≥ 0.5 或 HD95 降 ≥ 0.1 mm；同 seed 内比较。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_seedcheck.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_r50_seedcheck.ps1
#   ... -Only G2,B2024
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilR50Seed" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$base = @(
    "train.py", "--dataset", "Carotid", "--subset", "SPARC",
    "--vit_layers", "0",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC"
$f3 = @("--arch", "baseline", "--bdfr_v2", "1", "--bdfr_ddbe", "1", "--bdfr_acam", "1",
        "--bdfr_wave_op", "lap")

$jobs = @(
    # 1) γ 修正版 R-GEF：唯一改动是 ini_gamma 0 → 0.1
    @{ n = "G2"; tag = "r50_rgef_gm10";
       a = $base + @("--seed", "1234", "--arch", "baseline", "--edge_fusion", "rel", "--edge_gamma", "0.1");
       s = "${pre}_egrel_gm10_c2d" },

    # 2/3) 基线 A 的两个复核 seed（与 F3 配对，否则 Δ 不可比）
    @{ n = "B2024"; tag = "r50_base_s2024";
       a = $base + @("--seed", "2024", "--arch", "baseline");
       s = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_s2024_SPARC_c2d" },
    @{ n = "B3407"; tag = "r50_base_s3407";
       a = $base + @("--seed", "3407", "--arch", "baseline");
       s = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_s3407_SPARC_c2d" },

    # 4/5) F3（DDBE×4 + ACAM + 拉普拉斯）的 seed 复核
    @{ n = "F3s2024"; tag = "r50_ddbe_lap_s2024";
       a = $base + @("--seed", "2024") + $f3;
       s = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_s2024_SPARC_bdfr2_na_bwlap_c2d" },
    @{ n = "F3s3407"; tag = "r50_ddbe_lap_s3407";
       a = $base + @("--seed", "3407") + $f3;
       s = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_s3407_SPARC_bdfr2_na_bwlap_c2d" }
)

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n.ToUpper() }
    if (-not $jobs) { Write-Host "[seed] -Only '$Only' matched nothing"; exit 1 }
}

$statusPath = Join-Path $logDir "queue_c2d_r50_seedcheck_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while (-not $DryRun) {
        if (-not (Get-TrainingProcess)) { break }
        Write-Host "[seed][$($job.n)] another train.py is running, wait 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.s
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[seed][$($job.n)] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
    Write-Host "[seed][$($job.n)] $Python $($job.a -join ' ')"
    Write-Host "[seed][$($job.n)] snapshot: $($job.s)"
    Write-Host "[seed][$($job.n)] log: $jobLog"
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
    Write-Host "[seed][$($job.n)] end exit=$code ${minutes}min"
}
$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[seed] queue finished"
