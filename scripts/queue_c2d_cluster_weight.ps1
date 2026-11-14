# 2026-11-13 队列：聚类版样本加权（全训练集 k-means，k=2）。
#
# 分组口径（scripts/build_cluster_weights.py）：
#   特征 [rel_echo, cnr]，标准化参数与簇中心只在 train 上拟合，test 只 assign
#   低回声簇（簇中心 rel_echo 更小）权重 1.5，高回声簇 0.5，严格 3:1、均值 1
#   两种粒度：
#     cluster_slice  ：每张切片独立归簇
#     cluster_patient：同一患者切片取特征中位数归簇，整例共用权重
#   训练侧与 §8 完全一致（样本级 0.5CE+0.5Dice，batch 内归一化），
#   因此与 §8 是单变量对比：只换"权重表怎么来"。
#
# 离线分组统计（A0 基线逐切片指标，仅诊断）：
#   SPARC slice  ：低 243 切片(69%) 90.36 / 0.144 / 0.734 | 高 107 切片(31%) 90.90 / 0.146 / 0.617
#   SPARC patient：低 206 切片(59%) 90.44 / 0.137 / 0.666 | 高 144 切片(41%) 90.58 / 0.158 / 0.774  ← 方向不一致
#   ZN    slice  ：低 350 切片(78%) 91.89 / 0.125 / 0.589 | 高  98 切片(22%) 92.36 / 0.117 / 0.493
#   ZN    patient：低 342 切片(76%) 91.88 / 0.125 / 0.594 | 高 106 切片(24%) 92.38 / 0.117 / 0.479
#   → 主判是 slice 粒度（两库方向一致）；patient 粒度作为对照，SPARC 上已出现 HD 反向。
#
# 对照：A0 SPARC 90.4903 / 0.145 / 0.7062；A0 Zhongnan 92.0000 / 0.123 / 0.5650
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cluster_weight.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cluster_weight.ps1
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

$statusPath = Join-Path $logDir "queue_c2d_cluster_weight_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$jobs = @(
    @{ n = "CL-slice-SPARC"; subset = "SPARC"; mode = "cluster_slice"; tag = "r50_cl_slice_sparc"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lsw_swcluster_slice_c2d" },
    @{ n = "CL-slice-Zhongnan"; subset = "Zhongnan"; mode = "cluster_slice"; tag = "r50_cl_slice_zn"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lsw_swcluster_slice_c2d" },
    @{ n = "CL-patient-SPARC"; subset = "SPARC"; mode = "cluster_patient"; tag = "r50_cl_patient_sparc"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lsw_swcluster_patient_c2d" },
    @{ n = "CL-patient-Zhongnan"; subset = "Zhongnan"; mode = "cluster_patient"; tag = "r50_cl_patient_zn"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lsw_swcluster_patient_c2d" }
)

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[$($job.n)] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $argsList = @(
        "train.py", "--dataset", "Carotid", "--subset", $job.subset,
        "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
        "--max_epochs", "200", "--eval_test_every", "5",
        "--arch", "baseline", "--vit_layers", "0", "--n_skip", "3", "--img_size", "224",
        "--loss", "sw", "--sw_mode", $job.mode
    )
    $snapshot = Join-Path $modelDir $job.snap
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
Write-Host "[c2d cluster_weight] queue finished"
