# 2026-11-13 队列：样本级损失加权（回声分簇 / 边界对比度）两臂 × 两子集。
#
# 动机（老师提出的方案）：把斑块按回声聚成高低两类，低回声抬权、高回声降权；
# 边界与周围环境相似（对比度低）的切片抬权，明显的降权。
# 离线先验（logs/analysis/error_slices_*.csv，基线预测 + 逐切片指标）：
#   回声簇：SPARC 低 90.2/0.144/0.741 vs 高 91.0/0.146/0.646；Zhongnan 低 92.1/0.117/0.585
#           vs 高 91.8/0.132/0.534 —— 两库方向不一致，Dice 差 <0.01。
#   边界对比度（CNR 下/上 1/3）：SPARC 89.5/0.161/0.889 vs 90.9/0.145/0.624；
#           Zhongnan 92.2/0.117/0.586 vs 92.4/0.115/0.482 —— HD 两库同向，唯一有信号的项。
# 因此本队列两臂分开跑，直接看是哪一项在起作用（而非合成一个双臂都解释不清的结果）。
#
# 权重口径（scripts/build_sample_weights.py，参考分布只在 train 上拟合）：
#   指标 → train 内难度分位 u（越小越难 → u 越大）→ w = exp(γ(u−0.5))，γ = ln(3)
#   即最难的切片权重是最易的 3 倍，分档之间连续过渡；再归一化到 train 均值 1
#   echo：rel_echo 越小越难；cnr：cnr 越小越难
#   损失：--loss sw = 样本级 0.5CE+0.5Dice，w 在 batch 内归一化（w 全 1 时与 A0 数值一致）
#
# 对照臂（同 seed 1234 同协议，HD 为 medpy 口径，epoch_199.pth）：
#   A0 SPARC   Dice 90.4903 / ASSD 0.145 / HD 0.7062
#   A0 Zhongnan Dice 92.0000 / ASSD 0.123 / HD 0.5650
#
# 判定（§0）：ΔDice ≥ +0.5 且 ASSD / HD 同向不变差才算通过；
# 另报低对比度子组（cnr 下 1/3）的 Dice / HD，作为机制是否生效的旁证。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_sample_weight.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_sample_weight.ps1
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

$statusPath = Join-Path $logDir "queue_c2d_sample_weight_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$jobs = @(
    @{ n = "SW-echo-SPARC"; subset = "SPARC"; mode = "echo"; tag = "r50_sw_echo_sparc"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lsw_swecho_c2d" },
    @{ n = "SW-cnr-SPARC"; subset = "SPARC"; mode = "cnr"; tag = "r50_sw_cnr_sparc"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lsw_swcnr_c2d" },
    @{ n = "SW-echo-Zhongnan"; subset = "Zhongnan"; mode = "echo"; tag = "r50_sw_echo_zn"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lsw_swecho_c2d" },
    @{ n = "SW-cnr-Zhongnan"; subset = "Zhongnan"; mode = "cnr"; tag = "r50_sw_cnr_zn"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lsw_swcnr_c2d" }
)

foreach ($job in $jobs) {
    # 等 GPU 空出来再开跑，绝不与其它训练抢卡
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
Write-Host "[c2d sample_weight] queue finished"
