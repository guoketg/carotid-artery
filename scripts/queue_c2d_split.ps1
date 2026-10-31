# 2026-10-30 队列：拆支归因（A/B）+ U-Net 边缘算子横扫（C×5）+ FrCFusion（D）。
# 对应 docs/experiments.md §1.9b，8 个 job 串行，统一口径 $common：
#   SPARC / seed 1234 / bs4 + ga2（等效 bs8）/ 200 epoch / --eval_test_every 5
#
#   A cnn_only   --vit_layers 0           纯 R50 局部支（ViT 段清零，patch_emb 保留）
#   B vit_only   --vit_name ViT-B_16 --n_skip 0   纯 ViT 全局支（解码器无 skip）
#   C unet_edge  --arch unet --edge_in 1 --edge_op {wavelet,wavelet_lam,energy,sobel,laplacian}
#                同一个 U-Net（60.298 M），只换输入第二通道的边缘算子
#   D dual_frc   --arch dual --fuse_op frc   Syn-Net FrCFusion 四级全开
#
# 快照名（train.py L150-211 拼接规则）：
#   A → ${pre}_SPARC_c2d                （pre 里已含 _vitL0）
#   B → TU_pretrain_ViT-B_16_skip0_epo200_bs4_ga2_224_SPARC_c2d
#   C → ${pre}_SPARC_edge_unet_eo<op>_c2d
#   D → ${pre}_SPARC_dual_L1234_frc_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_split.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_split.ps1
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

# Prevent system sleep while training.
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilSplit" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"

$jobs = @(
    @{ Name = "A_cnn_only";  Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--vit_layers", "0") + $common;
       Snapshot = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_c2d" },
    @{ Name = "B_vit_only";  Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--vit_name", "ViT-B_16", "--n_skip", "0") + $common;
       Snapshot = "TU_pretrain_ViT-B_16_skip0_epo200_bs4_ga2_224_SPARC_c2d" },
    @{ Name = "C1_wavelet";  Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--edge_in", "1", "--edge_op", "wavelet") + $common;
       Snapshot = "${pre}_SPARC_edge_unet_eowavelet_c2d" },
    @{ Name = "C2_wavelet_lam"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--edge_in", "1", "--edge_op", "wavelet_lam") + $common;
       Snapshot = "${pre}_SPARC_edge_unet_eowavelet_lam_c2d" },
    @{ Name = "C3_energy";   Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--edge_in", "1", "--edge_op", "energy") + $common;
       Snapshot = "${pre}_SPARC_edge_unet_eoenergy_c2d" },
    @{ Name = "C4_sobel";    Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--edge_in", "1", "--edge_op", "sobel") + $common;
       Snapshot = "${pre}_SPARC_edge_unet_eosobel_c2d" },
    @{ Name = "C5_laplacian"; Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "unet", "--edge_in", "1", "--edge_op", "laplacian") + $common;
       Snapshot = "${pre}_SPARC_edge_unet_eolaplacian_c2d" },
    @{ Name = "D_dual_frc";  Args = @("train.py", "--subset", "SPARC", "--seed", "1234", "--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "frc") + $common;
       Snapshot = "${pre}_SPARC_dual_L1234_frc_c2d" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

$statusPath = Join-Path $logDir "queue_c2d_split_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    while ((Get-TrainingProcess) -and -not $DryRun) {
        Write-Host "[split] another train.py is running, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.Snapshot
    if ((Test-Path $snapshot) -and -not $DryRun) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[split] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.Name)_$ts.log"
    Write-Host "[split] ===== start $($job.Name) ====="
    Write-Host "[split] command: $Python $($job.Args -join ' ')"
    Write-Host "[split] snapshot: $($job.Snapshot)"
    Write-Host "[split] log: $jobLog"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @($job.Args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code,$jobLog" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[split] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[split] all jobs finished"
