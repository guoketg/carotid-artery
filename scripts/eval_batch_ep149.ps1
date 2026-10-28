# 批量重算 ep150（epoch_149.pth）逐切片指标并做病例级 Wilcoxon 配对（experiments.md §1.6 口径）。
# 背景：trainer.py 末尾的 metrics.csv 是 ep199，与本项目的 ep150 判定点错配。
# 每个 job：eval_carotid.py 产出 <snapshot>\metrics_ep149.csv；变体 job 额外与对应 seed 的 A0 基线 compare。
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\eval_batch_ep149.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\eval_batch_ep149.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\eval_batch_ep149.ps1 -Skip dauto1234
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [string]$Skip = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Baseline 列 = 与该 job 配对比较的 A0 快照（空 = 自身就是基线，不 compare）
# 模型参数必须与 train.py 快照名严格对应（train.py L110-153 拼接规则）
$jobs = @(
    @{ Name = "b1234";      Snap = "${pre}_SPARC_c2d";                          Args = @(); Baseline = "" },
    @{ Name = "b2024";      Snap = "${pre}_s2024_SPARC_c2d";                    Args = @(); Baseline = "" },
    @{ Name = "b3407";      Snap = "${pre}_s3407_SPARC_c2d";                    Args = @(); Baseline = "" },
    @{ Name = "dauto1234";  Snap = "${pre}_SPARC_dual_L1234_auto_c2d";          Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto"); Baseline = "b1234" },
    @{ Name = "dauto2024";  Snap = "${pre}_s2024_SPARC_dual_L1234_auto_c2d";    Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto"); Baseline = "b2024" },
    @{ Name = "dauto3407";  Snap = "${pre}_s3407_SPARC_dual_L1234_auto_c2d";    Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto"); Baseline = "b3407" },
    @{ Name = "gate";       Snap = "${pre}_SPARC_dual_L1234_gate_c2d";          Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "gate"); Baseline = "b1234" },
    @{ Name = "deform";     Snap = "${pre}_SPARC_dual_L1234_deform_c2d";        Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "deform"); Baseline = "b1234" },
    @{ Name = "ddbe3";      Snap = "${pre}_SPARC_ddbe3_c2d";                    Args = @("--ddbe", "3"); Baseline = "b1234" },
    @{ Name = "edge";       Snap = "${pre}_SPARC_edge_c2d";                     Args = @("--edge_in", "1"); Baseline = "b1234" },
    @{ Name = "edgedual";   Snap = "${pre}_SPARC_edge_dual_L1234_auto_c2d";     Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto", "--edge_in", "1"); Baseline = "b1234" },
    # 批次 0+1（experiments.md §1.9，queue_c2d_repro.ps1）：ddbe4 / ddbefull / faff / mainll
    @{ Name = "ddbe4";      Snap = "${pre}_SPARC_ddbe4_c2d";                    Args = @("--ddbe", "4"); Baseline = "b1234" },
    @{ Name = "ddbefull";   Snap = "${pre}_SPARC_ddbe4_acam_c2d";               Args = @("--ddbe", "4", "--acam", "1"); Baseline = "b1234" },
    @{ Name = "faff";       Snap = "${pre}_SPARC_dual_L1234_aff_c2d";           Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "aff"); Baseline = "b1234" },
    @{ Name = "mainll";     Snap = "${pre}_SPARC_dual_L1234_auto_mainLL_c2d";   Args = @("--arch", "dual", "--levels", "1,2,3,4", "--fuse_op", "auto", "--main_input", "ll"); Baseline = "b1234" }
)

if ($Only -ne "") {
    $jobs = @($jobs | Where-Object { $_.Name -eq $Only })
    if ($jobs.Count -eq 0) { Write-Error "unknown job: $Only"; exit 2 }
}

if ($Skip -ne "") {
    $skipNames = @($Skip -split "," | ForEach-Object { $_.Trim() })
    $jobs = @($jobs | Where-Object { $skipNames -notcontains $_.Name })
    Write-Host "[eval149] skipping: $($skipNames -join ', ')"
}

$statusPath = Join-Path $logDir "eval_batch_ep149_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code" | Out-File -FilePath $statusPath -Encoding utf8
}

# 先解析基线 CSV 路径（基线 job 必须先跑或已有产物）
$snapOf = @{}
foreach ($j in $jobs) { $snapOf[$j.Name] = $j.Snap }

foreach ($job in $jobs) {
    $snap = Join-Path $modelDir $job.Snap
    $ckpt = Join-Path $snap "epoch_149.pth"
    $out = Join-Path $snap "metrics_ep149.csv"
    if (-not (Test-Path $ckpt)) {
        Write-Warning "[eval149] $($job.Name): missing $ckpt, skipped"
        continue
    }

    $args2 = @("scripts\eval_carotid.py", "--subset", "SPARC") + $job.Args + @("--ckpt", $ckpt, "--out", $out)
    if ($job.Baseline -ne "") {
        $baseSnap = $snapOf[$job.Baseline]
        $baseCsv = Join-Path (Join-Path $modelDir $baseSnap) "metrics_ep149.csv"
        if (Test-Path $baseCsv) {
            $args2 += @("--baseline_csv", $baseCsv)
        } else {
            Write-Warning "[eval149] $($job.Name): baseline csv missing ($baseCsv), compare skipped"
        }
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "eval149_$($job.Name)_$ts.log"
    Write-Host "[eval149] ===== $($job.Name) ====="
    Write-Host "[eval149] command: $Python $($args2 -join ' ')"

    if ($DryRun) { continue }

    $start = Get-Date
    Push-Location $Root
    & $Python @args2 *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.Name),$start,$end,$minutes,$code" | Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[eval149] ===== end $($job.Name) exit=$code ${minutes}min ====="
}

Write-Host "[eval149] all done, status: $statusPath"
