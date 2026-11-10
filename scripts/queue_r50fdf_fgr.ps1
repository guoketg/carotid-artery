# 2026-11-09 队列：A.7 组合协同三臂渐进 + FGR 新模块 + ddbectl 去分解对照。
#
# 背景（用户决策）：FDFUNet 的 A.7 减法消融（UNet* 88.97 / 完整 89.98 / no_drdac 89.85 /
# no_mfdf 89.95 / no_asca 89.81）显示单模块不承载增益，但减法口径排除不掉「组合协同」；
# 且此前 §5(9) FEM-on-R50 判负只测了 MFDF+ASCA、只放解码器、还包了 γ 残差——
# 口径与论文不同。本轮把三模块按论文口径渐进上 R50 验证协同：
#   A1 --r50_fdf drdac      : 编码器四级 + DRDAC（残差零初始化恒等起步，TF32 下
#                             pass-through conv 不精确，必须走 x + branch）
#   A2 --r50_fdf drdac_mfdf : + 编码器四级/解码器四级 FEM（MFDF only，论文口径 inline）
#   A3 --r50_fdf full       : FEM 开 ASCA（完整 FDFUNet 模块组）
#   FGR --fgr 1             : 自研频门改写（拉普拉斯频带 + FcaNet 式 DCT 谱门 +
#                             1×1 改写，fuse 零初始化恒等起步），挂解码器四级
#   CTL --ddbe 4 --ddbe_hf_op plain : DDBE 去分解对照（三支全吃普通特征，参数量与
#                             haar 逐位一致），归因「拼接改写 vs 频率内容」
#                             （results.md §7(3) 机制问题）
#
# 对照：R50 单骨干基线（--vit_layers 0）90.4903 / ASSD 0.14473 / HD 0.7062（seed 1234）
# 判定（§0 统一协议）：|ΔDice| ≥ 0.5 或 HD95 降 ≥ 0.1 mm，终点 ep200。
# 恒等起步冒烟（scripts/smoke_r50_fdf.py 2026-11-09）：CTL/A1/FGR 与基线 Δlogits==0，
# A2/A3 非恒等属 FEM inline 口径预期；参数量 CTL +9.26M / A1 +16.58M / A2 +21.69M /
# A3 +29.02M / FGR +0.69M。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_r50fdf_fgr.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_r50fdf_fgr.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$statusPath = Join-Path $logDir "queue_r50fdf_fgr_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "CTL"; tag = "ddbectl"
       a = @("--arch", "baseline", "--vit_layers", "0",
             "--ddbe", "4", "--ddbe_hf_op", "plain", "--acam", "0")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_ddbe4_ddhfplain_c2d" },
    @{ n = "A1"; tag = "r50drdac"
       a = @("--arch", "baseline", "--vit_layers", "0", "--r50_fdf", "drdac")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_r50fdf_dr_c2d" },
    @{ n = "A2"; tag = "r50mfdf"
       a = @("--arch", "baseline", "--vit_layers", "0", "--r50_fdf", "drdac_mfdf")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_r50fdf_mf_c2d" },
    @{ n = "A3"; tag = "r50full"
       a = @("--arch", "baseline", "--vit_layers", "0", "--r50_fdf", "full")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_r50fdf_full_c2d" },
    @{ n = "FGR"; tag = "fgr"
       a = @("--arch", "baseline", "--vit_layers", "0", "--fgr", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fgr_c2d" }
)

foreach ($job in $jobs) {
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
Write-Host "[r50fdf/fgr] queue finished"
