# 【已收口，不重挂 · 2026-11-17 口径修订后重判】
#   本队列 5 臂跑在 `--arch baseline --vit_layers 0`（R50 四级编码器 + DecoderCup，`vit_layers=0` 不含 ViT）上。
#   **2026-11-17 用户确认：这正是默认的 ResNet50 对外口径**（基线 90.4903 / 0.14473 / HD 0.7062），
#   所以宿主写法本身没问题、train.py 也会放行（被拦的只是 `--vit_layers>=1` 的完整 TransUNet）。
#   本队列不重挂的理由是**已经全部收口**（2026-11-16 11:30，结果见 docs/results.md §10.3），不是封存宿主。
#   若确需重跑，请沿用此 arch 写法，并把日志/快照与既有读数分开。
#   规则见 CLAUDE.md 第 18/20 条。
#
# tl_w 剂量消融：R50 + DecoderCup（ImageNet21k 预训练 ResNet-50 + DecoderCup, vit_layers=0）上的 cvartail。
#
# 背景（2026-11-15）：
#   - U-Net 上 `--loss cvartail --tl_w 0.1` = 87.37 / 0.200 / HD 0.96（+0.82 / −0.18，双门槛过）
#   - R50 上逐字同参 = 90.50 / 0.142 / HD 0.65（+0.01 / −0.053，判负）；s2024 同向（−0.27 / −0.018）
#   - 逐例配对（scripts/analyze_cvar_paired.py，零训练）显示效应**只落在 A0 HD 最难的 Q4**
#     （s1234 −0.309、s2024 −0.164），Q1~Q3 反而 +0.02~+0.05 → 净额被抵消，不是"没作用"
#   - 量级探针（scripts/probe_cvar_magnitude.py，零训练）在终点 checkpoint 上测得
#     λ=0.1 时尾部项占主损失梯度的份额：UNet-A0 34.9%、R50-A0 19.2%（rho 3.49 vs 1.92），
#     即**同一 λ 在 R50 上前期只拿到约一半剂量**；按份额对齐 R50 需要 λ≈0.18~0.2。
#
# 本批四臂（全部 R50 vitL0 / SPARC / seed 1234 / bs4 ga2 / ep200 / 单变量，只动 tl_w 或 tl_q）：
#   R50-cvar-w0.05  λ=0.05  向下探，确认单调性（0.1 已近似等于 0 效应）
#   R50-cvar-w0.2   λ=0.20  探针反解的对齐点（与 U-Net 成功臂同梯度份额）
#   R50-cvar-w0.4   λ=0.40  2× 对齐点
#   R50-cvar-w1.0   λ=1.00  5×，测剂量上限与 Dice 是否开始掉
#   R50-cvar-q0.02  λ=0.20 + q=0.02  选择器臂：探针显示 q=0.1 选中的像素里只有 20% 是真误差
#                                （k=5018 px 远大于误差带），q=0.02 提到 47~59%
#
# 判定：主判 ΔHD ≤ −0.1 mm（配对 A0 90.4903 / 0.14473 / HD 0.7062）；护栏 ΔDice ≥ −0.3。
#   若 λ 从 0.2 加到 1.0 仍不过线且 Dice 单调下滑 → "剂量不是瓶颈"成立，问题在选择器/误差类型；
#   若某 λ 过线 → 收口到该 λ，再做 3 seed（42/317/1029，须先经用户同意）。
#
# 成本：R50 单臂约 45–50 min，五臂合计约 4 h。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvar_lambda_r50.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvar_lambda_r50.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cvar_lambda_r50.ps1 -Only W0.2,Q0.02
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string[]]$Only = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerCvarLam -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
[Win32.PowerCvarLam]::SetThreadExecutionState([uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：优先级 48（在 queue_c2d_rhd 50 之后、lossv2_unet 45 之前，用户直接指派）
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 48

$statusPath = Join-Path $logDir "queue_c2d_cvar_lambda_r50_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "baseline", "--vit_layers", "0",
    "--loss", "cvartail", "--tl_area_w", "0.05"
)
$baseR50 = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC"

$jobs = @(
    @{ n = "W0.05"; tag = "r50_cvar_w0.05"
       a = $common + @("--tl_w", "0.05", "--tl_q", "0.1")
       snap = "${baseR50}_lcvartail_W0.05Q0.1A0.05_c2d" },

    @{ n = "W0.2"; tag = "r50_cvar_w0.2"
       a = $common + @("--tl_w", "0.2", "--tl_q", "0.1")
       snap = "${baseR50}_lcvartail_W0.2Q0.1A0.05_c2d" },

    @{ n = "W0.4"; tag = "r50_cvar_w0.4"
       a = $common + @("--tl_w", "0.4", "--tl_q", "0.1")
       snap = "${baseR50}_lcvartail_W0.4Q0.1A0.05_c2d" },

    @{ n = "W1.0"; tag = "r50_cvar_w1.0"
       a = $common + @("--tl_w", "1.0", "--tl_q", "0.1")
       snap = "${baseR50}_lcvartail_W1.0Q0.1A0.05_c2d" },

    @{ n = "Q0.02"; tag = "r50_cvar_w0.2_q0.02"
       a = $common + @("--tl_w", "0.2", "--tl_q", "0.02")
       snap = "${baseR50}_lcvartail_W0.2Q0.02A0.05_c2d" }
)

if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($s in $Only) { $sel += @($s -split '[,;\s]+' | Where-Object { $_ }) }
    $jobs = @($jobs | Where-Object { $sel -contains $_.n })
    if ($jobs.Count -eq 0) {
        Write-Host "[cvar-lambda] -Only 未匹配任何臂：$($sel -join ',')"
        exit 1
    }
    Write-Host "[cvar-lambda] resume mode, jobs: $(($jobs | ForEach-Object { $_.n }) -join ',')"
}

foreach ($job in $jobs) {
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.snap
    $argsList = @("train.py") + $job.a
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
Write-Host "[cvar-lambda] queue finished"
