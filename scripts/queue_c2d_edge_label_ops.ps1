# 2026-11-18 队列：标签侧边缘算子批次（边缘监督分支），先 R50 后 U-Net。
#
# 用户指派（2026-11-18 晚）：
#   ① sobel 要作用在**标签**上（不是原图），先 ResNet 后 U-Net；
#   ② 再排 morph k=7——用户印象里"边缘标签比较粗、分割出来的也粗"，要验证粗带是否有用。
# 之前一版误排成 --edge_in 1 --edge_op sobel（原图高通通道），已撤且从未开跑。
#
# 四臂，只改边缘目标的两个自由度（配方 / 带宽），结构、权重、种子、协议逐字不动：
#   EB-SOBEL-R50 : --edge_mode sobel_orig（原尺寸 sobel → nearest 放大，带随放大倍数浮动）
#   EB-MORPH7-R50: --edge_mode morph224 --edge_kernel 7（224 上固定 3 px 内 + 3 px 外）
#   EB-SOBEL-U / EB-MORPH7-U : 同上，换 U-Net
#
# 带宽实测（SPARC test 分位抽样 3 例，GT 面积 19.4%）：
#   k=3  单边 1 px  边缘占比 2.44%（已跑过的骨架臂）
#   k=5  单边 2 px  4.69%
#   k=7  单边 3 px  6.90%（= k=3 的 2.83 倍）
#   k=9  单边 4 px  9.11%
#   k=11 单边 5 px 11.32%（与 sobel_orig 的 2.8%~13.5% 量级相当）
#
# 口径注意：224 网格上对标签做 3×3 Sobel 与 3×3 形态学梯度的非零点集完全相同，所以
#   sobel_orig 与 morph 的差别只是在"原尺寸算再放大（带宽浮动 2.8%~13.5%）"vs"224 上固定像素宽"，
#   不是算子族差异；梯度族 Sobel/Scharr/Prewitt/Roberts 在二值图上一律退化为同一件事。
#
# 对照（同 seed 1234、同协议，不重跑）：
#   UNet + morph224 k=3 边缘分支 86.65 / 0.216 / HD 1.08（对 U-Net 基线 +0.10）
#   R50  + morph224 k=3 边缘分支 90.41 / 0.144 / HD 0.69（对 R50 基线 −0.08）
#   两臂均在噪声内；容量混杂照旧（边缘臂多一个完整解码器，UNet +22.6M / R50 +7.4M）。
#
# 判定（§0）：单 seed，ΔDice >= +0.5 或 ΔHD <= -0.1 mm 才算过线。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_label_ops.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_label_ops.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 12

Add-Type -Namespace Win32 -Name Power4 -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
[Win32.Power4]::SetThreadExecutionState([uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)) | Out-Null

$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$statusPath = Join-Path $logDir "queue_c2d_edge_label_ops_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
$ebBase = @("--edge_branch", "1", "--edge_branch_w", "1.0")
$sobelArm = $ebBase + @("--edge_mode", "sobel_orig")
$morph7Arm = $ebBase + @("--edge_mode", "morph224", "--edge_kernel", "7")

$jobs = @(
    @{ n = "EB-SOBEL-R50";  tag = "r50_edge_label_sobel"
       a = @("--arch", "baseline", "--vit_layers", "0") + $sobelArm },
    @{ n = "EB-MORPH7-R50"; tag = "r50_edge_label_morph7"
       a = @("--arch", "baseline", "--vit_layers", "0") + $morph7Arm },
    @{ n = "EB-SOBEL-U";    tag = "unet_edge_label_sobel"
       a = @("--arch", "unet") + $sobelArm },
    @{ n = "EB-MORPH7-U";   tag = "unet_edge_label_morph7"
       a = @("--arch", "unet") + $morph7Arm }
)

foreach ($job in $jobs) {
    if ($Only -and $job.n -notlike "*$Only*") { continue }

    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
        Start-Sleep -Seconds 60
    }

    $argsList = @("train.py") + $common + $job.a
    Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
    if ($DryRun) { continue }

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
Write-Host "[edge_label_ops] queue finished"
