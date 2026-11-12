# 2026-11-12 队列：UNet 弱骨干上的两条损失改进臂（筛选阶段，各 1 臂单变量）。
#
# 背景（docs/low-score-samples.md §5 + 讨论结论）：
#   误差集中在贴 GT 的 1 px 边界带（带深 0.13/0.095 mm，梯度仅 1.90x 全图均值），
#   高 HD 尾部来自大斑块单点越界；而 CNR 与 Dice 的 Spearman 相关为
#   0.142(p=0.074) / -0.024(p=0.741)，两库方向不一致，故不做"低 CNR 加权"。
#   改为两条与失效模式一一对应的单变量损失：
#     LFA  --loss farfp   : 0.5CE+0.5Dice+0.1*远处假阳性（对准 HD 尾部；
#                          该项在 D1 领域损失里只有 w=0.1 且被边界 CE 稀释，从未单独成臂）
#     LBC  --loss bndsoft : 边界带软标签 CE(eps=0.25, radius=1) + 硬 Dice
#                          （低可分辨带降低监督强度，而不是提高权重）
#   两条都不叠加 --domain_loss，保证与基线的单变量差异可归因。
#
# 对照：UNet 基线 86.55（..._SPARC_unet_c2d，口径见 docs/results.md）
# 判定（§0 统一协议，HD 用 medpy 口径）：|ΔDice| >= 0.5 且 HD 同向，终点 ep200，单 seed 1234。
# 筛选过线的胜者再单独搬到 R50 复验，不在本队列内。
#
# 冒烟：venv\Scripts\python.exe scripts\smoke_loss_unet.py
#   （eps=0 退化为 CE、边界带几何、eps=1 全带 -> ln2、远 FP 项行为、梯度有限）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_loss_unet.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_loss_unet.ps1
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

function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            $_.CommandLine -and
            ($_.CommandLine -like "*queue_nopre_ddbe_fgr*" -or
             $_.CommandLine -like "*queue_c2d_head_hr*")
        }
}

$statusPath = Join-Path $logDir "queue_c2d_loss_unet_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "LFA"; tag = "unet_lfarfp"
       a = @("--arch", "unet", "--loss", "farfp")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_lfarfp_unet_c2d" },
    @{ n = "LBC"; tag = "unet_lbndsoft"
       a = @("--arch", "unet", "--loss", "bndsoft")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_lbndsoft_unet_c2d" }
)

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s..."
        Start-Sleep -Seconds 60
    }

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
Write-Host "[loss/unet] queue finished"
