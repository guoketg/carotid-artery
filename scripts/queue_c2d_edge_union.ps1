# 2026-11-02 第四队列：只跑「有正向先验」的两条线（不跑分析型/负结果实验）
#
# 位置：接力在 queue_c2d_loss_baselines.ps1 之后（12GB 显存只允许一个 train.py）。
#
# 选这两条的唯一理由：它们是全项目里仅有的、在别的骨干上见过正向增益的空位。
#   E1/E2  A0 + 梯度边缘算子（U-Net 上 sobel +0.97 / laplacian +1.35；A0+edge(wavelet)=90.32 零增益，
#          但梯度算子从未在 A0 上测过 —— 单变量、便宜、期望值最高的未测空位）
#   E3     多中心联合训练：SPARC 350 + Zhongnan 448 = 798 张，训练内 test 仍是 SPARC test（160），
#          这样 ep200 读数可以直接和 A0 的 90.32 比；训完再用 eval_cross_center.py 补 Zhongnan test。
#
# 口径一律 §0：bs4 + ga2 / 200 ep / seed 1234 / ep200 读数 / 224×224 / ImageNet21k 预训练。
#
# 快照名（train.py 自动拼）：
#   E1  ${pre}_SPARC_edge_eosobel_c2d
#   E2  ${pre}_SPARC_edge_eolaplacian_c2d
#   E3  ${pre}_SPARC_mcUnion_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_union.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_edge_union.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilEdgeUnion" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-PreviousQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_c2d_loss_baselines.ps1*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$common = @("train.py", "--subset", "SPARC", "--batch_size", "4", "--grad_accum", "2",
            "--max_epochs", "200", "--eval_test_every", "5")

$jobs = @(
    # 先跑期望值最高的：联合训练 798 张（train 798 行 = SPARC 350 + Zhongnan 448，test 仍是 SPARC 160）
    @{ name = "MC_union";   args = $common + @("--arch", "baseline",
                                              "--list_dir", "./lists/lists_Carotid2D_union");
       snap = "${pre}_SPARC_mcUnion_c2d" },
    @{ name = "EDGE_sobel"; args = $common + @("--arch", "baseline", "--edge_in", "1",
                                              "--edge_op", "sobel");
       snap = "${pre}_SPARC_edge_eosobel_c2d" },
    @{ name = "EDGE_lap";   args = $common + @("--arch", "baseline", "--edge_in", "1",
                                              "--edge_op", "laplacian");
       snap = "${pre}_SPARC_edge_eolaplacian_c2d" }
)

$statusPath = Join-Path $logDir "queue_c2d_edge_union_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

Write-Host "[edgeunion] $($jobs.Count) jobs, serial"
foreach ($j in $jobs) { Write-Host ("  {0,-12} -> {1}" -f $j.name, $j.snap) }
if ($DryRun) {
    Write-Host "[edgeunion] dry-run only, waiting for queue_c2d_loss_baselines: $((Get-PreviousQueue | Measure-Object).Count) proc"
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "[edgeunion] queue finished"
    exit 0
}

while (Get-PreviousQueue) {
    Write-Host "[edgeunion] waiting for queue_c2d_loss_baselines to finish (60s)..."
    Start-Sleep -Seconds 60
}
while (Get-TrainingProcess) {
    Write-Host "[edgeunion] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

foreach ($j in $jobs) {
    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[edgeunion] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.name)_$ts.log"
    Write-Host "[edgeunion] ===== start $($j.name) ====="
    Write-Host "[edgeunion] log: $jobLog"
    Write-Host "[edgeunion] cmd: $Python $($j.args -join ' ')"

    $start = Get-Date
    Push-Location $Root
    & $Python @($j.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[edgeunion] ===== done $($j.name) exit=$code ($mins min) ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[edgeunion] queue finished"
