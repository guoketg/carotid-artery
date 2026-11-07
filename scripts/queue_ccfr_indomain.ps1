# 第七队列：CCFR 在「同中心 in-domain」上的两个复现
#
# 背景（为什么要跑这两个）：
#   第六/六点五队列的 C1–C5 都在 SPARC 上，得到的读数是「flow 精修贡献 ≈ 0」
#   （C5 Stage1 90.85 → Stage2 90.83）。这个结论有三个数据侧的疑点：
#     1) SPARC 只有 101 个训练病例（3.5 张/例，同一斑块相邻帧），有效样本多样性低；
#     2) SPARC test 只有 43 例，±2 点以内靠切片数堆出来的"差异"不可判；
#     3) 论文公开可得的 Ar-PlaqSegm1 从未在本项目里跑过，缺一个能直接对表的数。
#   因此本队列只做「最有价值」的两件事，且都在各自数据集内部训练/测试：
#     Z1  Zhongnan 448/187（348/148 例，1.3 张/例）—— 独立病例最多，功效最高
#     A1  Ar-PlaqSegm1 316/24（论文公开数据集，scripts/prepare_arplaq.py 转换）
#   两者都用 C5 配置（论文保真版：预训练 R50 编码器 + 灰度复制 3 通道），
#   即目前唯一被证实有效的骨干；w/o flow 的读数来自同一 job 的 stage1_curve.csv，不另跑。
#
# 口径（与 C5 完全一致，只换 --subset）：
#   Stage1 200 ep（ep200 读数），Stage2 20 ep / K=5，bs4 + ga2，224²，seed 1234。
#
# 快照名（train.py 自动拼）：
#   Z1  ${pre}_Zhongnan_ccfr_pr50_c2d
#   A1  ${pre}_ArPlaqSegm1_ccfr_pr50_c2d
#
# 读数限制（必须写进结论）：
#   - Ar-PlaqSegm1 原始 541 对里有大量空掩膜（实测 train 177/493、test 24/48），
#     按"有斑块"过滤后只剩 316/24，测试集 24 张，只能做点估计，不能做显著性判断；
#     且该数据集斑块极小（占图中位 1.5%，224² 上约 600 像素），与 SPARC/Zhongnan 的 15–20% 完全不同量级。
#   - 该子集 nii 头 spacing 记 1.0 mm（公开图统一缩放到 800×800，真实 mm/px 未知），
#     HD95/ASSD 数值单位是像素，只能子集内部比，不能和论文的 mm 对表。
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr_indomain.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_ccfr_indomain.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilCCFRID" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$common = @("train.py", "--arch", "ccfr", "--dataset", "Carotid",
            "--batch_size", "4", "--grad_accum", "2", "--max_epochs", "200",
            "--eval_test_every", "5", "--list_dir", "./lists/lists_Carotid2D",
            "--num_workers", "4", "--img_size", "224", "--ccfr_pretrain", "1")
$tail = @("--ccfr_stage", "both", "--seed", "1234",
          "--stage1_epochs", "200", "--stage1_pick", "200",
          "--stage2_epochs", "20", "--flow_steps", "5")

$jobs = @(
    @{ name = "Z1_Zhongnan_pr50";    args = $common + @("--subset", "Zhongnan") + $tail;
       snap = "${pre}_Zhongnan_ccfr_pr50_c2d" },
    @{ name = "A1_ArPlaqSegm1_pr50"; args = $common + @("--subset", "ArPlaqSegm1") + $tail;
       snap = "${pre}_ArPlaqSegm1_ccfr_pr50_c2d" }
)

$statusPath = Join-Path $logDir "queue_ccfr_indomain_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

Write-Host "[ccfr-in] $($jobs.Count) jobs, serial"
foreach ($j in $jobs) { Write-Host ("  {0,-20} -> {1}" -f $j.name, $j.snap) }
if ($DryRun) {
    Write-Host "[ccfr-in] dry-run only; train.py running: $((Get-TrainingProcess | Measure-Object).Count) proc"
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "[ccfr-in] queue finished"
    exit 0
}

while (Get-TrainingProcess) {
    Write-Host "[ccfr-in] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

foreach ($j in $jobs) {
    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[ccfr-in] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.name)_$ts.log"
    Write-Host "[ccfr-in] ===== start $($j.name) ====="
    Write-Host "[ccfr-in] log: $jobLog"
    Write-Host "[ccfr-in] cmd: $Python $($j.args -join ' ')"

    $start = Get-Date
    Push-Location $Root
    & $Python @($j.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[ccfr-in] ===== done $($j.name) exit=$code ($mins min) ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[ccfr-in] queue finished"
