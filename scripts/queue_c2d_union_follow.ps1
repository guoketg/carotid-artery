# 2026-11-02 第五队列：联合训练（798 张）的稳定性 + 与领域损失叠加
#
# 背景：第四队列给出全项目第一个过门槛的正向结果
#   A0 (350 张)            90.32 / HD 0.64 / |dA| 2.11   ← §0 基准
#   MC_union (798 张)      91.06 / HD 0.60 / |dA| 2.00   ← +0.74，过 |ΔDice|≥0.5 门槛
#   MC_union → Zhongnan    92.28（A0→Zhongnan 90.75，+1.53；D1→Zhongnan 91.18）
#
# 本队列只做两件为「涨点结论站得住」服务的事：
#   U1  UNION_s2024  联合训练换 seed 2024 —— +0.74 是否只是 seed 运气（§0 噪声带 ±0.3）
#   U2  UNION_dl     联合训练 + 领域损失（D1 口径）—— 两条正向杠杆能否叠加
#
# 口径一律 §0：bs4 + ga2 / 200 ep / ep200 读数 / 224×224 / ImageNet21k 预训练；
# train 798（SPARC 350 + Zhongnan 448），test 仍是 SPARC test 160，可与 A0 直接比。
#
# 快照名（train.py 自动拼）：
#   U1  ${pre}_s2024_SPARC_mcUnion_c2d
#   U2  ${pre}_SPARC_dl_mcUnion_c2d
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_union_follow.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_union_follow.ps1
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
$sleepUtil = Add-Type -MemberDefinition $sig -Name "SleepUtilUnionFollow" -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
$sleepUtil::SetThreadExecutionState($keepAwake) | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224"
$common = @("train.py", "--subset", "SPARC", "--batch_size", "4", "--grad_accum", "2",
            "--max_epochs", "200", "--eval_test_every", "5",
            "--list_dir", "./lists/lists_Carotid2D_union")

$jobs = @(
    @{ name = "UNION_s2024"; args = $common + @("--arch", "baseline", "--seed", "2024");
       snap = "${pre}_s2024_SPARC_mcUnion_c2d" },
    @{ name = "UNION_dl";    args = $common + @("--arch", "baseline", "--domain_loss", "1");
       snap = "${pre}_SPARC_dl_mcUnion_c2d" }
)

$statusPath = Join-Path $logDir "queue_c2d_union_follow_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

Write-Host "[unionfollow] $($jobs.Count) jobs, serial"
foreach ($j in $jobs) { Write-Host ("  {0,-12} -> {1}" -f $j.name, $j.snap) }
if ($DryRun) {
    Write-Host "[unionfollow] dry-run only; train.py running: $((Get-TrainingProcess | Measure-Object).Count) proc"
    $sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "[unionfollow] queue finished"
    exit 0
}

while (Get-TrainingProcess) {
    Write-Host "[unionfollow] another train.py is running, waiting 60s..."
    Start-Sleep -Seconds 60
}

foreach ($j in $jobs) {
    $snapshot = Join-Path $modelDir $j.snap
    if (Test-Path $snapshot) {
        $bak = "${snapshot}_bak_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $snapshot -Destination $bak
        Write-Host "[unionfollow] archived previous run to $bak"
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($j.name)_$ts.log"
    Write-Host "[unionfollow] ===== start $($j.name) ====="
    Write-Host "[unionfollow] log: $jobLog"
    Write-Host "[unionfollow] cmd: $Python $($j.args -join ' ')"

    $start = Get-Date
    Push-Location $Root
    & $Python @($j.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    $mins = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($j.name),$($start.ToString('yyyy-MM-dd HH:mm:ss')),$($end.ToString('yyyy-MM-dd HH:mm:ss')),$mins,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[unionfollow] ===== done $($j.name) exit=$code ($mins min) ====="
}

$sleepUtil::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-Host "[unionfollow] queue finished"
