# 2026-11-20 队列：R50 + DecoderCup 上 FGR(解码器四级) + RHD sym 的融合两臂（I6 / I2）
#
# 宿主口径（CLAUDE.md 第 18 / 20 条）：--arch baseline --vit_layers 0
#   = ImageNet21k 预训练 ResNet50 四级编码器 + DecoderCup，不含 ViT（vit_layers=0 时 Transformer 段一行不建）
#   配对基线 A0 = 90.4903 / 0.14473 / HD 0.70624（SPARC 官方划分 n=160，seed 1234，bs4+ga2，200 ep）
#
# 背景（用户 2026-11-20 直接指派「两个 I2 I6 都安排」）：
#   该宿主上 FGR 与 RHD 从未叠加过（快照目录零 run，scripts 里也没有同一行同时出现 --fgr 与 --loss rhd 的命令）。
#   两个加数在本宿主上的单臂读数：
#     FGR 解码器位 90.7955 / 0.13912 / HD 0.69732（Δ +0.305 / −0.009，未过线，results.md §5(16)）
#     RHD sym I6   90.6363 / 0.141 / HD 0.67325（Δ +0.146 / −0.033，不过线，§10.2a）
#     RHD sym I2   90.8209 / 0.139 / HD 0.65457（Δ +0.331 / −0.052，不过线，§10.2a）
#   注意 FGR 在 R50 上只能用解码器位：编码器位（--fgr_pos enc）在本宿主是 88.85 / HD 0.8234（−1.64 / +0.117）判负。
#
# 两臂（只差 --rhd_iters，其余逐字相同：fgr 1、rhd_term sym、rhd_w 0.5、seed 1234、bs4+ga2、ep200）：
#   FGR-RHD-I6 = --fgr 1 + --loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 6
#   FGR-RHD-I2 = --fgr 1 + --loss rhd --rhd_term sym --rhd_w 0.5 --rhd_iters 2
#
# 跑前登记的主判（两档一致）：对「同档 RHD 单臂」判可加性——ΔDice >= +0.5 且 ΔHD <= −0.1 mm。
#   I2 的机制预期要写死：224 网格上 iters=2 截断在 2 px ≈ 0.14 mm（列）/ 0.21 mm（行），
#   远小于本宿主 HD 0.65~0.71 mm，因此 I2 这一档大概率压不动最坏点，主用途是归因（FGR +0.305 能否无损相加）。
#   对 A0 的绝对读数照记，但不作为过线依据（叠加臂含两项改动，不能归因给单个模块）。
#
# 快照（train.py 自动拼 _fgr + _lrhd_TsymW<w>I<iters>）：
#   ..._vitL0_epo200_bs4_ga2_224_SPARC_fgr_lrhd_TsymW0.5I6_c2d
#   ..._vitL0_epo200_bs4_ga2_224_SPARC_fgr_lrhd_TsymW0.5I2_c2d
#
# 用法（先 -DryRun 核对命令行与快照名）：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fgr_rhd_r50.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fgr_rhd_r50.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_fgr_rhd_r50.ps1 -Only FGR-RHD-I2
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerFgrRhd -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerFgrRhd]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：严格全序优先级（本队列 = 19，见 queue_priority.ps1；唯一优先级来源）
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 19

$statusPath = Join-Path $logDir "queue_c2d_fgr_rhd_r50_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
# 结构项：FGR 解码器四级（默认位置，不加 --fgr_pos）
$struct = @("--arch", "baseline", "--vit_layers", "0", "--fgr", "1")
$rhdI6 = @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
$rhdI2 = @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "2")

$jobs = @(
    @{ n = "FGR-RHD-I6"; tag = "r50_fgr_rhd_i6"
       a = $struct + $rhdI6
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fgr_lrhd_TsymW0.5I6_c2d" },
    @{ n = "FGR-RHD-I2"; tag = "r50_fgr_rhd_i2"
       a = $struct + $rhdI2
       snap = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_fgr_lrhd_TsymW0.5I2_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[fgrrhd] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
}

foreach ($job in $jobs) {
    # 两道闸：① 队列全序（优先级 + PID）→ 只让最高优先级实例过关；
    # ② 一次性启动锁（CreateNew 原子）→ 堵住「判断通过」与「起 python」之间的竞态窗口，
    #    防止两个实例同时起 trainer 把显存挤爆。锁在本次训练期间持有，训练结束即释放。
    $lock = $null
    while (-not $DryRun) {
        $lock = Enter-LaunchLock -TimeoutSec 20
        if ((Get-TrainingProcess) -or (Get-OtherQueue)) {
            Exit-LaunchLock $lock; $lock = $null
            Write-Host "[$($job.n)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
            Start-Sleep -Seconds 60
            continue
        }
        break
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
    Exit-LaunchLock $lock
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.n)] end exit=$code ${minutes}min"
}
Write-Host "[fgrrhd] queue finished"
