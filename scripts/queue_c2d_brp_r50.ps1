# 2026-11-18 编码器分带重投影（BRP）在 R50 上的位置消融：用户直接指派「ResNet 先排上队 + 深层/浅层消融」。
#
# 为什么这个模块值得单独排队（而不是再试一个门控回注）
#   R50 + DecoderCup（A0 = 90.4903 / 0.14473 / HD 0.7062）上，注入式编码器改写**全部判负**：
#     FGR 编码器四级 −1.64（§7(5)）、FDFUNet 残差频域支 +0.03（§5(5)）、多尺度融合 −0.37（§5(6)）、
#     FEM / R-GEF / MREF / ABF / FreqFusion 也全负（§5(2)(7)-(11)）。
#   而同一批机制挂在**解码器**上是正的：DDBE 四级 Haar +0.435 / HD 0.6292、单尺度拉普拉斯
#     +0.506 / HD 0.6581（§5(4)），且「去分解只留拼接」的 plain 对照仅 +0.060 → 增益来自频率内容而非拼接。
#   位置研究结论（§7(4)）：位置本身不是主变量（enc 88.62 ≈ dec 88.25），差别在「主链改写 vs skip 副本」。
#   所以这次要分辨的是：编码器位置是**被形态**（门控回注 whole 核）否掉的，还是被「编码器」本身否掉的。
#   BRP 只搬形态：分带 → 拼接 → 1×1 重投影 → 零初始化残差，4C²+28C/级。
#
# 五臂（单 seed 1234，SPARC 官方划分 n=160，200 epoch 终点）：
#   臂 1 浅层两级（112²/64 + 55²/256，+287,488）：用户要求的「浅层」端
#   臂 2 深层两级（28²/512 + 14²/1024，+5,285,888）：用户要求的「深层」端
#   臂 3 四级全挂（+5,573,376）：与 FGR 编码器位置（+11.05M）同宿主对照，看改写面而非容量
#   臂 4 浅层两级 + 跨级门（浅级门 → 深级门单向，只传门不传内容，+2 个标量）：
#        设计主张臂；只传门是为了规避 EFF 那种跨尺度低频内容搬运（R50 上 −1.14，§5(13)）
#   臂 5 浅层两级 + 随机频带对照（同参数、零新增参数）：排除「多出来的 4C² 容量」这一解释
#
# 判定（docs/experiments.md §0 口径，跑前登记，对 A0 90.4903 / 0.14473 / 0.7062）：
#   过线 = |ΔDice| 与方向一致且 ΔDice ≥ +0.5，或 ΔHD ≤ −0.1 mm；
#   臂 3 另与 FGR 编码器位置 88.85 / 0.1793 / HD 0.8234（同宿主同 seed）对照：
#     若 臂 3 明显高于 88.85，则「编码器位置被形态否掉」成立，改写形态可继续做；
#     若 臂 3 ≈ 或低于 88.85，则结论是「编码器主链改写这条路在 R50 上关掉」，不再排同类臂。
#   浅 vs 深（臂 1 对 臂 2）是本次主读：浅层为正、深层为负 → 位置问题；
#     两端同为负 → 形态问题，停该线；两端同为正 → 深或浅的容量效应交给臂 5 判。
#   单 seed 过线才按 42/317/1029 三 seed 复核（须用户同意）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_brp_r50.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_brp_r50.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_brp_r50.ps1 -Only R50-BRP-浅层两级
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string[]]$Only = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerBrp -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerBrp]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：本队列 = 26（低于 FGR+cvartail 三 seed 那条 30，高于 distloss 的 20），见 queue_priority.ps1
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 26

$statusPath = Join-Path $logDir "queue_c2d_brp_r50_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

# 基线口径：R50 + DecoderCup（--arch baseline 必须配 --vit_layers 0，否则是已封存的完整 TransUNet）
$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "baseline", "--vit_layers", "0"
)
$baseR50 = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC"

$jobs = @(
    @{ n = "R50-BRP-浅层两级"; tag = "brp12"
       a = $common + @("--brp", "1", "--brp_levels", "1,2")
       snap = "${baseR50}_brp12_c2d" },

    @{ n = "R50-BRP-深层两级"; tag = "brp34"
       a = $common + @("--brp", "1", "--brp_levels", "3,4")
       snap = "${baseR50}_brp34_c2d" },

    @{ n = "R50-BRP-四级全挂"; tag = "brp1234"
       a = $common + @("--brp", "1", "--brp_levels", "1,2,3,4")
       snap = "${baseR50}_brp1234_c2d" },

    @{ n = "R50-BRP-浅层加跨级门"; tag = "brp12g"
       a = $common + @("--brp", "1", "--brp_levels", "1,2", "--brp_gate", "1")
       snap = "${baseR50}_brp12g_c2d" },

    @{ n = "R50-BRP-浅层随机频带对照"; tag = "brp12ctl"
       a = $common + @("--brp", "1", "--brp_levels", "1,2", "--brp_control", "1")
       snap = "${baseR50}_brp12ctl_c2d" }
)

if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($s in $Only) { $sel += @($s -split '[,;\s]+' | Where-Object { $_ }) }
    $jobs = @($jobs | Where-Object { $sel -contains $_.n })
    if ($jobs.Count -eq 0) {
        Write-Host "[brp] -Only 未匹配任何臂：$($sel -join ',')"
        exit 1
    }
    Write-Host "[brp] resume mode, jobs: $(($jobs | ForEach-Object { $_.n }) -join ',')"
}

foreach ($job in $jobs) {
    # GPU 严格串行：等当前训练结束、且没有更高优先级的队列
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s... blocked by: $(Get-BlockReason)"
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
Write-Host "[brp] queue finished"
