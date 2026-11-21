# 【2026-11-18 用户指令解除停挂，已重新挂载并实跑】
#   原始停挂理由（2026-11-17），其中第 ① 条按现行口径更正：
#   ① 旧写法说「2 臂 CLW-echo-R-SPARC / CLW-echo-R-ZN 用 TransUNet 线、违反第 18 条」不成立：
#      这两臂是 --arch baseline --vit_layers 0，即 R50 + DecoderCup 默认宿主，现行第 18 条放行。
#   ② 设计属「数据集级聚类 + 固定权重表」，与 CLAUDE.md 第 19 条相冲突（该条为用户 2026-11-17
#      明确要求），同类 8 臂实测 ΔDice 全在 −0.04~+0.14（docs/results.md §8/§8.2）。
#   2026-11-18 用户明确要求「把它跑起来」→ 本次按 override 执行。执行理由：补齐 U-Net 弱骨干上
#      样本加权此前从未测过的缺口（§8/§8.2 的八臂全部在 R50 宿主上）。第 19 条作为通用规则保留。
#   首次真跑暴露的脚本缺陷：New-Arm 的 $arch 原为 [string]，数组被强转成单参数，四臂全 exit 2，
#      已改为 [string[]]。
#
# 2026-11-17 队列：按「斑块回声」分簇的样本加权，两数据集 × 两骨干，共 4 臂串行。
#
# 权重表（scripts/build_echo_cnr_groups.py 生成，规范版）：
#   echo = GT 内像素归一化灰度的**均值**（不减环带）；CNR 单独算，不再混进回声
#   分档 = train 中位数二分（<= 阈值为 low，> 阈值为 high），test 只查表
#   权重 = 低回声档 1.5 / 高回声档 0.5（ratio=3，两档各 50%，train 均值恰为 1）
#   表：logs\analysis\sample_weights_<Subset>_echo_group.csv
#
# 训练侧口径不变：--loss sw（样本级 0.5CE+0.5Dice，batch 内归一化），
# 与各自宿主基线的唯一差别就是这张权重表。
#
# 宿主基线（同协议 / 同 seed 1234 / bs4+ga2 / 200ep / SPARC 或 Zhongnan 全测试集）：
#   U-Net（from scratch）        SPARC 86.5455 / 0.22500 / HD 1.13618
#                                Zhongnan 90.0347 / 0.15421 / HD 0.71573
#   R50 + DecoderCup             SPARC 90.4903 / 0.14500 / HD 0.70620
#   （--arch baseline --vit_layers 0）  Zhongnan 92.0000 / 0.12300 / HD 0.56500
#
# 离线两档难度（A0 逐切片指标，仅诊断）：SPARC echo 低 90.50 / 高 90.48（差 0.02）；
#   Zhongnan echo 低 92.07 / 高 91.94（差 0.13）。即分档后两档难度几乎相同，
#   这组训练读数用于确认"权重表来源正确与否"对结果没有影响。
#
# 判定：ΔDice ≥ +0.5 且 HD 同向不变差（对各自宿主基线，单 seed）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cluster_weight_v2.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cluster_weight_v2.ps1 -Only CLW-echo-U-SPARC
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cluster_weight_v2.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string[]]$Only = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerClW -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerClW]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：优先级 7（低于 resnet_loss 8，高于 stack 5），以 queue_priority.ps1 的表为准。
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 7

$statusPath = Join-Path $logDir "queue_c2d_cluster_weight_v2_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

function New-Arm {
    # $arch 必须是 [string[]]：若写成 [string]，PowerShell 会把 @("--arch","baseline",
    # "--vit_layers","0") 强转成单参数 "--arch baseline --vit_layers 0"，argparse 报
    # "unrecognized arguments" 并 exit 2（2026-11-18 首次真跑时踩到，四臂全废）。
    param([string]$name, [string]$subset, [string[]]$arch, [string]$snap)
    $a = @(
        "--dataset", "Carotid", "--subset", $subset,
        "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
        "--max_epochs", "200", "--eval_test_every", "5"
    )
    $a += [string[]]$arch
    $a += @("--loss", "sw", "--sw_mode", "cluster_slice",
            "--sw_file", (Join-Path $Root ("logs\analysis\sample_weights_{0}_echo_group.csv" -f $subset)),
            "--tag", "echoG")
    return @{ n = $name; tag = ("clw_echo_" + $subset.ToLower()); a = $a; s = $snap }
}

$u = @("--arch", "unet")
$r = @("--arch", "baseline", "--vit_layers", "0")   # R50 编码器 + DecoderCup

$jobs = @(
    (New-Arm "CLW-echo-U-SPARC" "SPARC" $u "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_lsw_swcluster_slice_unet_echoG_c2d"),
    (New-Arm "CLW-echo-R-SPARC" "SPARC" $r "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC_lsw_swcluster_slice_echoG_c2d"),
    (New-Arm "CLW-echo-U-ZN"    "Zhongnan" $u "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_lsw_swcluster_slice_unet_echoG_c2d"),
    (New-Arm "CLW-echo-R-ZN"    "Zhongnan" $r "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_Zhongnan_lsw_swcluster_slice_echoG_c2d")
)

if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($s in $Only) { $sel += @($s -split '[,;\s]+' | Where-Object { $_ }) }
    $jobs = @($jobs | Where-Object { $sel -contains $_.n })
    if ($jobs.Count -eq 0) {
        Write-Host "[cluster-weight-v2] -Only 未匹配任何臂：$($sel -join ',')"
        exit 1
    }
    Write-Host "[cluster-weight-v2] resume mode, jobs: $(($jobs | ForEach-Object { $_.n }) -join ',')"
}

foreach ($job in $jobs) {
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] GPU or other queue busy, waiting 60s... blocked by: $(Get-BlockReason)"
        Start-Sleep -Seconds 60
    }

    $snapshot = Join-Path $modelDir $job.s
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
Write-Host "[cluster-weight-v2] queue finished"
