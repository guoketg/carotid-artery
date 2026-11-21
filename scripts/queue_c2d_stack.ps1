# 2026-11-15 叠加批（stack）：把各自「单变量已过线 / 幅度最大」的结构臂与损失臂配到一起，2 臂单 seed。
#
# 选臂规则（本项目累积证据，见 docs/results.md）：
#   ① 两个单臂必须都在**同一宿主、同一 seed** 上各自过线或幅度最大；
#   ② 机制作用面不重叠（特征改写 vs 优化目标），避免 DDBE+FGR 那种「两个改写机器抢同一份频率内容」的互覆
#      （R50 上同点串联比两个单挂臂都差：§5(17) 相对 DDBE −0.293、相对 FGR −0.164）；
#   ③ 不做「结构 + 结构」：R50 上所有注入式融合（FreqFusion / FEM / R-GEF / MREF / ABF / FDFUNet 三模块）
#      全部判负（§5(2)(5)-(11)(13)-(15)），U-Net 上 DDBE 单挂本身也只有 −0.21（ddbe4_unet 86.34 / HD 1.089）。
#
# 臂 1（U-FGREnc-RHD，弱骨干，本次主臂）：
#   U-Net + FGR 编码器位置 + 区域 HD 损失（sym, I=6）
#   单臂读数（同宿主 SPARC seed 1234 配对）：FGREnc 88.62 / 0.176 / HD 0.93（+2.075 / −0.206，§7(5)）
#                                        RHDsym 88.21 / 0.181 / HD 0.88（+1.66 / −0.25，§10.1）
#   两者是「改特征」与「改优化目标」，无参数/内容重叠；且先例支持可加：FGREnc+域损失 89.07 = FGREnc +0.45，
#   与域损失单挂（D2 87.01，+0.46）几乎相等（§10.2）→ 损失项在 FGR 之上**没有被吃掉**。
#   若完全可加预期 ≈ 89.9~90.3 / HD 0.75 左右；若打平 FGREnc（≈88.6），说明两者修的是同一批误差。
#   同族参照：FGREnc+cvartail 3 seed 队列（queue_c2d_fgr_cvartail_3seed，cvartail 单挂 +0.82）——本臂用更强的
#   损失（RHD +1.66）检验同一问题，两者结果一起看。
#
# 臂 2（R50-BDFRlap-RHD，强骨干，论文主线）：
#   R50 vitL0 + BDFR-Net 忠实解码器（DDBE×4，skip 融合 concat，单尺度拉普拉斯）+ 区域 HD 损失（sym, I=6）
#   单臂读数（同宿主 SPARC seed 1234 配对 A0 90.4903 / 0.14473 / HD 0.7062）：
#     BDFR2-na-bwlap 90.9957 / 0.13577 / HD 0.65807（+0.505 / −0.048，§5(4)，跨数据集 Zhongnan 同向 +0.555 §7(1)）
#     全项目在 R50 上唯一 Dice 过线的结构臂，也是唯一跨中心方向一致的正信号。
#   臂 2 的损失参照随 RHD-I6（queue_c2d_rhd，2026-11-15 08:54 起跑）落盘后确定：
#     RHD-I6 过线（ΔHD ≤ −0.1 且 ΔDice ≥ −0.3）→ 本臂判「两项可加」；
#     RHD-I6 判负（R50 上像 cvartail/DL 一样归零）→ 本臂改判「结构改写之上，中性的距离损失是否仍有增量」，
#       仍是有效问题（特征统计被改写后，损失对距离剖面的约束可能才生效），不入废臂。
#
# 判定（docs/experiments.md §0，跑前登记）：
#   臂 1 主判 ΔDice ≥ +0.5（对 FGREnc 88.62）且 ΔHD ≤ −0.1 mm（对 0.93）；次判对 RHD 单挂 88.21 / 0.88。
#   臂 2 主判 ΔDice ≥ +0.5（对 90.9957）且 ΔHD ≤ −0.1 mm（对 0.65807）；次判对 A0 90.4903 / 0.7062。
#   过线的臂再按 3 seed（42/317/1029）复核报均值±sd；5-seed 集成不参与判定（只作内部诊断）。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_stack.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_stack.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_stack.ps1 -Only U-FGREnc-RHD
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string[]]$Only = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerStack -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerStack]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：严格全序优先级，本队列 = 5（最低，排在当日全部队列之后），见 queue_priority.ps1
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 5

$statusPath = Join-Path $logDir "queue_c2d_stack_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
# #82 主用式：LAHsym（两侧距离剖面按各自质量归一），I=6 覆盖本数据 HD 尾部（中位 5.4 px）
$rhd = @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")

$baseUnet = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC"
$baseR50 = "TU_pretrain_R50-ViT-B_16_skip3_vitL0_epo200_bs4_ga2_224_SPARC"

$jobs = @(
    @{ n = "U-FGREnc-RHD"; tag = "unet_fgre_rhdsym"
       a = $common + @("--arch", "unet", "--fgr", "1", "--fgr_pos", "enc") + $rhd
       snap = "${baseUnet}_fgre_lrhd_TsymW0.5I6_unet_c2d" },

    @{ n = "R50-BDFRlap-RHD"; tag = "r50_bdfr2na_bwlap_rhdsym"
       a = $common + @("--arch", "baseline", "--vit_layers", "0",
                       "--bdfr_v2", "1", "--bdfr_wave_op", "lap") + $rhd
       snap = "${baseR50}_bdfr2_na_bwlap_lrhd_TsymW0.5I6_c2d" }
)

# -Only <臂名列表>：只跑指定臂（接续/单人复核用；臂名见上面 n 字段）
# 注意：用 powershell -File 调用时 "A,B" 会作为**单个字符串**传入（-File 不做逗号分割），
# 故这里再按 , ; 空格 拆一次，两种调用方式都能用。
if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($s in $Only) { $sel += @($s -split '[,;\s]+' | Where-Object { $_ }) }
    $jobs = @($jobs | Where-Object { $sel -contains $_.n })
    if ($jobs.Count -eq 0) {
        Write-Host "[stack] -Only 未匹配任何臂：$($sel -join ',')"
        exit 1
    }
    Write-Host "[stack] resume mode, jobs: $(($jobs | ForEach-Object { $_.n }) -join ',')"
}

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
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
Write-Host "[stack] queue finished"
