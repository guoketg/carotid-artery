# 2026-11-16 纯 ResNet50 骨干上的损失横评（12 臂单 seed，串行）
#
# 骨干口径（与「R50-TransUNet」严格区分）：
#   --arch wmren --wmren_ablate unet_dec --wmren_enc r50
#     = ResNet50 四级（112²/56²/28²/14²，ImageNet21k 预训练）直连普通 UNet 解码器；
#       不含 ViT、不含 DecoderCup。参数量 31.00M。
#   与此前所有「--arch baseline --vit_layers 0」的 R50 臂不是同一个宿主，读数不得跨表比。
#
# 宿主基线（2026-11-08 已有，同协议 / 同 seed / 同解码器，本队列不重跑）：
#   R50U = 89.6309 / ASSD 0.16380 / HD 0.82617
#   （快照 TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_wmren_abunetdec_encr50_c2d）
#   该线上此前只测过 WRMM 消融，**损失函数覆盖 = 0/12**。
#
# 判定：ΔDice ≥ +0.5 或 ΔHD ≤ −0.1 mm（单 seed 配对，参照上面 R50U）。
# 目的是回答：「距离/边界/一致性/加权类损失在纯 ResNet 骨干上还剩几条有效」，
# 而不是把 U-Net 的结论外推过来（骨干强弱依赖已被反复证实）。
#
# 13 臂（只动损失，骨干/超参/seed 逐字不动；第 13 臂为 2026-11-17 追加）：
#   1  RES-cvar     cvartail：CVaR 距离尾部 + 面积一致（U-Net +0.82 / R50T −0.16）
#   1b RES-cvar-w0.2 cvartail 剂量推到探针反解的对齐点 λ=0.2（同一宿主上唯一微正的一档；
#                    R50T 上是 +0.11 / −0.011，此次为纯 R50 的单变量补齐，用户 2026-11-17 指派）
#   2  RES-rhd6     RHD sym I6：两侧距离剖面均值（U-Net +1.66，最强）
#   3  RES-rhd2     RHD sym I2：同上，论文默认 iters
#   4  RES-rhdh     RHD h I6：soft-max 压最坏一点（U-Net +0.64）
#   5  RES-rhdah    RHD ah I6：两侧先 alpha-LogSumExp 再取最大（从未跑过的新成员）
#   6  RES-bnd      边界图加权（Kervadec，U-Net +0.77）
#   7  RES-lfa      远处假阳性（U-Net −0.28）
#   8  RES-lbc      边界带软标签（U-Net +0.16）
#   9  RES-dl       域损失组合：0.5·边界CE + 0.5·Dice + 面积 + 远FP（R50T −0.06）
#   10 RES-cons     斑点扰动一致性（U-Net +1.39，零参数）
#   11 RES-swecho   样本加权·低回声簇（R50T +0.08）
#   12 RES-swcnr    样本加权·低对比度（R50T +0.08）
#
# 未纳入本批（需改代码，等 GPU 空档再排）：
#   --sdf aux（几何场回归）/ --ds 1（多尺度深监督）目前 parser 只放开 baseline 与 unet 两种解码器
#   （train.py L469-L472），接 unet_dec 要给 wmren 加辅助头，属于网络改动；且这两项在已有骨干上
#   分别只有 +0.36 / +0.33（U-Net）与 +0.18 / +0.02（R50T），优先级低于本批 12 臂。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_resnet_loss.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_resnet_loss.ps1 -Only RES-rhd6,RES-cons
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_resnet_loss.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string[]]$Only = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerResLoss -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerResLoss]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：严格全序优先级。本队列 = 8（高于叠加批的 5，故本批 12 臂先跑、叠加批在后），
# 见 scripts/queue_priority.ps1。注意：实际优先级一律由该脚本的优先级表推出，
# 这里手写的数字只作一致性自检；2026-11-17 的事故就是这里写 3、表里写 8 导致互锁。
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 8

$statusPath = Join-Path $logDir "queue_c2d_resnet_loss_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$base = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
# 纯 ResNet50 宿主：R50 四级直连 unet 解码器（无 ViT / 无 DecoderCup）
$u = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_enc", "r50")
$pre = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC"
$post = "_wmren_abunetdec_encr50_c2d"

$jobs = @(
    @{ n = "RES-cvar";   tag = "res_cvar";
       a = $base + $u + @("--loss", "cvartail", "--tl_w", "0.1", "--tl_q", "0.1", "--tl_area_w", "0.05");
       s = "${pre}_lcvartail_W0.1Q0.1A0.05${post}" },

    # 剂量对齐点：与 RES-cvar 只差 tl_w（0.1 → 0.2），其余（宿主/协议/seed/tl_q/tl_area_w）逐字相同
    @{ n = "RES-cvar-w0.2"; tag = "res_cvar_w0.2";
       a = $base + $u + @("--loss", "cvartail", "--tl_w", "0.2", "--tl_q", "0.1", "--tl_area_w", "0.05");
       s = "${pre}_lcvartail_W0.2Q0.1A0.05${post}" },

    @{ n = "RES-rhd6";   tag = "res_rhd_sym_i6";
       a = $base + $u + @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6");
       s = "${pre}_lrhd_TsymW0.5I6${post}" },

    @{ n = "RES-rhd2";   tag = "res_rhd_sym_i2";
       a = $base + $u + @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "2");
       s = "${pre}_lrhd_TsymW0.5I2${post}" },

    @{ n = "RES-rhdh";   tag = "res_rhd_h_i6";
       a = $base + $u + @("--loss", "rhd", "--rhd_term", "h", "--rhd_w", "0.5", "--rhd_iters", "6");
       s = "${pre}_lrhd_ThW0.5I6${post}" },

    @{ n = "RES-rhdah";  tag = "res_rhd_ah_i6";
       a = $base + $u + @("--loss", "rhd", "--rhd_term", "ah", "--rhd_w", "0.5", "--rhd_iters", "6");
       s = "${pre}_lrhd_TahW0.5I6${post}" },

    @{ n = "RES-bnd";    tag = "res_boundary";
       a = $base + $u + @("--loss", "boundary", "--bl_w", "5.0");
       s = "${pre}_lboundary${post}" },

    @{ n = "RES-lfa";    tag = "res_farfp";
       a = $base + $u + @("--loss", "farfp", "--far_w", "0.1");
       s = "${pre}_lfarfp${post}" },

    @{ n = "RES-lbc";    tag = "res_bndsoft";
       a = $base + $u + @("--loss", "bndsoft", "--bnd_eps", "0.25", "--bnd_radius", "1");
       s = "${pre}_lbndsoft${post}" },

    @{ n = "RES-dl";     tag = "res_domain_loss";
       a = $base + $u + @("--domain_loss", "1", "--dl_lambda", "3.0", "--dl_radius", "2",
                          "--dl_area_w", "0.05", "--dl_far_w", "0.1");
       s = "${pre}_dl${post}" },

    @{ n = "RES-cons";   tag = "res_cons";
       a = $base + $u + @("--cons_w", "0.2", "--cons_sigma", "0.08", "--cons_gain", "0.1",
                          "--cons_warmup", "20");
       s = "${pre}${post}_cons_w0.2_s0.08" },

    @{ n = "RES-swecho"; tag = "res_sw_echo";
       a = $base + $u + @("--loss", "sw", "--sw_mode", "echo");
       s = "${pre}_lsw_swecho${post}" },

    @{ n = "RES-swcnr";  tag = "res_sw_cnr";
       a = $base + $u + @("--loss", "sw", "--sw_mode", "cnr");
       s = "${pre}_lsw_swcnr${post}" }
)

# -Only <臂名列表>：只跑指定臂（接续用）。-File 调用时 "A,B" 会作为单个字符串传入，故再拆一次。
if ($Only.Count -gt 0) {
    $sel = @()
    foreach ($s in $Only) { $sel += @($s -split '[,;\s]+' | Where-Object { $_ }) }
    $jobs = @($jobs | Where-Object { $sel -contains $_.n })
    if ($jobs.Count -eq 0) {
        Write-Host "[resnet-loss] -Only 未匹配任何臂：$($sel -join ',')"
        exit 1
    }
    Write-Host "[resnet-loss] resume mode, jobs: $(($jobs | ForEach-Object { $_.n }) -join ',')"
}

foreach ($job in $jobs) {
    # 等 GPU 与其它队列都空出来再开跑，绝不与其它训练抢卡
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
Write-Host "[resnet-loss] queue finished"
