# 2026-11-20 队列：解码器 WRMM+lap（R50 编码器 + 普通 U-Net 解码器，单 seed 1234）
#
# 用户 2026-11-20 指派：「把 WRMM+lap 放在 ResNet+U-Net 的解码器上，SPARC / Zhongnan 各跑一条」。
#
# 动机：WRMM 此前只出现在编码器（wrmm 从零 / r50wrmm 四级同级精炼）。r50wrmm 的实测是
#   结构单挂 Haar 89.7390 / lap 89.6490，对 R50 基线 90.4903 为 −0.75 ~ −0.84，
#   「R50 已经编码的低频 + 小波支再分一次频 = 冗余」。
#   解码器上采样路径是 R50 完全没碰过的区域，本臂把 WRMM 镜像搬到解码器，
#   即「ResNet 占编码器、WRMM 占解码器」，绕开位置重叠。
#
# 宿主口径（CLAUDE.md 第 18 条：Δ 只对同宿主同 seed 同协议）
#   --arch wmren --wmren_ablate unet_dec --wmren_enc r50
#   = ImageNet21k 预训练 ResNet50 四级 + 普通 U-Net 解码器（cat+DBB），31.00M
#   配对基线（§13 表 13-A，11-08 已有）：89.6309 / 0.16380 / HD 0.82617
#
# 挂法（networks/unet_wmren.py 的 dec_wrmm）：
#   WRDM 走 stride=1 同级精炼，挂在每个解码级融合块输出之后：
#   28²/512 → 56²/256 → 112²/128 → 224²/64（与编码器四个 WRMM 镜像对称）
#   参数量：31.00M → 52.41M（+21.41M，主要在 28²/512 那一级的 14.9M）
#
# 判定（跑前登记）
#   主判：对本宿主 R50U 基线 89.6309 / 0.82617，ΔDice >= +0.5 且 ΔHD <= -0.1 mm。
#   归因提醒：本臂参数量 +69%，即便过线也不能直接记「解码器小波有效」。
#   若过线，下一步必须补两条同参数孪生才可下结论：
#     ① --wmren_dec_wrmm 1 + use_wave=False（同结构、同参数量、去掉小波支）；
#     ② --wmren_dec_wrmm 2（+1.29M 的浅两级版），看增益是否集中在浅层。
#   不过线则记「解码器侧小波精炼无效」，与 r50wrmm 编码器侧的负向合并成一条机制结论。
#   Zhongnan 格无同宿主结构参照，只报绝对读数 + 与 SPARC 的方向一致性。
#
# 快照名（train.py 自动拼）
#   ..._224_SPARC_wmren_abunetdec_wolap_encr50_dwrmm1_c2d
#   ..._224_Zhongnan_wmren_abunetdec_wolap_encr50_dwrmm1_c2d
#
# 用法：先 -DryRun 核对命令行与快照名
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wrmm_dec.ps1 -DryRun
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerWrmmDec -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerWrmmDec]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：严格全序优先级（本队列 = 19，见 queue_priority.ps1；唯一优先级来源）
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 19

$statusPath = Join-Path $logDir "queue_c2d_wrmm_dec_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5",
    "--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_enc", "r50",
    "--wmren_wave_op", "lap", "--wmren_dec_wrmm", "1"
)

$jobs = @(
    @{ n = "WRMM-Dec-SPARC"; tag = "wrmm_dec_sparc"; sub = "SPARC"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_wmren_abunetdec_wolap_encr50_dwrmm1_c2d" },
    @{ n = "WRMM-Dec-ZN"; tag = "wrmm_dec_zn"; sub = "Zhongnan"
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_wmren_abunetdec_wolap_encr50_dwrmm1_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[wrmmdec] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
}

foreach ($job in $jobs) {
    # 两道闸：① 队列全序（优先级 + PID）→ 只让最高优先级实例过关；
    # ② 一次性启动锁（CreateNew 原子）→ 堵住「判断通过」与「起 python」之间的竞态窗口。
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
    $argsList = @("train.py") + $common + @("--subset", $job.sub)
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
Write-Host "[wrmmdec] queue finished"
