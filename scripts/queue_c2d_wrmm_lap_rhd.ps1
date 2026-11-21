# 2026-11-19 队列：WRMM 线「结构（WRDM 小波支换拉普拉斯）+ 损失（RHD）」四臂，单 seed 1234
#
# 用户 11-19 指派：WRMM 编码器 + lap + RHD，U-Net 侧与 ResNet 侧都要，且 SPARC / Zhongnan 两个训练集都要；
# RHD 的 iters 按宿主分：U-Net 侧（弱/从零骨干）= 6，ResNet 侧（预训练强骨干）= 2。
#
# 宿主与参照（CLAUDE.md 第 18 条：Δ 只对同宿主、同 seed、同协议）
#   臂 1/2 U-Net 侧 = --arch wmren --wmren_ablate unet_dec（WRMM 编码器随机初始化，72.17M，
#        解码器是普通 U-Net 解码器；即台账里的 K2）。结构单挂参照 88.7524 / 0.17097 / HD 0.81731
#        （WRMM 换 lap，§6 表 6-C，+0.978 / −0.162 对 Haar 87.7742 / 0.19230 / 0.97899）。
#   臂 3/4 ResNet 侧 = 同上再 --wmren_enc r50wrmm（R50 四级并联 WRMM 精修 + unet_dec）。
#        结构单挂参照 89.6490 / 0.15856 / HD 0.81286（RW_lap，表 6-B，对 Haar r50wrmm 89.7390 / 0.83206 为 −0.090 / −0.019）。
#
# 为什么这两个宿主上的 RHD 是真新实验：此前 RHD 在 wmren 架构上跑过的四条全在 --wmren_enc r50
# （R50 编码器、WRMM 全不建的 89.6309 控制臂），sym I2/I6 与 h/ah 四档全负（−0.07 ~ −0.36 Dice，HD 全在 ±0.06 内，
# 从没进过 results.md）。带 WRMM 的两个宿主（K2、r50wrmm）RHD 零 run。
#
# 判定（跑前登记，单 seed 只作筛选）
#   主判：对各自宿主的结构单挂臂（臂 1/2 对 88.7524 / 0.81731，臂 3/4 对 89.6490 / 0.81286），
#         要求 ΔDice >= +0.5 且 ΔHD <= -0.1 mm（RHD 在结构改写之上是否可加）。
#   次判：对同宿主 RHD 单挂（本次没有，故不作次判；若主判不过，结论只能是「不可加」，不能反推 RHD 无效）。
#   Zhongnan 训练臂的参照：Zhongnan 上同宿主无结构臂读数，只报绝对读数 + 与 SPARC 臂的方向一致性，不判门槛。
#
# 快照名（train.py 自动拼：损失段在前、arch/ablate/wo/enc 在后）
#   臂 1 ..._224_SPARC_lrhd_TsymW0.5I6_wmren_abunetdec_wolap_c2d
#   臂 2 ..._224_Zhongnan_lrhd_TsymW0.5I6_wmren_abunetdec_wolap_c2d
#   臂 3 ..._224_SPARC_lrhd_TsymW0.5I2_wmren_abunetdec_wolap_encr50wrmm_c2d
#   臂 4 ..._224_Zhongnan_lrhd_TsymW0.5I2_wmren_abunetdec_wolap_encr50wrmm_c2d
#
# 用法：先 -DryRun 核对；正式跑用 -Only 逐臂挂（不写 -Only 则四臂串行）
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name PowerWrmmLap -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.PowerWrmmLap]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 队列互斥：严格全序优先级（本队列 = 17，见 queue_priority.ps1；唯一优先级来源）
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 17

$statusPath = Join-Path $logDir "queue_c2d_wrmm_lap_rhd_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)
# 结构项：WRMM 编码器 + 普通 U-Net 解码器 + WRDM 左支换拉普拉斯
$struct = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_wave_op", "lap")
$rhdI6 = @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "6")
$rhdI2 = @("--loss", "rhd", "--rhd_term", "sym", "--rhd_w", "0.5", "--rhd_iters", "2")

$jobs = @(
    @{ n = "WRMM-K2-SPARC-I6"; tag = "wrmm_k2_lap_rhd_i6_sparc"; sub = "SPARC"
       a = $struct + $rhdI6
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_lrhd_TsymW0.5I6_wmren_abunetdec_wolap_c2d" },
    @{ n = "WRMM-K2-ZN-I6"; tag = "wrmm_k2_lap_rhd_i6_zn"; sub = "Zhongnan"
       a = $struct + $rhdI6
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_lrhd_TsymW0.5I6_wmren_abunetdec_wolap_c2d" },
    @{ n = "WRMM-R50W-SPARC-I2"; tag = "wrmm_r50w_lap_rhd_i2_sparc"; sub = "SPARC"
       a = $struct + @("--wmren_enc", "r50wrmm") + $rhdI2
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_lrhd_TsymW0.5I2_wmren_abunetdec_wolap_encr50wrmm_c2d" },
    @{ n = "WRMM-R50W-ZN-I2"; tag = "wrmm_r50w_lap_rhd_i2_zn"; sub = "Zhongnan"
       a = $struct + @("--wmren_enc", "r50wrmm") + $rhdI2
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_Zhongnan_lrhd_TsymW0.5I2_wmren_abunetdec_wolap_encr50wrmm_c2d" }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.n -or $keep -contains $_.tag })
    Write-Host "[wrmmlap] -Only 过滤后剩 $($jobs.Count) 臂：$(($jobs | ForEach-Object { $_.n }) -join ', ')"
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
    $argsList = @("train.py") + $common + @("--subset", $job.sub) + $job.a
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
Write-Host "[wrmmlap] queue finished"
