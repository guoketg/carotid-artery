# 2026-11-19 队列：跨中心矩阵（纯推理，不训练）
#
# 用户 11-19 指派：「两个数据集 Zhongnan 以及 SPARC 都要做，跨数据集验证也要做」。
# 本脚本补两件事：
#   A. 基线四格（U-Net / ResNet50 × SPARC↔Zhongnan）里缺的三格。域内读数各训练快照自带，
#      缺的是 R50 的 SPARC→ZN（cross_center_eval.csv 里那条 90.75 是完整 TransUNet `_SPARC_c2d`，
#      不是 vitL0，不能当 A0 的零样本分母）、U-Net 的 ZN→SPARC、R50 的 ZN→SPARC。
#   B. 融合臂的双向跨中心（快照缺席则跳过，等训练收口后再跑本脚本即可）。
#
# 优先级 2：刻意排在所有训练队列之后，永不阻塞训练（CLAUDE.md 第 16 条串行）。
#  eval_cross_center.py 自己有 _wait_free_mem（最多等 30 min 显存），拿到空闲窗口才建模型。
#
# 去重：读 logs/queue/cross_center_eval.csv，同 (快照, 测试集) 已有一行就跳过，不重复写。
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_cross_center_matrix.ps1
#       -DryRun 只打印将要执行的命令；-Only 过滤（按 tag）
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 2

$statusPath = Join-Path $logDir "queue_c2d_cross_center_matrix_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$csv = Join-Path $logDir "cross_center_eval.csv"
$done = @()
if (Test-Path $csv) {
    $done = @(Import-Csv $csv | ForEach-Object { "$($_.ckpt)|$($_.test_subset)" })
}

$P = "TU_pretrain_R50-ViT-B_16_"
$E = "_epo200_bs4_ga2_224"
# 每个条目：快照目录名、训练集、测试集、以及重建模型所需的 eval 参数
$jobs = @(
    # ---- A. 基线四格 ----
    @{ tag="A0-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3_vitL0${E}_SPARC_c2d"
       a=@("--arch","baseline","--vit_layers","0") },
    @{ tag="A0-ZN-SPARC"; train="Zhongnan"; test="SPARC"
       snap="${P}skip3_vitL0${E}_Zhongnan_c2d"
       a=@("--arch","baseline","--vit_layers","0") },
    @{ tag="UNet-ZN-SPARC"; train="Zhongnan"; test="SPARC"
       snap="${P}skip3${E}_Zhongnan_unet_c2d"
       a=@("--arch","unet") },
    @{ tag="UNet-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3${E}_SPARC_unet_c2d"
       a=@("--arch","unet") },
    @{ tag="K2-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3${E}_SPARC_wmren_abunetdec_c2d"
       a=@("--arch","wmren","--wmren_ablate","unet_dec") },
    # ---- B. DDBE+lap(+RHD) 双臂，双向 ----
    @{ tag="DDBElap-UNet-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3${E}_SPARC_ddbe4_ddhflap_unet_c2d"
       a=@("--arch","unet","--ddbe","4","--ddbe_hf_op","lap") },
    @{ tag="DDBElap-UNet-ZN-SPARC"; train="Zhongnan"; test="SPARC"
       snap="${P}skip3${E}_Zhongnan_ddbe4_ddhflap_unet_c2d"
       a=@("--arch","unet","--ddbe","4","--ddbe_hf_op","lap") },
    @{ tag="DDBElapRHD-UNet-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3${E}_SPARC_ddbe4_ddhflap_lrhd_TsymW0.5I6_unet_c2d"
       a=@("--arch","unet","--ddbe","4","--ddbe_hf_op","lap") },
    @{ tag="DDBElapRHD-UNet-ZN-SPARC"; train="Zhongnan"; test="SPARC"
       snap="${P}skip3${E}_Zhongnan_ddbe4_ddhflap_lrhd_TsymW0.5I6_unet_c2d"
       a=@("--arch","unet","--ddbe","4","--ddbe_hf_op","lap") },
    @{ tag="DDBElapRHD-R50-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3_vitL0${E}_SPARC_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d"
       a=@("--arch","baseline","--vit_layers","0","--bdfr_v2","1","--bdfr_wave_op","lap") },
    @{ tag="DDBElapRHD-R50-ZN-SPARC"; train="Zhongnan"; test="SPARC"
       snap="${P}skip3_vitL0${E}_Zhongnan_bdfr2_na_bwlap_lrhd_TsymW0.5I2_c2d"
       a=@("--arch","baseline","--vit_layers","0","--bdfr_v2","1","--bdfr_wave_op","lap") },
    # ---- B. WRMM lap+RHD 双臂，双向 ----
    @{ tag="WRMElapRHD-K2-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3${E}_SPARC_lrhd_TsymW0.5I6_wmren_abunetdec_wolap_c2d"
       a=@("--arch","wmren","--wmren_ablate","unet_dec","--wmren_wave_op","lap") },
    @{ tag="WRMElapRHD-K2-ZN-SPARC"; train="Zhongnan"; test="SPARC"
       snap="${P}skip3${E}_Zhongnan_lrhd_TsymW0.5I6_wmren_abunetdec_wolap_c2d"
       a=@("--arch","wmren","--wmren_ablate","unet_dec","--wmren_wave_op","lap") },
    @{ tag="WRMElapRHD-R50W-SPARC-ZN"; train="SPARC"; test="Zhongnan"
       snap="${P}skip3${E}_SPARC_lrhd_TsymW0.5I2_wmren_abunetdec_wolap_encr50wrmm_c2d"
       a=@("--arch","wmren","--wmren_ablate","unet_dec","--wmren_wave_op","lap","--wmren_enc","r50wrmm") },
    @{ tag="WRMElapRHD-R50W-ZN-SPARC"; train="Zhongnan"; test="SPARC"
       snap="${P}skip3${E}_Zhongnan_lrhd_TsymW0.5I2_wmren_abunetdec_wolap_encr50wrmm_c2d"
       a=@("--arch","wmren","--wmren_ablate","unet_dec","--wmren_wave_op","lap","--wmren_enc","r50wrmm") }
)

if ($Only) {
    $keep = $Only -split '[,;\s]+' | Where-Object { $_ }
    $jobs = @($jobs | Where-Object { $keep -contains $_.tag })
    Write-Host "[xcenter] -Only 过滤后剩 $($jobs.Count) 条：$(($jobs | ForEach-Object { $_.tag }) -join ', ')"
}

$ran = 0; $skipped = 0
foreach ($job in $jobs) {
    $snapPath = "E:\mkl\code\python_code\model\TU_Carotid224\$($job.snap)"
    $ck = Join-Path $snapPath "epoch_199.pth"
    if (-not (Test-Path $ck)) {
        Write-Host "[$($job.tag)] skip: 快照还没有 epoch_199.pth（$($job.snap)）"
        $skipped++; continue
    }
    if ($done -contains "$($job.snap)|$($job.test)") {
        Write-Host "[$($job.tag)] skip: cross_center_eval.csv 已有 $($job.snap) -> $($job.test)"
        $skipped++; continue
    }

    $lock = $null
    while (-not $DryRun) {
        $lock = Enter-LaunchLock -TimeoutSec 20
        if ((Get-TrainingProcess) -or (Get-OtherQueue)) {
            Exit-LaunchLock $lock; $lock = $null
            Write-Host "[$($job.tag)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
            Start-Sleep -Seconds 60
            continue
        }
        break
    }

    $argsList = @("eval_cross_center.py") + $job.a +
                @("--ckpt", $snapPath, "--train_subset", $job.train,
                  "--test_subset", $job.test, "--epoch", "199")
    Write-Host "[$($job.tag)] $Python $($argsList -join ' ')"
    if ($DryRun) { continue }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "cross_center_$($job.tag)_$ts.log"
    $start = Get-Date
    Push-Location $Root
    & $Python @argsList *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    Exit-LaunchLock $lock
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.tag)] end exit=$code ${minutes}min"
    $ran++
    # 本轮新写完的条目要计入去重表，避免同轮重复
    $done += "$($job.snap)|$($job.test)"
}
Write-Host "[xcenter] finished: ran=$ran skipped=$skipped"
