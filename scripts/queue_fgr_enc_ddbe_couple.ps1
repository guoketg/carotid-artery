# 排队：编码器 FGR 与解码器 DDBE 的跨位置耦合（2026-11-14 设计，见
# docs/module-suite-design.md「编码器 FGR ↔ 解码器 DDBE 跨位置耦合」一节）。
#
# 背景读数（弱骨干 U-Net，SPARC seed 1234，同协议）：
#   基线 86.5455 / HD 1.13618；DDBE4 只挂解码器 86.34；FGR(enc) 88.62 / HD 0.93；
#   FGR(dec)+DDBE4 同点串联 88.3863 / HD 0.86406。
# 两处缺口：① 弱骨干上「DDBE 挂解码器、FGR 挂编码器」这种错位组合从未跑过；
#           ② 两个模块此前要么同点串联（互覆）、要么各挂其位却互不通信。
#
# 本轮两臂（参数量逐位相同 = +40,138,328，唯一变量是「门有没有接过去」）：
#   UNetComboEnc : --ddbe 4 --fgr 1 --fgr_pos enc
#                  编码器四级 FGR 改写主链，解码器四级 DDBE 各自独立做事（对照臂）
#   UNetCoupleEnc: 上一行 + --fgr_ddbe_couple 1
#                  编码器每个 FGR 的逐通道谱门（前 C 管 hf、后 C 管 lf）按分辨率
#                  投递到解码器同级 DDBE 的 HF/LF 两支；CTX 支与全部卷积不变，
#                  接口零参数。DDBE.fuse 仍是唯一零点 → 恒等起步。
#
# 判定（§0 统一协议，HD 用 medpy 口径，单 seed 1234，终点 ep200）：
#   主判 UNetCoupleEnc vs UNetComboEnc：ΔDice ≥ +0.5 或 ΔHD ≤ −0.1 才算「耦合真的增益」；
#   若两臂落在噪声带内 → 结论「位置错开就已足够，编码器门传给解码器是多余接口」。
#   次判对 UNet+FGR(enc) 88.62 / HD 0.93：只有超过它才说明 DDBE 在这一位形下不是负贡献。
#
# 冒烟（scripts/smoke_fgr_enc_ddbe_couple.py）：恒等起步逐位 0、两臂参数增量相同、
# band_gate 生效且 None 口径不变、门按分辨率配对、梯度链路（第 0 步 fuse 有梯度、
# 1 步后 FGR 门非零）、三组非法旗标组合均 parser.error。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_enc_ddbe_couple.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\queue_fgr_enc_ddbe_couple.ps1
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]1
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
[Win32.Power]::SetThreadExecutionState($keepAwake) | Out-Null

$logDir = Join-Path $Root "logs\queue"
$modelDir = "E:\mkl\code\python_code\model\TU_Carotid224"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-OtherQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_*.ps1*" -and
                       $_.CommandLine -notlike "*queue_fgr_enc_ddbe_couple.ps1*" }
}

$statusPath = Join-Path $logDir "queue_fgr_enc_ddbe_couple_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @(
    "--dataset", "Carotid", "--subset", "SPARC",
    "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

$jobs = @(
    @{ n = "UNetComboEnc"; tag = "unetddbe4fgre"
       a = @("--arch", "unet", "--ddbe", "4", "--fgr", "1", "--fgr_pos", "enc")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_ddbe4_fgre_unet_c2d" },
    @{ n = "UNetCoupleEnc"; tag = "unetddbe4fgrecpl"
       a = @("--arch", "unet", "--ddbe", "4", "--fgr", "1", "--fgr_pos", "enc",
             "--fgr_ddbe_couple", "1")
       snap = "TU_pretrain_R50-ViT-B_16_skip3_epo200_bs4_ga2_224_SPARC_ddbe4_fgre_cpl_unet_c2d" }
)

foreach ($job in $jobs) {
    # 等 GPU 空出来再开跑，绝不与其它训练抢卡（硬约束 16）
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.n)] another train.py / queue is running, waiting 60s..."
        Start-Sleep -Seconds 60
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
    $end = Get-Date
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.n)_$($job.tag),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.n)] end exit=$code ${minutes}min"
}
Write-Host "[fgr-enc / ddbe-dec couple] queue finished"
