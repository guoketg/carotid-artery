# 2026-11-20 队列：解码器 WRMM 五臂（用户直接指派「把 WRMM 放到解码器上，Zhongnan/SPARC ×
# ResNet/Unet 都试，先看它是不是真的有效」）+ 重跑 R50+FGR+RHD I6。
#
# 代码（2026-11-20 新增，冒烟 Temp\smoke_dec_wrmm.py 全绿）：
#   --dec_wrmm 0/1/2/3 = 把 unet_wmren 编码器的 WRDM 镜像搬到解码器，stride=1 同级精炼，
#                        挂在每级 DBB 之后；1=四级全挂，2=浅两级，3=只 224²。
#   --dec_wrmm_op haar/lap
#   实测参数增量：unet 60.30M→80.16M（+19.86M，四级 512/256/128/64）；
#                 DecoderCup 20.22M→25.15M（+4.92M，四级 256/128/64/16）。
#   关掉时与历史臂逐位一致（同 seed 基础权重 diff=0、forward max|d|=0）。
#
# 宿主与配对基线（CLAUDE.md 第 18 条：Δ 只对同宿主、同 seed、同协议）
#   臂 1/2 = ResNet + DecoderCup（--arch baseline --vit_layers 0），就是 A0/Z0 那条主线模型
#           SPARC A0 90.4903 / 0.14473 / 0.70624；Zhongnan Z0 92.0041 / 0.12266 / 0.56511
#   臂 3/4 = U-Net（--arch unet）
#           SPARC 86.5455 / 0.22500 / 1.13618；Zhongnan 90.0347 / 0.15421 / 0.71573
#
# 判定（跑前登记，单 seed 只作筛选）
#   每个臂对各自宿主基线：ΔDice >= +0.5 且 ΔHD <= -0.1 mm 才过线。
#   不过线即「解码器 WRMM 在该宿主无效」，不换档位、不换算子；过线再扫档位 2/3 与 lap。
#
# 为什么先跑解码器 WRMM 而不是编码器版：WRMM 此前只在编码器（K2 / r50wrmm），
# r50wrmm 对 R50 基线 −0.75~−0.84，判为「预训练编码器已编码的低频被再分一次频 = 冗余」。
# 解码器上采样路径是编码器完全没碰过的区域，这条臂才是「换位置」而不是「换剂量」。
#
# 臂 5 = 重跑 2026-11-20 20:17 中断的 R50+FGR+RHD I6（epoch 140 停住、无 metrics，
#        被停队列守护进程时连带终止；ep140 中期 90.47 / HD 0.66 不能当结论）。
#
# 顺序：臂 1 → 2 → 3 → 4 → 5（先给 R50+DecoderCup 两条最快读数，再投两条 2.5~3.5h 的 U-Net）
# 用法：先 -DryRun 核对；逐臂挂用 -Only 1（编号见 StatusName）
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [switch]$DryRun,
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "queue_priority.ps1")
Set-QueuePriority 22

$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusPath = Join-Path $logDir "queue_c2d_dec_wrmm_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

$common = @("train.py", "--dataset", "Carotid",
            "--seed", "1234", "--batch_size", "4", "--grad_accum", "2",
            "--max_epochs", "200", "--eval_test_every", "5")

$jobs = @(
    @{ n = "1"; name = "DWRMM-R50DEC-SPARC"
       args = $common + @("--subset", "SPARC", "--arch", "baseline", "--vit_layers", "0", "--dec_wrmm", "1") },
    @{ n = "2"; name = "DWRMM-R50DEC-ZN"
       args = $common + @("--subset", "Zhongnan", "--arch", "baseline", "--vit_layers", "0", "--dec_wrmm", "1") },
    @{ n = "3"; name = "DWRMM-UNET-SPARC"
       args = $common + @("--subset", "SPARC", "--arch", "unet", "--dec_wrmm", "1") },
    @{ n = "4"; name = "DWRMM-UNET-ZN"
       args = $common + @("--subset", "Zhongnan", "--arch", "unet", "--dec_wrmm", "1") },
    @{ n = "5"; name = "FGR-RHD-I6-SPARC-RERUN"
       args = $common + @("--subset", "SPARC", "--arch", "baseline", "--vit_layers", "0",
                          "--fgr", "1", "--loss", "rhd", "--rhd_term", "sym",
                          "--rhd_w", "0.5", "--rhd_iters", "6") }
)

foreach ($job in $jobs) {
    if ($Only -and $job.n -ne $Only) { continue }
    while (-not $DryRun -and ((Get-TrainingProcess) -or (Get-OtherQueue))) {
        Write-Host "[$($job.name)] busy, waiting 60s... blocked by: $(Get-BlockReason)"
        Start-Sleep -Seconds 60
    }
    Write-Host "[$($job.name)] $Python $($job.args -join ' ')"
    if ($DryRun) { continue }

    $lock = Enter-LaunchLock
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $safe = ($job.name -replace '-', '_').ToLower()
    $jobLog = Join-Path $logDir "${safe}_$ts.log"
    $start = Get-Date
    Push-Location $Root
    & $Python @($job.args) *> $jobLog
    $code = $LASTEXITCODE
    Pop-Location
    $end = Get-Date
    Exit-LaunchLock -LockPath $lock
    $minutes = [math]::Round(($end - $start).TotalMinutes, 1)
    "$($job.name),$start,$end,$minutes,$code,$jobLog" |
        Out-File -FilePath $statusPath -Append -Encoding utf8
    Write-Host "[$($job.name)] end exit=$code ${minutes}min"
}
