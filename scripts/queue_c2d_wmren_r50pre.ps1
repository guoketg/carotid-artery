# R50 宿主两臂：R50 全段（ImageNet21k 预训练，不截断）+ 两种接法（docs/experiments.md §1.14）。
#
# 两臂都**全部**使用 R50：conv_root(7×7 stem) + block1/2/3（3/4/6 个残差单元），
# 出 112²/64、56²/256、28²/512、14²/1024 四级特征，没有任何 stage 被截断。
# 随机初始化对照（--no_pretrain）全部排除：文献里的 ResNet 一律加载预训练，
# 从零训不是本任务的对照条件。
#
# 臂表：
#   F1  R50 全段四级特征 + 每级 WRMM stride=1 同级精炼 + unet_dec + 预训练
#       → ..._abunetdec_encr50wrmm_pretrain_c2d
#   E1  R50 全段四级特征直接进 unet_dec + 预训练（WRMM 不参与，别人的方案）
#       → ..._abunetdec_encr50_pretrain_c2d       31.00 M
#
# 读数：
#   F1 − E1      = 「在 R50 提的特征上再加自己的 WRMM 编码器（精炼）」值多少
#   F1 / E1 对 WMREN 参照（WRMM 编码器 + U-Net 解码器，去 REM = 87.77 / 0.1923 / HD 0.979）
#                = 换成强骨干（R50 预训练）值多少
#   注：`--wmren_enc r50deep`（R50 只嫁接深两级）按用户要求排除，不排期。
#
# 判定（§0）：|ΔDice| ≥ 0.5 或 HD 降 ≥ 0.1 mm 才算有效；单 seed 噪声带 ±0.3。
#
# 串行策略：先等 leaveout 队列（M1/M2）进程退出，再走常规 train.py 等待，
# 避免两个队列同时启动抢显存。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\queue_c2d_wmren_r50pre.ps1
#   ... -Only "E1"   只跑指定任务
#   ... -DryRun      只打印命令
param(
    [string]$Python = "E:\mkl\code\python_code\carotid-artery\venv\Scripts\python.exe",
    [string]$Root = "E:\mkl\code\python_code\carotid-artery",
    [string]$Only = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Root "logs\queue"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" }
}

function Get-LeaveoutQueue {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object { $_.CommandLine -like "*queue_c2d_wmren_leaveout.ps1*" }
}

$common = @(
    "--dataset", "Carotid",
    "--batch_size", "4", "--grad_accum", "2",
    "--max_epochs", "200", "--eval_test_every", "5"
)

# name, extra args, log tag
$jobs = @(
    @{ n = "F1"; tag = "wmren_encr50wrmm_pre"; a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_enc", "r50wrmm") },
    @{ n = "E1"; tag = "wmren_encr50_pre"; a = @("--arch", "wmren", "--wmren_ablate", "unet_dec", "--wmren_enc", "r50") }
)

if ($Only) {
    $keep = $Only.Split(",") | ForEach-Object { $_.Trim().ToUpper() }
    $jobs = $jobs | Where-Object { $keep -contains $_.n.ToUpper() }
    if (-not $jobs) { Write-Host "[r50pre] -Only '$Only' matched nothing"; exit 1 }
}

$statusPath = Join-Path $logDir "queue_c2d_wmren_r50pre_status.csv"
if (-not (Test-Path $statusPath)) {
    "job,start,end,minutes,exit_code,log" | Out-File -FilePath $statusPath -Encoding utf8
}

foreach ($job in $jobs) {
    if (-not $DryRun) {
        while (Get-LeaveoutQueue) {
            Write-Host "[$($job.n)] leaveout queue still running, wait 60s..."
            Start-Sleep -Seconds 60
        }
        while (Get-TrainingProcess) {
            Write-Host "[$($job.n)] another train.py is running, wait 60s..."
            Start-Sleep -Seconds 60
        }
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $jobLog = Join-Path $logDir "$($job.tag)_$ts.log"
    $argsList = @("train.py", "--subset", "SPARC", "--seed", "1234") + $common + $job.a

    Write-Host "[$($job.n)] $Python $($argsList -join ' ')"
    Write-Host "[$($job.n)] log: $jobLog"
    if ($DryRun) { continue }

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
Write-Host "[r50pre] queue finished"
