# 骨干口径审计（2026-11-17 连续两次误用 TransUNet 后新增，规则见 CLAUDE.md 第 18/20 条）
#
# 为什么需要它：
#   现行口径（2026-11-17 修订，用户确认）：默认两骨干 = ResNet50（= --arch baseline --vit_layers 0，
#   即 R50 + DecoderCup，90.4903 那一行）与 U-Net（--arch unet，from scratch）。
#   被封存的只有**完整 TransUNet**（--arch baseline 且 --vit_layers>=1，含 12 层 ViT + DecoderCup）。
#   历史事故：① 2026-11-16 08:55 的 cvartail λ 消融 5 臂误跑在当时的错误口径上；
#   ② 2026-11-17 15:24 挂载的 cluster_weight_v2 队列里仍有 2 臂属于误用宿主。
#   两次都是「文档写了、执行没落实」，所以把检查做成可执行脚本 + train.py 侧代码级拦截。
#
# 判定：一行里出现 --arch baseline 但没有同行的 --vit_layers 0 → 视为完整 TransUNet 线（违规）。
#
# 用法（挂任何新队列前必跑，退出码 0 才允许启动）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_backbone_policy.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_backbone_policy.ps1 -Static
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_backbone_policy.ps1 -Live
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_backbone_policy.ps1 -Scripts scripts\queue_c2d_resnet_loss.ps1
#
# 退出码：0 = 无 error（历史脚本列表属 informational）；1 = 有 error，禁止启动/必须终止事故进程。

param(
    [string[]]$Scripts = @(),
    [switch]$Static,
    [switch]$Live
)

$ErrorActionPreference = "Continue"
if (-not $Static -and -not $Live -and $Scripts.Count -eq 0) { $Static = $true; $Live = $true }

$errors = @()
$warns = @()
function Add-Err { param([string]$m) $script:errors += $m }

# 匹配三种写法：--arch", "baseline" / --arch baseline / --arch="baseline"
$pattern = '--arch\W{0,4}baseline'
# 同一行带 --vit_layers 0 = R50 + DecoderCup = 默认 ResNet50 宿主，不算违规
$r50Pattern = '--vit_layers\W{0,4}0'
$selfName = "check_backbone_policy.ps1"

if ($Static) {
    Write-Host "=== 1) 静态检查：scripts\*.ps1 里仍写死完整 TransUNet（--arch baseline 且无 --vit_layers 0）的脚本 ==="
    $all = @(Get-ChildItem $PSScriptRoot -Filter "*.ps1" | Where-Object { $_.Name -ne $selfName } |
             Sort-Object Name)
    $rows = @()
    $exempt = @()
    foreach ($f in $all) {
        # 只算代码行：注释里的说明不算违规；同行写 --vit_layers 0 的 R50+DecoderCup 臂也不算违规
        $hits = @(Select-String -LiteralPath $f.FullName -Pattern $pattern |
                  Where-Object { $_.Line -notmatch '^\s*#' -and $_.Line -notmatch $r50Pattern })
        # 2026-11-19：显式写了 --allow_legacy_tu 的行 = 用户已授权的完整 TransUNet 臂，
        # 不算违规，但必须单独列出（绝不静默豁免），否则「授权了」和「忘了写 -vit_layers 0」分不开。
        $ok = @($hits | Where-Object { $_.Line -match '--allow_legacy_tu' })
        $hits = @($hits | Where-Object { $_.Line -notmatch '--allow_legacy_tu' })
        if ($ok.Count -gt 0) {
            $exempt += [pscustomobject]@{ script = $f.Name; lines = ($ok | ForEach-Object { $_.LineNumber }) -join ',' }
        }
        if ($hits.Count -gt 0) {
            $rows += [pscustomobject]@{
                script = $f.Name
                hits   = $hits.Count
                first  = ($hits | ForEach-Object { $_.LineNumber } | Sort-Object | Select-Object -First 1)
            }
        }
    }
    if ($exempt.Count -gt 0) {
        Write-Host "以下行写明了 --allow_legacy_tu（用户已授权的完整 TransUNet 臂），按豁免处理："
        $exempt | Format-Table -AutoSize | Out-String | Write-Host
    }
    if ($rows.Count -eq 0) {
        Write-Host "未发现使用完整 TransUNet 线的队列脚本。"
    } else {
        $rows | Format-Table -AutoSize | Out-String | Write-Host
        Write-Host "以上 $($rows.Count) 个脚本含「--arch baseline 但同行没有 --vit_layers 0」的臂，属已封存的完整 TransUNet 线，**不得再挂载**；"
        Write-Host "要复用必须显式写成默认 ResNet50 宿主（--arch baseline --vit_layers 0，R50 + DecoderCup）或 U-Net（--arch unet）。"
        Write-Host "（历史脚本的静态存在不算 error，但任何人挂载它们都视为违规；-Scripts 显式点名则直接判 error。）"
    }
}

if ($Live) {
    Write-Host ""
    Write-Host "=== 2) 运行态检查：存活 python 进程是否在跑完整 TransUNet 线 ==="
    $liveProcs = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" |
                   Where-Object {
                       $_.CommandLine -and
                       ($_.CommandLine -match '--arch\s+baseline') -and
                       ($_.CommandLine -notmatch '--vit_layers\s+0')
                   })
    if ($liveProcs.Count -eq 0) {
        Write-Host "无 python 进程使用完整 TransUNet 线（R50 + DecoderCup 的 --vit_layers 0 臂属默认宿主，放行）。"
    } else {
        foreach ($p in $liveProcs) {
            $cmd = $p.CommandLine
            if ($cmd.Length -gt 160) { $cmd = $cmd.Substring(0, 160) + "..." }
            Write-Host ("  PID {0}  start {1}" -f $p.ProcessId, $p.CreationDate)
            Write-Host ("    {0}" -f $cmd)
            Add-Err "PID $($p.ProcessId) 正在用 --arch baseline 且无 --vit_layers 0（完整 TransUNet 线）训练：立即终止，并改到 --arch baseline --vit_layers 0 或 --arch unet 重排"
        }
    }
}

if ($Scripts.Count -gt 0) {
    Write-Host ""
    Write-Host "=== 3) 拟挂载脚本检查（点名即须 exit 0）==="
    foreach ($s in $Scripts) {
        foreach ($one in ($s -split '[,;\s]+' | Where-Object { $_ })) {
            $path = $one
            if (-not (Test-Path -LiteralPath $path)) { $path = Join-Path $PSScriptRoot (Split-Path $one -Leaf) }
            if (-not (Test-Path -LiteralPath $path)) { Add-Err "找不到脚本：$one"; continue }
            $hits = @(Select-String -LiteralPath $path -Pattern $pattern |
                      Where-Object { $_.Line -notmatch '^\s*#' -and $_.Line -notmatch $r50Pattern })
            if ($hits.Count -gt 0) {
                Add-Err "$(Split-Path $path -Leaf)：含 $($hits.Count) 处完整 TransUNet 线路由（--arch baseline 而同行无 --vit_layers 0，行 $(($hits | ForEach-Object { $_.LineNumber }) -join ','))，不得挂载"
            } else {
                Write-Host "  OK  $(Split-Path $path -Leaf)"
            }
        }
    }
}

Write-Host ""
if ($warns.Count -gt 0) { Write-Host "WARNING:"; $warns | ForEach-Object { Write-Host "  - $_" } }
if ($errors.Count -gt 0) {
    Write-Host "ERROR:"
    $errors | ForEach-Object { Write-Host "  - $_" }
    Write-Host ""
    Write-Host "审计未通过：把宿主改成 --arch baseline --vit_layers 0（R50 + DecoderCup）或 --arch unet，或终止违规进程后重排。"
    exit 1
}
Write-Host "审计通过（无 error）。"
exit 0
