# 队列侧运行体检：从 log 里读出「是否真的跑成了」，而不只看 exit_code。
#
# 背景：2026-11-16 的 unet_sdfaux_cons_ds 从 ep75 起 dice=0.00% 一直挂到 ep200，
# 脚本仍 exit 0 并把 epoch_199.pth 落盘，队列状态 CSV 记的是成功。训练侧的
# NaN 守卫（trainer.py::_check_finite_or_skip）负责让这种情况非零退出；本函数
# 负责事后核对：只要 log 里出现 nan 行或最终 test summary 的 dice 塌到 <1%，
# 该次运行即判 COLLAPSED / NAN，不进结果台账。
#
# 用法（队列脚本里加两行）：
#   . (Join-Path $PSScriptRoot 'queue_health.ps1')
#   $health = Get-RunHealth -LogPath $logPath
#   # 之后把 $health.Verdict 一起写进状态 CSV

function Get-RunHealth {
    param(
        [Parameter(Mandatory = $true)][string]$LogPath,
        [double]$MinDicePct = 1.0
    )

    $result = [pscustomobject]@{
        Log     = $LogPath
        DicePct = $null
        NanHits = 0
        Verdict = 'NO_LOG'
        SizeMB  = 0
        MTime   = $null
    }
    if (-not (Test-Path -LiteralPath $LogPath)) { return $result }

    $item = Get-Item -LiteralPath $LogPath
    $result.SizeMB = [math]::Round($item.Length / 1MB, 2)
    $result.MTime = $item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')

    $hits = Select-String -LiteralPath $LogPath -Pattern 'test summary|loss : nan|loss: nan' -Encoding UTF8
    if ($null -ne $hits) {
        $result.NanHits = @($hits | Where-Object { $_.Line -match 'nan' }).Count
        $summary = $hits | Where-Object { $_.Line -match 'test summary' } | Select-Object -Last 1
        if ($summary) {
            $line = $summary.Line
            # 新版打印：dice=89.37%±4.86%
            if ($line -match 'dice=([0-9]+(?:\.[0-9]+)?)%') {
                $result.DicePct = [double]$Matches[1]
            }
            # 旧版打印：{'dice': {'mean': 0.9046172..., 'std': ...
            elseif ($line -match "'dice':\s*\{" + "'mean':\s*([0-9.]+)") {
                $result.DicePct = [math]::Round([double]$Matches[1] * 100, 2)
            }
        }
    }

    if ($result.NanHits -gt 0 -and ($null -eq $result.DicePct -or $result.DicePct -lt $MinDicePct)) {
        $result.Verdict = 'COLLAPSED_NAN'
    }
    elseif ($null -ne $result.DicePct -and $result.DicePct -lt $MinDicePct) {
        $result.Verdict = 'COLLAPSED'
    }
    elseif ($null -ne $result.DicePct) {
        $result.Verdict = 'OK'
    }
    elseif ($item.Length -gt 0) {
        $result.Verdict = 'NO_SUMMARY'
    }
    else {
        $result.Verdict = 'EMPTY'
    }

    return $result
}
