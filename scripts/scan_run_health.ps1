# 对 logs/queue 下全部历史日志做一次运行体检，输出 CSV 供补账/剔废。
#
# 用法：pwsh -NoProfile -File scripts\scan_run_health.ps1
# 输出：logs\analysis\run_health_<yyyyMMdd>.csv（log,dice_pct,nan_hits,verdict,size_mb,mtime）
# 判据见 scripts\queue_health.ps1 头部注释：dice < 1% 或出现 nan 行 → 该次运行不进台账。

param(
    [string]$LogDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'logs\queue'),
    [string]$OutDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'logs\analysis')
)

. (Join-Path $PSScriptRoot 'queue_health.ps1')

if (-not (Test-Path -LiteralPath $LogDir)) { throw "日志目录不存在：$LogDir" }
if (-not (Test-Path -LiteralPath $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$logs = Get-ChildItem -LiteralPath $LogDir -Filter '*.log' -File | Sort-Object LastWriteTime
$rows = foreach ($log in $logs) { Get-RunHealth -LogPath $log.FullName }

$csv = Join-Path $OutDir ('run_health_' + (Get-Date -Format 'yyyyMMdd') + '.csv')
$rows | Select-Object @{n = 'log'; e = { Split-Path -Leaf $_.Log } },
    @{n = 'dice_pct'; e = { $_.DicePct } },
    @{n = 'nan_hits'; e = { $_.NanHits } },
    @{n = 'verdict'; e = { $_.Verdict } },
    @{n = 'size_mb'; e = { $_.SizeMB } },
    @{n = 'mtime'; e = { $_.MTime } } |
    Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding utf8

$summary = $rows | Group-Object Verdict | Sort-Object Name |
    Select-Object @{n = 'verdict'; e = { $_.Name } }, @{n = 'count'; e = { $_.Count } }
$bad = $rows | Where-Object { $_.Verdict -in 'COLLAPSED', 'COLLAPSED_NAN' } |
    Select-Object Log, DicePct, NanHits, Verdict, SizeMB

"scan: {0} 份日志 -> {1}" -f $logs.Count, $csv
$summary | Format-Table -AutoSize | Out-String -Width 200
if ($bad) {
    "塌陷的 run（dice < 1%，读数作废）："
    $bad | Format-Table -AutoSize | Out-String -Width 200
}
