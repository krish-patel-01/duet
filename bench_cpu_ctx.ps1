# CPU-only generation speed at several context lengths, for a given ggml-cpu-alderlake.dll.
# usage: bench_cpu_ctx.ps1 [-Dll path] [-Depths 64,1024,4096] [-Gen 64] [-Repeat 2]
param(
    [string]$Dll = '',
    [string]$Model = (Join-Path $PSScriptRoot 'models\MiniCPM5-1B-Q4_K_M.gguf'),
    [int[]]$Depths = @(64, 1024, 2048, 4096),
    [int]$Gen = 64,
    [int]$Repeat = 2,
    [int]$Threads = 6
)
$bin = Join-Path $PSScriptRoot 'bin'
$exe = Join-Path $bin 'hybrid.exe'
$target = Join-Path $bin 'ggml-cpu-alderlake.dll'
$saved = $null
if ($Dll) {
    # keep the backup outside bin\: the backend loader tries every ggml-cpu-*.dll it finds there
    $saved = Join-Path $PSScriptRoot 'dll-variants\bench-saved.dll'
    Copy-Item $target $saved -Force
    Copy-Item $Dll $target -Force
}
try {
    $rows = foreach ($d in $Depths) {
        $best = 0.0
        for ($r = 0; $r -lt $Repeat; $r++) {
            $line = cmd /c "`"$exe`" -m `"$Model`" --mode cpu -t $Threads --bench $d -n $Gen 2>nul" | Select-String '^BENCH'
            if ($line.Line -match 'gen_tps=([\d.]+)') { $best = [math]::Max($best, [double]$Matches[1]) }
        }
        [pscustomobject]@{ Context = $d; 'Gen tok/s (best)' = $best }
    }
    $rows | Format-Table -AutoSize
} finally {
    if ($saved) { Copy-Item $saved $target -Force; Remove-Item $saved }
}
