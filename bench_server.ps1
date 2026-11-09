# Benchmarks llama-server (the engine the D: drive launcher uses) with a given runtime folder
# and extra arguments. Reports the server's own timings: prompt tok/s and generation tok/s.
#
# usage: bench_server.ps1 -ServerDir D:\runtime\win-x64-v2\vulkan -ExtraArgs '-ngl','0' -Label test
param(
    [Parameter(Mandatory)] [string]$ServerDir,
    [string[]]$ExtraArgs = @(),
    [string]$Label = '',
    [string]$Model = (Join-Path $PSScriptRoot 'models\MiniCPM5-1B-Q4_K_M.gguf'),
    [int[]]$PromptRepeats = @(4, 40, 110),   # ~70, ~700, ~1900 prompt tokens
    [int]$Gen = 200,
    [int]$Repeat = 2,
    [int]$Port = 39299
)
$exe = Join-Path $ServerDir 'llama-server.exe'
$para = 'The history of computing is a story of steady improvement in speed, memory and cost. Early machines filled rooms; later ones fit in a pocket. '
$log = Join-Path $env:TEMP "bench_server_$Port.log"
$args = @('-m', "`"$Model`"", '--host', '127.0.0.1', '--port', "$Port", '-c', '8192', '--load-mode', 'none',
          '--reasoning', 'off', '-np', '1') + $ExtraArgs
$p = Start-Process -FilePath $exe -ArgumentList $args -WindowStyle Hidden -PassThru -RedirectStandardError $log -RedirectStandardOutput "$log.out"
try {
    $ok = $false
    foreach ($i in 1..240) {
        if ($p.HasExited) { throw "server exited: $(Get-Content $log -Tail 5 -ErrorAction SilentlyContinue)" }
        try { if ((Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ok = $true; break } } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ok) { throw 'server not ready' }
    # warm-up (first request compiles GPU pipelines etc.)
    $null = Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' `
        -Body (@{ prompt = $para * 2; n_predict = 8; cache_prompt = $false } | ConvertTo-Json)
    foreach ($r in $PromptRepeats) {
        $best = $null
        for ($k = 0; $k -lt $Repeat; $k++) {
            # a unique prefix per run so nothing is reused from the cache
            $prompt = "Run $k-$r. " + ($para * $r) + "`nContinue the story:"
            $body = @{ prompt = $prompt; n_predict = $Gen; cache_prompt = $false; temperature = 0; ignore_eos = $true } | ConvertTo-Json
            $res = Invoke-RestMethod "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body $body
            $t = $res.timings
            if (-not $best -or $t.predicted_per_second -gt $best.predicted_per_second) { $best = $t }
        }
        [pscustomobject]@{
            Config = $Label; 'Prompt tok' = $best.prompt_n
            'Prompt tok/s' = [math]::Round($best.prompt_per_second, 1)
            'Gen tok/s' = [math]::Round($best.predicted_per_second, 1)
            'Total s' = [math]::Round(($best.prompt_ms + $best.predicted_ms) / 1000, 2)
        }
    }
} finally {
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
}
