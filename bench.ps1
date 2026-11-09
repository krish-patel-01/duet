# Compares hybrid vs CPU-only vs GPU-only on several prompt lengths.
# Run with the charger plugged in; on battery Windows throttles the CPU and iGPU.
# Each case runs -Repeat times (modes interleaved) and the median is reported.
# Modes: cpu-stock = CPU with the prebuilt llama.cpp CPU backend (dll-variants\prebuilt.dll),
#        cpu = CPU with our backend (fast decode attention), gpu = Iris Xe only, hybrid = both.
# -SwitchAt passes a fixed switch point to hybrid instead of calibrating on every run.
param(
    [string]$Model = (Join-Path $PSScriptRoot 'models\MiniCPM5-1B-Q4_K_M.gguf'),
    [int[]]$Prompts = @(50, 500, 2000, 4000),
    [int]$Gen = 300,
    [int]$Repeat = 3,
    [string[]]$Modes = @('cpu-stock', 'cpu', 'gpu', 'hybrid'),
    [int]$SwitchAt = -1
)
$exe = Join-Path $PSScriptRoot 'bin\hybrid.exe'
$cpuDll = Join-Path $PSScriptRoot 'bin\ggml-cpu-alderlake.dll'
$stockDll = Join-Path $PSScriptRoot 'dll-variants\prebuilt.dll'
$oursDll = Join-Path $PSScriptRoot 'dll-variants\ours-saved.dll'
Copy-Item $cpuDll $oursDll -Force

$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bat -and $bat.BatteryStatus -eq 1) { Write-Warning "On battery ($($bat.EstimatedChargeRemaining)%): numbers will be throttled." }

function Run-One($mode, $p) {
    $extra = ''
    if ($mode -eq 'hybrid') { $extra = if ($SwitchAt -ge 0) { "--switch-at $SwitchAt" } else { '--calibrate' } }
    $m = $mode
    if ($mode -eq 'cpu-stock') { $m = 'cpu'; Copy-Item $stockDll $cpuDll -Force }
    # stderr is dropped inside cmd: PowerShell 5.1 turns native stderr into errors and loses the output
    $line = cmd /c "`"$exe`" -m `"$Model`" --mode $m --bench $p -n $Gen $extra 2>nul" | Select-String '^BENCH'
    if ($mode -eq 'cpu-stock') { Copy-Item $oursDll $cpuDll -Force }
    $kv = @{}
    foreach ($t in ($line.Line -split ' ')) { if ($t -match '^(\w+)=(.+)$') { $kv[$Matches[1]] = $Matches[2] } }
    $kv
}
function Median($xs) { $s = @($xs | Sort-Object); $s[[int][math]::Floor(($s.Count - 1) / 2)] }

$runs = @{}
try {
    for ($r = 1; $r -le $Repeat; $r++) {
        foreach ($p in $Prompts) {
            foreach ($mode in $Modes) {
                $k = "$p|$mode"
                if (-not $runs[$k]) { $runs[$k] = @() }
                $runs[$k] += , (Run-One $mode $p)
            }
        }
    }
} finally {
    Copy-Item $oursDll $cpuDll -Force
    Remove-Item $oursDll
}

$rows = foreach ($p in $Prompts) {
    foreach ($mode in $Modes) {
        $rs = $runs["$p|$mode"]
        [pscustomobject]@{
            Prompt          = $p
            Mode            = $mode
            'Prompt tok/s'  = Median ($rs | ForEach-Object { [double]$_.prompt_tps })
            'First token s' = Median ($rs | ForEach-Object { [double]$_.ttft })
            'Gen tok/s'     = Median ($rs | ForEach-Object { [double]$_.gen_tps })
            'Total s'       = Median ($rs | ForEach-Object { [double]$_.total })
            'Total range'   = '{0:N1}-{1:N1}' -f ($rs | ForEach-Object { [double]$_.total } | Measure-Object -Minimum -Maximum | ForEach-Object { $_.Minimum, $_.Maximum })
            'CPU/GPU tok'   = ($rs | ForEach-Object { "$($_.cpu_tok)/$($_.gpu_tok)" }) -join ' '
        }
    }
}
$rows | Format-Table -AutoSize
