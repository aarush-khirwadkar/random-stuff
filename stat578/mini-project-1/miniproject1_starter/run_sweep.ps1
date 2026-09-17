# Runs the SimCLR temperature and batch-size sweeps back to back.
# Usage (from miniproject1_starter):   .\run_sweep.ps1
# A run whose encoder.pt already exists is skipped, so you can re-launch after an interruption.
# Console output of each run is also saved to <out>/train.log.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# Each entry: output folder, then the extra flags for train.py.
# T=0.5 with batch 256 is the default config, so runs/simclr already covers simclr_T0.5 and simclr_b256.
$jobs = @(
    @{ out = "runs/simclr_T0.1"; args = @("--temperature", "0.1") },
    @{ out = "runs/simclr_T0.2"; args = @("--temperature", "0.2") },
    @{ out = "runs/simclr_T1.0"; args = @("--temperature", "1.0") },
    @{ out = "runs/simclr_b128"; args = @("--batch_size", "128") },
    @{ out = "runs/simclr_b512"; args = @("--batch_size", "512") }
)

foreach ($j in $jobs) {
    $out = $j.out
    if (Test-Path "$out/encoder.pt") {
        Write-Host "[skip] $out already has encoder.pt"
        continue
    }
    New-Item -ItemType Directory -Force $out | Out-Null
    $start = Get-Date
    Write-Host "[start] $out  $(Get-Date -Format 'HH:mm')"
    # cmd merges stderr so PowerShell 5.1 does not wrap warnings as errors; -u makes python
    # flush each epoch line immediately instead of buffering until the run ends.
    $log = "$out/train.log"
    cmd /c "python -u train.py --method simclr $($j.args -join ' ') --out $out 2>&1" |
        ForEach-Object { Write-Host $_; Add-Content -Path $log -Value $_ -Encoding utf8 }
    $mins = [int]((Get-Date) - $start).TotalMinutes
    Write-Host "[done]  $out  ($mins min)"
}

Write-Host "all runs finished $(Get-Date -Format 'HH:mm')"
