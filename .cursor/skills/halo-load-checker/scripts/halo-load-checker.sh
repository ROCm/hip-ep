#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
set -euo pipefail
FILTER="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PS1="${SCRIPT_DIR}/halo-load-checker.ps1"
PS1_WIN="$(cygpath -w "$PS1" 2>/dev/null || echo "$PS1")"
section() { echo ""; echo "=== $1 ==="; }
section "Time / host"
date
hostname 2>/dev/null || true
section "CPU / memory (Windows via PowerShell)"
powershell.exe -NoProfile -Command "
  \$cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 2).CounterSamples[-1].CookedValue
  \$os = Get-CimInstance Win32_OperatingSystem
  \$total = [math]::Round(\$os.TotalVisibleMemorySize/1MB,2)
  \$free = [math]::Round(\$os.FreePhysicalMemory/1MB,2)
  \$used = \$total - \$free
  \$pct = if (\$total -gt 0) { [math]::Round(100*\$used/\$total,1) } else { 0 }
  Write-Host ('CPU total: {0:N1}%' -f \$cpu)
  Write-Host ('RAM used: {0} GB / {1} GB ({2}% used)' -f \$used, \$total, \$pct)
" 2>/dev/null || echo "PowerShell metrics unavailable"
section "GPU (AMD)"
powershell.exe -NoProfile -File "$PS1_WIN" -GpuOnly 2>/dev/null || {
  echo "GPU section failed; run halo-load-checker.ps1 from PowerShell"
}
section "Top processes (PowerShell, by CPU)"
powershell.exe -NoProfile -Command "
  Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name,Id,
    @{N='CPU_s';E={[math]::Round(\$_.CPU,1)}},
    @{N='WS_MB';E={[math]::Round(\$_.WorkingSet64/1MB,1)}} | Format-Table -AutoSize
" 2>/dev/null || ps aux 2>/dev/null | head -15
if [[ -n "$FILTER" ]]; then
  section "Filter: $FILTER"
  powershell.exe -NoProfile -Command "
    Get-Process | Where-Object { \$_.Name -like '*${FILTER}*' } |
      Select-Object Name,Id,CPU | Format-Table -AutoSize
  " 2>/dev/null || true
fi
section "Build / benchmark hints"
for name in model_benchmark onnxruntime_perf_test hip-compiler MSBuild cl; do
  if tasklist 2>/dev/null | grep -qi "${name}\.exe"; then
    echo "RUNNING: ${name}.exe"
  fi
done
