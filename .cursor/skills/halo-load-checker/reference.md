<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Manual one-liners

## PowerShell

```powershell

Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 2

$os = Get-CimInstance Win32_OperatingSystem

# Total/free in KB from WMI; divide by 1MB for GB

Get-Process | Sort-Object CPU -Descending | Select-Object -First 15 Name, CPU, WorkingSet

```

### GPU (THERock on Halo)

```powershell

$bin = "$env:USERPROFILE\workspace\build\hip-ep\_therock\bin"

& "$bin\rocm-smi.exe"          # often missing on Windows THERock

& "$bin\hipInfo.exe"           # device + memInfo.total/free

& "$bin\clinfo.exe"            # OpenCL board name / device count

Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1

```

Full snapshot (includes fallbacks):

```powershell

& "$env:USERPROFILE\workspace\.cursor\skills\halo-load-checker\scripts\halo-load-checker.ps1"

& "$env:USERPROFILE\workspace\.cursor\skills\halo-load-checker\scripts\halo-load-checker.ps1" -GpuOnly

```

## gpu-bash

```bash

export BUILD_DIR="${BUILD_DIR:-$HOME/workspace/build/hip-ep}"

export THEROCK_DIST="${THEROCK_DIST:-$BUILD_DIR/_therock}"

export PATH="$THEROCK_DIST/bin:$PATH"

rocm-smi --showuse 2>/dev/null || "$THEROCK_DIST/bin/hipInfo.exe"

bash "$HOME/workspace/.cursor/skills/halo-load-checker/scripts/halo-load-checker.sh"

```

## MSYS2 UCRT64 (no htop)

```bash

pacman -S procps-ng

top

```

## Kill stuck benchmarks

```powershell

taskkill /F /IM model_benchmark.exe

```
