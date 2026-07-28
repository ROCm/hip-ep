<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Halo load checker (CPU, memory, GPU)

Follow the **halo-load-checker** project skill (`.cursor/skills/halo-load-checker/SKILL.md` in this workspace).

1. Run a **point-in-time** snapshot yourself (do not only tell me what to run):
   - PowerShell: `& ".cursor/skills/halo-load-checker/scripts/halo-load-checker.ps1"`
   - gpu-bash: `bash ".cursor/skills/halo-load-checker/scripts/halo-load-checker.sh"`
2. Optional process filter from my message: add `-Filter "<substring>"` (PowerShell) or pass the substring as the first argument to the `.sh` script.
3. Summarize **at snapshot time**:
   - **CPU:** total processor utilization (%)
   - **Memory:** used / total (GB) and percent used
   - **GPU:** `rocm-smi` / `amd-smi` if present; else **hipInfo** (device + mem free/total), **clinfo** (OpenCL board), and **Windows GPU engine utilization %** when counters work; otherwise state unavailable and mention Task Manager
4. Optionally include top CPU/memory processes or filter matches if relevant.
