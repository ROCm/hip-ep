<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---

name: halo-load-checker

description: >-

  Reports CPU, memory, and GPU load on the dev machine (Halo) at the moment the

  snapshot runs. Use when the user asks for halo-load-checker, machine
  load, CPU/GPU activity, rocm-smi, whether a build/benchmark is using the GPU, or
  /halo-load-checker.

---

# Halo load checker

Run a **point-in-time** snapshot yourself (do not only tell the user what to run). Report **CPU, memory, and GPU load right then**; optional process lists are supplementary.

**Project location:** `~/workspace/.cursor/skills/halo-load-checker/`

**Slash command:** `/halo-load-checker` via `.cursor/commands/halo-load-checker.md`.

## Quick run

**PowerShell** (works from Cursor terminal on Windows):

```powershell

& "$env:USERPROFILE\workspace\.cursor\skills\halo-load-checker\scripts\halo-load-checker.ps1"

```

**gpu-bash / Git Bash**:

```bash

bash "$HOME/workspace/.cursor/skills/halo-load-checker/scripts/halo-load-checker.sh"

```

Optional: pass a process name substring to highlight (e.g. `model_benchmark`, `cmake`, `hip-compiler`):

```powershell

& "$env:USERPROFILE\workspace\.cursor\skills\halo-load-checker\scripts\halo-load-checker.ps1" -Filter "onnxruntime"

```

## GPU reporting (priority order)

On Halo, THERock often has **`_therock/bin`** but **no** `rocm-smi` / `amd-smi`. The script handles that:

| Priority | Source | What you get |

|----------|--------|----------------|

| 1 | `rocm-smi.exe` in `_therock/bin` | Full ROCm SMI output |

| 2 | `amd-smi.exe` | `amd-smi monitor` |

| 3 | **Fallback** (Windows THERock) | `hipInfo.exe`: Name, `gcnArchName`, `memInfo.total` / `memInfo.free` (VRAM pressure / idle vs in use) |

| 3b | `clinfo.exe` | OpenCL device count + board name (stack sees GPU) |

| 3c | Windows perf counters | Max **GPU Engine** utilization % (3D/compute); else Task Manager hint |

Summarize for the user: **SMI** → utilization + VRAM if present; **hipInfo fallback** → device + memory free/total and approximate “in use”; **Windows counters** → engine utilization % at snapshot.

## Agent workflow

1. Run **one** snapshot script (PowerShell on Windows unless the user is clearly in gpu-bash only).

2. For a long job, run again while the job is active if the user cares about GPU/CPU **during** the run.

3. Summarize **at snapshot time**:

   - **CPU:** total processor utilization (%)

   - **Memory:** used / total (GB) and how full that is

   - **GPU:** SMI output, or hipInfo memory lines + Windows GPU engine %, or clearly unavailable

4. Optionally add top processes or `-Filter` matches if relevant.

5. If no `_therock/bin` at all, suggest hip-ep configure (`build/hip-ep/_therock`) — see `bashrc-gpu-ep.sh`.

## Environment hints (this workspace)

| Variable | Typical path |

|----------|----------------|

| `BUILD_DIR` | `~/workspace/build/hip-ep` |

| `THEROCK_DIST` | `$BUILD_DIR/_therock` when present |

| `HIP_BUILD_BIN` | `$BUILD_DIR/bin/RelWithDebInfo` |

MSYS2 **does not ship `htop`**; use these scripts or `procps-ng` → `top` in UCRT64.

## Output format for the user

```markdown

## Halo load checker

**Time:** …

**CPU:** … (% busy at snapshot)

**Memory:** … (used / total)

**GPU:** … (SMI, or hipInfo VRAM + Windows engine %, or unavailable)

### Top CPU (optional)

…

### Top memory (optional)

…

### Notable processes (filter, optional)

…

```

## Troubleshooting

| Symptom | Action |

|---------|--------|

| `rocm-smi` not found | Normal on Windows THERock; script falls back to hipInfo/clinfo + perf counters |

| hipInfo/clinfo fail | Check AMD driver; ensure `$THEROCK_DIST/bin` on PATH for manual runs |

| GPU engine % missing | Task Manager → Performance → GPU |

| No `_therock/bin` | Configure/build hip-ep so CMake populates `build/hip-ep/_therock` |

| Permission errors | Run from user shell, not elevated unless required |

See [reference.md](reference.md) for manual one-liners.
