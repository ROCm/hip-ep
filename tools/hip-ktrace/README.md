<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# ktrace

ktrace is a standalone Windows HIP system profiler. It combines injected
HIP/ROCTx callbacks, an external shared-memory collector, optional Windows ETW,
and independent post-processing. Adding the `kernel` mode with
`-a/--rgp-auto-capture` also drives RdpCaptureApi to produce an `.rgp` file for
Radeon GPU Profiler, either on its own or alongside a ktrace capture.

## Current capabilities

- Trace a root process and its descendant tree through suspended launch and DLL
  injection, without modifying or rebuilding the target application.
- Capture HIP Runtime APIs, kernel dispatches, memory copies, barriers, hardware
  queues, streams, correlation IDs, and decoded API arguments; flow edges link
  Host API calls to GPU operations.
- Record kernel names plus occupancy, VGPR, scratch, LDS, and grid/block
  execution properties.
- Capture ROCTx marks, nested thread ranges, and cross-thread ranges, with
  optional inferred `prefill`/`decode` phases.
- Sample VRAM usage; optional `--etw` adds Windows CPU scheduling, context
  switches, sampled callstacks, and EXE/DLL ImageLoad mappings.
- Preserve committed records across fast, forced, and `_Exit` target teardown
  through `ktrace_collector.exe`, a shared MPSC ring, and a binary journal,
  with dropped-record and ring high-watermark diagnostics.
- Export ROCm 7.2.1-shaped CSV/JSON, text performance reports, Chrome Trace
  JSON, and native Perfetto `.pftrace`; JSON and pftrace can be generated
  separately or in parallel from the same event stream.
- Optionally combine ktrace with RGP/SQTT, SPM counters, and instruction tracing.
- Support Windows x64, with synthetic workloads built for the architectures
  selected by `GPU_ARCH`.

## Usage

```bat
:: Legacy ktrace: HIP API + GPU commands
run.bat system -- application.exe args...

:: Add CPU scheduling and sampled stacks (run elevated)
run.bat system --etw -- application.exe args...

:: Native pftrace only, with every host API and correlation flow
run.bat system --chrome-pftrace --flows -- application.exe args...

:: Generate matching JSON and pftrace from one event stream
run.bat system --chrome-json --chrome-pftrace --flows -- application.exe args...

:: Set a 10-minute timeout for system mode
run.bat system --timeout-minutes 10 -- application.exe args...

:: RGP/SQTT only
run.bat kernel -a dispatch:64:256 -- application.exe args...

:: Combined ktrace + RGP
run.bat system kernel -a dispatch:64:256 -- application.exe args...

:: RGP with SPM counters, instruction tracing, and the largest SQTT buffer.
:: These are opt-in; keep the capture window small when they are enabled.
run.bat kernel -a dispatch:1000:16 --rgp-counter-collection --rgp-instruction-tracing --rgp-sqtt-buffer-size maximum -- application.exe args...

:: Preserve private spool files without post-processing
run.bat system --capture-only -- application.exe args...
```

Omitting the mode defaults to `system`. `--` is required before the target
command. Run `run.bat -h` for all options.
For every traced HIP process, the launcher prints the loaded `amdhip64` path
and file version; the collector also prints the numeric HIP runtime version.

### Modes and argument grouping

`system` and `kernel` are independent switches rather than alternatives, so
naming both performs a dual capture from a single run of the target. Each mode
may be followed by its own options, which keeps the two sides readable:

```bat
run.bat system [args] kernel [args] -- application.exe [args]
```

The grouping is a readability convention, not a parsing rule. `run.bat`
consumes its own options wherever they appear and forwards everything else to
`ktrace.exe`, which resolves each option by name: `-a` and `--rgp-*` configure
the RGP capture; launch, timeout, ETW, and output behavior use launcher flags;
record limits, API arguments, VRAM sampling, and other tracing details use
`KTRACE_*` environment variables.
Everything after `--` is the target command line and is never interpreted, so
the target's own options cannot collide with ktrace's.

| Argument | Consumed by |
| --- | --- |
| `system`, `kernel` | `run.bat` mode selection |
| `--capture-only`, `--chrome-json`, `--chrome-pftrace`, `--flows` | `run.bat` post-processing |
| `-a`, `--rgp-*` | `ktrace.exe`, RGP capture |
| `--etw`, `--verbose`, `--timeout-minutes`, `--working-directory`, `-o` | `ktrace.exe` launcher |
| everything after `--` | the target application |

Options not consumed by `run.bat` are forwarded to `ktrace.exe`, which rejects
unknown launcher or RGP options rather than silently ignoring them.

## Optional RGP dual capture

```bat
run.bat system kernel -a dispatch:64:256 -- bin\workload.exe --rounds 2000 --size 1024
```

Here:

- `-a dispatch:64:256` tells RGP to capture 256 GPU dispatches starting at dispatch 64.
- `--` separates ktrace/RGP options from the target command, and is required.
- `bin\workload.exe` is the synthetic HIP target being traced.
- `--rounds 2000` is passed to `workload.exe` and runs 2,000 rounds. Each round
  launches 13 kernels, for 26,000 kernel dispatches in total. Six of the 13 are
  GEMMs of differing tile shape, so one capture can rank them against each
  other. Run `workload.exe --verify` separately to check they all agree; it
  exits after the check, so it does not combine with a capture.
- `--size 1024` makes each kernel large enough that the run takes about three
  seconds. The default `--size 256` finishes in under a tenth of a second, which
  is too short for the driver to dump the trace and fails the capture.

Direct invocation:

```bat
set "KTRACE_OUTPUT_DIR=%CD%\out\combined"
set "KTRACE_OUTPUT_FILE=trace"
bin\ktrace.exe -a dispatch:64:256 -o "%KTRACE_OUTPUT_DIR%\trace_capture.rgp" -- app.exe args
bin\ktrace_postprocess.exe all "%KTRACE_OUTPUT_DIR%"
```

Run `bin\ktrace.exe --help` for dispatch selection, capture delay,
instruction tracing, SPM counters, shader instrumentation, and SQTT buffer
options. Without `-o`, the RGP file is written as
`KTRACE_OUTPUT_DIR\<KTRACE_OUTPUT_FILE>_capture.rgp`.
`system` does not load RDP. `kernel` requires `-a` but does not force any
SQTT feature flags.

### RGP feature flags

These are opt-in. Without them a capture records the SQTT timeline only, which
is what most profiling needs and what allows large capture windows.

| Flag | Effect | Cost |
| --- | --- | --- |
| `--rgp-counter-collection` | Adds SPM (streaming performance counter) data | Moderate |
| `--rgp-instruction-tracing` | Adds per-instruction timing, enabling the RGP instruction timeline | Very high — roughly 1 MB of SQTT data per dispatch |
| `--rgp-sqtt-buffer-size <size>` | Sets the driver's SQTT buffer: `minimum`, `low`, `default`, `high`, `maximum` | Reserves more GPU-visible memory |
| `--rgp-shader-instrumentation` | Adds shader instrumentation | Varies |
| `--rgp-auto-capture-delay <ms>` | Delays a dispatch auto-capture, for applications whose GPU work starts late | None |

Example with all three enabled:

```bat
run.bat kernel -a dispatch:1000:16 ^
    --rgp-counter-collection ^
    --rgp-instruction-tracing ^
    --rgp-sqtt-buffer-size maximum ^
    -- bin\workload.exe --rounds 2000 --size 1024
```

These flags may appear in any order before `--`.

Enabling instruction tracing requires shrinking the capture window: raising
`--rgp-sqtt-buffer-size` to `maximum` does not compensate, because the limit is
reached well before a few hundred dispatches.

Two things make a capture fail with
`There was an error trying to collect a trace` (`result=-2`):

1. The target finishes too quickly. The driver needs the process to stay alive
   after the capture window to dump the trace. A workload that exits in well
   under a second fails even with a small capture window. Give the target real
   work — for `bin\workload.exe`, `--size 1024` raises per-kernel cost enough
   that 2,000 rounds run for about three seconds, which is sufficient.
2. The capture window is too large *for the enabled features*. Instruction
   tracing dominates the SQTT volume, and the limit is on total data, not on a
   dispatch count: what fits depends on which kernels land inside the window.
   Treat the dispatch count as a dial to turn down when a capture fails, not as
   a number with a documented safe value.

The dispatch start index is not a constraint on its own; starting at dispatch 8
works as well as starting at 1,000.

Capture size grows about linearly with the window. Measured on gfx1151 with
`bin\workload.exe --rounds 2000 --size 1024`, whose rounds are GEMM-heavy:

| Features | Window | Capture |
| --- | --- | --- |
| counters + instruction tracing + `maximum` buffer | `dispatch:1000:64` | 117 MB |
| counters + instruction tracing + `maximum` buffer | `dispatch:1000:256` | 462 MB |
| counters + instruction tracing + `maximum` buffer | `dispatch:1000:512` | 916 MB |
| defaults (no extra features) | `dispatch:1000:1000` | 157 MB |

That is roughly 1.8 MB per dispatch with instruction tracing on, against 0.16 MB
without it. Estimate from that before capturing: a window wide enough to be
interesting is also wide enough to fill a disk. Start small and widen.

Use `--rgp-auto-capture-delay` for applications whose GPU work starts late.

ktrace writes the `.rgp` and stops there. Analyse it with Radeon GPU Profiler
from [RDTS for Windows](https://gpuopen.com/rdts-windows/). Extract the RDTS
archive, start `RadeonGPUProfiler.exe`, and open the generated `.rgp` file. It
can also be opened directly from a command prompt:

```bat
"<RDTS directory>\RadeonGPUProfiler.exe" "out\trace-<timestamp>\trace_capture.rgp"
```

Radeon GPU Profiler displays the SQTT timeline, shader and pipeline details,
SPM counters, and instruction timing recorded in the capture.

The combined launcher supports Windows x64 only. A single RDP context can be
active at a time, and enabling RGP plus ktrace callbacks adds capture overhead.
On timeout, it aborts RGP capture and terminates the root process it created.

## One-command runner

`run.bat` provides two capture modes that may be combined. It
creates a timestamped output directory and runs the target:

```bat
run.bat system -- application.exe args...
run.bat kernel -a dispatch:64:256 -- application.exe args...
run.bat system kernel -a dispatch:64:256 -- application.exe args...
```

- `system`: legacy ktrace, recording HIP Runtime API and GPU commands/kernels.
- `kernel`: RGP/SQTT only, producing `.rgp` without injecting the ktrace tracer.
- `system kernel`: both at once, from a single run of the target.

`kernel` names the RGP feature area rather than the capture itself, so it needs
an action before anything runs. Capture is currently the only action, requested
with `-a`; reading back an existing `.rgp` is not implemented yet. Naming
`kernel` without `-a` therefore reports what is missing and exits instead of
running the target.

`system` produces full `ktrace_report.txt` and
`ktrace_perfetto.json`. `kernel` produces only the `.rgp`.
JSON is the default system trace format; add `--chrome-pftrace` for native
pftrace, or specify both `--chrome-json --chrome-pftrace` to generate both from
the same event stream.
The launcher timeout defaults to 30 minutes. In system mode, pass
`--timeout-minutes N --` before the target. The old `--timeout-ms` spelling is
retained only as a compatibility alias.

`system` copies private spool files to `out\trace-<timestamp>\raw\` before
finalize, so original data remains available after public files are generated.

Add `--capture-only` after the mode to write spool files directly under `raw\`
and skip finalize, analysis, and Perfetto export:

```bat
run.bat system --capture-only -- application.exe args...
run.bat kernel --capture-only -a dispatch:64:256 -- application.exe args...
run.bat system kernel --capture-only -a dispatch:64:256 -- application.exe args...
```

Post-process later with:

```bat
bin\ktrace_postprocess.exe finalize out\trace-<timestamp>\raw
bin\ktrace_postprocess.exe all out\trace-<timestamp>\raw
```

## ROCTx annotations

After `build.bat`, the ROCTx files are:

```text
src\external\roctx\roctx.h
build\roctx64.lib
bin\roctx64.dll
bin\roctx_workload.exe
```

Applications can annotate semantic ranges:

```cpp
#include <roctx.h>

roctxMark("startup");
roctxRangePush("prefill");
// HIP work
roctxRangePop();
```

Link the application with `roctx64.lib` and make `roctx64.dll` available at
runtime. The system mode produces:

```text
trace_roctx_trace.csv
raw\.trace_<pid>_ops.csv.roctx.csv
```

`ktrace_report.txt` summarizes range counts and durations.
`ktrace_perfetto.json` displays nested ranges, cross-thread ranges, and marks
on their CPU thread tracks.

The tracer currently supports the bundled lightweight `roctx64.dll` and legacy
ROCTx libraries that export `roctxRegisterTracerCallback`. The newer
rocprofiler-sdk API-table registration path is not yet integrated.

Self-test:

```bat
run.bat system -- bin\roctx_workload.exe --rounds 200
```

Trace an executable:

```bat
run.bat system -- "C:\absolute\path\to\application.exe" --arg1 value1
```

Trace a batch file through `cmd.exe`:

```bat
run.bat system -- C:\Windows\System32\cmd.exe /c C:\absolute\path\to\script.bat
```

Trace a Python script:

```bat
run.bat system -- "C:\path\to\python.exe" "C:\absolute\path\to\script.py" --arg1 value1
```

The result is written under `out\trace-<timestamp>\` and includes
`ktrace_report.txt` and `ktrace_perfetto.json`. Use absolute paths for the
target, script, and Python interpreter.

`run.bat` enables API arguments and 100 ms VRAM sampling, captures without
record limits, then runs the native C++ `all` pipeline. For each GPU-owning
process it creates:

```text
<prefix>_agent_info.csv
<prefix>_hip_api_trace.csv
<prefix>_kernel_trace.csv
<prefix>_kernel_trace_readable.csv
<prefix>_memory_copy_trace.csv
<prefix>_memory_allocation_trace.csv
<prefix>_roctx_trace.csv
<prefix>_ktrace_launch_info.csv
<prefix>_hip_runtime_info.csv
<prefix>_results.json
<prefix>_ktrace_extensions.json
ktrace_report.txt
ktrace_perfetto.json
ktrace_perfetto.pftrace   (when --chrome-pftrace is selected)
```

`<prefix>_roctx_trace.csv` is emitted only when the target uses ROCTx.

With multiple GPU-owning processes, report and Perfetto filenames gain a
prefix to prevent collisions. Full capture can produce several gigabytes.

## Recommended workflow

This README is a reference for what each mode and flag does. The procedure for
an actual investigation — global timeline first, then a targeted per-instruction
capture of one kernel — is written up in
[`skills/ktrace-profiling/SKILL.md`](skills/ktrace-profiling/SKILL.md),
including why ROCTx and disabling autotune are load-bearing rather than
optional. [`skills/README.md`](skills/README.md) lists the agent skills and how
to install them into an AI coding assistant such as Cursor or Claude Code.

In short: capture the whole run with `system` mode and rank kernels by GPU time
in `ktrace_report.txt`; disable autotune so the dispatch order is reproducible;
then capture that kernel's dispatch window with `kernel` mode plus SPM and
instruction tracing, and analyse it with Radeon GPU Profiler from RDTS for
Windows.

## Quick self-test

First check that the GEMM variants agree with the reference kernel:

```bat
bin\workload.exe --size 1024 --verify
```

`--verify` exits after the check without running the rounds, so keep it out
of capture commands: the process would end before the driver can dump an RGP
trace, and the system capture would contain almost no dispatches.

Then capture:

```bat
set "KTRACE_OUTPUT_DIR=%CD%\out\probe"
set "KTRACE_OUTPUT_FILE=probe"
set "KTRACE_API_ARGS=1"
if exist "%KTRACE_OUTPUT_DIR%" rmdir /s /q "%KTRACE_OUTPUT_DIR%"

bin\ktrace.exe -a dispatch:64:256 -- bin\workload.exe --rounds 2000 --size 1024
bin\ktrace_postprocess.exe all "%KTRACE_OUTPUT_DIR%" --chrome-json --chrome-pftrace
python python\ktrace_cli.py validate --output-dir "%KTRACE_OUTPUT_DIR%" --prefix probe
```

This is a combined ktrace + RGP test: `-a` also creates `probe_capture.rgp`.
For a system-only test, remove `-a dispatch:64:256`; shorter runs are then
acceptable because the driver does not need to dump an RGP capture.

Add `--graph` to exercise HIP graph replay:

```bat
if exist "%KTRACE_OUTPUT_DIR%" rmdir /s /q "%KTRACE_OUTPUT_DIR%"
bin\ktrace.exe -a dispatch:64:256 -- bin\workload.exe --rounds 2000 --size 1024 --graph
bin\ktrace_postprocess.exe all "%KTRACE_OUTPUT_DIR%" --chrome-json --chrome-pftrace
```

Results are written to `out\probe\`.

## Output

Each capture produces ROCm 7.2.1 rocprofv3-shaped data:

| File | Contents |
| --- | --- |
| `<prefix>_agent_info.csv` | One row per agent (CPU and GPU) describing the hardware: `Cu_Count`, `Simd_Per_Cu`, `Wave_Front_Size`, `Max_Waves_Per_Cu`, `Lds_Size_In_Kb`, `Local_Mem_Size`, clocks, `Gfx_Target_Version`, PCI `Vendor_Id`/`Device_Id`, and the agent `Name`. This is what the occupancy math in the report is derived from. |
| `<prefix>_hip_api_trace.csv` | One row per HIP Runtime API call on the host: `Domain`, `Function`, `Process_Id`, `Thread_Id`, `Correlation_Id`, `Start_Timestamp`, `End_Timestamp`. `Correlation_Id` is what links a host call to the GPU operation it produced. |
| `<prefix>_kernel_trace.csv` | One row per kernel dispatch: `Agent_Id`, `Queue_Id`, `Stream_Id`, `Dispatch_Id`, `Kernel_Name` (mangled), `Correlation_Id`, begin/end timestamps, and the launch shape — `Workgroup_Size_{X,Y,Z}`, `Grid_Size_{X,Y,Z}`, `VGPR_Count`, `Accum_VGPR_Count`, `SGPR_Count`, `LDS_Block_Size`, `Scratch_Size`. |
| `<prefix>_kernel_trace_readable.csv` | Identical columns and row order to `kernel_trace.csv`; only `Kernel_Name` differs, demangled to a readable signature. Use it for reading, and the mangled file for tooling that matches symbols. |
| `<prefix>_memory_copy_trace.csv` | One row per memory copy: `Direction` (host/device combination), `Stream_Id`, `Source_Agent_Id`, `Destination_Agent_Id`, `Correlation_Id`, begin/end timestamps. |
| `<prefix>_memory_allocation_trace.csv` | One row per allocation or free: `Operation`, `Agent_Id`, `Allocation_Size`, `Address`, `Correlation_Id`, begin/end timestamps. |
| `<prefix>_roctx_trace.csv` | One row per ROCTx mark or range: `Operation`, `Process_Id`, `Thread_Id`, `Range_Id`, `Timestamp`, `Message`. Written only when the target uses ROCTx. |
| `<prefix>_results.json` | Run metadata in the `rocprofiler-sdk-tool` envelope: hostname, Windows release/version, machine name, `pid`, `init_time`/`fini_time`, and the resolved output/raw paths. It describes the run, not the events. |
| `<prefix>_cpu.etl` | Raw Windows ETW trace produced by `run.bat system --etw -- ...`. |
| `<prefix>_cpu_sched.csv` | ETW context switches and CPU scheduling intervals. |
| `<prefix>_cpu_samples.csv` | ETW sampled instruction pointers and callstacks. |
| `<prefix>_cpu_images.csv` | EXE/DLL ImageLoad mappings used to symbolize sampled CPU stacks. |

A file is omitted rather than written empty, so an absent
`memory_copy_trace.csv` means the target performed no copies. Timestamps are
nanoseconds on the same clock as the Perfetto timeline, whose `t=0` is the QPC
sample taken immediately before `ResumeThread`.

ktrace also produces:

- `<prefix>_ktrace_launch_info.csv`: root-process `ResumeThread` QPC timestamp
- `<prefix>_hip_runtime_info.csv`: `Process_Id`, and the path and file version of
  the `amdhip64` actually loaded by that process. Worth checking first when a
  capture behaves unexpectedly, since the loaded runtime is often not the one
  on `PATH`: Windows resolves `System32` before `PATH`, and only the
  application's own directory takes precedence over it.
- `<prefix>_ktrace_extensions.json`: occupancy, VRAM, API arguments, and field quality
- `<prefix>_ktrace_report.txt`: text analysis report, including the ktrace version
- `<prefix>_ktrace_chrome.json`: Perfetto/Chrome Trace
- `<prefix>_ktrace_perfetto.pftrace`: native Perfetto TrackEvent protobuf

A single GPU-owning process is the common case, and its report and trace are
written without a prefix as `ktrace_report.txt`, `ktrace_perfetto.json`, and
`ktrace_perfetto.pftrace`. The prefixed names above appear once a capture
contains several GPU-owning processes.

Open `ktrace_perfetto.json` from a single-process `run.bat`/`all` capture at
https://ui.perfetto.dev. A standalone `chrome` or `export` command instead
defaults to `<prefix>_ktrace_chrome.json`.
For new captures, timeline `t=0` is the QPC timestamp sampled immediately
before the launcher calls `ResumeThread`; this keeps process startup, HIP
initialization, and GPU work on one stable clock. Older captures without launch
metadata fall back to the first captured HIP/GPU event.
Perfetto orders the top-level groups as GPU, HIP API, then ROCTx. ROCTx marks
and ranges are split into tracks by their original OS thread IDs.

Capture I/O is handled by `ktrace_collector.exe` through a versioned shared
memory ring and a temporary binary journal. CSV formatting runs only after the
target exits, so it cannot make the live collector fall behind. The injected
DLL performs no file I/O in this mode, so committed records remain available
after normal, fast, or forced target exit. Ring capacity is 262,144 records;
the summary and console report its peak occupancy. Set
`KTRACE_IPC_MODE=inline` only for compatibility comparisons with the old
in-process writer.
`_results.json` contains the capture host name; inspect or anonymize it before
publishing a trace.

## Analyze an existing capture

Python:

```bat
python python\ktrace_cli.py analyze out\my_trace\my_trace_kernel_trace.csv
python python\ktrace_cli.py chrome out\my_trace\my_trace_kernel_trace.csv --all
python python\ktrace_cli.py perfetto out\my_trace\my_trace_kernel_trace.csv --all --flows
python python\ktrace_cli.py export out\my_trace\my_trace_kernel_trace.csv ^
  --chrome-json out\my_trace\trace.json ^
  --chrome-pftrace out\my_trace\trace.pftrace --all --flows
python python\ktrace_cli.py validate --output-dir out\my_trace --prefix my_trace
```

C++:

```bat
bin\ktrace_postprocess.exe analyze out\my_trace\my_trace_kernel_trace.csv
bin\ktrace_postprocess.exe chrome out\my_trace\my_trace_kernel_trace.csv --all
bin\ktrace_postprocess.exe export out\my_trace\my_trace_kernel_trace.csv ^
  --chrome-json out\my_trace\trace.json ^
  --chrome-pftrace out\my_trace\trace.pftrace --all --flows
```

Python's `perfetto` command has no same-named C++ subcommand; use C++
`export --chrome-pftrace` or `all --chrome-pftrace` instead.

The input can also be `<prefix>_results.json` or the prefix shared by the output
files.

`--all` exports the whole GPU timeline but limits host slices to blocking APIs
and omits flow arrows. Add `--flows` to retain every host API and the
API-to-GPU correlation arrows: `chrome <capture> --all --flows`. This can make
large traces substantially bigger.

`--chrome-json` and `--chrome-pftrace` are independent. Specifying both makes
the C++ scanner fan out the same logical events to buffered JSON and pftrace
writers; specifying neither keeps the legacy JSON default. Native pftrace
preserves counters, event arguments, metadata, flows, and callstacks.

### Inferred LLM phases

When kernel names contain `prefill` or `decode`, ktrace uses them as anchors to
infer phase ranges. A second streaming pass assigns every kernel and standard
memory-copy command inside those ranges to the phase, including generic GEMV,
transpose, cast, and elementwise kernels whose names do not contain a phase
word. Adjacent phase boundaries are expanded to their midpoint when the gap is
at most 0.5 seconds. Captures without phase anchors omit phase reporting rather
than guessing.

Kernel-name resolution first loads `bin\amd_comgr_3.dll` and calls AMD COMGR's
official
[`amd_comgr_demangle_symbol_name`](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr)
API. If COMGR is unavailable or fails, ktrace falls back to `llvm-cxxfilt.exe`
(`CXXFILT`, `PATH`, `LLVM_ROOT`/`LLVM_HOME`, or `HIP`) and then its built-in
simplified parser.

## Common environment variables

- `KTRACE_OUTPUT_DIR`: capture directory
- `KTRACE_OUTPUT_FILE`: output file prefix
- `KTRACE_MAX`: optional GPU command limit; unset or `0` means unlimited
- `KTRACE_API_MAX`: optional HIP API record limit; unset or `0` means unlimited
- `KTRACE_NO_API`: disable the HIP API domain
- `KTRACE_API_ARGS`: record launch, memory, and stream arguments
- `KTRACE_VRAM_MS`: VRAM sampling interval; `0` disables sampling
- `KTRACE_IPC_MODE`: `collector` (default) or legacy `inline`
- `KTRACE_ETW`: set to `1` to request Windows CPU scheduling/profile capture;
  kernel ETW requires an elevated process with `SeSystemProfilePrivilege`
- `_NT_SYMBOL_PATH`: optional DbgHelp path for CPU stacks, for example
  `C:\build;srv*C:\Symbols*https://msdl.microsoft.com/download/symbols`
- `AMD_COMGR`: optional full path to `amd_comgr_3.dll`; the bundled DLL is preferred
- `CXXFILT`: optional `llvm-cxxfilt.exe` fallback path

## Build

Requirements:

- Windows 11
- Visual Studio 2022 C++ toolchain
- CMake 3.24+ and Ninja
- A HIP toolchain, either TheRock or ROCm
- HIP headers matching the runtime actually loaded by the target process
- Python 3.10+

`build.bat` enables optional RGP support by default. On the first build it
downloads RDTS from `https://gpuopen.com/rdts-windows/` and places
`RdpCaptureApi.dll`, `GPUPerfAPICounters-x64.dll`, `amd_comgr_3.dll`, and
`amdgpu_dis.dll` in `bin\`. Set
`KTRACE_BUILD_RDP=OFF` to build the tracer, probe, and post-processing tools
without RGP support.

Set up the build environment:

```bat
set "VCVARS=C:\path\to\VisualStudio\VC\Auxiliary\Build\vcvars64.bat"
set "HIP=C:\path\to\hip-toolchain"
set "HIP_HEADERS=C:\path\to\runtime-matched-hip\include"
set "GPU_ARCH=gfx1100,gfx1103,gfx1150,gfx1151,gfx1152,gfx1200,gfx1201"
set "LLVM_ROOT=C:\path\to\llvm"
```

`VCVARS` is optional when running from a Visual Studio x64 Native Tools prompt.
`GPU_ARCH` has no default. It accepts one architecture or a comma/semicolon-
separated list to build a fat binary.
`LLVM_ROOT` only locates the optional `llvm-cxxfilt.exe` fallback. Normal
demangling uses `bin\amd_comgr_3.dll`, so it can be omitted.

`HIP` provides `hipcc.exe`. `HIP_HEADERS` defines the HIP API ABI used by the
tracer and must match the runtime loaded by the target process, not merely the
toolchain used to compile the application.

When the toolchain and target runtime are the same version, for example both
HIP 7.11, they can share one installation:

```bat
set "HIP=C:\path\to\hip-toolchain-7.11"
set "HIP_HEADERS=%HIP%\include"
```

If the application is compiled with HIP 7.11 but still loads HIP 7.2 from
System32, the settings must remain separate: `HIP` points to the 7.11
toolchain, while `HIP_HEADERS` points to headers matching the 7.2 runtime.

Build all targets:

```bat
build.bat
```

To build without the synthetic HIP workload:

```bat
set "KTRACE_BUILD_WORKLOAD=OFF"
build.bat
```

This mode does not require `HIP` or `GPU_ARCH`, but capture targets still
require runtime-matched `HIP_HEADERS`.

To build only the independent C++ post-processing tool, without any HIP SDK or
headers:

```bat
set "KTRACE_BUILD_CAPTURE=OFF"
set "KTRACE_BUILD_POSTPROCESS=ON"
set "KTRACE_BUILD_WORKLOAD=OFF"
build.bat
```

Artifacts are written to `bin\`:

- `ktrace_tracer.dll`: tracer injected into the target process
- `ktrace_collector.exe`: external shared-memory/ETW collector and spool writer
- `roctx64.dll`: Windows ROCTx runtime
- `ktrace.exe`: unified launcher and injector; `-a` additionally enables RGP
- `workload.exe`: synthetic HIP self-test workload
- `roctx_workload.exe`: ROCTx mark, nested-range, and cross-thread-range self-test
- `version_probe.exe`: runtime/header compatibility probe
- `ktrace_postprocess.exe`: independent native finalize/analyze/Perfetto tool
- `RdpCaptureApi.dll`: RDP capture API from RDTS, loaded only with `-a`
- `GPUPerfAPICounters-x64.dll`: GPUPerfAPI counter definitions from RDTS
- `amd_comgr_3.dll`: COMGR runtime from RDTS; usable for kernel-name demangling
- `amdgpu_dis.dll`: AMDGPU disassembly runtime from RDTS

The RDTS DLLs are present only when RGP support is enabled.
