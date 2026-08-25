# T0.2 spike — linkage and runtime smoke

Throwaway spike per `docs/design/hybrid-npu-gpu-tasks.md` (Phase 0, T0.2). Do not
productionize anything here. Companion to `spikes/npu/registration/` (T0.1).

## What this spike is trying to prove

Per the task text: "Prove three toolchains coexist. Build a minimal DLL against
the prebuilt DynamicDispatch with its own runtime settings, exporting one C
function that runs one trivial DD operator. Load it from a process that has
already loaded HIP, ORT, and this repository's EP DLL. Allocate through the EP
allocator, free through it, and run the DD operator, in that order and
repeatedly." Watched for: heap corruption across the runtime boundary, under
an application-verifier-class heap check.

**This machine has no NPU** (per the task assignment). That does not make the
task empty: everything that does not require actually driving an XDNA device
was built and, further than T0.1's spike managed, largely *run* here — see
"What ran here" below for exactly how far, and "What did not run" for the
precise wall this session hit and why.

## Two programs, matching the design doc's toolchain split

| Program | Toolchain | Purpose |
|---|---|---|
| `dd_op_dll/` | DD's own settings: dynamic CRT (`/MD`), DD's own protobuf/nlohmann_json | Exports one C function (`dd_linkage_run_trivial_op`) that runs one DD operator |
| `harness/` | hip-ep's own settings: static CRT (`/MT`, the repo default) | Links HIP + ORT directly, loads this repo's built EP DLL through ORT's real plugin-EP mechanism, obtains the real `HipGpuAllocator`, then `LoadLibrary`s `dd_op_dll` and drives the allocate → run-op → free loop |

The two are separate CMake projects on purpose — mirroring
[the design doc's shim boundary](../../../docs/design/hybrid-npu-gpu-design.md#packaging-and-the-shim-boundary):
the DD-linked artifact is its own DLL, built with DD's toolchain, loaded at
runtime by name, never linked into the harness.

## A discovery worth recording up front

T0.1's spike (`spikes/npu/registration/README.md`, written one day earlier on
what is presumably this same or a similarly-provisioned machine) found *no*
DynamicDispatch/XRT/ryzen_mm SDK reachable from this workspace and had to
guess at `find_package()` incantations. That is no longer true: a full,
already-built hybrid-llm checkout with a real `install/` tree exists at
`D:\develop\git\gitenterprise\hybrid-llm\install`, including:

- `install/lib/cmake/DynamicDispatch/DynamicDispatchConfig.cmake` (a real
  `find_package(DynamicDispatch CONFIG)` target, `DynamicDispatch::dyn_dispatch_core`,
  built `SHARED`)
- `install/include/ryzenai/dynamic_dispatch/...` — the real public DD headers
- `D:\develop\drivers\XRT\currentxrt\share\cmake\XRT\xrt-config.cmake` — a real
  `find_package(XRT CONFIG)` target

This unblocks a real build (not a stub) against DynamicDispatch on this
machine, which is what everything below uses. A maintainer revisiting T0.1's
"cannot compile the NPU leg here" finding should know this tree now exists —
that is not this task's job to act on, only to flag; see T0.1's own README for
its still-open gate.

## What ran here

### Build: both sides compile and link against the real SDKs

```bash
# dd_op_dll (DD's own /MD toolchain)
cd spikes/npu/linkage/dd_op_dll
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
  -DHYBRID_LLM_INSTALL_DIR="D:/develop/git/gitenterprise/hybrid-llm/install" \
  -DNLOHMANN_JSON_CMAKE_DIR="D:/develop/drivers/ROCM/therock/share/cmake/nlohmann_json/.." \
  -DXRT_DIR="D:/develop/drivers/XRT/currentxrt"
cmake --build build --config RelWithDebInfo   # NOT Release -- see gotcha #1 below

# harness (hip-ep's own /MT toolchain)
cd ../harness
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
  -DHIP_ARCHITECTURES=gfx1036 \
  -DORT_INCLUDE_DIR="E:/develop/onnx-hipdnn/workspace/prebuilt-local/include/onnxruntime" \
  -DORT_LIB="E:/develop/onnx-hipdnn/workspace/prebuilt-local/lib/onnxruntime.lib"
cmake --build build --config Release
```

Both configure and build cleanly. Three non-obvious gotchas hit along the way,
recorded here so the NPU host doesn't lose time on them again:

**Gotcha 1 — build DD's DLL as `RelWithDebInfo`, not `Release`.**
`DynamicDispatch-targets-relwithdebinfo.cmake` is the only per-config file the
installed package ships; there is no `-release.cmake` variant. Building
`Release` configures fine but fails to *link* with four `LNK2019 unresolved
external symbol` errors for every `ryzenai::*` symbol used, because CMake has
no `IMPORTED_IMPLIB` mapping for that configuration and silently drops the
import library. This looks exactly like a missing-export problem (see gotcha
2) until you check the configuration.

**Gotcha 2 — `ryzenai::elw_add<uint16_t, uint16_t, uint16_t>` is not
exported by this particular prebuilt `dyn_dispatch_core.dll`, even though
its explicit template instantiation is present in DynamicDispatch's *source*
tree** (`src/ops/dmacompiler/elwadd/elwadd.cpp:1085`). A `dumpbin /exports`
against `install/lib/dyn_dispatch_core.lib` confirms zero `elw_add` symbols in
1857 exported names, while `ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>`
*is* exported. This spike switched to `elw_mul` for that reason alone, not
because it is more representative — see `dd_op_dll/dd_linkage_op.cpp`'s
comment. (`elw_mul`'s constructor still requires the dtype string
`"bfloat16"`, not `"uint16"`, even though the C++ storage type is
`uint16_t` — DD's dtype tag and its C++ template parameter are independent;
passing the wrong string throws `DD_THROW` immediately.)

**Gotcha 3 — `onnxruntime.dll` version shadowing.** This machine has a
system-installed `C:\Windows\System32\onnxruntime.dll` (ORT 1.17.1, no plugin-EP
API). Windows' DLL search order checks the directory containing the loading
module and `System32` *before* `PATH`, so simply prepending the correct
`onnxruntime.dll`'s directory to `PATH` is not enough — the harness loaded
System32's copy and immediately hit `"requested API version [25] is not
available, only [1, 17] are supported"`. Fixed by copying the matching
`onnxruntime.dll` (and `onnxruntime_providers_shared.dll`) next to
`npu_linkage_harness.exe` itself, which *is* checked first. Do the same on the
NPU host if it has any other ORT install on `PATH`/`System32`.

### Run: HIP + ORT + this repo's EP DLL coexist; the real EP allocator survives 500 alloc/free cycles

```bash
cd spikes/npu/linkage/harness/build/Release
cp <prebuilt-local>/bin/onnxruntime.dll <prebuilt-local>/bin/onnxruntime_providers_shared.dll .
unset HIP_DEVICE_LIB_PATH   # see gotcha #4
export PATH="<therock>/bin:<prebuilt-local>/bin:<hybrid-llm install>/bin:<XRT currentxrt>:$PATH"
export DD_ROOT="<hybrid-llm>/DynamicDispatch"   # only matters once dd_op_dll loads, see below
appverif -enable Heaps Handles Leak -for npu_linkage_harness.exe   # optional, see below
./npu_linkage_harness.exe \
  --ep-dll "<prebuilt-local>/bin/onnxruntime_morphizen_ep.dll" \
  --dd-dll "<repo>/spikes/npu/linkage/dd_op_dll/build/RelWithDebInfo/dd_linkage_op.dll" \
  --iterations 500
appverif -disable * -for npu_linkage_harness.exe   # cleanup
```

Actual output on this machine (500 iterations, Application Verifier `Heaps
Handles Leak` enabled for the process):

```
[harness] hipGetDeviceCount -> no error, count=2
[harness] LoadLibrary(.../dd_linkage_op.dll) failed: 126 -- continuing WITHOUT the DD leg (EP allocator loop still runs).
[harness] registered EP library <prebuilt-local>/bin/onnxruntime_morphizen_ep.dll
[harness] EP device: CPUExecutionProvider
[harness] EP device: DmlExecutionProvider
[harness] EP device: DmlExecutionProvider
[harness] EP device: MorphiZenExecutionProvider
[harness] EP device: MorphiZenExecutionProvider
[harness] obtained HipGpuAllocator via CreateSharedAllocator
[harness] loop done: 500 iterations, 0 DD-op ok, 0 DD-op failed (dd_run_trivial_op NOT loaded -- see above), no crash
[harness] HARNESS COMPLETED, NO CRASH
```

Exit code 0, no Application Verifier break-in, no crash, across 500
allocate-through-the-real-`HipGpuAllocator` / free cycles, in a process that
directly linked HIP and ORT and dynamically loaded the actual, already-built
`onnxruntime_morphizen_ep.dll` through ORT's standard plugin-EP registration
path (`RegisterExecutionProviderLibrary` → `GetEpDevices` →
`Env::CreateSharedAllocator`, which calls
`MorphiZenEpFactory::CreateAllocatorImpl` exactly as production does, without
needing a session — see the note on `Ort::Session`/Issue #031 below). That is
two of the task's three toolchains coexisting under a heap check, doing real
work, repeatedly, with no corruption observed.

**Gotcha 4 — a pre-existing local environment trap, already documented by
T0.1's README, reproduced identically here:** `HIP_DEVICE_LIB_PATH` was set to
a *different* ROCm/TheRock install than the one providing `amdhip64_7.dll`.
Before unsetting it, every call into `HipGpuAllocator::Alloc` (i.e. the first
`hipHostMalloc`) segfaulted the harness. This is a local machine
misconfiguration, not a project bug — see
`spikes/npu/registration/README.md`'s "A local environment trap hit while
running this" for the first occurrence.

**Why the allocator path skips `Ort::Session`.** `morphizen/ort-bridge/test/src/test-hello-ep.cpp`'s
`CreateSession` test is currently `GTEST_SKIP()`-marked ("Target auto-discovery
failure, see Issue #031"). Rather than depend on that unrelated, already-known-broken
path, this harness uses `OrtApi::CreateSharedAllocator` (ORT 1.23+, `env.h`
`\since Version 1.23`), which takes an `OrtEpDevice*` directly and returns the
factory's real allocator without creating a session at all. This is a
supported ORT API, not a workaround specific to this spike — but it does mean
this spike does not exercise session creation, only the allocator.

## What did not run: the DD leg

`LoadLibrary(dd_linkage_op.dll)` fails with `126` (`ERROR_MOD_NOT_FOUND`) on
this machine. `dumpbin /dependents` on `dd_linkage_op.dll` and, in turn, on
`dyn_dispatch_core.dll` traces this to a **hard, non-delay-load dependency on
`spdlog.dll`** that `dyn_dispatch_core.dll` itself carries (`xrt_coreutil.dll`
is only a *delay-load* dependency and is not the problem — a standalone
`ctypes.WinDLL()` load of it alone succeeds). `spdlog` is one of
`DynamicDispatch/env.yml`'s conda dependencies, built into whatever conda
environment produced this specific `dyn_dispatch_core.dll`; that environment
is not this machine, and a search of the entire `D:` drive plus every local
conda env (`modelbuilder`, `onnxmodelgen`, `onnxmodifier`) found no
`spdlog.dll` anywhere. Unlike `nlohmann_json` (header-only — any
version-compatible copy satisfies the CMake `find_dependency`, which is why
substituting a ROCm-vendored copy for that one was fine), `spdlog.dll` is a
compiled shared library whose ABI must match what `dyn_dispatch_core.dll` was
actually linked against; there is nothing to substitute it with here, and
building one from source to match an unknown exact conda-pinned version is
not a reasonable use of this session's scope.

This is **not** the NPU-hardware wall the task expects to eventually hit — it
is one level earlier, a packaging/redistribution gap specific to this
dev-build install tree. The actual NPU host is expected to have a complete,
working RyzenAI/DynamicDispatch install (that is what makes it usable for
Phase 5+ at all), which would include `spdlog.dll` alongside the rest of the
runtime it depends on. **This blocker is not expected to reproduce there.**

The harness was deliberately written so this failure is non-fatal (see
`harness_main.cpp`'s comment at the `LoadLibrary` call): everything provable
without the DD leg still runs and is reported above, rather than the whole
run aborting on the one part that can't complete here.

## What remains for the gfx1151 NPU host

1. Confirm `spdlog.dll` (or an equivalent complete redistribution of that
   `dyn_dispatch_core.dll` build) is resolvable there — expected to already be
   true given a working RyzenAI install.
2. Re-run the exact command in "Run" above, unchanged, with `--dd-dll`
   pointing at a `dd_op_dll.dll` rebuilt against that host's DynamicDispatch
   install (reconfigure `HYBRID_LLM_INSTALL_DIR`/`XRT_DIR` for that host's
   paths; the CMake files here take them as variables for exactly this
   reason).
3. Expect `dd_linkage_run_trivial_op` to now load and run
   `ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>("bfloat16", load_xrt=true,
   {})`'s constructor for real, which is the point that actually opens an XRT
   context against a real XDNA device — the genuine hardware-dependent step
   this spike could not reach here. If it throws or crashes there, capture
   the exact exception text (`dd_run_trivial_op`'s `err_msg` is designed to
   carry it back through the C ABI without an exception ever crossing it) —
   that is itself useful evidence for T0.3.
4. Re-run under Application Verifier (`Heaps Handles Leak` was used here; add
   more checks as desired) with all three toolchains actually exercising real
   work in the loop, for a sustained iteration count (hundreds to thousands).
   **This is what actually satisfies the task's gate** — what ran on this
   machine is real progress on two of the three legs, not a substitute for
   it.
5. If it passes clean: the gate is met. If Application Verifier reports heap
   corruption: that is exactly the failure mode this task exists to catch,
   and per the design doc's stop condition ("Toolchains cannot coexist in one
   process... this is architectural"), escalate rather than try to route
   around it.

## Design invariants this spike's own code follows

- `dd_linkage_op.h` is the only file both sides see -- deliberately narrow (a
  raw pointer, a size, an error buffer, no C++ types), mirroring the
  production shim boundary described in
  [Packaging and the shim boundary](../../../docs/design/hybrid-npu-gpu-design.md#packaging-and-the-shim-boundary).
  This spike does not implement that boundary; it only smoke-tests that a
  boundary shaped like it can coexist with HIP + ORT + this repo's EP DLL in
  one process.
- **No exception crosses the DLL boundary.** `dd_linkage_run_trivial_op`
  catches everything (`std::exception` and `...`) and reports failure as a
  return code plus a message string, per the same rule that governs the
  output-allocator callback and the eventual production shim.
- The harness never links `dd_op_dll`'s import library; it `LoadLibrary`s +
  `GetProcAddress`es it by name at runtime, matching how the production shim
  is loaded.
- The buffer handed to the DD operator comes from the real
  `morphizen::HipGpuAllocator`, obtained the same way ORT itself would obtain
  it (`CreateSharedAllocator` against a real `OrtEpDevice`) -- not a
  hand-rolled stand-in allocator.
- This spike does not implement or claim zero-copy: the DD operator's `Tensor`
  interface stages internally (DD's own `xrt::bo` copy-in/copy-out), which is
  DD's business, not a boundary copy introduced by this repository. Zero-copy
  registration is T0.1's job, not T0.2's.

## Gate status

**Not met — partially demonstrated.** The task's gate is "no crash, no heap
corruption, over a sustained loop" with all three toolchains (HIP, ORT +
this repo's EP DLL, and DD's own `/MD` build) coexisting and doing real work.
On this machine:

- HIP + ORT + this repo's real, already-built EP DLL coexist in one process,
  and 500 allocate/free cycles through the real production `HipGpuAllocator`
  ran clean under Application Verifier's heap checks. **This much is
  demonstrated, not just built.**
- The third toolchain (DD's `/MD` DLL) builds successfully against a real
  prebuilt DynamicDispatch but cannot be *loaded* into the process on this
  machine, for a documented, non-NPU-related reason (`spdlog.dll` missing
  from this dev-build install tree). It has therefore not yet been exercised
  in the same process as the other two.
- No candidate NPU hardware exists on this machine at all, so even with
  `spdlog.dll` resolved, the DD operator's actual device-open call could not
  have been reached here regardless.

Phase 1 should not start on the assumption that three-way coexistence is
proven until the steps under "What remains for the gfx1151 NPU host" above
have actually been run there.
