# T0.1 spike — zero-copy registration, both directions

Throwaway spike per `docs/design/hybrid-npu-gpu-tasks.md` (Phase 0, T0.1). Do not
productionize anything here. This directory is the "written finding" plus retained
program the task asks for; see the note at the bottom about where the finding
actually lives (not appended to the tasks doc — see there for why).

## What this spike is trying to prove

Candidate A ("HIP owns", per the design doc's [Ownership: three
candidates](../../../docs/design/hybrid-npu-gpu-design.md#ownership-three-candidates)):
allocate host-mapped memory the same way morphizen's EP allocator does
(`hipHostMalloc(Mapped|Coherent)`), then hand that pointer *directly* to
DynamicDispatch's `bind_bo()` and confirm the NPU can read/write it in place —
no RyzenMM-allocated intermediate, no copy anywhere at the boundary.

The task text calls out the specific untested combination: *"whether XDNA accepts
a pointer the HIP driver has already pinned."* Every reference call site that
uses `bind_bo()` binds a RyzenMM-allocated pointer, not a HIP one — this spike is
the first thing to try the other combination.

## Two programs, two very different statuses

| Program | Needs | Status on this machine |
|---|---|---|
| `npu_registration_alloc_probe` | HIP only | **Built and ran. Results below.** |
| `npu_registration_npu_probe` | HIP + DynamicDispatch + XRT (+ ryzen_mm) | **Does not compile here — see "What did not run".** |

This machine is a developer laptop/workstation with two HIP-visible GPUs
(an integrated `gfx1036` and a discrete Radeon PRO W7900 `gfx1100`) and no
XDNA/NPU hardware at all. It is not the gfx1151 NPU host the task's
verification requires. `docs/remote-dev-workflow.md`'s gfx1151 remote and the
"NPU host" referenced by T0.1 are believed to be the same machine, which this
session has no SSH access to.

## What ran here: `npu_registration_alloc_probe`

Answers the two sub-questions from the T0.1 task text that don't need
XRT/DynamicDispatch, plus the CPU↔GPU half of the aliasing check.

Build (from this directory):

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 "-DHIP_ARCHITECTURES=gfx1036;gfx1100"
cmake --build build --config Release
./build/Release/npu_registration_alloc_probe.exe
```

(Substitute `-DHIP_ARCHITECTURES` for whatever the local box's HIP devices
report — `gfx1151` on the NPU host.)

### Results (this machine, 2026-08-19)

```
HIP device count: 2
Device 0: AMD Radeon(TM) Graphics (gcnArchName=gfx1036)

=== Alignment probe (140 size classes + 1 large) ===
Result: every hipHostMalloc(Mapped|Coherent) pointer, across all 140 size
classes and one large exact-size allocation, was 4 KiB aligned.

=== Freelist reissue probe ===
pooled class 70 (3538944 B): address was stable across 5 free/alloc cycles
  -> a registration made once for this physical pointer stays valid across
     every logical ORT Alloc()/Free() that maps to this size class -- no
     re-registration needed on the fast path.
large (33554432 B): first == after free+realloc (driver reused the VA range)
  -> any cached registration for the old pointer must still be treated as
     invalid once freed; the registry must resolve the new pointer as a
     fresh interval rather than assume the old registration still applies
     just because the address happened to come back the same.

=== CPU<->GPU aliasing probe ===
host ptr == device ptr (identical VA -- consistent with UMA on this box)
Result: GPU kernel wrote in place; CPU read the mutation with no explicit
copy: PASS

=== alloc_probe: ALL PASS ===
```

**Findings, confirmed on real hardware (not inferred):**

1. Every `hipHostMalloc(Mapped|Coherent)` pointer the allocator's exact
   size-class table can produce (128 B .. 16 MB, 140 classes) came back 4 KiB
   aligned, as did an unpooled 32 MB allocation. Matches the design doc's
   claim that this is free with `hipHostMalloc`.
2. A pooled buffer's address survives repeated free()/alloc() cycles at the
   same size class with **zero** HIP calls on the fast path (this mirrors
   `HipGpuAllocator`'s actual free-list code, not just a description of it —
   see the comment in `alloc_probe.cpp`). Consequence for the registry design:
   a registration is amortized across many logical ORT allocations for
   pooled sizes, not repeated per `Alloc()`.
3. An unpooled (>16 MB) buffer's address is **not** guaranteed stable across
   free+realloc (on this run the driver happened to reuse the VA, but nothing
   guarantees that) — the registry must not assume an old registration is
   still valid without re-resolving.
4. A HIP kernel mutated a `hipHostMalloc(Mapped)` buffer in place via
   `hipHostGetDevicePointer`, and the CPU observed the mutation through the
   *original* pointer with no explicit copy anywhere in between — the CPU↔GPU
   half of the "confirm addresses alias, not merely that values agree" bar
   the task sets.

**Caveat:** this ran on `gfx1036`/`gfx1100`, neither of which is `gfx1151`.
Per the design doc's [registered-memory
rule](../../../docs/design/hybrid-npu-gpu-design.md#the-registered-memory-rule),
`gfx1150`-family UMA behavior can silently mask bugs that only show up on
`gfx1151`'s true device memory. Findings 1, 2, and 4 above are about
`hipHostMalloc`/host-mapped-memory behavior specifically (not about
`hipMalloc`/device memory), so they are not expected to be arch-sensitive in
the way the registered-memory rule warns about — but that expectation itself
needs confirming on `gfx1151`, not assumed.

### A local environment trap hit while running this, worth recording

`hipHostMalloc(Mapped)` segfaulted unconditionally on this machine (even a
`hipGetDeviceCount`-only program was fine; adding one `hipHostMalloc` call
crashed regardless of which of the two GPUs was selected). Root cause: this
machine had `HIP_DEVICE_LIB_PATH` pointing at a *different* ROCm/TheRock
install (`D:\develop\drivers\ROCM\therock\...`) than the one providing
`amdhip64_7.dll` (`ROCM_PATH=E:\develop\hipdnn\dist\therock`). Unsetting
`HIP_DEVICE_LIB_PATH` fixed it immediately. This is a pre-existing local
environment inconsistency unrelated to hip-ep or to Candidate A — recorded
here only so it doesn't cost someone else an hour if they hit the same
segfault on this or a similarly-configured box.

## What did not run: `npu_registration_npu_probe`

This is Candidate A's actual NPU leg — the untested combination the gate
exists to check. It needs DynamicDispatch, XRT, and (as a fallback path
comparison) `ryzen_mm`, none of which this workspace vendors and none of
which are installed as CMake packages here (XRT itself resolves via
`find_package` — a `D:\develop\drivers\XRT\currentxrt` install exists on this
machine — but DynamicDispatch and `ryzen_mm` do not, and neither the DD/XRT
headers nor `ops/op_interface.hpp` are reachable from this workspace or from
the read-only hybrid-llm reference checkout used to write this file — see
"Provenance" below). `CMakeLists.txt`'s `find_package(...)` calls correctly
find nothing and skip the target — this was verified by configuring and
confirming only `npu_registration_alloc_probe` and `gpu_touch` appear in the
build, not by disabling or faking the check.

**Provenance of `npu_probe.cpp`'s API calls.** This workspace has no
DynamicDispatch/XRT/RyzenMM SDK and no hybrid-llm checkout, but a read-only
partial checkout of the hybrid-llm reference tree was available on this
machine outside the hip-ep workspace
(`onnx_custom_ops/hybrid_llm/npu/{binary_elemwise_npu_kernel.hpp,mul.hpp,mul.cpp,npu_op.cpp,matmulnbits.cpp}`).
`npu_probe.cpp`'s call sequence — `bind_bo(ptr, len)`,
`set_tensor_shape`/`get_buffer_reqs`, `run(ins, outs)`, the
`ElwMul<bfloat16,bfloat16,bfloat16>` instantiation, `getCommonAttrs()`'s exact
attribute map — is cribbed from those files, not invented. What is **not**
verified, because that tree does not vendor DynamicDispatch's own headers
either (it consumes `DynamicDispatch::dyn_dispatch_core` as an external
package the same way this spike's `CMakeLists.txt` tries to):

- The real include paths for `::Tensor`, `OpArgMap`, `NPUBufferSpan` (best
  guesses in `npu_probe.cpp`, annotated inline).
- Whether `bind_bo` has a third `read_only` parameter — the task text mentions
  `bind_bo(void*, size, read_only)`, but every call site found in the
  reference tree uses the two-argument form. Try two-argument first.
- The `DDKernel` constructor's first `bool` argument's real meaning (always
  passed as `true` at reference call sites; never explained there).
- The exact `find_package` names/config-module locations for
  `DynamicDispatch` and `ryzen_mm` on the NPU host (XRT's resolved cleanly
  here via its installed `XRTConfig.cmake`; there was nothing to crib the
  other two from).

### To finish this spike on the gfx1151 NPU host

1. `del %TEMP%\morphizen_mlir_*` is irrelevant here (no compiled model
   involved), but do invalidate/rebuild if DynamicDispatch or the kernel
   library itself changed since last built.
2. Confirm/replace the three `find_package()` calls in `CMakeLists.txt` with
   however that host's build actually locates DynamicDispatch/XRT/ryzen_mm
   (vcpkg toolchain file, explicit `<Pkg>_DIR`, etc.) so
   `npu_registration_npu_probe`'s `if(TARGET ...)` guard sees real targets.
3. Reconcile the "not verified" list above against
   `DynamicDispatch/include/ops/op_interface.hpp` directly. Expect at least
   one include-path fix.
4. Configure with `-DHIP_ARCHITECTURES=gfx1151`, build, run
   `npu_registration_npu_probe.exe`.
5. The gate is met only if that program prints both `PASS` lines (NPU
   compute-in-place, then GPU-mutates-NPU-output-in-place) **and** the
   `bind_bo` timing line is sane (not, e.g., seconds — that would suggest a
   staging copy hiding inside the SDK rather than true registration).
6. Read the two `sync(...)` timing lines, which are a separate question from
   the `bind_bo` one. `bind_bo` happens once; `sync` happens on every
   inference. On a UMA part an imported buffer should need cache maintenance
   at most, so an implied throughput near DRAM bandwidth means XRT is moving
   the bytes — a per-inference boundary copy that the EP's own counters
   cannot see, because it happens below them. Every value would still
   compare equal and zero copy would be silently false. If that is what the
   numbers show, record it as a finding against Decision 3 rather than
   passing the gate: it changes which ownership candidate is viable, not
   merely how fast this one is.
6. If it instead throws, crashes, or an SDK error indicates XDNA rejected the
   HIP-pinned pointer: that is exactly the "none of the three candidates
   work" outcome the task's gate calls out as a stop condition, not a
   fallback signal. Try Candidate B or C (see the design doc's Ownership
   section) before concluding zero-copy is unreachable.
7. Record the outcome and the chosen Decision 3 owner — see "Where this
   finding actually lives" below.

## Design invariants this spike's own code follows

- No boundary copy is introduced anywhere in either probe: `alloc_probe`'s
  GPU kernel operates on the `hipHostGetDevicePointer` view of the *same*
  allocation; `npu_probe` reads NPU output back through the original
  `hipHostMalloc` pointer, never through a copy, and hands the *same* buffer
  to the GPU-touch kernel afterward.
- `gpu_touch.hip`'s kernel launch clears `hipGetLastError()` before the
  launch and returns the post-launch status rather than assuming success.
- `gpu_touch.hip` is a separate hipcc-compiled translation unit specifically
  so `npu_probe.cpp` (which links MSVC-built DynamicDispatch/XRT/ryzen_mm
  static libraries) never needs clang-cl and MSVC object code in the same
  TU. Proving that MSVC/clang-cl coexistence works *at all* in one binary is
  T0.2's job ("prove three toolchains coexist"), not T0.1's; this spike
  sidesteps the question rather than accidentally depending on its answer.

## Where this finding actually lives

`docs/design/hybrid-npu-gpu-tasks.md` asks Phase 0 spikes to append their
finding to that same file. The agent instructions for this task explicitly
forbid editing either `docs/design/hybrid-npu-gpu-design.md` or
`docs/design/hybrid-npu-gpu-tasks.md`, so the finding lives here instead. A
maintainer should fold a summary (both the local half's results and the
"gate not yet met — see below" status) into the tasks doc's T0.1 entry by
hand.

## Gate status

**Not met.** The task's gate requires "one candidate demonstrably works, with
aliasing proven rather than inferred, and Decision 3 is recorded with the
chosen owner." This session proved the CPU↔GPU half of Candidate A's
allocation strategy on non-NPU hardware. It did not — and on this machine,
could not — exercise the NPU leg, so:

- Decision 3 (memory ownership) is **still open**.
- No candidate has been demonstrated end-to-end (CPU, GPU, *and* NPU aliasing
  the same physical allocation).
- Phase 1 should not start on the assumption that Candidate A (or any
  candidate) works until `npu_registration_npu_probe` has actually run on
  the gfx1151 NPU host and the steps under "To finish this spike" above have
  been completed.
