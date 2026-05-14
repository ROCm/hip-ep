# Per-gfx backend DLL

This document describes how heavy ROCm kernels (Composable Kernel today;
future hipBLASLt-tuned kernels, etc.) are isolated into a per-gfx-arch
DLL loaded at runtime by `model.dll`, and the vtable C ABI that connects
the two.

> **Authoritative for the backend ABI, runtime client, and CK build
> details.** CLAUDE.md links here for those topics; new sessions should
> read this doc end-to-end before touching `lib/Backend/`,
> `lib/Runtime/real/hip_backend_client.{h,cpp}`, `lib/Runtime/real/ck_conv.cpp`,
> or the `HIPBackendVTable` contract in `lib/Backend/hipdnn_ep_backend.h`.

---

## Why a separate DLL

The EP DLL invokes `lld::lldMain` in-process to link each compiled
`model.dll` (`lib/Target/LLVM/DLLLinker.cpp`). Pointing lld at
`device_conv_operations.lib` (TheRock ships it at ~2.6 GB) silently hangs
and takes down the python process with no traceback (the cpptrace handlers
don't catch it). So we built CK once, with MSVC `link.exe`, into a single
SHARED DLL that ends up ~6 MB after dead-code stripping. Every compiled
`model.dll` then loads that DLL via `LoadLibrary` / `dlopen` -- the heavy
link cost is paid ONCE at backend-DLL build time, not on every model
compile in the EP pipeline.

The DLL is named `hip-backend-gfx<ARCH>.dll` on Windows or
`libhip-backend-gfx<ARCH>.so` on Linux (e.g. `hip-backend-gfx1151.dll`).
Multiple per-arch DLLs can coexist in `install/dist/bin/`; the runtime
client (`hip::Backend`, see below) picks the right one for the current
device via `hipGetDeviceProperties().gcnArchName`.

---

## C ABI: single export, vtable-shaped

`lib/Backend/hipdnn_ep_backend.h` is the single source of truth.

The backend exposes ONE symbol: `HIPBackendAPI`, a pointer to a static
`HIPBackendVTable` instance. `lib/Backend/hip_backend.def` is permanently
a 1-line file:

```
EXPORTS
    HIPBackendAPI DATA
```

The `DATA` keyword is mandatory for variable exports on MSVC -- without
it the linker treats the symbol as a function and the client's
`GetProcAddress` returns a thunk; dereferencing crashes.

Vtable layout (`HIPBackendVTable` in the header):

```c
typedef struct HIPBackendVTable {
  /* Header. Stable across all ABI versions. */
  uint32_t      abi_version;     /* == HIP_BACKEND_API_VERSION */
  const char   *arch;            /* e.g. "gfx1151"; static storage */

  /* Lifecycle. NULL => no-op. */
  int         (*init)(void);
  void        (*shutdown)(void);

  /* Configuration. NULL => unsupported (older backend). */
  void        (*set_scratch_provider)(void *ctx,
                                      hip_backend_scratch_provider_fn provider);

  /* Ops. NULL => not implemented. APPEND-ONLY. */
  hip_backend_conv_fwd_fp16_nchw_fn conv_fwd_fp16_nchw;
} HIPBackendVTable;
```

### ABI evolution rule

- **Appending a new op slot at the END is additive** -- DOES NOT bump
  `HIP_BACKEND_API_VERSION`. Older clients walk only the prefix they
  know; newer clients null-check the slot before calling so a backend
  that hasn't implemented the op yet is reported as "not implemented"
  rather than crashing.
- **Removing, reordering, or changing a slot's signature MUST bump
  `HIP_BACKEND_API_VERSION`.** Mismatched versions are refused at
  backend-load time.

The first slot's position is fixed forever (`abi_version` first, `arch`
second) so the mismatch check itself doesn't depend on the rest of the
layout.

---

## Runtime client: `hip::Backend` + `hip::GetBackend`

`lib/Runtime/real/hip_backend_client.{h,cpp}` defines a small RAII wrapper.

`hip::Backend`'s ctor:
1. Detects gfx via `hipGetDeviceProperties(&props, 0).gcnArchName`,
   stripping any `:xnack-` / `:sramecc+` target-id suffix.
2. Constructs the DLL filename (`hip-backend-<arch>.dll` on Windows,
   `libhip-backend-<arch>.so` on Linux) and loads it via a tiny portable
   shim (Win32 `LoadLibraryA`/`GetProcAddress`/`FreeLibrary` declared
   inline so `<windows.h>` doesn't enter bitcode TUs; `dlfcn.h` on
   Linux).
3. Resolves `HIP_BACKEND_API_SYMBOL` (`"HIPBackendAPI"`); the dlsym
   result is the address of the exported pointer variable, so we
   dereference once to get the vtable pointer.
4. Validates `vtable->abi_version == HIP_BACKEND_API_VERSION` and that
   `vtable->arch` matches the device arch we detected.
5. Calls `vtable->init()` if non-null.

The dtor calls `vtable->shutdown()` (if non-null) and then `FreeLibrary`
/ `dlclose`.

Per-op methods (today: `Backend::Conv`, `Backend::SetScratchProvider`)
null-check the vtable slot and throw `std::runtime_error` if the backend
doesn't implement the slot or returns a non-zero status. The
`extern "C" wrap_*` shims that generated MLIR code targets wrap the call
in try/catch and convert exceptions to `return -1` so the C ABI generated
code expects is preserved.

### Lifetime: weak_ptr-cached, anchored on RuntimeState

`hip::GetBackend()` returns `std::shared_ptr<Backend>`, lazy-constructed
on first call. An internal `std::weak_ptr` re-uses the existing instance
across concurrent callers; when the last `shared_ptr` drops, the
backend's dtor unloads the DLL.

`inference_init` (`lib/Runtime/hipdnn_ep_runtime_state.cpp`) parks one
strong ref in `RuntimeState::backend_holder` (real `std::shared_ptr<hip::Backend>`
field, since RuntimeState is now allocated via `new`/`delete`), so the
backend lives at least as long as the session. Op-site code calls
`hip::GetBackend()` directly via the weak_ptr cache without worrying
about lifetime; the holder field only exists so backend lifetime tracks
session lifetime cleanly.

### Failure handling

`hip::GetBackend()` failures from `inference_init` are caught and demoted
to a `[hip_ep] backend not available: ...` warning -- subsequent
`wrap_*` calls that need the backend will surface the same error via a
per-op `[<op>] ...` line + `return -1`. Models that don't use the backend
keep working.

---

## Scratch buffer: provided by model.dll

The backend does not own GPU scratch. At session init, model.dll
registers a `hip_backend_scratch_provider_fn` callback via the vtable's
`set_scratch_provider` slot:

```c
typedef void *(*hip_backend_scratch_provider_fn)(void *ctx,
                                                 size_t needed_bytes);
```

The backend stores the `(ctx, provider)` pair globally. Op
implementations (e.g. `lib/Backend/conv.cpp`) call
`backend_get_scratch(needed_bytes)` -- a thin file-static helper in
`lib/Backend/main.cpp` that forwards to the registered provider.

In the EP runtime, the registered provider is
`hipdnn_ep_runtime_scratch_provider` (in
`lib/Runtime/hipdnn_ep_runtime_state.cpp`):

```cpp
void *hipdnn_ep_runtime_scratch_provider(void *ctx, size_t needed_bytes) {
  auto *state = static_cast<RuntimeState *>(ctx);
  if (!state || needed_bytes == 0) return nullptr;
  if (hipdnn_ep_state_ensure_workspace(state, needed_bytes) != 0)
    return nullptr;
  return hipdnn_ep_state_get_workspace(state);
}
```

So the backend pulls scratch from the model's existing shared workspace
(matmul / GQA / conv all share one grow-on-demand buffer; serialized on
the HIP stream so reuse is safe). This eliminates the
`hipMalloc`'d-by-the-backend scratch that used to live in
`lib/Backend/conv.cpp::ConvState::scratch`, halving the long-lived GPU
allocations on processes that load the backend at all.

### Concurrency caveat

The backend stores ONE `(ctx, provider)` globally. With concurrent
sessions in the same process, the most recent `SetScratchProvider` call
wins. This is safe because:

- `RuntimeState` is documented as not thread-safe (one inference per
  state at a time).
- OGA-style workloads are sequential.
- Each session's `state_init` runs before that session's first op
  dispatch, so the provider is correctly set when needed.

If concurrent sessions are needed in the future, route the provider
through a per-call argument instead of a setter.

---

## DLL discovery

No compile-time arch baking. `hip::Backend::Backend()` constructs the
filename from the runtime-detected `gcnArchName` and lets the OS resolve
it via the standard search path. `install/dist/bin` is already on PATH
for the EP DLL itself, which covers the backend lookup. Moving a model
between hosts with different gfx requires no rebuild as long as a
backend DLL for the target arch is present.

If a given build doesn't ship a backend for the device's gfx,
`inference_init` logs a `[hip_ep] backend not available: ...` warning
and continues -- ops that need it will fail at dispatch with a
`[ck_conv] ...` (or per-op) line.

---

## CK layout inside the backend

`lib/Backend/conv.cpp` instantiates CK's
`DeviceGroupedConvFwdMultipleABD<2, NHWGC, GKYXC, ..., NHWGK, F16, F16,
..., F16, PassThrough, PassThrough, PassThrough>`. WMMA cshufflev3 fp16
instances ship for `NHWGC / GKYXC / NHWGK` only on RDNA3+. Per call, the
backend internally:

1. Transposes the activation NCHW → NHWC (G dim collapses to C since
   G·C_per_g = C).
2. Materializes a GKYXC weight copy (negligible cost vs the conv itself
   for typical shapes; revisit if profiling shows otherwise).
3. Iterates `Factory::GetInstances()` and picks the first instance whose
   `IsSupportedArgument` returns true (cached per shape).
4. Optional bias add (NHWC channels-innermost makes a per-token
   broadcast trivial).
5. Detransposes NHWC → NCHW into the caller's output buffer.

The per-shape cache (`DeviceOp` + `BaseInvoker`) is a process-global
mutex-protected `unordered_map<ConvShapeKey, ConvCacheEntry>`. Cache
entries hold `unique_ptr`s; clearing the cache (in `backend_shutdown`)
runs the CK kernel-handle dtors transitively.

Scratch comes from the model.dll's pool via the registered provider
(see "Scratch buffer" above) -- nothing GPU-allocated lives in
`lib/Backend/conv.cpp`'s state.

---

## CK build details that bite

1. **`CK_ENABLE_DL_KERNELS` is undef'd in TheRock's CK build**
   (`install/therock/include/ck/config.h`) so fp32 NHWGC has zero
   precompiled instances. Only fp16/bf16 WMMA cshufflev3 (RDNA3+) and
   XDL (CDNA-only) instances ship. The backend is therefore fp16-only
   on gfx1151.
2. **CK headers don't bitcode-compile under bare clang `-emit-llvm`
   host-only mode** (`__clz`, `bhalf_t`, `std::byte`, template-`auto`).
   Hence the split: the backend DLL is built by hipcc, the model.dll
   bitcode TU only sees the C vtable.
3. **CRT mismatch**: CK is built upstream against the dynamic CRT (/MD).
   model.dll uses static CRT (/MT) per project convention. The backend
   DLL overrides the project-wide /MT with `MSVC_RUNTIME_LIBRARY
   "MultiThreadedDLL"`; hipcc-compiled .obj files get
   `-D_DLL -D_MT --dependent-lib=msvcrt` for the same reason. CRT
   mismatch across the dlsym boundary is harmless (only data crosses).
4. **`hip_backend.def` is mandatory**. Without a .def file, the SHARED
   library would have zero exports (CMake's default for SHARED without
   dllexport markup). Use a .def file rather than
   `WINDOWS_EXPORT_ALL_SYMBOLS TRUE` so internal helpers stay private.
   Adding a new public op does NOT require editing this file -- ops are
   added by appending a slot to `HIPBackendVTable` and filling it in in
   `main.cpp`'s `g_vtable` initializer.
5. **Imported targets carry hipcc-only flags**. `hip::host` +
   `composable_kernel::*` carry `INTERFACE_COMPILE_OPTIONS` like `-x hip`
   that propagate to the SHARED-lib's auto-generated dummy/shell .cpp
   via `target_link_libraries`, where MSVC `cl.exe` rejects them.
   Workaround: link raw `.lib` paths (read from the imported targets via
   `get_target_property(... IMPORTED_LOCATION_RELEASE)`) instead of the
   imported targets themselves at the SHARED-lib link stage. The
   hipcc-compiled .obj files already embed the right hip/CK calls.
6. **Backend headers don't auto-track in `_hip_compile_sources`**. The
   `_hip_compile_sources` helper from
   `3rd-party/custom_kernels/cmake/hip_utils.cmake` doesn't track header
   deps, so editing a header in `lib/Backend/` requires
   `python build.py --clean` or a `touch` on a tracked source.
7. **`main.cpp` is plain C++, NOT compiled by hipcc.** It contains the
   static `g_vtable` with function-pointer initializers. Compiling it
   through hipcc would also emit a device-side object containing
   `g_vtable`, and the device linker would (correctly) reject the
   host-only target functions as undefined-on-device. `main.cpp` lives
   directly in the SHARED `add_library` source list and is compiled by
   `cl.exe`. Only TUs that actually contain HIP code (HIP intrinsics or
   `__global__` kernels) belong in `BACKEND_HIP_SOURCES`.

---

## Adding a new op (recipe)

Each step is mandatory; missing any one produces silent wrong results or
link failures.

1. **Append a function-pointer field at the END of `HIPBackendVTable`**
   in `lib/Backend/hipdnn_ep_backend.h` (e.g.
   `hip_backend_gemm_fp16_fn gemm_fp16;`). Add a matching `*_fn` typedef
   above the struct.
2. **Implement the op** in a new `lib/Backend/<op>.cpp` (or extend an
   existing file). Keep the impl at file-static linkage; expose a
   single forward decl matching `extern int backend_<op>_impl(...)`.
   Add the source to `BACKEND_HIP_SOURCES` in
   `lib/Backend/CMakeLists.txt`.
3. **Plug the impl into `g_vtable`** in `lib/Backend/main.cpp`.
4. **Add a method to `hip::Backend`** in
   `lib/Runtime/real/hip_backend_client.{h,cpp}` that null-checks the slot and
   throws `std::runtime_error` on missing-or-failure.
5. **Implement a new `wrap_*` extern-C shim** in a bitcode TU under
   `lib/Runtime/real/` that catches and converts to return-code (mirror
   `lib/Runtime/real/ck_conv.cpp`). Declare in
   `lib/Runtime/hipdnn_ep_runtime.h`. Register in
   `lib/Runtime/CMakeLists.txt`'s `compile_to_bitcode` list.
6. **Hook into MLIR lowering** if the op is reachable from a MLIR
   pattern (typically `lib/Conversion/HipToLLVM/<Op>Lowering.cpp`).
7. **DO NOT touch `lib/Backend/hip_backend.def`** -- it stays a 1-line
   file forever.
8. **Bump `HIP_BACKEND_API_VERSION` ONLY if you remove or change an
   existing slot's signature.** Appending is purely additive.

### Importing from Composable Kernel

The first three steps are CK-specific:

1. **Find what CK ships in TheRock.** Headers live at
   `install/therock/include/ck/`. Precompiled instances are organized
   by op family in
   `install/therock/include/ck/library/tensor_operation_instance/gpu/{grouped_conv_fwd,grouped_conv_bwd_data,...}/`
   -- the `.hpp` files are forward-declarations, the actual instance
   code is in `device_*_operations.lib` etc. under
   `install/therock/lib/`. **Right now only `device_conv_operations.lib`
   and `utility.lib` are shipped** (others -- `device_gemm_operations`,
   `device_mha_operations`, `device_contraction_operations`,
   `device_reduction_operations`, `device_other_operations` -- are
   listed in `composable_kernelConfig.cmake` but the .lib files aren't
   in TheRock's distribution). Adding a non-conv op may require building
   the CK lib from upstream first.
2. **Pick the abstract template.** Each op has a `Device<Op>` base class
   in `ck/tensor_operation/gpu/device/`. For conv:
   `DeviceGroupedConvFwdMultipleABD<...>` (most general -- supports
   bias/elementwise fusion via the `D` slot). For GEMM:
   `DeviceGemm_Xdl_CShuffle_V3<...>` or `DeviceGemmMultipleD_<...>`.
   For attention: `DeviceMHA<...>` family. The abstract base has
   pure-virtual `MakeArgumentPointer` / `MakeInvokerPointer`; our code
   only ever holds a `unique_ptr<DeviceOp>`, never instantiates the
   concrete template.
3. **Use the instance factory.**
   `DeviceOperationInstanceFactory<DeviceOp<...>>::GetInstances()`
   returns a `vector<unique_ptr<DeviceOp>>` of all precompiled instances
   matching the type tuple. Iterate with `IsSupportedArgument(arg)` to
   filter, pick the first (or autotune by timing) -- pattern is in
   `lib/Backend/conv.cpp::lookup_or_build_locked`.

CK execution backend availability per arch:

| Backend          | Arches            | Dtypes        | Macro                  |
|------------------|-------------------|---------------|------------------------|
| WMMA cshufflev3  | RDNA3+ (gfx11xx)  | fp16, bf16    | `CK_USE_WMMA`          |
| XDL              | CDNA (gfx9xx DC)  | all dtypes    | `CK_USE_XDL`           |
| DL (no tensor)   | -                 | -             | `CK_ENABLE_DL_KERNELS` |

TheRock's CK has WMMA + XDL but not DL. WMMA cshufflev3 ships fp16/bf16
only (no fp32). XDL ships all dtypes but won't run on RDNA. The factory's
`if constexpr` branches gate which `add_*_instances` calls survive at
compile time -- if your type tuple has zero matching branches you get an
empty vector and `IsSupportedArgument` will never return true.

CK uses 5D `lengths/strides` arrays even for 2D ops, ordered
`[G, N, C/G, H, W]` for activations -- but the *strides* reflect the
actual in-memory layout (`NHWGC`, `NCHW`, etc.), not the logical dim
order. Layout typedefs are in
`ck/tensor_operation/gpu/device/tensor_layout.hpp` --
`convolution::NHWGC`, `convolution::GKYXC`, `gemm::RowMajor`,
`gemm::ColumnMajor`. Stride helpers are in `lib/Backend/conv.cpp`'s
`a_strides()`, `b_strides()`, `e_strides()` -- model new ops on those.

The third element op slot (`CDEElementwiseOperation`) controls output
fusion: `PassThrough` for plain conv/gemm, `Bilinear` (alpha\*A\*B +
beta\*C), `AddRelu`, `Scale`, `ScaleAdd`, etc. -- defined in
`ck/tensor_operation/gpu/element/element_wise_operation.hpp`. Each
fusion is a separate precompiled instance set; switching ops at runtime
means picking a different `DeviceOp<...>` factory specialization.

---

## Adding a new gfx target

Build with `-DHIP_ARCHITECTURES=<arch>` (e.g. `gfx1100`, `gfx1200`); each
build produces ONE `hip-backend-<arch>.dll` in `install/dist/bin/`.
Multiple arch builds can coexist in the same `install/dist/bin/` and the
model.dll auto-detects the device's gfx via `hipGetDeviceProperties` --
no rebuild needed when moving a model between hosts with different gfx,
only when the backend itself needs new arch coverage.

Per-arch CK instance availability differs (e.g. CDNA gfx9xx prefer XDL
paths instead of WMMA cshufflev3); future per-arch backends may need to
instantiate different `DeviceGroupedConvFwdMultipleABD` template
parameters in `lib/Backend/conv.cpp`. The vtable surface stays the same.

---

## Troubleshooting

On first successful load, `[hip::Backend] loaded hip-backend-gfx1151.dll
(arch gfx1151, ABI v1)` is printed to stderr by the wrapper ctor.

Failure modes manifest as `std::runtime_error` thrown from
`Backend::Backend()`:

- `failed to load backend DLL: ...` -> the DLL search failed (PATH
  wrong, or this build didn't ship a backend for the device's gfx).
- `is missing required export 'HIPBackendAPI'` -> .def file regression
  (should be exactly one DATA export). Verify with
  `dumpbin /EXPORTS install/dist/bin/hip-backend-gfx<ARCH>.dll`.
- `ABI version N does not match expected M` -> client and DLL were
  built from different commits, rebuild.
- `reports arch X but device is Y` -> wrong DLL on PATH (filename
  collision or PATH ordering).

At inference_init time, exceptions from `hip::GetBackend()` are caught
and demoted to a `[hip_ep] backend not available: ...` warning --
subsequent op calls (e.g. `wrap_ckConvForward`) will surface the same
error via `[<op>] ...` + `return -1`.

The dispatch shim propagates the -1 and the MLIR-generated code
currently discards the return value, so dispatch failure manifests as
zero-filled output (the caller-provided buffer never got written).
**Always check stderr first.**

When `wrap_ckConvForward` returns -1 with `[ep_backend] scratch provider
unavailable or returned null for N bytes`, model.dll's
`hip::Backend::SetScratchProvider` either wasn't called (state_init
failed before reaching that point) or the underlying
`hipdnn_ep_state_ensure_workspace` grow returned an error (out of GPU
memory). Verify `state_init` ran fully -- the
`[hip::Backend] loaded ...` line should appear before any compute.

---

## Verifying exports

```
dumpbin /EXPORTS install/dist/bin/hip-backend-gfx<ARCH>.dll
```

Must list exactly ONE export: `HIPBackendAPI`. Anything else means the
.def file regressed (someone added a per-symbol export instead of a
vtable slot).
