<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Runtime Module Registry

Mechanism the runtime uses for per-session, per-operator state.

## Code map

| Path | Contents |
|---|---|
| `lib/Runtime/module_registry.h` | `OpModuleSpec`, fn-pointer typedefs, SFINAE detectors, `make_op_module_spec<T>`, `HIPDNN_OP_MODULE` macro, free-function decls. |
| `lib/Runtime/module_registry.cpp` | Process-global `spec_table()` + mutex; `SlotEntry` and `ModuleRegistry` definitions; lifecycle and lookup implementations. |
| `lib/Runtime/growable_buffer.h` | `hipdnn_ep::GrowableDeviceBuffer` / `GrowablePinnedBuffer`. |
| `lib/Runtime/runtime_state_internal.h` | Forward-declares `ModuleRegistry`; `RuntimeState` ends with `hipdnn_ep::ModuleRegistry *modules;`. |
| `lib/Runtime/hipdnn_ep_runtime_state.cpp` | Create / destroy / `begin_compute` fan-out; defines `hipdnn_ep::get_module_registry`. |
| `lib/Runtime/real/causal_conv_with_state.cpp` | `CausalConvState`. |
| `lib/Runtime/real/matmul_nbits.cpp` | `ZpUnpackState` (shared with `qmoe`). |
| `lib/Runtime/real/gqa.cpp` | `GqaGemmState`, `GqaSeqlensCache`. |
| `lib/Runtime/real/qmoe.cpp` | `QmoeState` + C-ABI shims. |
| `test/runtime_steady_state/runtime_steady_state.cpp` | Native CTest target `RuntimeSteadyState`. |

## Op-module contract

A C++ struct authored by the op owner:

- **Required:** `explicit T(RuntimeState *)` constructor. Destructor if any
  member needs explicit teardown.
- **Optional, SFINAE-detected** (exact signature required, or detection
  silently fails):
  - `void begin_compute(RuntimeState *)` — fires at the top of every
    `Compute()`, for cache invalidation.
  - `void end_compute(RuntimeState *)` — detected and dispatched, but
    nothing calls it today.

## `HIPDNN_OP_MODULE` macro

`make_op_module_spec<T>` always installs `init_fn` (`new T(state)`) and
`destroy_fn` (`delete`). Optional fn-pointers are installed via
`if constexpr` on the SFINAE detectors.

```cpp
#define HIPDNN_OP_MODULE(ACCESSOR, NAME, STATE_T)                              \
  static STATE_T *ACCESSOR(::RuntimeState *state) {                            \
    static const ::hipdnn_ep::OpModuleSpec hipdnn_ep_module_spec_ =            \
        ::hipdnn_ep::make_op_module_spec<STATE_T>(NAME);                       \
    static const int hipdnn_ep_module_slot_ =                                  \
        ::hipdnn_ep::register_op_module(&hipdnn_ep_module_spec_);              \
    return static_cast<STATE_T *>(                                             \
        ::hipdnn_ep::op_module_get(::hipdnn_ep::get_module_registry(state),    \
                                   state, hipdnn_ep_module_slot_));            \
  }
```

The two function-local statics initialize once per TU; after that
every call reduces to `op_module_get(reg, state, cached_slot)`.

## Process-global spec table

`spec_table()` (Construct-On-First-Use, mutex-guarded). `register_op_module`:

1. Aborts (with stderr message) if `name` / `init_fn` / `destroy_fn` are
   null, or if `name` duplicates an existing entry.
2. Appends the spec pointer and returns the new slot id.

The spec pointer must remain valid for the lifetime of the process —
the macro satisfies this with a function-local static.

## Per-`RuntimeState` `ModuleRegistry`

```cpp
struct SlotEntry {
  void *state_ptr = nullptr;
  OpBeginComputeFn begin_compute_fn = nullptr;
  OpEndComputeFn end_compute_fn = nullptr;
  OpDestroyFn destroy_fn = nullptr;
  const char *name = nullptr;
};

struct ModuleRegistry {
  std::vector<SlotEntry> slots;
};
```

The slot entry caches the spec's fn-pointers at first access, so the
per-Compute() fan-out and cleanup never re-enter the mutex-guarded
spec table.

- **`module_registry_create`** — empty registry.
- **`module_registry_destroy`** — iterates slots in reverse and calls
  `destroy_fn` on every populated entry. `nullptr` is a no-op.
- **`module_registry_begin_compute`** — forward iterate; call each
  cached `begin_compute_fn` when non-null. Allocation-free, lock-free.
- **`op_module_get`** — hot path: bounds check, load, null branch.
  Cold path (first access for `(session, slot)`): resize, call
  `spec->init_fn`, cache fn-pointers + state pointer into the slot.

```cpp
void *op_module_get(ModuleRegistry *reg, RuntimeState *state, int slot_id) {
  if (!reg || slot_id < 0) return nullptr;
  if (static_cast<size_t>(slot_id) < reg->slots.size()) {
    void *p = reg->slots[slot_id].state_ptr;
    if (p) return p;
  }
  if (static_cast<size_t>(slot_id) >= reg->slots.size())
    reg->slots.resize(static_cast<size_t>(slot_id) + 1, SlotEntry{});
  const OpModuleSpec *spec = get_op_module_spec(slot_id);
  if (!spec || !spec->init_fn) return nullptr;
  void *p = spec->init_fn(state);
  if (!p) return nullptr;
  SlotEntry &slot = reg->slots[slot_id];
  slot.state_ptr        = p;
  slot.begin_compute_fn = spec->begin_compute_fn;
  slot.end_compute_fn   = spec->end_compute_fn;
  slot.destroy_fn       = spec->destroy_fn;
  slot.name             = spec->name;
  return p;
}
```

## `RuntimeState` integration

```cpp
namespace hipdnn_ep { struct ModuleRegistry; }

struct RuntimeState {
  // ... existing flat fields (stream, handles, constants, pool,
  //     workspace, op_profile, device_error_flag, hipdnn_handle,
  //     hipdnn_graph_registry) ...
  hipdnn_ep::ModuleRegistry *modules;   // tail position: preserves
                                        // existing field offsets
};
```

`hipdnn_ep::get_module_registry` is defined in
`hipdnn_ep_runtime_state.cpp` (where `RuntimeState`'s layout is in
scope), not in `module_registry.cpp`:

```cpp
namespace hipdnn_ep {
ModuleRegistry *get_module_registry(RuntimeState *state) {
  return state ? state->modules : nullptr;
}
}
```

It inlines through `llvm-link` in the bitcode build.

## Compute() lifecycle

| Phase | What happens |
|---|---|
| Session init (`initialize_state_handles`) | `state->modules = module_registry_create()`. No slots populated yet. |
| First `wrap_*` call per slot | Cold path of `op_module_get`: `new T(state)`, cache fn-pointers + state ptr. |
| Subsequent `wrap_*` calls | Hot path: bounds + load + null branch. |
| EP enters `Compute()` | EP calls `hipdnn_ep_runtime_begin_compute(state)` (`extern "C"`, `__declspec(dllexport)`). Fans out via `module_registry_begin_compute`. |
| Session cleanup | After shared stream sync: `module_registry_destroy`. Library handles (`stream`, `miopen_handle`, `hipblas_handle`) are still live, so destructors may call `hipFree` / `hipblasLt*Destroy`. |

There is no end-of-Compute hook caller today.

## Steady-state cost

After warmup, per call:

- `my_op_module(state)`: one bounds compare, one load, one null branch.
  No mutex, no allocation, no string op.
- `hipdnn_ep_runtime_begin_compute`: iterate `slots.size()` entries
  (currently ≤ 5). Non-null `begin_compute_fn`: one indirect call.
  Null: one load + one null branch.

CTest target `RuntimeSteadyState` asserts that `init_fn` fires exactly
once per slot across warmup + 1000 simulated inferences.

## `growable_buffer.h`

Header-only RAII helpers. Two classes with the same API:

```cpp
namespace hipdnn_ep {
class GrowableDeviceBuffer {       // hipMalloc / hipFree
public:
  explicit GrowableDeviceBuffer(hipStream_t stream = nullptr);
  ~GrowableDeviceBuffer();
  void   set_stream(hipStream_t stream);
  int    grow(size_t needed);      // 0 ok, -1 alloc failure
  void  *data() const noexcept;
  size_t size() const noexcept;
};
class GrowablePinnedBuffer { /* hipHostMalloc / hipHostFree, same API */ };
}
```

- 1.5× growth on `grow(needed)` (cold-start sizes to exactly `needed`).
- `hipStreamSynchronize(stream_)` before each free (when stream is set).
- Destructor frees without pre-sync — relies on `hipdnn_ep_state_cleanup`
  having already synced the stream before module destruction runs.
- Move-only.

Used by composition. Not registered as modules themselves.

## The five op-modules in the tree

| Slot name | State type | TU | Notes |
|---|---|---|---|
| `"causal_conv_with_state"` | `CausalConvState` | `real/causal_conv_with_state.cpp` | `unordered_map<CausalConvKey, CausalConvCacheEntry, ...>`. Destructor frees MIOpen descriptors via the file-local `destroyEntry`. |
| `"zp_unpack"` | `hipdnn_ep_real::ZpUnpackState` | `real/matmul_nbits.cpp` | Two `unordered_map<const void *, std::pair<void *, size_t>>` for the asym AWQ `(u8, fp16)` unpacked zero_points buffers, plus a `std::mutex`. Destructor `hipFree`s every cached buffer. Reached by both `wrap_matmul_nbits` and `wrap_qmoe` via `lookup_or_unpack_zp_u8` / `lookup_or_convert_zp_fp16` — single slot, no duplication. |
| `"gqa_seqlens_cache"` | `GqaSeqlensCache` | `real/gqa.cpp` | POD (`bool valid; int32_t val; const void *ptr;`). `begin_compute` resets `valid` and `ptr`. First GQA layer in a `Compute()` pays the D2H to read `seqlens_k[0]`; later layers reuse it. |
| `"gqa_gemm_cache"` | `GqaGemmState` | `real/gqa.cpp` | `unordered_map<GqaGemmKey, GqaGemmCacheEntry, ...>`. Destructor frees the hipBLASLt matrix layouts (`layA..layD`) and matmul descriptor. |
| `"qmoe"` | `QmoeState` | `real/qmoe.cpp` | `GrowableDeviceBuffer scratch`, `GrowablePinnedBuffer host_scratch`. Constructor calls `set_stream(rs->stream)` on both. |

## C-ABI shims (`hipdnn_ep_state_*_qmoe_*`)

`hipdnn_ep_runtime.h` declares four `extern "C"` functions that predate
the registry; their signatures cannot change because pre-built
model.dlls call them from `inference_compute` bitcode:

```c
void *hipdnn_ep_state_get_qmoe_scratch(RuntimeState *state);
int   hipdnn_ep_state_ensure_qmoe_scratch(RuntimeState *state, size_t needed_size);
void *hipdnn_ep_state_get_qmoe_host_scratch(RuntimeState *state);
int   hipdnn_ep_state_ensure_qmoe_host_scratch(RuntimeState *state, size_t needed_size);
```

The bodies live in `real/qmoe.cpp` and delegate to the `QmoeState`
slot:

```cpp
extern "C" int hipdnn_ep_state_ensure_qmoe_scratch(RuntimeState *state,
                                                   size_t needed_size) {
  if (!state) return -1;
  if (needed_size == 0) return 0;
  QmoeState *m = qmoe_module(state);
  if (!m) return -1;
  return m->scratch.grow(needed_size);
}
```

`wrap_qmoe` itself also goes through these shims — same call path
across host code and generated bitcode.

## Smoke test

`test/runtime_steady_state/runtime_steady_state.cpp` is a native
(non-GPU, non-bitcode) test that links `module_registry.cpp` directly.
It uses two fake state types:

- `FullHooksState` — defines `begin_compute` + destructor (exercises
  the `has_begin_compute_` SFINAE branch; absent `end_compute` keeps
  that one null even when the rest are wired).
- `MinimalState` — only the required constructor + destructor.

It stubs `get_module_registry` locally and uses a one-int
`::RuntimeState` (the registry never dereferences the pointer).

Seven cases:

1. SFINAE installs `begin_compute_fn` for `FullHooksState`, leaves
   `end_compute_fn` null on both types.
2. `register_op_module` assigns monotonic slot ids; out-of-range
   `get_op_module_spec` returns null.
3. First `op_module_get` calls the ctor once; 100 follow-up calls
   return the same pointer without re-invoking it.
4. Null registry / negative slot / out-of-range slot /
   `module_registry_destroy(nullptr)` /
   `module_registry_begin_compute(nullptr, &s)` all behave gracefully.
5. `begin_compute` fires only for populated slots whose spec defines
   the hook; the current `RuntimeState *` is forwarded each call.
6. Destructors fire exactly once at `module_registry_destroy`.
7. After warmup, 1000 simulated inferences leave the init counter at
   1 and produce exactly 1000 `begin_compute` invocations.

## Build glue

`lib/Runtime/CMakeLists.txt`:

- `module_registry.h` and `growable_buffer.h` are in the universal
  `DEPENDS` list of `compile_to_bitcode`.
- `module_registry.cpp` compiles to `runtime_module_registry.bc` and
  is added to both the real and mock runtime link lists.

`test/runtime_steady_state/CMakeLists.txt` builds
`test-runtime-steady-state` as a native C++17 executable linking only
`module_registry.cpp`, and registers it as the `RuntimeSteadyState`
CTest target. The root `CMakeLists.txt` pulls the subdirectory in via
`add_subdirectory(test/runtime_steady_state)`.

## Threading model

`RuntimeState` is single-threaded per session. `op_module_get`'s cold
path is not locked; concurrent first-time accesses to the same
`(session, slot)` would race the `slots.resize`. Multi-threaded
inference on a single `RuntimeState` would need a `std::call_once` or
CAS only at lazy-init.

The process-global spec table is mutex-guarded because multiple
model.dlls register their modules concurrently from static-init.

## Adding a new op-module

In the op's `.cpp` (no edits to any framework file):

1. Define `MyOpState` (anonymous namespace). Constructor takes
   `RuntimeState *`. Add a destructor if needed.
2. Optionally add `begin_compute(RuntimeState *)` or
   `end_compute(RuntimeState *)`. Exact signature required for SFINAE
   pickup.
3. `HIPDNN_OP_MODULE(my_op_module, "my_op", MyOpState);` at namespace
   scope. The name must be unique across the DLL.
4. Inside `wrap_*`: `auto *s = my_op_module(state);` — null only on
   alloc failure.

If the state owns grow-on-demand device or pinned-host memory,
compose `GrowableDeviceBuffer` / `GrowablePinnedBuffer` rather than
hand-rolling the grow logic.

### On-disk persistence (algo caches, autotune results, etc.)

Persistence is **out of scope for the module registry**. Op-modules
hold derived state — descriptors, algorithm choices, sized scratch
buffers — that is cheap to rebuild on the first hot call. Serializing
the in-memory layout would couple every disk artifact to ABI; the
opaque HIP/MIOpen/hipBLASLt handles inside descriptor caches are not
portable across processes, driver versions, or GPU resets anyway.

When a specific op genuinely benefits from cross-process caching
(e.g. autotune results that take seconds to converge), follow the
matmul_nbits autotune convention:

- Persist **algo-selection metadata only**, never descriptors or
  buffers (`int` indices, kernel-config tuples, hash → choice map).
- Key on something **content-derived and version-aware** — e.g.
  `{model_hash, driver_version, GPU_arch, shape_signature}`.
- Read/write the file directly from the op's `.cpp` (typically a
  small JSON or binary blob next to the executable, the way
  `_results.json` is handled for matmul_nbits WMMA tuning).
- Consult the file inside the op's `constructor` if you want
  cold-start speedup, and update it from the destructor (or
  opportunistically inside `wrap_*` after a new tuning result).

This keeps persistence **opt-in per op**, the on-disk format
decoupled from in-memory state, and the framework contract minimal.
The `HIPDNN_OP_MODULE` macro intentionally has no `save_fn` /
`load_fn` slot.

## Fields that stay flat on `RuntimeState`

The registry handles only **per-op** state. These framework-tier
fields stay as direct members of `RuntimeState`:

| Field(s) | Why |
|---|---|
| `stream`, `miopen_handle`, `hipblas_handle` | Library handles used by every op. |
| `gpu_constants_blob`, `gpu_constants`, `num_constants`, `constants_is_shared`, `shared_constants_mapping`, `shared_constants_view` | Loaded once at `inference_init`. Not op-specific. |
| `pool_base`, `pool_size`, `buffer_offsets`, `num_buffers` | Compiled-graph memory pool emitted by the compiler. |
| `workspace`, `workspace_size` | Shared scratch across many ops. |
| `op_profile` | Cross-op profiling state (`HIPDNN_EP_PERF=1`). |
| `device_error_flag` | Single `int *` used by every kernel. |
| `hipdnn_handle`, `hipdnn_graph_registry` | Attached by the EP after `inference_init` returns; lifecycle does not match the registry's model. |
