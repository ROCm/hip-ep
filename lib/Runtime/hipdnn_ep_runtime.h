/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EP_RUNTIME_H
#define HIP_EP_RUNTIME_H

#include "hipdnn_ep_errors.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Backend-Independent Data Type Identifiers
//===----------------------------------------------------------------------===//
//
// These are our own values -- do NOT assume they match MIOpen, cuDNN, or any
// other library's enum. Each backend provides an explicit mapping function
// (e.g. hipdnn_ep_to_miopen_type in real/elementwise.cpp) to convert these
// to library-specific types.
//
// To add a new type:
//   1. Add #define here
//   2. Update hipdnn_ep_datatype_size() and hipdnn_ep_datatype_name()
//   3. Update compiler mapping getHipdnnDataType() in HipToLLVM.cpp
//   4. Update each backend mapping function
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_DATATYPE_FLOAT 0    // f32, 4 bytes
#define HIPDNN_EP_DATATYPE_HALF 1     // f16, 2 bytes
#define HIPDNN_EP_DATATYPE_BFLOAT16 2 // bf16, 2 bytes
#define HIPDNN_EP_DATATYPE_INT32 3    // i32, 4 bytes
#define HIPDNN_EP_DATATYPE_INT64 4    // i64, 8 bytes
#define HIPDNN_EP_DATATYPE_INT8 5     // i8, 1 byte
#define HIPDNN_EP_DATATYPE_DOUBLE 6   // f64, 8 bytes
#define HIPDNN_EP_DATATYPE_UINT8 7    // ui8, 1 byte
#define HIPDNN_EP_DATATYPE_INT16 8    // i16, 2 byte

//===----------------------------------------------------------------------===//
// Backend-Independent Tensor Operation Identifiers
//===----------------------------------------------------------------------===//
//
// Same design as data types above -- our own values, mapped explicitly to
// library-specific ops in each backend (e.g. miopenTensorOpMul).
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_TENSOR_OP_MUL 0 // element-wise multiply
#define HIPDNN_EP_TENSOR_OP_ADD 1 // element-wise add
#define HIPDNN_EP_TENSOR_OP_MIN 2 // element-wise min
#define HIPDNN_EP_TENSOR_OP_MAX 3 // element-wise max

static inline const char *hipdnn_ep_tensor_op_name(int64_t op) {
  switch (op) {
  case HIPDNN_EP_TENSOR_OP_MUL:
    return "mul";
  case HIPDNN_EP_TENSOR_OP_ADD:
    return "add";
  case HIPDNN_EP_TENSOR_OP_MIN:
    return "min";
  case HIPDNN_EP_TENSOR_OP_MAX:
    return "max";
  default:
    return "unknown";
  }
}

static inline int64_t hipdnn_ep_datatype_size(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return 4;
  case HIPDNN_EP_DATATYPE_HALF:
    return 2;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return 2;
  case HIPDNN_EP_DATATYPE_INT32:
    return 4;
  case HIPDNN_EP_DATATYPE_INT64:
    return 8;
  case HIPDNN_EP_DATATYPE_INT8:
    return 1;
  case HIPDNN_EP_DATATYPE_UINT8:
    return 1;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return 8;
  case HIPDNN_EP_DATATYPE_INT16:
    return 2;
  default:
    return -1;
  }
}

static inline const char *hipdnn_ep_datatype_name(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return "f32";
  case HIPDNN_EP_DATATYPE_HALF:
    return "f16";
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return "bf16";
  case HIPDNN_EP_DATATYPE_INT32:
    return "i32";
  case HIPDNN_EP_DATATYPE_INT64:
    return "i64";
  case HIPDNN_EP_DATATYPE_INT8:
    return "i8";
  case HIPDNN_EP_DATATYPE_UINT8:
    return "ui8";
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return "f64";
  case HIPDNN_EP_DATATYPE_INT16:
    return "i16";
  default:
    return "unknown";
  }
}

//===----------------------------------------------------------------------===//
// Backend-Independent Activation Mode Identifiers
//===----------------------------------------------------------------------===//
//
// Same pattern as HIPDNN_EP_DATATYPE_* above. Each backend provides an explicit
// mapping function (e.g. hipdnn_ep_to_miopen_activation in
// real/activation.cpp).
//
// To add a new activation:
//   1. Add #define here
//   2. Update hipdnn_ep_activation_name()
//   3. Update each backend mapping function
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_ACTIVATION_SIGMOID 0
#define HIPDNN_EP_ACTIVATION_RELU 1
#define HIPDNN_EP_ACTIVATION_TANH 2
#define HIPDNN_EP_ACTIVATION_SOFTPLUS 3

static inline const char *hipdnn_ep_activation_name(int64_t activation_mode) {
  switch (activation_mode) {
  case HIPDNN_EP_ACTIVATION_SIGMOID:
    return "sigmoid";
  case HIPDNN_EP_ACTIVATION_RELU:
    return "relu";
  case HIPDNN_EP_ACTIVATION_TANH:
    return "tanh";
  case HIPDNN_EP_ACTIVATION_SOFTPLUS:
    return "softplus";
  default:
    return "unknown";
  }
}

//===----------------------------------------------------------------------===//
// Global pool reduction modes (must match kGlobalPool* in HipToLLVMUtils.h).
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_GLOBAL_POOL_AVERAGE 0
#define HIPDNN_EP_GLOBAL_POOL_MAX 1
#define HIPDNN_EP_GLOBAL_POOL_LP 2

static inline const char *hipdnn_ep_global_pool_mode_name(int64_t mode) {
  switch (mode) {
  case HIPDNN_EP_GLOBAL_POOL_AVERAGE:
    return "global_avg_pool";
  case HIPDNN_EP_GLOBAL_POOL_MAX:
    return "global_max_pool";
  case HIPDNN_EP_GLOBAL_POOL_LP:
    return "global_lp_pool";
  default:
    return "global_pool_unknown";
  }
}

//===----------------------------------------------------------------------===//
// Window-pool reduction modes (must match kPool* in HipToLLVMUtils.h and the
// pool_mode constants used in OnnxToHip/PoolConversion.cpp).
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_POOL_AVERAGE 0
#define HIPDNN_EP_POOL_MAX 1
#define HIPDNN_EP_POOL_LP 2

static inline const char *hipdnn_ep_pool_mode_name(int64_t mode) {
  switch (mode) {
  case HIPDNN_EP_POOL_AVERAGE:
    return "avg_pool";
  case HIPDNN_EP_POOL_MAX:
    return "max_pool";
  case HIPDNN_EP_POOL_LP:
    return "lp_pool";
  default:
    return "pool_unknown";
  }
}

// Opaque handle for runtime state
typedef struct RuntimeState RuntimeState;

//===----------------------------------------------------------------------===//
// RuntimeState: Opaque Execution State
//===----------------------------------------------------------------------===//
//
// RuntimeState encapsulates GPU execution resources (stream, library handles,
// model constants). Generated code treats it as opaque void*, runtime library
// owns the internal structure.
//
// Design rationale: Opaque pointer pattern allows runtime to evolve internal
// layout without breaking generated code.
//
// Lifecycle: init -> use -> cleanup (must call in this order)
// Thread safety: Not thread-safe (one inference per state at a time)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Output Allocator Contract
//===----------------------------------------------------------------------===//
//
// The EP sets an output allocator before inference_compute; the generated
// main_graph gets each output buffer from it once the output shape is known
// (this is what hip.alloc_output -> hipdnn_ep_alloc_output does).
//
// Call flow:
//   MlirCustomOp::Compute
//     -> hipdnn_ep_set_output_allocator(state, &alloc)   (EP installs)
//     -> inference_compute(state, inputs)                (2-arg, no out span)
//        -> main_graph -> main_graph_internal -> ...
//           -> hipdnn_ep_alloc_output(state, out_idx, shape, rank, elem)
//                -> alloc.allocate(self, ...)            (= EP callback)
//                     -> GetOutput(shape)               (GPU zero-copy /
//                                                         host scratch + D2H)
//     -> set_output_allocator(nullptr); completeness check; host D2H
//
// ABI: this struct crosses the model.dll <-> EP boundary. Its layout is a fixed
// contract, locked by static_asserts and mirrored by an identical EP-side copy
// -- same convention as tensor_t below. There is intentionally no size/version
// field: a layout change is an ABI break, handled by rebuilding the model.dll
// (deleting the stale cached DLLs), the same as any other runtime change.
// A model.dll built before this contract simply lacks the exported setter, so
// the EP's GetProcAddress returns null and it no-ops.
typedef struct {
  void *self; // opaque EP context (borrowed; runtime never owns/frees)
  void *(*allocate)(void *self, int64_t out_idx, const int64_t *shape,
                    int64_t rank, int64_t elem_size);
} hipdnn_output_allocator_t;

// Compile-time layout lock (mirrors the tensor_t static_assert idiom below).
// The EP-side copy must carry the same asserts.
static_assert(offsetof(hipdnn_output_allocator_t, self) == 0,
              "self must remain first -- update all "
              "hipdnn_output_allocator_t copies");

// Export attribute for runtime entry points the EP resolves by name. dllexport
// keeps the symbol alive through LLVM optimization in the bitcode build (the
// export_symbols /EXPORT list in CompilerDriver.cpp runs after opt and cannot
// resurrect an uncalled symbol). Must be applied to BOTH the declaration and
// the definition: output_allocator.cpp is also compiled natively by MSVC for
// the GPU-free unit test, and MSVC rejects a decl/def mismatch (C2375). The
// unit-test build defines HIPDNN_EP_RT_NO_EXPORT to drop the attribute.
#if defined(_WIN32) && !defined(HIPDNN_EP_RT_NO_EXPORT)
#define HIPDNN_EP_RT_EXPORT __declspec(dllexport)
#else
#define HIPDNN_EP_RT_EXPORT
#endif

// EP -> model.dll (exported), installs the allocator before inference_compute.
// `state` is RuntimeState* to match every other state entry point; the EP side
// resolves this by name and treats state as an opaque void* (pointer-
// compatible), exactly like hipdnn_ep_runtime_begin_compute.
HIPDNN_EP_RT_EXPORT void
hipdnn_ep_set_output_allocator(RuntimeState *state,
                               const hipdnn_output_allocator_t *allocator);

// generated main_graph -> runtime (internal), forwards to the installed
// callback. Returns a generic address-space-0 device pointer (the lowering
// casts to the memref's address space). Returns null if none is installed.
void *hipdnn_ep_alloc_output(RuntimeState *state, int64_t out_idx,
                             const int64_t *shape, int64_t rank,
                             int64_t elem_size);

// Initialize runtime state with external constant storage via FileSystem.
// Used when compiled with hip_compile_with_fs.
// Reads constants_filename and constant_sizes from the FlatBuffers blob
// (HipModelMetaInfo schema), opens the file via fs, reads each constant
// sequentially, uploads each to GPU via hipMalloc+hipMemcpy.
//   out_state:     Pointer to receive allocated RuntimeState
//   fs:            morphizen::FileSystem* (void* for C ABI) - must not be null
//   metadata_blob: FlatBuffers binary blob (HipModelMetaInfo) baked into DLL
//   blob_size:     Size of metadata_blob in bytes
// Return codes: 0=success, 1=alloc/read error, 2-11=GPU/runtime init error
int hipdnn_ep_state_init_with_fs(RuntimeState **out_state, void *fs,
                                 const void *metadata_blob, size_t blob_size);

// Cleanup runtime state (destroys handles, frees memory)
// Best-effort cleanup - continues even if individual operations fail
// Returns 0 always (best-effort)
int hipdnn_ep_state_cleanup(RuntimeState *state);

// Get GPU stream from state (for passing to HIP operations)
// Returns: hipStream_t cast to void* (NULL on error)
// Ownership: Caller does NOT own stream (destroyed in cleanup)
void *hipdnn_ep_state_get_stream(RuntimeState *state);

// Current inference session stream (hipStream_t as void*), set per Compute by
// hipdnn_ep_runtime_begin_compute. Lets ABI-fixed runtime helpers that don't
// receive `state` -- notably memrefCopy (MLIR memref.copy lowering) -- issue
// their work on the session stream instead of the default/null stream (0).
// Default-stream work serializes with the session stream (legacy implicit
// sync) and shows up as GPU idle in the prefill profile. Returns NULL before
// the first begin_compute on this thread (callers fall back to stream 0).
//
// Defined in the natively-compiled tls_stream.cpp (NOT runtime.bc) so the
// thread_local storage stays out of the JIT'd module; exported so the JIT's
// process search generator can resolve it from the host.
HIPDNN_EP_RT_EXPORT void *hipdnn_ep_get_current_stream(void);

// Publishes the session stream for hipdnn_ep_get_current_stream. Called by
// hipdnn_ep_runtime_begin_compute (and cleared on stream teardown). Lives in
// tls_stream.cpp alongside the getter.
HIPDNN_EP_RT_EXPORT void hipdnn_ep_set_current_stream(void *stream);

// Get MIOpen handle from state (for MIOpen operations)
// Returns: miopenHandle_t cast to void* (NULL on error)
// Ownership: Caller does NOT own handle (destroyed in cleanup)
void *hipdnn_ep_state_get_miopen_handle(RuntimeState *state);

// Get hipBLASLt handle from state (for GEMM operations)
// Returns: hipblasLtHandle_t cast to void* (NULL on error)
// Ownership: Caller does NOT own handle (destroyed in cleanup)
void *hipdnn_ep_state_get_hipblas_handle(RuntimeState *state);

// Get buffer from memory pool by index
// Returns: GPU pointer at pool_base + buffer_offsets[index] (NULL on error)
// Ownership: Caller does NOT own pointer (freed in cleanup)
void *hipdnn_ep_get_buffer_from_pool(RuntimeState *state, size_t index);

// Get the base pointer of one of the runtime's GPU memory pools, growing it
// if needed. Called from PoolAllocs-generated code, once per emitted
// hip.get_pool, at the start of each inference.
//
// `domain_id` selects which pool to access: hip-pool-allocs partitions the
// function's pooled allocs into independent dominance domains and emits one
// hip.get_pool per domain (id starts at 0). Domain 0 inherits the legacy
// single-pool semantics — its pool was eagerly sized by hipdnn_ep_pool_init
// using the static buffer offsets, so single-domain models are bit-identical
// to the pre-multi-domain runtime. Domains 1..N start with size 0 and grow
// lazily on their first call here.
//
// When `needed_size` exceeds the selected domain's current allocation, that
// pool is grown via stream-sync + hipFree + hipMalloc. Pools never shrink and
// are independent across domains: growing domain N does not touch domain M.
//
// There is no compile-time cap on `domain_id`: the per-domain arrays are
// themselves grown on demand the first time a higher id is seen (a cold-path
// event on the first inference). A negative `domain_id` returns NULL with a
// stderr diagnostic (it would indicate a compiler bug — ids start at 0).
//
// Returns: GPU base pointer for the selected domain (NULL on bad domain_id
//          or allocation failure).
void *hipdnn_ep_get_pool_base(RuntimeState *state, int domain_id,
                              size_t needed_size);

// Get the host-mapped scratch buffer base, growing it if needed. Called from
// hip.get_host_scratch (emitted by hip-materialize-host-scalars) once per
// inference for tiny host-fed scalar memrefs that would otherwise land in the
// GPU pool. Memory is hipHostMalloc(hipHostMallocMapped) - host-writable AND
// GPU-readable via the device pointer mapping. Grow semantics mirror
// hipdnn_ep_get_pool_base: stream-synced hipHostFree + hipHostMalloc; never
// shrinks.
// Returns: host-mapped base pointer (NULL on allocation failure)
void *hipdnn_ep_get_host_scratch_base(RuntimeState *state, size_t needed_size);

// Shared workspace management (lazily grown, reused across MatMul/GQA/Conv)
void *hipdnn_ep_state_get_workspace(RuntimeState *state);
size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state);
int hipdnn_ep_state_ensure_workspace(RuntimeState *state, size_t needed_size);

// Per-session scratch for wrap_qmoe transient buffers (device + pinned-host
// mirror for routing readback). Replaces the per-call hipMalloc/hipFree storm
// (8 buffers x N MoE layers per inference). Same grow-on-demand policy as
// the shared workspace; never shrinks. See runtime_state_internal.h for
// rationale.
void *hipdnn_ep_state_get_qmoe_scratch(RuntimeState *state);
int hipdnn_ep_state_ensure_qmoe_scratch(RuntimeState *state,
                                        size_t needed_size);
void *hipdnn_ep_state_get_qmoe_host_scratch(RuntimeState *state);
int hipdnn_ep_state_ensure_qmoe_host_scratch(RuntimeState *state,
                                             size_t needed_size);

// Per-session MIOpen convolution workspace pool (used by
// wrap_miopenConvolutionForward for both 2D and the H=1 1D conv path). Lazily
// grown via hipdnn_ep_state_ensure_conv_scratch (same policy as qmoe_scratch
// above: never shrinks, freed in hipdnn_ep_state_cleanup). Single buffer
// reused across all conv calls in the session -- safe because the stream is
// serialised. See runtime_state_internal.h for design rationale.
void *hipdnn_ep_state_get_conv_scratch(RuntimeState *state);
int hipdnn_ep_state_ensure_conv_scratch(RuntimeState *state,
                                        size_t needed_size);

// Per-session scratch for the W4A8 dp4a matmul_nbits decode path
// (hip_matmul_nbits_dp4a). One contiguous device buffer holding the quantized
// activation row (int8) plus the per-group activation scales (float). Lazily
// grown via hipdnn_ep_state_ensure_matmul_dp4a_scratch (same policy as
// conv_scratch: never shrinks, freed in hipdnn_ep_state_cleanup). Single buffer
// reused across all matmul_nbits calls in the session -- safe because the
// stream is serialised. See runtime_state_internal.h for design rationale.
void *hipdnn_ep_state_get_matmul_dp4a_scratch(RuntimeState *state);
int hipdnn_ep_state_ensure_matmul_dp4a_scratch(RuntimeState *state,
                                               size_t needed_size);

// Per-op state slots (see docs/design/op-state-slots-design.md). The generated
// @hipdnn_ep_op_states_init_fn (built by --generate-op-state-init) calls
// _alloc once, then per stateful op calls its construct symbol; each construct
// stores its built state via hipdnn_ep_op_state_set (plain C++, declared in
// op_state.h, since it is bitcode-internal -- never called by generated IR).
// _get (also in op_state.h) reaches a slot from an op's runtime entry. Cleanup
// walks the array via each object's deletor.
bool hipdnn_ep_op_states_alloc(RuntimeState *state, int64_t n);

// Device-side runtime error flag (set by kernels, observed by wrappers).
// Intended for operators that detect runtime-invalid inputs on GPU (e.g. Range
// delta==0) and need to propagate an error code back through main_graph.
void *hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *state);
int hipdnn_ep_state_reset_error_flag(RuntimeState *state);
int hipdnn_ep_state_read_and_clear_error_flag(RuntimeState *state);
// Mark the start of a new Compute() call. Invalidates per-forward-pass
// runtime caches -- today: the GQA seqlens_k cache (see
// runtime_state_internal.h for the canonical list).
//
// Contract: the EP-side caller must invoke this exactly once at the top
// of every MlirCustomOp::Compute(), before any input marshaling. The
// pairing with Compute() is what defines "forward pass" for cache
// invalidation purposes.
//
// Cost: writes a small number of fields on RuntimeState (no allocation,
// no GPU work). Safe to call unconditionally on the hot path.
//
// Backwards compatibility: model.dlls predating this export are loaded
// with a null cached function pointer on the EP side and the call is
// silently skipped. Such DLLs MUST run with HIPDNN_EP_GQA_CACHE_SEQLENS=0;
// otherwise stale seqlens_k values from the previous forward pass would
// silently corrupt decode output from token 2 onward. The mismatch is
// detected at session creation by InferenceState::create() and produces
// a LOG(WARNING) containing the substring "GQA seqlens_k cache is
// enabled ... but the loaded model.dll does not export
// hipdnn_ep_runtime_begin_compute" -- greppable across logs and code.
void hipdnn_ep_runtime_begin_compute(RuntimeState *state);

// Resolve and print the per-op profile table (HIPDNN_EP_PERF).
//
// Contract: the EP must invoke this AFTER the Compute() wall-clock window
// has closed. The resolve (one hipEventElapsedTime per profiled op + map
// aggregation + fprintf) used to run inside hipdnn_ep_stream_sync on the
// hot path, where its cost inflated the very [PERF SUMMARY] numbers the
// table explains. A no-op (single null check) when perf is disabled, so it
// is safe to call unconditionally.
//
// Backwards compatibility: same as hipdnn_ep_runtime_begin_compute -- a
// model.dll predating this export resolves to a null pointer EP-side and the
// call is skipped (inference unaffected; the per-op block just won't print).
void hipdnn_ep_runtime_flush_op_profile(RuntimeState *state);

// Record an outer (whole-scope) CPU timing sample into the per-op profile
// table (HIPDNN_EP_PERF). Called by the EP from MlirCustomOp::Compute AFTER
// inference_compute returns (so it survives the per-inference op_profile_reset
// that happens during input marshaling) and BEFORE
// hipdnn_ep_runtime_flush_op_profile resolves the table.
//
// `cpu_ms` MUST be a std::chrono::steady_clock measurement (the same clock the
// per-op `cpu (ms)` column uses) and `cpu_start_us` the absolute steady_clock
// microseconds at which the scope began (same axis as the chrome-trace CPU
// track). The intent is bubble detection: this records a CPU-only table row AND
// an enclosing chrome-trace span so the per-op spans nest inside it -- the
// uncovered width is the CPU time not attributed to any wrapper (marshaling,
// launch/dispatch overhead, inter-op gaps, fence). Do NOT feed a hipEvent-
// derived (GPU-timer) value here -- that clock is distinct from steady_clock,
// so mixing them would conflate async offset and clock skew with the bubble.
//
// Backwards compatibility: same contract as hipdnn_ep_runtime_flush_op_profile
// -- a model.dll predating this export resolves to a null pointer EP-side and
// the call is skipped. No-op (single null check) when perf is disabled, so it
// is safe to call unconditionally.
void hipdnn_ep_runtime_add_cpu_profile(RuntimeState *state, const char *name,
                                       double cpu_start_us, double cpu_ms);

// Initialize memory pool in runtime state
// Called by generated inference_init after creating RuntimeState
// Parameters:
//   state: Runtime state to initialize pool in
//   pool_size: Total size of memory pool in bytes
//   buffer_offsets: Array of offsets for each buffer
//   num_buffers: Number of buffers
// Returns: 0=success, non-zero=error
int hipdnn_ep_pool_init(RuntimeState *state, size_t pool_size,
                        const size_t *buffer_offsets, size_t num_buffers);

//===----------------------------------------------------------------------===//
// Inference API Types (for generated interface)
//===----------------------------------------------------------------------===//

// Memory placement of a tensor's `data` pointer. Values are 1:1 with ORT's
// OrtMemoryInfoDeviceType (onnxruntime_c_api.h) so MlirCustomOp can write
// the ORT value straight into tensor_t.memory_type with no remapping.
//
// Today the runtime only special-cases TENSOR_MEMORY_GPU (alias path,
// avoids the per-inference H2D / D2H copy on AMD APU iGPU mapped-pinned
// memory). CPU / FPGA / NPU all fall through to the legacy host H2D / D2H
// path, preserving existing behaviour for hip-test, hip-onnx-runner,
// and any other host-input caller.
//
// Must match the matching enum in
// `backend-mlir-compiler/custom-op-mlir/src/custom_op_mlir.hpp`.
enum {
  TENSOR_MEMORY_CPU = 0, // == OrtMemoryInfoDeviceType_CPU
  TENSOR_MEMORY_GPU = 1, // == OrtMemoryInfoDeviceType_GPU  (alias path; only
                         // mode optimized today)
  TENSOR_MEMORY_FPGA =
      2, // == OrtMemoryInfoDeviceType_FPGA (treated as host today)
  TENSOR_MEMORY_NPU =
      3, // == OrtMemoryInfoDeviceType_NPU  (treated as host today)
};

// Represents a tensor with data pointer and shape information.
//
// Memory ownership: caller-owned. `memory_type` selects the data pointer's
// placement (see the enum above) and tells the runtime whether to copy
// H2D/D2H or alias the caller's GPU-accessible buffer.
//
// tensor_t is the wire-protocol ABI between three components that are
// intentionally kept decoupled (compiler-emitted bitcode, EP runtime,
// hip-test harness), so we re-declare it here instead of sharing a
// header. The static_assert block below catches any layout drift at
// compile time. Sibling copies live at:
//   * `backend-mlir-compiler/custom-op-mlir/src/custom_op_mlir.hpp` (compiler)
//   * `tools/hip-test/hip-test.cpp`                           (test
//   driver)
typedef struct {
  void *data;          // Data pointer (host or GPU-accessible per memory_type)
  int64_t *shape;      // Array of dimension sizes
  size_t rank;         // Number of dimensions
  size_t element_size; // Bytes per element (e.g. 4=float32, 2=float16, 8=int64)
  int memory_type;     // One of TENSOR_MEMORY_CPU / _GPU / _FPGA / _NPU
} tensor_t;

// Compile-time guard for the wire-protocol ABI described above. The same
// three asserts live in each of the three sibling headers; if you reorder
// / add / remove a field in one copy and forget to mirror it in the others,
// at least one of them fails to build. Per-field offsets (not raw sizeof)
// because trailing padding after `memory_type` is compiler-defined and not
// part of what the JITted per-model code actually reads.
static_assert(offsetof(tensor_t, data) == 0,
              "tensor_t.data must remain the first field");
static_assert(offsetof(tensor_t, shape) == sizeof(void *),
              "tensor_t.shape moved -- update all three tensor_t copies");
static_assert(offsetof(tensor_t, memory_type) ==
                  offsetof(tensor_t, element_size) + sizeof(size_t),
              "tensor_t.memory_type moved -- update all three tensor_t "
              "copies");

// Represents a span of tensors (inputs or outputs)
typedef struct {
  tensor_t *data; // Array of tensors
  size_t count;   // Number of tensors
} span_t;

// Represents a prepared tensor with GPU buffer and metadata
// Used internally by tensor preparation helpers
typedef struct {
  void *gpu_ptr;      // GPU memory (allocated, from pool, or aliased)
  void *host_ptr;     // Host memory (from tensor_t.data)
  int64_t *shape_ptr; // Shape array (from tensor_t.shape) for memref building
  size_t rank;        // Tensor rank (for validation)
  size_t size_bytes;  // Buffer size
  bool is_pooled;     // Internal: true if from pool, false if allocated
  // Internal: true if gpu_ptr aliases caller's GPU-accessible memory
  // (tensor_t.memory_type == TENSOR_MEMORY_GPU). When set, free_input skips
  // pool_release/hipFree because the memory is owned by the caller, not by us.
  bool is_aliased;
} TensorBuffer;

//===----------------------------------------------------------------------===//
// Constant Access (used by generated inference code)
//===----------------------------------------------------------------------===//

// Get GPU pointer for constant at index
// Returns: GPU pointer (NULL if index out of range or state invalid)
// Ownership: Caller does NOT own pointer (freed in state_cleanup)
void *hipdnn_ep_constant_get(RuntimeState *state, int64_t index);

//===----------------------------------------------------------------------===//
// Tensor Preparation Helpers (allocation-strategy agnostic)
//===----------------------------------------------------------------------===//
//
// These helpers abstract tensor preparation logic (parsing, validation,
// GPU allocation, H2D/D2H transfer) from the generated code.
//
// Design principle: The runtime handles allocation strategy internally.
// Generated code is allocation-strategy agnostic.
//
// Element size: Read from tensor_t.element_size, set by the EP caller.
//===----------------------------------------------------------------------===//

// Prepare input tensor: parse, validate, get/allocate GPU buffer, H2D transfer
//
// Parameters:
//   state: Runtime state (provides stream, may contain pre-allocated buffers)
//   inputs: Span of input tensors
//   index: Which tensor to prepare (0-based)
//   expected_rank: Compile-time known rank (from module metadata)
//   out_buffer: Output TensorBuffer to populate
int hipdnn_ep_tensor_prepare_input(RuntimeState *state, span_t *inputs,
                                   size_t index, size_t expected_rank,
                                   TensorBuffer *out_buffer);

// Release input tensor buffer (no D2H transfer needed)
//
// Parameters:
//   state: Runtime state
//   buffer: TensorBuffer from prepare_input
void hipdnn_ep_tensor_free_input(RuntimeState *state, TensorBuffer *buffer);

// Synchronize the GPU stream and print PERF/profile timing (if enabled).
// Called by generated code after compute, before free_input.
int hipdnn_ep_stream_sync(RuntimeState *state);

// Per-operator profiling state accessor (OpProfileState*, gated on
// HIPDNN_EP_PERF)
void *hipdnn_ep_state_get_op_profile(RuntimeState *state);

// NOTE: the GQA GEMM descriptor cache (GqaGemmCache) formerly lived in
// RuntimeState::gqa_gemm_cache with a hipdnn_ep_gqa_gemm_cache_destroy teardown
// shim here. It is now per-instance: each gqa instance owns one in its GqaState
// op-state slot, so concurrent sessions (and distinct GQA layers) no longer
// share it. See docs/design/op-state-slots-design.md.

// NOTE: the MultiHeadAttention GEMM descriptor cache (MhaGemmCache) formerly
// lived in RuntimeState::mha_gemm_cache with a hipdnn_ep_mha_gemm_cache_destroy
// teardown shim here. It is now per-instance: each multi_head_attention
// instance owns one in its MhaState op-state slot. See
// docs/design/op-state-slots-design.md.

// NOTE: the CausalConvWithState descriptor/algo cache (CausalConvCache)
// formerly lived in RuntimeState::causal_conv_cache with a
// hipdnn_ep_causal_conv_cache_destroy teardown shim here. It is now
// per-instance: each causal_conv_with_state instance owns one in its
// CausalConvState op-state slot, so concurrent sessions no longer share it.
// See docs/design/op-state-slots-design.md.

// Asym zero_points unpack cache lifecycle (qmoe-owned RuntimeState cache;
// matmul_nbits keeps a per-instance cache in its op-state slot).
void hipdnn_ep_zp_unpack_cache_destroy(void *cache);

// TensorBuffer Field Accessors (Opaque Pattern)
//===----------------------------------------------------------------------===//
//
// These accessors allow generated code to extract fields from TensorBuffer
// without knowing its internal layout. This maintains abstraction and allows
// the struct definition to evolve without breaking generated code.
//
// Design: TensorBuffer is opaque to generated MLIR code, accessed only via
// these functions (same pattern as RuntimeState accessors above).
//===----------------------------------------------------------------------===//

// Get GPU pointer from TensorBuffer
// Returns: GPU memory pointer (NULL on error)
void *hipdnn_ep_tensor_buffer_get_gpu_ptr(TensorBuffer *buffer);

// Get host pointer from TensorBuffer
// Returns: Host memory pointer (NULL on error)
void *hipdnn_ep_tensor_buffer_get_host_ptr(TensorBuffer *buffer);

// Get shape array pointer from TensorBuffer
// Returns: Pointer to int64_t shape array (NULL on error)
int64_t *hipdnn_ep_tensor_buffer_get_shape_ptr(TensorBuffer *buffer);

// Get rank from TensorBuffer
// Returns: Tensor rank (number of dimensions)
size_t hipdnn_ep_tensor_buffer_get_rank(TensorBuffer *buffer);

// Get buffer size in bytes from TensorBuffer
// Returns: Size in bytes
size_t hipdnn_ep_tensor_buffer_get_size_bytes(TensorBuffer *buffer);

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

// GPU D2D memcpy (hipMemcpyAsync); called from generated LLVM IR.
// Follows opaque RuntimeState pattern - extracts stream internally
//
// Parameters:
//   state: Runtime state (provides GPU stream)
//   dst_ptr: Destination GPU buffer pointer
//   src_ptr: Source GPU buffer pointer
//   size_bytes: Number of bytes to copy
//
// Return codes:
//   0 = success
//   -1 = copy failed
int wrap_hipMemcpyAsync(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                        size_t size_bytes);

/// 2D pitched device copy (e.g. strided memref → dense output). Width is in
/// bytes; pitches are row pitches (hipMemcpy2DAsync semantics).
int wrap_hipMemcpy2DAsync(RuntimeState *state, void *dst_ptr, size_t dst_pitch,
                          const void *src_ptr, size_t src_pitch, size_t width,
                          size_t height);

/// Parallel strided D2D copy via a single kernel launch (element units). Fast
/// path for a pitched copy with very thin rows, where hipMemcpy2DAsync
/// degenerates into one micro-transfer per row. `height` rows, each copying
/// `row_elems` contiguous elements; outer strides are `*_pitch_elems`
/// (elements). Falls back to hipMemcpy2DAsync internally on kernel failure.
int wrap_strided_copy(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                      int64_t elem_bytes, int64_t height,
                      int64_t src_pitch_elems, int64_t dst_pitch_elems,
                      int64_t row_elems);

//===----------------------------------------------------------------------===//
// Library Operations (MIOpen, hipBLAS)
//===----------------------------------------------------------------------===//

int wrap_im2d2col(RuntimeState *state, const void *input, int64_t data_type,
                  int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw,
                  int64_t pad_top, int64_t pad_bottom, int64_t pad_left,
                  int64_t pad_right, int64_t stride_h, int64_t stride_w,
                  int64_t dilation_h, int64_t dilation_w, void *output,
                  int64_t out_h, int64_t out_w);

// MIOpen convolution forward operation
// Full wrapper with descriptor creation, algorithm finding, workspace
// management. Follows opaque RuntimeState pattern - extracts handle/stream
// internally. Parameters match generated LLVM IR from HipToLLVM pass.
//
// `data_type` is a HIPDNN_EP_DATATYPE_* enum value applied uniformly to the
// input / weights / output tensor descriptors — MIOpen requires all three to
// share the same element type. The host-side lowering derives this from the
// hip.conv result memref's element type.
int wrap_miopenConvolutionForward(
    RuntimeState
        *state, // RuntimeState (opaque - extracts handle/stream internally)
    const void *input,   // Input tensor GPU pointer
    int64_t input_n,     // Input batch size
    int64_t input_c,     // Input channels
    int64_t input_h,     // Input height
    int64_t input_w,     // Input width
    const void *weights, // Weights tensor GPU pointer
    int64_t weights_k,   // Output channels (number of filters)
    const void *bias,    // Bias tensor GPU pointer (nullable)
    void *output,        // Output tensor GPU pointer (in-place)
    int64_t output_h,    // Output height
    int64_t output_w,    // Output width
    int64_t kernel_h,    // Kernel height
    int64_t kernel_w,    // Kernel width
    int64_t stride_h,    // Stride height
    int64_t stride_w,    // Stride width
    int64_t pad_top,     // Padding top
    int64_t pad_left,    // Padding left
    int64_t pad_bottom,  // Padding bottom
    int64_t pad_right,   // Padding right
    int64_t dilation_h,  // Dilation height
    int64_t dilation_w,  // Dilation width
    int64_t group,       // Number of groups
    int64_t data_type);  // HIPDNN_EP_DATATYPE_* for I/O and weights

// MIOpen transposed convolution (deconvolution) wrapper
// Uses MIOpen's miopenTranspose convolution mode. Follows the opaque
// RuntimeState pattern - extracts handle/stream internally.
// Weight layout is ONNX ConvTranspose's [C, M/group, kH, kW] (input channels
// first); M/group is derived from output_c and group inside the wrapper.
int wrap_miopenConvolutionTranspose(
    RuntimeState *state, // RuntimeState (opaque)
    const void *input,   // Input tensor GPU pointer [N, C, H, W]
    int64_t input_n,     // Input batch size
    int64_t input_c,     // Input channels (C)
    int64_t input_h,     // Input height
    int64_t input_w,     // Input width
    const void *weights, // Weights GPU pointer [C, M/group, kH, kW]
    const void *bias,    // Bias GPU pointer (nullable) [M]
    void *output,        // Output tensor GPU pointer (in-place) [N, M, H', W']
    int64_t output_c,    // Output channels (M)
    int64_t output_h,    // Output height
    int64_t output_w,    // Output width
    int64_t kernel_h,    // Kernel height
    int64_t kernel_w,    // Kernel width
    int64_t stride_h,    // Stride height
    int64_t stride_w,    // Stride width
    int64_t pad_top,     // Padding top
    int64_t pad_left,    // Padding left
    int64_t pad_bottom,  // Padding bottom
    int64_t pad_right,   // Padding right
    int64_t dilation_h,  // Dilation height
    int64_t dilation_w,  // Dilation width
    int64_t output_padding_h, // Output padding height (ONNX "adjs")
    int64_t output_padding_w, // Output padding width
    int64_t group,            // Number of groups
    int64_t data_type);       // HIPDNN_EP_DATATYPE_* element type

// hipBLASLt GEMM operation wrapper
// Called by generated IR for matrix multiplication operations
int wrap_hipblasLtGemm(void *handle, // hipBLASLt handle
                       void *stream, // HIP stream
                       int64_t m, int64_t n, int64_t k,
                       const void *alpha, // Scalar alpha
                       const void *A,     // Matrix A GPU pointer
                       const void *B,     // Matrix B GPU pointer
                       const void *beta,  // Scalar beta
                       void *C);          // Matrix C GPU pointer (in/out)

// MatMul operation wrapper (batched matrix multiplication)
// Called by generated IR for onnx.MatMul lowering
// Computes output = A @ B for each batch
// A: [batch_count x M x K], B: [K x N] (broadcast) or [batch_count x K x N]
// output: [batch_count x M x N]
//
// `b_batch_stride` is hipBLASLt's STRIDED_BATCH_OFFSET on layA when
// `batch_count > 1`: the per-batch advance in elements through B. It MUST be:
//   * 0   when B is a broadcast weight — one matrix reused across all
//         batches. Includes both rank-2 `[K, N]` and rank-N
//         `[1, ..., 1, K, N]` (any leading-dim product == 1).
//   * K*N when B is per-batch — leading-dim product > 1, so the buffer
//         actually holds multiple `[K, N]` matrices laid out contiguously.
// Mis-setting this to K*N for a broadcast B causes hipBLASLt to step K*N
// elements past the end of the weight buffer on every batch beyond the
// first, reading uninitialised memory into the GEMM and producing wrong
// (often NaN) outputs for batch > 0. For batch_count == 1 the value is
// ignored. Always pass an exact stride; the compiler computes 0 vs K*N
// at compile time when B's leading dims are static, else at runtime.
int wrap_hipblasLtMatmul(
    RuntimeState *state,
    int op_state_slot,       // per-instance op-state slot (shared algo table)
    const void *A,           // Matrix A GPU pointer
    const void *B,           // Matrix B GPU pointer
    void *output,            // Output GPU pointer
    int64_t M,               // Rows of A (per batch)
    int64_t N,               // Columns of B
    int64_t K,               // Columns of A / Rows of B
    int64_t batch_count,     // Number of batches
    int64_t elem_size,       // Element size in bytes (2=f16, 4=f32)
    int64_t b_batch_stride); // 0 = broadcast (any rank); K*N = per-batch

// GroupQueryAttention operation wrapper (Full MS spec)
// Called by generated IR for onnx.Custom(GroupQueryAttention) lowering
// GQA runtime wrapper following the complete Microsoft ONNX Runtime
// specification (14 inputs + 12 attributes).  Supports separate Q/K/V and
// packed QKV paths, optional RoPE, KV cache management, local window
// attention (local_window_size), and smooth softmax (head_sink /
// smooth_softmax).
int wrap_group_query_attention(
    RuntimeState *state,
    int op_state_slot, // per-instance op-state slot (GEMM descriptor cache)
    // Inputs 1-7 (core GQA)
    void *query, void *key, void *value, void *past_key, void *past_value,
    void *seqlens_k, void *total_seq_len,
    // Inputs 8-10 (RoPE)
    void *cos_cache, void *sin_cache, void *position_ids,
    // Inputs 11-14 (advanced features)
    void *attention_bias, void *head_sink, void *k_scale, void *v_scale,
    // Outputs
    void *output, void *present_key, void *present_value, void *output_qk,
    // Attributes (13)
    int64_t num_heads, int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width,
    // Whisper bidirectional-attention flag: when non-zero, the causal mask
    // step is skipped. Default 0 preserves Llama / gpt-oss behaviour.
    int32_t no_causal,
    // Shape values (6)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t past_buf_seq, int64_t head_dim, int64_t element_size_bytes,
    int64_t attn_bias_batch, int64_t attn_bias_num_heads);

// MultiHeadAttention operation wrapper (com.microsoft.MultiHeadAttention v1).
// Called by generated IR for onnx.Custom(MultiHeadAttention) lowering.
//
// Today this is a stub: the function only logs its parameters and throws a
// std::runtime_error indicating the op is not yet implemented. The full MS
// MultiHeadAttention spec covers 1-10 inputs (query, optional key/value,
// bias, key_padding_mask, attention_bias, past_key/value, past_seq_len,
// cache_indirection) and 1-4 outputs (output, optional present_key/value,
// qk). See lib/Conversion/OnnxToHip/MultiHeadAttentionConversion.cpp for
// the input layout the compiler emits.
//
// Optional pointer args: pass nullptr if the corresponding input/output is
// absent. Shape parameters describe the query layout the compiler observed:
//   query_rank      = 3 (standard [B, S, hidden]) or
//                     5 (packed QKV [B, S_kv, num_heads, 3, head_size])
//   query_hidden    = query.shape[2] when rank==3 (else 0)
//   head_size       = query.shape[-1] when rank==5 (else 0; runtime derives
//                                                   from hidden / num_heads)
//   seq_len_kv      = key.shape[1] when key is provided else 0
//                     (self-attention: == seq_len_q at runtime)
//   v_hidden        = value.shape[2] when value is provided else 0
int wrap_multi_head_attention(
    RuntimeState *state,
    int op_state_slot, // per-instance op-state slot (GEMM descriptor cache)
    // Inputs 1-10 (10 pointers - some may be nullptr)
    void *query, void *key, void *value, void *bias, void *key_padding_mask,
    void *attention_bias, void *past_key, void *past_value,
    void *past_sequence_length, void *cache_indirection,
    // Outputs 1-4 (last 3 may be nullptr)
    void *output, void *present_key, void *present_value, void *qk,
    // Attributes (4)
    int64_t num_heads, float mask_filter_value, float scale,
    int64_t unidirectional,
    // Shape info (8: batch, seq_q, seq_kv, query_hidden, v_hidden, head_size,
    //              query_rank, element_size_bytes)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t query_hidden, int64_t v_hidden, int64_t head_size,
    int64_t query_rank, int64_t element_size_bytes);

// Generic MIOpen tensor operation wrapper with per-operand 4D shapes.
// Computes output = op(lhs, rhs) element-wise via miopenOpTensor.
// Each operand is described by 4D shape (N, C, H, W) to enable MIOpen-native
// broadcasting: dims of 1 are broadcast against the corresponding larger dim.
//   tensor_op: HIPDNN_EP_TENSOR_OP_* constant (mul, add, min, max)
//   data_type: HIPDNN_EP_DATATYPE_* constant identifying the element type
int wrap_miopenOpTensor(RuntimeState *state, int op_state_slot, void *lhs,
                        void *rhs, void *output, int64_t lhs_n, int64_t lhs_c,
                        int64_t lhs_h, int64_t lhs_w, int64_t rhs_n,
                        int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
                        int64_t out_n, int64_t out_c, int64_t out_h,
                        int64_t out_w, int64_t data_type, int64_t tensor_op);

// Element-wise subtraction with 4D ONNX broadcast (rank <= 4).
// Computes output = lhs - rhs; materialises broadcast via hip_expand when
// an operand shape differs from the output shape. Sub is not commutative --
// operands are never swapped.
int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                         void *output, int64_t lhs_n, int64_t lhs_c,
                         int64_t lhs_h, int64_t lhs_w, int64_t rhs_n,
                         int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
                         int64_t out_n, int64_t out_c, int64_t out_h,
                         int64_t out_w, int64_t data_type);

// Element-wise Where wrapper (NumPy-style multidirectional broadcasting,
// arbitrary rank). Computes output[i] = condition[i] ? x[i] : y[i] with
// per-operand broadcasting.
//
// Each operand is described by its own (shape, rank) pair: the shape array
// holds `rank` i64 dims in row-major order. Operand shapes are left-padded
// with 1s up to `out_rank` by the runtime, and dims of 1 are broadcast
// against the corresponding larger output dim. No fixed layout is assumed;
// any rank up to HIP_WHERE_MAX_RANK is supported.
//
//   condition: bool tensor (1 byte per element)
//   x, y, output: same data_type (HIPDNN_EP_DATATYPE_*)
int wrap_where(RuntimeState *state, void *condition, void *x, void *y,
               void *output, const int64_t *cond_shape, int64_t cond_rank,
               const int64_t *x_shape, int64_t x_rank, const int64_t *y_shape,
               int64_t y_rank, const int64_t *out_shape, int64_t out_rank,
               int64_t data_type);

// Unified power entry: output = f(input; alpha, beta, gamma).
// alpha, beta, gamma match the MIOpen POWER activation tuple where the
// MIOpen path is used. data_type is HIPDNN_EP_DATATYPE_* (FLOAT=0, HALF=1,
// BFLOAT16=2).
//
// LLVM lowering always calls this symbol. For (0, 1, -1) and (0, 1, 0.5) the
// runtime uses HIP elementwise reciprocal and sqrt kernels (ONNX semantics).
// Other (alpha, beta, gamma) tuples use miopenActivationPOWER /
// miopenActivationForward.
int wrap_power(RuntimeState *state, void *input, void *output,
               int64_t num_elements, int64_t data_type, double alpha,
               double beta, double gamma);

// Gather operation wrapper.
// `axis_size` = data.shape[axis]; `inner_size` = product of
// data.shape[axis+1:]. outer_size is derived as data_num_elements / (axis_size
// * inner_size).
int wrap_gather(RuntimeState *state, void *data, void *indices, void *output,
                int64_t axis, int64_t data_num_elements,
                int64_t indices_num_elements, int64_t output_num_elements,
                int64_t axis_size, int64_t inner_size,
                int64_t element_size_bytes, int64_t indices_element_size_bytes);

int wrap_gather_elements(RuntimeState *state, void *data, void *indices,
                         void *output, int64_t axis, int64_t rank,
                         const int64_t *data_shape,
                         const int64_t *indices_shape, int64_t num_elements,
                         int64_t element_size_bytes,
                         int64_t indices_element_size_bytes);

int wrap_top_k(RuntimeState *state, void *x, void *k, void *values,
               void *indices, int64_t axis, int64_t largest, int64_t sorted,
               int64_t rank, const int64_t *x_shape, int64_t num_elements,
               int64_t element_size_bytes);

int wrap_scatter_elements(RuntimeState *state, void *data, void *indices,
                          void *updates, void *output, int64_t axis,
                          int64_t reduction_id, int64_t rank,
                          const int64_t *data_shape,
                          const int64_t *indices_shape, int64_t num_updates,
                          int64_t element_size_bytes,
                          int64_t indices_element_size_bytes);

int wrap_compress(RuntimeState *state, void *input, void *condition,
                  void *output, int64_t flatten, int64_t axis,
                  int64_t input_rank, int64_t output_rank,
                  const int64_t *input_shape, const int64_t *output_shape,
                  int64_t condition_len, int64_t num_output_elements,
                  int64_t element_size_bytes);

int wrap_one_hot(RuntimeState *state, void *indices, void *depth, void *values,
                 void *output, int64_t axis, int64_t indices_rank,
                 int64_t output_rank, const int64_t *indices_shape,
                 const int64_t *output_shape, int64_t num_indices,
                 int64_t num_output_elements, int64_t element_size_bytes,
                 int64_t indices_element_size_bytes,
                 int64_t depth_element_size_bytes);

// Range operation wrapper
int wrap_range(RuntimeState *state, void *start, void *limit, void *delta,
               void *output, int64_t output_num_elements, int64_t hip_dtype);

// Transpose operation wrapper (ONNX Transpose).
// Permutes the dimensions of `input` according to `perm` and writes the
// result to `output`.  `input_shape` and `perm` are host-side arrays of
// length `rank`; `num_elements` is the product of `input_shape`.
// `element_size_bytes` selects the kernel datapath (1/2/4/8 currently).
int wrap_transpose(RuntimeState *state, const void *input, void *output,
                   int64_t rank, const int64_t *input_shape,
                   const int64_t *perm, int64_t num_elements,
                   int64_t element_size_bytes);

// ReduceSum operation wrapper
// data_type: HIPDNN_EP_DATATYPE_* enum value identifying the element type.
// Supported types: HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_INT32,
//                  HIPDNN_EP_DATATYPE_INT64.
// `inner_size` = product of input dims AFTER the reduced axis (1 for a
// trailing/contiguous reduce); enables strided reduction over a non-trailing
// axis (e.g. NCHW channel-axis LayerNorm2d).
int wrap_reduce_sum(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes,
                    int64_t inner_size);

// ReduceMean operation wrapper
// data_type: HIPDNN_EP_DATATYPE_* enum value identifying the element type.
// Supported types: HIPDNN_EP_DATATYPE_HALF (ONNX ReduceMean is float-domain).
// The division by the reduced-element count happens in-kernel, so a dynamic
// reduce axis is tolerated.
// `inner_size` = product of input dims AFTER the reduced axis (1 for a
// trailing/contiguous reduce); enables strided reduction over a non-trailing
// axis (e.g. NCHW channel-axis LayerNorm2d).
int wrap_reduce_mean(RuntimeState *state, void *data, void *axes, void *output,
                     int64_t data_num_elements, int64_t output_num_elements,
                     int64_t axes_num_elements, int64_t data_type,
                     int64_t keepdims, int64_t noop_with_empty_axes,
                     int64_t inner_size);

// ReduceL2 operation wrapper
// data_type: HIPDNN_EP_DATATYPE_* enum value identifying the element type.
// Supported types: HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_FLOAT.
// Computes sqrt(sum(x^2)) in-kernel, so a dynamic reduce axis is tolerated.
// `inner_size` = product of input dims AFTER the reduced axis (1 for a
// trailing/contiguous reduce); enables strided reduction over a non-trailing
// axis.
int wrap_reduce_l2(RuntimeState *state, void *data, void *axes, void *output,
                   int64_t data_num_elements, int64_t output_num_elements,
                   int64_t axes_num_elements, int64_t data_type,
                   int64_t keepdims, int64_t noop_with_empty_axes,
                   int64_t inner_size);

// ReduceMax operation wrapper
// data_type: HIPDNN_EP_DATATYPE_* enum value identifying the element type.
int wrap_reduce_max(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes,
                    int64_t inner_size);

// ReduceMin operation wrapper
// data_type: HIPDNN_EP_DATATYPE_* enum value identifying the element type.
int wrap_reduce_min(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes,
                    int64_t inner_size);

// Cast operation wrapper (element type conversion)
// src_data_type and dst_data_type are HIPDNN_EP_DATATYPE_* enum values.
int wrap_cast(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t src_data_type,
              int64_t dst_data_type);

// Generic MIOpen activation wrapper
// Applies activation_mode (HIPDNN_EP_ACTIVATION_*) element-wise
// data_type: HIPDNN_EP_DATATYPE_* constant identifying the element type
int wrap_miopenActivationForward(RuntimeState *state, int op_state_slot,
                                 void *input, void *output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode);

// GELU activation wrapper (uses custom HIP kernel)
// Applies GELU element-wise with support for exact or approximate mode
// data_type: HIPDNN_EP_DATATYPE_* (supports FLOAT, HALF, BFLOAT16, DOUBLE)
// approximate: 0 = exact (erf), 1 = tanh approximation
int wrap_gelu(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type, int64_t approximate);

// Fused com.microsoft.BiasGelu: Gelu_erf(data + broadcast(bias)).
// data_type: HIPDNN_EP_DATATYPE_* (supports FLOAT, HALF, BFLOAT16, DOUBLE)
int wrap_bias_gelu(RuntimeState *state, void *data, void *bias, void *output,
                   int64_t num_elements, int64_t bias_len, int64_t data_type);

// Fused com.microsoft.FastGelu: tanh-approx Gelu on (input + broadcast(bias)).
// bias_len == 0 means no bias (bias pointer ignored).
int wrap_fast_gelu(RuntimeState *state, void *input, void *bias, void *output,
                   int64_t num_elements, int64_t bias_len, int64_t data_type);

// LeakyRelu activation wrapper (uses custom HIP kernel).
// data_type: HIPDNN_EP_DATATYPE_* (supports FLOAT, HALF, DOUBLE)
// alpha: slope for negative values (default 0.01 per ONNX spec)
int wrap_leaky_relu(RuntimeState *state, void *input, void *output,
                    int64_t num_elements, int64_t data_type, double alpha);

// Window-pool wrapper (uses custom HIP kernel).
// Generic ONNX MaxPool / AveragePool / LpPool over (N, C, D_1[, D_2[, D_3]])
// input with row-major output layout.  `pool_mode` (HIPDNN_EP_POOL_*) selects
// the per-window reduction: AVERAGE / MAX / LP.  When `has_indices=1` (MAX
// only), also writes per-output flat int64 indices into the unpadded input.
// `spatial_rank` selects how many of the three trailing axes
// (in_*, out_*, k*, s*, p*, dil*) are read; unused slots must be 1
// (kernel/dim) or 0 (pad).  `storage_order` and `ceil_mode` are pre-resolved
// at compile time and accepted only for ABI completeness.  `count_include_pad`
// is the AveragePool divisor selector; `p` is the LpPool norm exponent — both
// are ignored for the modes that don't use them.
// data_type: HIPDNN_EP_DATATYPE_* (supports FLOAT, HALF, BFLOAT16, DOUBLE).
int wrap_pool(RuntimeState *state, void *input, void *output, void *indices,
              int64_t data_type, int64_t pool_mode, int64_t spatial_rank,
              int64_t N, int64_t C, int64_t in0, int64_t in1, int64_t in2,
              int64_t out0, int64_t out1, int64_t out2, int64_t k0, int64_t k1,
              int64_t k2, int64_t s0, int64_t s1, int64_t s2, int64_t p0,
              int64_t p1, int64_t p2, int64_t dil0, int64_t dil1, int64_t dil2,
              int64_t storage_order, int64_t ceil_mode, int64_t has_indices,
              int64_t count_include_pad, int64_t p);
// Resize wrapper (uses custom HIP kernel).
// Spatial-axis-only resize over (N, C, D_1[, D_2[, D_3]]) input; (N, C)
// pass-through.  `mode` (0=nearest, 1=linear), `coord_transform`
// (0=half_pixel, 1=asymmetric, 2=align_corners) and `nearest_mode`
// (0=round_prefer_floor) are pre-resolved at compile time from the ONNX
// string attributes.  data_type: HIPDNN_EP_DATATYPE_* (FLOAT, HALF,
// BFLOAT16, DOUBLE).

int wrap_resize(RuntimeState *state, void *input, void *output,
                int64_t data_type, int64_t spatial_rank, int64_t N, int64_t C,
                int64_t in0, int64_t in1, int64_t in2, int64_t out0,
                int64_t out1, int64_t out2, int64_t mode,
                int64_t coord_transform, int64_t nearest_mode);

// Global pool wrapper (uses custom HIP kernel).
// Treats the data as a flat [outer, reduce_size] matrix and writes one
// reduced value per row into output. Covers ONNX GlobalAveragePool /
// GlobalMaxPool / GlobalLpPool of any input rank >= 3:
//   outer       = N * C                       (leading two input dims)
//   reduce_size = D_1 * D_2 * ... * D_k       (product of spatial dims)
// data_type: HIPDNN_EP_DATATYPE_* (supports FLOAT, HALF, BFLOAT16, DOUBLE)
// mode      : HIPDNN_EP_GLOBAL_POOL_* (AVERAGE / MAX / LP)
// p         : LP-norm exponent; only consumed when mode == LP, otherwise
//             ignored. ONNX spec requires `p >= 1`; values below that are
//             rejected upstream during ONNX→HIP conversion.
int wrap_global_pool(RuntimeState *state, void *input, void *output,
                     int64_t outer, int64_t reduce_size, int64_t data_type,
                     int64_t mode, int64_t p);

// Rotary embedding operation wrapper.
//
// Supports M-RoPE / partial rotary embedding (rotary_dim < head_dim) and the
// two standard input layouts:
//   is_bnsh == 0 : BSNH [batch, seq_len, num_heads, head_dim]
//                  (also covers 3D [batch, seq_len, num_heads*head_dim])
//   is_bnsh != 0 : BNSH [batch, num_heads, seq_len, head_dim]
//                  (ONNX com.microsoft.RotaryEmbedding 4D default; GQA K/V)
// position_ids may be NULL: native ai.onnx RotaryEmbedding (opset >= 23)
// without position_ids ships cos/sin already position-expanded to
// [batch, seq_len, rotary_dim/2], and the kernel indexes them by the flat
// token position b*seq+s instead of looking up position_ids.
int wrap_rotary_embedding(RuntimeState *state, void *input, void *position_ids,
                          void *cos_cache, void *sin_cache, void *output,
                          int64_t interleaved, int64_t batch_size,
                          int64_t seq_len, int64_t num_heads, int64_t head_dim,
                          int64_t rotary_dim, int64_t cos_cache_num_elements,
                          int64_t element_size_bytes, int64_t is_bnsh);

// SimplifiedLayerNormalization operation wrapper
int wrap_miopenT5LayerNormForward(RuntimeState *state, int op_state_slot,
                                  void *input, void *scale, void *output,
                                  int64_t input_num_elements,
                                  int64_t scale_num_elements,
                                  int64_t element_size_bytes, int64_t axis,
                                  float epsilon, int64_t stash_type);

// LayerNormalization operation wrapper (standard ONNX opset 17+)
// bias, mean, inv_std may be nullptr when optional inputs/outputs are absent
int wrap_layer_normalization(RuntimeState *state, void *input, void *scale,
                             void *bias, void *output, void *mean,
                             void *inv_std, int64_t input_num_elements,
                             int64_t scale_num_elements,
                             int64_t element_size_bytes, int64_t axis,
                             float epsilon, int64_t stash_type);

// SkipSimplifiedLayerNormalization operation wrapper (Full MS spec)
// Computes: input_skip_bias_sum = input + skip [+ bias]
//           output = RMSNorm(input_skip_bias_sum) * gamma
// bias and input_skip_bias_sum may be nullptr (optional per MS spec)
int wrap_skip_simplified_layer_norm(RuntimeState *state, int op_state_slot,
                                    void *input, void *skip, void *gamma,
                                    void *bias, void *output,
                                    void *input_skip_bias_sum,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes, float epsilon);

// MatMulNBits operation wrapper (quantized N-bit matrix multiplication)
// Dequantizes packed int4 weights and computes Y = A @ dequant(B)^T + bias
// A: [batch_count x M x K], B: [N x k_blocks x blob_size] (packed uint8)
// scales: [N x k_blocks], output: [batch_count x M x N]
// Optional: zero_points, g_idx (deprecated), bias - pass nullptr if absent
int wrap_matmul_nbits(
    RuntimeState *state,
    int op_state_slot,       // per-instance op-state slot (zp-cache home)
    const void *A,           // activation tensor
    const void *B,           // packed quantized weights
    const void *scales,      // per-block scale factors
    const void *zero_points, // per-block zero points (nullable)
    const void *g_idx,       // GPTQ group indices (nullable, deprecated)
    const void *bias,        // output bias [N] (nullable)
    void *output,            // result tensor
    int64_t M,               // rows per batch
    int64_t N,               // output columns
    int64_t K,               // inner dimension
    int64_t batch_count,     // number of batches
    int64_t bits,            // quantization bits (e.g. 4)
    int64_t block_size,      // quantization block size
    int64_t elem_size,       // element size in bytes
    int64_t zp_elem_size);   // zero_points element size: 1=uint8 packed, 2=fp16

// GatherBlockQuantized operation wrapper (com.microsoft).
// Gather + block-wise dequantize: gather rows from `data` along
// `gather_axis` using `indices`, then dequantize the gathered rows using
// per-block `scales` and optional `zero_points`. Handles sub-byte
// unpacking when `bits == 4` (two values per byte, low nibble first).
//
// Shapes are passed as int64_t* arrays + ranks because both `data` and
// `indices` rank are arbitrary per ONNX spec. zero_points is nullable —
// when null, default is 0 for int4/uint4, 2^(bits-1) for uint8.
//
// Currently a stub — see lib/Runtime/real/gather_block_quantized.cpp.
int wrap_gather_block_quantized(
    RuntimeState *state,
    const void *data,        // packed quantized [r-rank]
    const void *indices,     // i32/i64 [q-rank]
    const void *scales,      // dequant scales (T2)
    const void *zero_points, // dequant zero points (nullable)
    void *output,            // dequantized output [q + (r-1)-rank]
    const int64_t *data_shape, int64_t data_rank, const int64_t *indices_shape,
    int64_t indices_rank, const int64_t *scales_shape, int64_t scales_rank,
    const int64_t *output_shape, int64_t output_rank,
    int64_t bits,       // 4 or 8
    int64_t block_size, // power of 2, >= 16
    int64_t gather_axis, int64_t quantize_axis,
    int64_t data_dtype,    // HIPDNN_EP_DATATYPE_* (uint8 packed)
    int64_t indices_dtype, // INT32 / INT64
    int64_t scales_dtype); // FLOAT / HALF / BFLOAT16

// QMoE operation wrapper (quantized Mixture-of-Experts)
// Routes tokens to top-k experts, performs quantized MLP per expert,
// applies activation (e.g. SwiGLU), and combines results.
// Optional pointer args: pass nullptr if the corresponding input is absent.
int wrap_qmoe(
    RuntimeState *state,
    const void *input,           // [num_tokens, hidden_size]
    const void *router_probs,    // [num_tokens, num_experts]
    const void *router_weights,  // (nullable) ONNX v1.25+ routing weights
    const void *fc1_weights,     // [num_experts, fusion*inter, hidden/pack]
    const void *fc1_scales,      // [num_experts, fusion*inter, hidden/bs]
    const void *fc1_bias,        // (nullable) [num_experts, fusion*inter]
    const void *fc2_weights,     // [num_experts, hidden, inter/pack]
    const void *fc2_scales,      // [num_experts, hidden, inter/bs]
    const void *fc2_bias,        // (nullable) [num_experts, hidden]
    const void *fc3_weights,     // (nullable) unfused SwiGLU
    const void *fc3_scales,      // (nullable)
    const void *fc3_bias,        // (nullable)
    const void *fc1_zero_points, // (nullable) fc1 dequant zero points
    const void *fc2_zero_points, // (nullable) fc2 dequant zero points
    const void *fc3_zero_points, // (nullable) fc3 dequant zero points
    void *output,                // [num_tokens, hidden_size]
    int64_t num_tokens, int64_t hidden_size, int64_t inter_size,
    int64_t num_experts, int64_t k, int64_t expert_weight_bits,
    int64_t block_size, int64_t swiglu_fusion,
    int64_t activation_type, // 0=relu,1=gelu,2=silu,3=swiglu,4=identity
    float activation_alpha, float activation_beta, float swiglu_limit,
    int64_t normalize_routing_weights, int64_t elem_size);

// CausalConvWithState operation wrapper (stateful causal depthwise convolution)
// Used by Gated DeltaNet (Qwen3.5) and Mamba models.
// Performs causal depthwise convolution with carry state for incremental
// decode. The convolution is causal (looks only at current and past positions)
// and depthwise (each channel convolved independently). Input layout:
// channels-first (batch, channels, seq_len). Weight layout: (channels, 1,
// kernel_size) for 1D depthwise.
//   bias:       nullable - per-channel bias (channels)
//   past_state: nullable - carry state from previous step (batch, channels,
//   k-1) activation: 0=none, 1=silu/swish
int wrap_causal_conv_with_state(
    RuntimeState *state,
    int op_state_slot,  // per-instance op-state slot (descriptor cache home)
    const void *input,  // (batch, channels, seq_len)
    const void *weight, // (channels, 1, kernel_size)
    const void *bias,   // nullable, (channels)
    const void *past_state, // nullable, (batch, channels, kernel_size - 1)
    void *output,           // (batch, channels, seq_len)
    void *present_state,    // (batch, channels, kernel_size - 1)
    int64_t batch_size, int64_t channels, int64_t seq_len, int64_t kernel_size,
    int64_t ndim,
    int64_t activation, // 0=none, 1=silu/swish
    int64_t element_size_bytes);

//==============================================================================
// ONNX Gemm via hipBLASLt
//==============================================================================
// Y = alpha * op(A) * op(B) + beta * C
// op(A) shape: [M, K], op(B) shape: [K, N], C optional broadcastable to [M, N]
// LinearAttention operation wrapper (com.microsoft.LinearAttention)
// Unified linear attention with recurrent state for autoregressive decoding
// and prefill. Supports update rules: linear(0), gated(1), delta(2),
// gated_delta(3).
// All tensor inputs use packed 3D format [B, T, H*D] except past/present
// state which is 4D [B, H_kv, d_k, d_v].
// Head counts are three-way:
//   - Hq : query heads
//   - Hkv: KV state heads
//   - Nk : number of key heads in the packed key tensor (Nk divides H_kv when
//          multiple KV heads share a K head; Nk < Hkv).
// Supports both standard GQA (Hq >= Hkv, Hq % Hkv == 0) and inverse GQA
// (Hq < Hkv, Hkv % Hq == 0).  The output tensor's last dim is max(Hq, Hkv)*d_v,
// packed in Q-head order for standard GQA and KV-head order for inverse GQA.
// Optional pointer args: pass nullptr if the corresponding input is absent.
// Last arg `type` is HIPDNN_EP_DATATYPE_FLOAT (0), HIPDNN_EP_DATATYPE_HALF (1),
// or HIPDNN_EP_DATATYPE_BFLOAT16 (2).
//
// decay_per_key_dim / beta_per_head describe the layout of the optional
// decay / beta tensors so the runtime can pick the correct stride. Values
// are ignored when the corresponding pointer is nullptr; compilers should
// pass 0 in that case.
//   decay_per_key_dim = 1  -> decay is [B, T, H_kv * d_k] (GLA / RWKV-6)
//                     = 0  -> decay is [B, T, H_kv]       (DeltaNet / RetNet)
//   beta_per_head     = 1  -> beta  is [B, T, H_kv]
//                     = 0  -> beta  is [B, T, 1]          (broadcast over
//                     heads)
int wrap_linear_attention(
    RuntimeState *state,
    const void *query,      // [B, T, Hq * dk]
    const void *key,        // [B, T, Nk * dk]
    const void *value,      // [B, T, H_kv * d_v]
    const void *past_state, // [B, H_kv, d_k, d_v] (nullable)
    const void *decay,      // [B, T, H_kv * d_k] or [B, T, H_kv] (nullable)
    const void *beta,       // [B, T, H_kv] or [B, T, 1] (nullable)
    void *output,           // [B, T, max(H_q, H_kv) * d_v]
    void *present_state,    // [B, H_kv, d_k, d_v]
    int64_t Hq, int64_t Hkv, int64_t Nk, int64_t decay_per_key_dim,
    int64_t beta_per_head, float scale, int64_t chunk_size,
    int64_t update_rule, // 0=linear, 1=gated, 2=delta, 3=gated_delta
    int64_t B, int64_t seq_len, int64_t dk, int64_t dv, int64_t type);

//==============================================================================
// ONNX Gemm via hipBLASLt
//==============================================================================
// Y = alpha * op(A) * op(B) + beta * C
// op(A) shape: [M, K], op(B) shape: [K, N], C optional broadcastable to [M, N]
int wrap_gemm(RuntimeState *state, int op_state_slot, const void *A,
              const void *B, const void *C, void *output, int64_t M, int64_t N,
              int64_t K, float alpha, float beta, int64_t transA,
              int64_t transB, int64_t typeCode, int64_t cDim0, int64_t cDim1);

// Element-wise Equal. Operand shapes are passed as 4D (N, C, H, W), left-
// padded with 1 by the compiler; `data_type` is the INPUT (comparison
// operand) type and the output is always 1 byte per element. General ONNX
// multidirectional broadcast is handled by materialising any partially-
// broadcast operand to the output shape via hip_expand; scalar and same-shape
// operands take the kernel's direct path.
int wrap_equal(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
               int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
               int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
               int64_t out_h, int64_t out_w, int64_t data_type);

// Element-wise logical AND / OR on bool tensors (marshalled as 1-byte
// uint8 elements). Operand shapes are passed as 4D (N, C, H, W), left-padded
// with 1 by the compiler; ONNX multidirectional broadcast is materialised via
// hip_expand when an operand does not already match the output shape.
// `data_type` is a sentinel (i1 has no HIPDNN dtype slot) and is unused.
int wrap_or(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
            int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
            int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
            int64_t out_h, int64_t out_w, int64_t data_type);
int wrap_and(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
             int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
             int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
             int64_t out_h, int64_t out_w, int64_t data_type);

int wrap_abs(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type);
int wrap_neg(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type);
int wrap_not(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type);

// ONNX NonZero wrapper.
// Returns the indices of the non-zero elements of `input` in row-major
// order, packed into `output` as a [R, N] int64 tensor where R is the
// input rank and N is the (data-dependent) number of non-zero entries.
// The caller pre-allocates the output buffer with capacity `output_capacity`
// elements along the N dim (the OnnxToHip pass uses `input_num_elements`
// as the upper bound); columns beyond the true count are left undefined.
//
// The kernel writes the true count N into the device i32 scalar `count_ptr`.
// The generated code reads it back to the host via `hipdnn_ep_readback_i32`
// (lowered from hip.readback_dim) and slices `output` to [R, N] so consumers
// and the ORT-reported output shape see the real extent, not the capacity.
//
// `input_dims` is the host array of the R input extents (the runtime copies it
// to the device so the kernel can decompose flat indices into coordinates).
//
// `input_data_type` is the HIPDNN_EP_DATATYPE_* value of the input tensor.
// Bool (ONNX `tensor(bool)`) is marshalled as 1-byte uint8 by the EP and
// reuses the INT8 slot here.
int wrap_nonzero(RuntimeState *state, void *input, void *output,
                 int32_t *count_ptr, int64_t input_num_elements,
                 int64_t input_rank, const int64_t *input_dims,
                 int64_t output_capacity, int64_t input_data_type);

// Synchronize the GPU stream (so the producing kernel has completed), then
// copy a single device-resident i32 scalar back to the host and return it.
// Used by hip.readback_dim to turn a kernel-computed runtime extent (e.g.
// NonZero's non-zero count) into a host value that can size dynamic shapes.
int32_t hipdnn_ep_readback_i32(RuntimeState *state, const void *device_scalar);

// Synchronize the GPU stream, then copy a small device-resident scalar of
// arbitrary byte width (1/2/4/8) into the caller-provided host buffer. The
// generic counterpart to hipdnn_ep_readback_i32, used by hip.readback_scalar to
// materialise a GPU-computed scalar (i64/f32/f16/...) on the host for shape
// arithmetic — e.g. the limit/start/delta of a data-dependent onnx.Range whose
// trip count is computed host-side. See the runtime impl for why a bare
// memref.load of a GPU scalar is incorrect on true-device-memory targets.
void hipdnn_ep_readback_scalar(RuntimeState *state, void *host_dst,
                               const void *device_scalar, int64_t num_bytes);

// ONNX Size wrapper (dynamic-shape path only).
//
// Static-shape Size ops are folded into arith.constant at OnnxToHip time
// and never reach this symbol. For inputs with at least one dynamic dim
// the HipToLLVM lowering computes `num_elements = prod(input.shape)` as
// a runtime i64 value (compile-time constants for static dims, MemRef
// descriptor `sizes[]` for dynamic dims) and calls this wrapper, which
// stores the 8-byte value into the rank-0 i64 output buffer on the GPU.
int wrap_size(RuntimeState *state, void *output, int64_t num_elements);
int wrap_cos(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type);
int wrap_sin(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type);
int wrap_ceil(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type);
int wrap_exp(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type);

int wrap_log(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type);

// Element-wise division with 4D ONNX broadcast (rank <= 4, left-padded).
// Computes output = lhs / rhs; materialises broadcast via hip_expand when
// an operand shape differs from the output shape.
int wrap_div(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t lhs_n, int64_t lhs_c, int64_t lhs_h, int64_t lhs_w,
             int64_t rhs_n, int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
             int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
             int64_t data_type);

// CumSum operation wrapper (cumulative sum along an axis).
// `axis` is a rank-0 (scalar) GPU tensor whose i32/i64 value selects the
// reduction axis; the runtime is responsible for reading it (typically a
// single hipMemcpyAsync D2H or kernel-side load).
// `axis_dtype` is HIPDNN_EP_DATATYPE_INT32 / _INT64.
// `data_type` is HIPDNN_EP_DATATYPE_* of the data tensor.
int wrap_cumsum(RuntimeState *state, void *x, void *axis, void *y,
                const int64_t *data_shape, int64_t data_rank,
                int64_t num_elements, int64_t data_type, int64_t axis_dtype,
                int64_t exclusive, int64_t reverse);

// Pad operation wrapper (constant / reflect / edge / wrap modes).
// pads:           int64 1-D tensor [2 * num_axes]
//                 -- formatted as [x1_begin, ..., x1_end, ...]
// constant_value: nullable scalar tensor (only used when mode_id == 0)
// axes:           nullable int64 1-D tensor selecting axes; nullptr/empty
//                 means "all axes"
// mode_id:        0=constant, 1=reflect, 2=edge, 3=wrap
int wrap_pad(RuntimeState *state, void *data, void *pads, void *constant_value,
             void *axes, void *output, const int64_t *data_shape,
             int64_t data_rank, const int64_t *output_shape,
             int64_t output_rank, int64_t pads_num_elements,
             int64_t axes_num_elements, int64_t data_type, int64_t mode_id);

// Tile operation wrapper.
// repeats: int64 1-D tensor of length `data_rank`.
int wrap_tile(RuntimeState *state, void *input, void *repeats, void *output,
              const int64_t *input_shape, int64_t input_rank,
              const int64_t *output_shape, int64_t output_rank,
              int64_t data_type);

// Expand operation wrapper (NumPy-style broadcasting to a target shape).
int wrap_expand(RuntimeState *state, void *input, void *shape, void *output,
                const int64_t *input_shape, int64_t input_rank,
                const int64_t *output_shape, int64_t output_rank,
                int64_t data_type);

// ReduceProd operation wrapper. Same calling convention as wrap_reduce_sum
// / wrap_reduce_max.
int wrap_reduce_prod(RuntimeState *state, void *data, void *axes, void *output,
                     int64_t data_num_elements, int64_t output_num_elements,
                     int64_t axes_num_elements, int64_t data_type,
                     int64_t keepdims, int64_t noop_with_empty_axes,
                     int64_t inner_size);

// Less operation wrapper (element-wise C = A < B). Output is bool (1 byte);
// `data_type` is the INPUT (comparison operand) type. Operand shapes are
// passed as 4D (N, C, H, W), left-padded with 1 by the compiler; ONNX
// multidirectional broadcast is materialised via hip_expand when an operand
// does not already match the output shape.
int wrap_less(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
              int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
              int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
              int64_t out_h, int64_t out_w, int64_t data_type);

// GatherND operation wrapper. data_shape has rank `data_rank`; indices has
// rank `indices_rank` with last dim `indices_inner = indices_shape[-1]`.
int wrap_gather_nd(RuntimeState *state, void *data, void *indices, void *output,
                   const int64_t *data_shape, int64_t data_rank,
                   const int64_t *indices_shape, int64_t indices_rank,
                   const int64_t *output_shape, int64_t output_rank,
                   int64_t batch_dims, int64_t data_type);

// Sign operation wrapper (element-wise sign(x)).
int wrap_sign(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type);

// Mod operation wrapper (element-wise modulo / fmod).
//   fmod = 0  -> integer modulo (Python %); requires integer data_type
//   fmod = 1  -> C fmod         ; requires float data_type
int wrap_mod(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t num_elements, int64_t data_type, int64_t fmod);

// Slice operation wrapper (ONNX Slice native fallback).
//
// Today this is a stub: the OnnxToHip decompose pattern handles the common
// case (compile-time constant starts/ends/axes/steps with positive unit
// stride) by rewriting onnx.Slice to tensor.extract_slice, so this runtime
// entry is only called for non-constant-indices or negative-step Slices.
// The stub only logs its parameters and returns success — models that
// exercise it will produce incorrect Slice output but will still link and
// run end-to-end for IR-shape debugging.
//
// axes / steps may be nullptr when the corresponding optional input is absent.
int wrap_slice(RuntimeState *state, void *data, void *starts, void *ends,
               void *axes, void *steps, void *output, const int64_t *data_shape,
               int64_t data_rank, const int64_t *output_shape,
               int64_t output_rank, int64_t starts_num_elements,
               int64_t axes_num_elements, int64_t steps_num_elements,
               int64_t data_type);

// ScatterND: output = copy(data), then output[indices[i]] (reduction)
// updates[i].
//
// `reduction_id` encodes the ONNX `reduction` attribute as a small enum
// (must match ScatterNDOpLowering::reductionIdFromString):
//
//   0 = "none" (overwrite, default)
//   1 = "add"
//   2 = "mul"
//   3 = "min"
//   4 = "max"
//
// Today this is a stub: it only logs its parameters and returns success.
// The runtime is responsible (when implemented) for both the initial
// data -> output copy and the per-index scatter writes.
int wrap_scatter_nd(RuntimeState *state, void *data, void *indices,
                    void *updates, void *output, const int32_t *count_ptr,
                    const int64_t *data_shape, int64_t data_rank,
                    const int64_t *indices_shape, int64_t indices_rank,
                    const int64_t *updates_shape, int64_t updates_rank,
                    const int64_t *output_shape, int64_t output_rank,
                    int64_t reduction_id, int64_t data_type);

//===----------------------------------------------------------------------===//
// ONNX Loop Drivers
//===----------------------------------------------------------------------===//
//
// Body callback signature shared by both loop drivers. Emitted as a small
// trampoline LLVMFuncOp by the HipToLLVM lowering pass; one trampoline per
// `hip.loop`. The trampoline unpacks the descriptor-pointer arrays and
// invokes the outlined ONNX body function with its actual (variable-arity)
// positional memref-descriptor signature.
//
// Args:
//   state              : RuntimeState pointer.
//   iter_dev_ptr       : device int64_t holding the current iteration index.
//                        Driver writes the host iter value into this buffer
//                        each iter before calling the trampoline.
//   cond_dev_ptr       : device bool (i1, 1 byte) holding the current cond.
//                        Aliased between cond_in and cond_out -- the body
//                        reads it as cond_in and writes the next-iter cond
//                        in place. The driver re-reads it on the slow path
//                        to decide whether to continue.
//   loop_carried_descs : array of pointers to memref descriptors for each
//                        loop-carried slot. The trampoline passes the same
//                        descriptor for both the v_in and v_out positions
//                        of the body -- one buffer per slot, body reads-
//                        and-writes in place (safe under single-pass kernel
//                        semantics).
//   capture_descs      : array of pointers to memref descriptors for each
//                        captured value. Treated read-only by the body.
// Returns 0 on success, non-zero on body failure.
typedef int (*HipdnnEpLoopBodyFn)(RuntimeState *state, void *iter_dev_ptr,
                                  void *cond_dev_ptr, void **loop_carried_descs,
                                  void **capture_descs);

// Fast-path: counted loop. Selected by the HipToLLVM lowering when the
// outlining pass proves cond_out == cond_in (SSA-equality, i.e. the body's
// returned cond passes through unchanged). Skips per-iter D2H cond sync,
// which is the dominant cost in the slow path.
//
// Equivalent to `for (i = 0; i < max_trip_count; ++i) body(i)` -- the
// cheapest possible CPU-driven loop. Inspired by counted-loop hoisting in
// classic optimizing compilers; no analogue in ORT CUDA EP or MIGraphX
// (both always read cond_out every iter, even for trivially-counted loops).
int hipdnn_ep_run_counted_loop(RuntimeState *state, HipdnnEpLoopBodyFn body_fn,
                               int64_t max_trip_count, bool cond_init,
                               int32_t num_loop_carried, int32_t num_captures,
                               void **loop_carried_descs, void **capture_descs);

// Slow-path: dynamic-cond loop. Reads cond_out each iter via D2H sync
// (matches the behavior of ORT CUDA EP `LoopImpl::Execute` and MIGraphX
// `run_loop`). Used when the body's returned cond is the result of a body-
// internal computation rather than a passthrough of cond_in.
int hipdnn_ep_run_loop(RuntimeState *state, HipdnnEpLoopBodyFn body_fn,
                       int64_t max_trip_count, bool cond_init,
                       int32_t num_loop_carried, int32_t num_captures,
                       void **loop_carried_descs, void **capture_descs);

// ONNX If driver. Dispatches to exactly one of the two branch trampolines
// based on `cond`. Each trampoline writes branch outputs into the shared
// `output_descs` buffers (DPS / out-param ABI).
typedef int (*HipdnnEpIfBranchFn)(RuntimeState *state, void **output_descs,
                                  void **capture_descs);

int hipdnn_ep_run_if(RuntimeState *state, bool cond, HipdnnEpIfBranchFn then_fn,
                     HipdnnEpIfBranchFn else_fn, int32_t num_outputs,
                     int32_t num_captures, void **output_descs,
                     void **capture_descs);

//===----------------------------------------------------------------------===//
// Low-Level HIP Wrappers
//===----------------------------------------------------------------------===//

// HIP memory allocation wrapper with error handling
int wrap_hipMalloc(void **ptr, int64_t size);

// HIP memory free wrapper with error handling
int wrap_hipFree(void *ptr);

// HIP memory copy host-to-device wrapper
int wrap_hipMemcpyH2D(void *dst, const void *src, int64_t size, void *stream);

// HIP memory copy device-to-host wrapper
int wrap_hipMemcpyD2H(void *dst, const void *src, int64_t size, void *stream);

// HIP stream synchronization wrapper
int wrap_hipStreamSynchronize(void *stream);

#ifdef __cplusplus
}
#endif

#endif // HIP_EP_RUNTIME_H
