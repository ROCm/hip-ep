/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- runtime_state_internal.h - RuntimeState Internal Definition -------===//
//
// INTERNAL HEADER - Not part of public API
//
// This header defines the internal structure of RuntimeState.
// Only runtime implementation files should include this header.
//
// Public interface uses opaque pointer (see hipdnn_ep_runtime.h).
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
#define HIPDNN_EP_RUNTIME_STATE_INTERNAL_H

#include "runtime_types.h"

// Forward declaration of the op-module registry. Defined in
// module_registry.cpp; only accessed through the hipdnn_ep:: free functions.
// Lives at the bottom of the struct so it sits next to the other "opaque
// per-op state" pointers and the existing flat fields keep their offsets
// (no ABI churn for already-compiled model.dlls).
namespace hipdnn_ep {
struct ModuleRegistry;
} // namespace hipdnn_ep

// Internal runtime state structure
// This struct is opaque to generated code (passed as void*)
struct RuntimeState {
  hipStream_t stream;
  miopenHandle_t miopen_handle;
  hipblasLtHandle_t hipblas_handle;

  // Single allocation holding all constants as one blob.
  // gpu_constants[i] points into gpu_constants_blob at the offset stored in
  // ConstantInfo, so only one allocation/copy is needed at init time.
  // Always hipMalloc (VRAM) for both dGPU and iGPU.
  void *gpu_constants_blob;
  void **gpu_constants;
  size_t num_constants;

  // OGA pipeline shared constants cache: prefill and decode models share
  // the same constants blob via process-wide named shared memory + atomic
  // ref count. Set by try_attach_shared_constants when reusing another
  // model's blob; cleanup decrements ref_count and only the last
  // reference frees the GPU memory.
  bool constants_is_shared;
  void *shared_constants_mapping; // Win32 file mapping HANDLE
  void *shared_constants_view; // MapViewOfFile pointer (SharedConstantsMeta*)

  // Memory pooling support
  void *pool_base;        // Single large memory pool
  size_t pool_size;       // Total pool size in bytes
  size_t *buffer_offsets; // Offset for each buffer in the pool
  size_t num_buffers;     // Number of buffers in the pool

  // Shared workspace for operator temp buffers (MatMul GEMM ws, GQA pipeline).
  // Lazily grown via hipdnn_ep_state_ensure_workspace(); never shrinks.
  void *workspace;
  size_t workspace_size;

  // (Removed: qmoe_scratch, qmoe_host_scratch and their *_size fields.
  //  Both now live in the QmoeState op-module (lib/Runtime/real/qmoe.cpp)
  //  built on top of hipdnn_ep::GrowableDeviceBuffer /
  //  GrowablePinnedBuffer in growable_buffer.h. Allocated lazily on first
  //  wrap_qmoe call. The C-ABI getters / ensure helpers
  //  (hipdnn_ep_state_*_qmoe_*) are preserved in hipdnn_ep_runtime.h and
  //  now delegate through the QmoeState slot, so qmoe runtime bitcode is
  //  unaffected.)

  // (Removed: gqa_gemm_cache, causal_conv_cache, zp_unpack_cache.
  //  All three have moved into typed op-modules:
  //    - GqaGemmState     (lib/Runtime/real/gqa.cpp)
  //    - CausalConvState  (lib/Runtime/real/causal_conv_with_state.cpp)
  //    - ZpUnpackState    (lib/Runtime/real/matmul_nbits.cpp)
  //  Slots are allocated lazily inside the ModuleRegistry below.)

  // Per-operator profiling state (OpProfileState*, gated on HIPDNN_EP_PERF).
  // Allocated in state_init, freed in state_cleanup.
  void *op_profile;

  // Device-side error flag used by kernels to report runtime-invalid inputs.
  // 0 = no error, non-zero = error code (currently -1).
  int *device_error_flag;

  // hipDNN graph execution support.
  // Set by EP via hipdnn_graph_runtime_attach() after inference_init().
  // hipdnn_handle: hipdnnHandle_t cast to void* (owned by EP, not cleaned up
  // here) hipdnn_graph_registry: opaque GraphRegistry* (owned by EP, not
  // cleaned up here)
  void *hipdnn_handle;
  void *hipdnn_graph_registry;

  // (Removed: seqlens_k_cached_*. The per-Compute() seqlens_k cache now
  //  lives in the GqaSeqlensCache op-module (lib/Runtime/real/gqa.cpp).
  //  Its begin_compute() hook fires from hipdnn_ep_runtime_begin_compute
  //  through the module registry's begin_compute fan-out, so the
  //  invalidation contract documented above still holds end-to-end.)

  // Op-module registry. New ops add per-session state by registering through
  // module_registry.h's HIPDNN_OP_MODULE macro instead of growing this
  // struct. Created in initialize_state_handles; destroyed (along with every
  // populated slot) in hipdnn_ep_state_cleanup. The per-Compute()
  // invalidation iteration is also driven through here (see
  // hipdnn_ep_runtime_begin_compute). Existing flat fields above stay put
  // for now; they migrate to dedicated modules in later stages of the plan.
  hipdnn_ep::ModuleRegistry *modules;
};

#endif // HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
