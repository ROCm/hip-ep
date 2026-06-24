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
// For hipdnn_output_allocator_t (the output_allocator field below). Acyclic:
// hipdnn_ep_runtime.h only forward-declares RuntimeState; it does not include
// this internal header.
#include "hipdnn_ep_runtime.h"
// For `struct OpState` (the op_states array element type below). op_state.h is
// layout-agnostic (it only forward-declares RuntimeState), so including it here
// is acyclic.
#include "op_state.h"
// Unified Memory Manager (Phase 1). All session-scoped GPU/host buffers
// (pool domains, workspace, host-scalar scratch, qmoe host scratch) are now
// owned by MemoryManager and accessed through it. seqlens_k cache also lives
// in MM so begin_compute() is the single invalidation point.
#include "mm/memory_manager.h"

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

  // -------------------------------------------------------------------------
  // Unified Memory Manager (Phase 1+).
  //
  // All session-scoped memory — pool domains, shared workspace, host-scalar
  // scratch, qmoe host scratch — is now owned by `mm`. The extern C API
  // functions in hipdnn_ep_runtime.h remain unchanged; their implementations
  // in hipdnn_ep_runtime_state.cpp delegate to mm->*.
  //
  // The legacy flat fields below (pool_base, pool_size, workspace, …) are kept
  // during the Phase-1 transition period so that the EXISTING EXTERN C ABI
  // surface compiles unchanged. They are effectively aliases — the actual data
  // lives in mm, and these fields are unused once mm is created. They will be
  // removed in Phase 2 once all callers are proven to go through mm.
  // -------------------------------------------------------------------------

  MemoryManager *mm; // owns all session-scoped memory; created in init

  // LEGACY: kept for ABI transition. Do not use directly — go through mm.
  int num_pool_domains;   // mirrors mm->num_pool_domains()
  void **pool_base;       // unused after Phase 1 (mm owns the pools)
  size_t *pool_size;      // unused after Phase 1
  size_t *buffer_offsets; // unused after Phase 1 (mm->buffer_offsets_)
  size_t num_buffers;     // unused after Phase 1 (mm->num_buffers_)

  // LEGACY: kept for ABI transition. Do not use directly — go through mm.
  void *workspace;
  size_t workspace_size;

  // LEGACY: kept for ABI transition. Do not use directly — go through mm.
  void *host_scratch_base;
  size_t host_scratch_size;

  // Output allocator installed by the EP before inference_compute via
  // hipdnn_ep_set_output_allocator. hipdnn_ep_alloc_output forwards to
  // allocate(self, ...). Borrowed: `self` is EP-owned, never freed here.
  // allocate == nullptr means no allocator is installed (the classic pipeline
  // never calls alloc_output); zero-initialized in initialize_state_handles.
  hipdnn_output_allocator_t output_allocator;

  // LEGACY: kept for ABI transition. Do not use directly — go through mm.
  // Per-session scratch buffer for wrap_qmoe transient device buffers
  // (expert_indices, expert_weights, gather_buf, fc1_buf, act_buf, fc2_buf,
  // token_ids, token_wts -- 8 sub-buffers laid out at fixed offsets).
  //
  // Why this exists: pre-cache wrap_qmoe issued 8 hipMalloc + 8 hipFree per
  // call, every layer, every inference. On 24-layer gpt-oss-20b that's 192
  // mallocs + 192 frees per token; HIP's hipMalloc takes ~50 us each on
  // Windows, so the storm cost ~10-12 ms/token and bottlenecked decode TPS to
  // roughly half the Vulkan baseline on the same hardware.
  //
  // Layout policy: one contiguous buffer sized to fit ALL sub-buffers for the
  // largest (num_tokens, hidden, inter, k, num_experts, elem) shape ever seen
  // by this session. Sub-buffer offsets recomputed per-call (cheap arithmetic);
  // the buffer itself grows on demand via hipdnn_ep_state_ensure_qmoe_scratch
  // and never shrinks (mirrors the `workspace` field's policy). Shared by all
  // qmoe instances in the session: safe because the HIP stream is serialised,
  // so the next qmoe launches only after the previous one's kernels finish.
  //
  // Pinned host mirror is needed for the 24-bytes-per-layer D2H readback of
  // expert routing decisions (still required at decode pre-Phase-2). hipHost-
  // Malloc'd once with hipHostMallocDefault; reused across calls without sync.
  void *qmoe_scratch;
  size_t qmoe_scratch_size;
  void *qmoe_host_scratch; // pinned host mirror for D2H of expert idx/weights
  size_t qmoe_host_scratch_size;

  // Per-session scratch buffer for the MIOpen convolution workspace
  // (wrap_miopenConvolutionForward, both 2D and the H=1 1D conv path).
  //
  // The MIOpen forward-convolution Find API selects an algorithm whose
  // workspace requirement is shape-dependent (winograd/gemm/etc). Whisper's
  // encoder front-end runs the same two Conv shapes every inference
  // (Cin=128/Cout=1280 K=3 s=1, Cin=1280/Cout=1280 K=3 s=2), so a per-call
  // hipMalloc/hipFree of the workspace would be wasted work after the
  // first call. Same grow-on-demand policy as qmoe_scratch above: lazily
  // allocated on first use, never shrinks, freed in
  // hipdnn_ep_state_cleanup. Single-buffer reuse is safe because the HIP
  // stream is serialised -- the next conv launches only after the previous
  // miopenConvolutionForward + bias add have consumed the workspace.
  void *conv_scratch;
  size_t conv_scratch_size;

  // NOTE: the GQA GEMM descriptor cache (GqaGemmCache) formerly lived here as
  // gqa_gemm_cache. It is now per-op-instance: each gqa instance owns one in
  // its GqaState op-state slot (see op_states below and
  // docs/design/op-state-slots-design.md), so concurrent sessions (and
  // distinct GQA layers) no longer share one descriptor map.

  // NOTE: the MultiHeadAttention GEMM descriptor cache (MhaGemmCache) formerly
  // lived here as mha_gemm_cache. It is now per-op-instance: each
  // multi_head_attention instance owns one in its MhaState op-state slot (see
  // op_states below and docs/design/op-state-slots-design.md).

  // NOTE: the CausalConvWithState MIOpen descriptor + algorithm cache
  // (CausalConvCache) formerly lived here as causal_conv_cache. It is now
  // per-op-instance: each causal_conv_with_state instance owns one in its
  // CausalConvState op-state slot (see op_states below and
  // docs/design/op-state-slots-design.md), so concurrent instances no longer
  // share one descriptor cache.

  // Asym zero_points unpack cache (ZpUnpackCache*) used by wrap_qmoe.
  //
  // The asym AWQ path stores zero_points as packed nibbles [N, ceil(K/bs/2)].
  // Two unpacked layouts are needed (u8 [N, K/bs] for GEMV/naive, fp16 for
  // WMMA/col-major GEMV M>1), and naively the unpack kernel is launched on
  // every call. Since zero_points points into the model constants blob (stable
  // for the session lifetime), we cache the unpacked buffer per input pointer.
  // Lazily created on first asym call. Freed in hipdnn_ep_state_cleanup via
  // hipdnn_ep_zp_unpack_cache_destroy.
  //
  // NOTE: matmul_nbits no longer uses this shared cache -- it owns a
  // per-instance ZpUnpackCache in its MatmulNbitsState op-state slot. This
  // field is therefore qmoe-owned.
  void *zp_unpack_cache;

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

  // Per-Compute() seqlens_k cache — now lives in RuntimeState::mm.
  //
  // These three fields are kept here as ALIASES so that the existing code in
  // gqa.cpp (which reads/writes state->seqlens_k_cached_*) continues to
  // compile unchanged during Phase 1. They are populated from / flushed to
  // mm->seqlens_k_* by hipdnn_ep_runtime_begin_compute(). In Phase 2 the
  // gqa.cpp accessors will be redirected to mm directly and these removed.
  //
  // Invalidated by hipdnn_ep_runtime_begin_compute() via mm->begin_compute().
  bool seqlens_k_cached_valid;
  int32_t seqlens_k_cached_val;
  const void *seqlens_k_cached_ptr;

  // ONNX Loop driver state. Lazily allocated by hipdnn_ep_run_counted_loop /
  // hipdnn_ep_run_loop on first call; freed in hipdnn_ep_state_cleanup.
  //
  // iter -- stream-ordered per-iter update model:
  //   * `loop_iter_cpu_buf` is a pinned host array of int64 indexed by iter
  //     number (capacity grows on demand to max trip count seen so far).
  //     New tail entries are initialized to their own index on grow, then
  //     never written again -- the values 0,1,2,... are static.
  //   * `loop_iter_dev` is a real device buffer (hipMalloc) of a single
  //     int64; the body's `iter` memref descriptor points here.
  //   * Each iter does `hipMemcpyAsync(loop_iter_dev, &cpu_buf[i], 8, H2D,
  //     stream)` immediately before launching the body. Both ops are on
  //     the same stream, so the body kernel for iter i sees the value 'i'
  //     placed by the matching memcpy. Stream-ordered, no per-iter sync
  //     on the CPU side, no host-store-vs-kernel-launch race.
  //
  // The previous design used hipHostMalloc(hipHostMallocMapped) so a plain
  // host store + atomic_thread_fence(release) could "publish" the iter value
  // to the GPU. That works only if you also flush the entire pipeline
  // before reusing the buffer for the next iter -- HIP does NOT order plain
  // host stores against later stream submissions, so without a sync the
  // GPU sees whatever value is in the mapped page when the launched kernel
  // actually runs (typically the last host store = M-1 for all iters).
  // Switching to a stream-ordered hipMemcpyAsync from a per-iter slot
  // restores correctness with negligible perf cost (~1-2 µs / iter
  // memcpy enqueue, vs ~50-200 µs / hipStreamSynchronize).
  //
  // cond -- still host-mapped:
  //   * Host writes cond_init exactly once before the loop starts; the GPU
  //     reads cond_in on first iter, writes cond_out at end of each iter.
  //     The dynamic path then records `loop_event` on the stream and
  //     hipEventSynchronizes before reading loop_cond_host -- the event
  //     sync serialises kernel completion vs the next host read, so there
  //     is no iter-update race for cond.
  //
  // `loop_event` is reused across iters (hipEventCreateWithFlags +
  // hipEventDisableTiming -- we don't need timestamps, just synchronization)
  // to avoid hipEventCreate/Destroy per iter.
  //
  // Sharing one device iter slot per state is safe for non-nested Loops
  // only -- nested Loops would race on the inner driver overwriting the
  // outer's iter while the outer body still expects to read its own iter
  // after the inner returns. OnnxLoopOutlinePass walks each onnx.Loop
  // body for an inner onnx.Loop and emits a two-location error before
  // outlining runs, so any nested case is rejected at compile time.
  // A future P-extension can replace the single buffer with a small
  // per-depth stack and lift the constraint.
  void *loop_iter_cpu_buf; // hipHostMalloc(default)-allocated, int64[capacity]
  size_t loop_iter_capacity; // current size of loop_iter_cpu_buf (in int64s)
  void *loop_iter_dev;       // hipMalloc'd, sizeof(int64), passed to body
  void *loop_cond_host;      // hipHostMalloc-allocated, host-side view (int8*)
  void
      *loop_cond_dev; // hipHostGetDevicePointer(loop_cond_host), passed to body
  void *loop_event;   // hipEvent_t cast to void*; reused for cond-readback sync

  // Per-op state slots (see docs/design/op-state-slots-design.md). One entry
  // per stateful op instance, sized by --assign-op-state-slots and constructed
  // by the generated @hipdnn_ep_op_states_init_fn during inference_init. Each
  // entry's concrete type derives from OpState; cleanup walks the array calling
  // each object's `deletor`, so teardown needs no per-type knowledge. Managed
  // by hipdnn_ep_op_states_alloc / _set / _get in op_state.cpp.
  OpState **op_states;
  int num_op_states;
};

#endif // HIPDNN_EP_RUNTIME_STATE_INTERNAL_H
