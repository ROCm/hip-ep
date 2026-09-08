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

// Internal runtime state structure
// This struct is opaque to generated code (passed as void*)
struct RuntimeState {
  hipStream_t stream;
  hipblasLtHandle_t hipblas_handle;

  // Single allocation holding all constants as one blob.
  // gpu_constants[i] points into gpu_constants_blob at the offset stored in
  // ConstantInfo, so only one allocation/copy is needed at init time.
  // Always hipMalloc (VRAM) for both dGPU and iGPU.
  void *gpu_constants_blob;
  void **gpu_constants;
  size_t num_constants;

  // Memory pooling support — multi-domain.
  //
  // hip-pool-allocs partitions a function's pooled allocs into independent
  // dominance domains; each domain owns one contiguous GPU pool that grows on
  // demand. Domain 0 carries the legacy single-pool semantics (eagerly sized
  // by hipdnn_ep_pool_init using static offsets) so single-domain models are
  // bit-identical to the pre-multi-domain runtime. Domains 1..N start empty
  // and grow lazily on the first hipdnn_ep_get_pool_base(state, domain_id, ...)
  // call.
  //
  // The per-domain pool arrays are themselves grown on demand: there is no
  // compile-time cap on the domain count. pool_base/pool_size are heap arrays
  // of num_pool_domains entries, reallocated (zero-filling new slots) the first
  // time a higher domain_id is observed — by hipdnn_ep_get_pool_base for the
  // lazy domains and by hipdnn_ep_pool_init for domain 0. Every domain_id is
  // first seen on the cold first inference, so num_pool_domains stabilises
  // after that and no further array realloc happens at steady state — mirroring
  // the grow-on-demand contract of the individual pools. realloc-move is safe
  // because nothing caches &pool_base[i] across calls; pool_base[domain_id] is
  // re-derived from state on every access.
  int num_pool_domains;   // Number of slots currently allocated in the arrays
  void **pool_base;       // [num_pool_domains] per-domain GPU pool base ptrs
  size_t *pool_size;      // [num_pool_domains] per-domain pool size in bytes
  size_t *buffer_offsets; // Offsets for static buffers in domain 0
  size_t num_buffers;     // Static buffer count in domain 0

  // Shared workspace for operator temp buffers (MatMul GEMM ws, GQA pipeline).
  // Lazily grown via hipdnn_ep_state_ensure_workspace(); never shrinks.
  void *workspace;
  size_t workspace_size;

  // Host-mapped scratch buffer for tiny host-fed scalars routed away from the
  // GPU pool by hip-materialize-host-scalars.
  // hipHostMalloc(hipHostMallocMapped): host-writable AND GPU-readable.
  // Grow-on-demand via hipdnn_ep_get_host_scratch_base(); never shrinks.
  // hipHostFree'd in cleanup. Why: on some targets the regular GPU pool is
  // real device memory; host stores into it SEGV. Other targets silently
  // worked because hipMalloc returned UMA-mapped host memory there, masking
  // the bug.
  void *host_scratch_base;
  size_t host_scratch_size;

  // Output allocator installed by the EP before inference_compute via
  // hipdnn_ep_set_output_allocator. hipdnn_ep_alloc_output forwards to
  // allocate(self, ...). Borrowed: `self` is EP-owned, never freed here.
  // allocate == nullptr means no allocator has been installed yet;
  // zero-initialized in initialize_state_handles.
  hipdnn_output_allocator_t output_allocator;

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

  // Per-session scratch for wrap_qmoe_amd (com.amd QMoE / LatentMoE)
  // transient buffers. Deliberately a SEPARATE field from
  // qmoe_scratch above -- the two ops are independent pipelines (see
  // lib/Runtime/real/qmoe_amd.cpp) and must not share a buffer, so growing
  // one never invalidates offsets computed for the other. Same grow-on-
  // demand, never-shrink policy as qmoe_scratch; freed in
  // hipdnn_ep_state_cleanup.
  void *qmoe_amd_scratch;
  size_t qmoe_amd_scratch_size;
  void *qmoe_amd_host_scratch; // pinned host mirror for D2H of expert counts
  size_t qmoe_amd_host_scratch_size;

  // Per-session convolution workspace. Currently allocated by nobody: both
  // convolution directions now run on in-tree kernels (hip_conv,
  // hip_conv_transpose) that need no workspace at all. Kept only because the
  // accessors are
  // part of the runtime's exported surface. Same grow-on-demand policy as
  // qmoe_scratch above if it is ever wired up again: lazily allocated on first
  // use, never shrinks, freed in hipdnn_ep_state_cleanup.
  void *conv_scratch;
  size_t conv_scratch_size;

  // Per-session scratch for the W4A8 dp4a matmul_nbits decode path
  // (hip_matmul_nbits_dp4a). One contiguous device buffer holding the
  // per-token quantized activation (int8, K bytes, nibble-deinterleaved) plus
  // the per-group activation scales (float, ceil(K/block_size)). Same
  // grow-on-demand / never-shrink policy as conv_scratch; lazily allocated on
  // first dp4a call, freed in hipdnn_ep_state_cleanup. Single-buffer reuse is
  // safe because the HIP stream is serialised (the next matmul_nbits only
  // launches after the previous dp4a gemv has consumed the quantized row).
  void *matmul_dp4a_scratch;
  size_t matmul_dp4a_scratch_size;

  // Per-session scratch for the linear-attention chunk-parallel gated_delta
  // prefill (hip_linear_attention_prefill_chunked). One contiguous device
  // buffer holding the per-(head,chunk) Uloc/W/rlast/alast tiles and the
  // chunk-start states for one window of the sequence, plus the cross-window
  // carry of the recurrent state. Sized by the window rather than by seq_len,
  // so it stops growing once the sequence exceeds one window. Same
  // grow-on-demand / never-shrink policy as conv_scratch; lazily allocated on
  // first prefill, freed in hipdnn_ep_state_cleanup. Single-buffer reuse is
  // safe because the HIP stream is serialised (the three prefill passes consume
  // it before the next linear-attention layer launches). Replaces a
  // process-static buffer so the footprint is bounded to the session and
  // released on teardown.
  void *la_scratch;
  size_t la_scratch_size;

  // NOTE: the GQA GEMM descriptor cache (GqaGemmCache) formerly lived here as
  // gqa_gemm_cache. It is now per-op-instance: each gqa instance owns one in
  // its GqaState op-state slot (see op_states below and
  // docs/design/op-state-slots-design.md), so concurrent sessions (and
  // distinct GQA layers) no longer share one descriptor map.

  // NOTE: the MultiHeadAttention GEMM descriptor cache (MhaGemmCache) formerly
  // lived here as mha_gemm_cache. It is now per-op-instance: each
  // multi_head_attention instance owns one in its MhaState op-state slot (see
  // op_states below and docs/design/op-state-slots-design.md).

  // NOTE: the CausalConvWithState descriptor + algorithm cache
  // (CausalConvCache) formerly lived here as causal_conv_cache, then moved to
  // a per-op-instance CausalConvState op-state slot. It no longer exists at
  // all: the op runs entirely on custom kernels. See
  // docs/design/op-state-slots-design.md.

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

  // Session-owned GQA offline performance policy (GqaAutotunePolicy*).
  // Loaded once from gqa_autotune.fb through the EP FileSystem. The concrete
  // type lives in Kernels/hip/autotune/gqa/gqa_autotune.cpp to keep this
  // ABI-facing struct opaque.
  void *gqa_autotune_policy;

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

  // Per-Compute() cache for seqlens_k_val (decode hot path).
  //
  // Decode runs 32 GQA layers per token, all reading the same seqlens_k
  // from device memory. The decomposed-path readback in gqa.cpp issues a
  // hipMemcpyAsync(D2H) + hipStreamSynchronize per layer (31 redundant
  // pipeline stalls, tens of ms per token on long-context decode). The cache
  // is on by default
  // (HIPDNN_EP_GQA_CACHE_SEQLENS=1, set to 0 to disable): the first GQA
  // in a forward pass populates the cache and the remaining 31 layers
  // reuse it.
  //
  // Invalidated by hipdnn_ep_runtime_begin_compute() at the start of each
  // Compute(), called from the EP-side MlirCustomOp::Compute() entry.
  // If the symbol is not exported (older per-model bitcode), invalidation
  // does not happen and the cache is unsafe -- the EP logs a warning at
  // session creation and the user must set HIPDNN_EP_GQA_CACHE_SEQLENS=0.
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
