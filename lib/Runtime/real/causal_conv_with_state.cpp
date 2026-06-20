/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "cache_utils.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// CausalConvWithState: Stateful Causal Depthwise Convolution
//
// Used by Gated DeltaNet (Qwen3.5) and Mamba (Jamba, FalconMamba).
// Replaces the 3-op pattern (Concat + Conv + Slice) with a single fused op.
//
// Algorithm (1D, ndim=1):
//   1. Build virtual input by prepending past_state to input:
//      virtual = [past_state | input]  shape: (batch, channels, k-1 + L)
//   2. Depthwise causal convolution over virtual input:
//      For each output position t in [0, L):
//        output[b,c,t] = sum_{j=0}^{k-1} weight[c,0,j] * virtual[b,c,t+j]
//      Plus optional bias: output[b,c,t] += bias[c]
//   3. Extract present_state = last (k-1) positions of virtual input:
//      present_state[b,c,:] = virtual[b,c, L : L+k-1]
//      (equivalently the last k-1 elements of the concatenation)
//   4. Optional SiLU activation: output = output * sigmoid(output)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Per-shape MIOpen descriptor + algorithm cache
//
// The compiled model has fully static shapes, so for a given layer the tuple
// (dt, B, C, virtual_len, seq_len, k, has_bias, has_act) is invariant for
// every Compute() call. Without this cache we would:
//   - create+destroy 4-6 MIOpen descriptors per call,
//   - call miopenInitConvolutionDescriptor + GroupCount per call,
//   - run miopenFindConvolutionForwardAlgorithm (which actually launches
//     candidate kernels and synchronizes) per call.
//
// All of that is amortized to once per shape per session here.
//===----------------------------------------------------------------------===//

namespace {

struct CausalConvKey {
  int64_t dt;          // miopenDataType_t value
  int64_t batch;       // B
  int64_t channels;    // C
  int64_t virtual_len; // state_len + seq_len
  int64_t seq_len;     // L
  int64_t kernel_size; // k
  int64_t has_bias;    // 0 / 1
  int64_t activation;  // 0 = none, 1 = SiLU

  bool operator==(const CausalConvKey &o) const {
    return dt == o.dt && batch == o.batch && channels == o.channels &&
           virtual_len == o.virtual_len && seq_len == o.seq_len &&
           kernel_size == o.kernel_size && has_bias == o.has_bias &&
           activation == o.activation;
  }
};

struct CausalConvKeyHash {
  size_t operator()(const CausalConvKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.dt);
    hash_combine_val(h, k.batch);
    hash_combine_val(h, k.channels);
    hash_combine_val(h, k.virtual_len);
    hash_combine_val(h, k.seq_len);
    hash_combine_val(h, k.kernel_size);
    hash_combine_val(h, k.has_bias);
    hash_combine_val(h, k.activation);
    return h;
  }
};

struct CausalConvCacheEntry {
  miopenTensorDescriptor_t inDesc = nullptr;
  miopenTensorDescriptor_t wDesc = nullptr;
  miopenTensorDescriptor_t outDesc = nullptr;
  miopenTensorDescriptor_t biasDesc = nullptr; // null if has_bias == 0
  miopenConvolutionDescriptor_t convDesc = nullptr;
  miopenActivationDescriptor_t actDesc = nullptr; // null if activation == 0
  miopenConvFwdAlgorithm_t algo;                  // valid iff algo_valid
  size_t algo_workspace_size = 0;                 // workspace bytes for algo
  bool algo_valid = false;
};

struct CausalConvCache {
  std::unordered_map<CausalConvKey, CausalConvCacheEntry, CausalConvKeyHash>
      entries;
  // Destroys every cached MIOpen descriptor/algo entry. Defined out-of-line
  // below (after destroyEntry). Runs when the owning op-state slot is torn
  // down (CausalConvState's deletor).
  ~CausalConvCache();
};

// Build the static (shape-only) descriptors. The convolution algorithm and its
// workspace size are filled in lazily on first use, once the input/output
// pointers are known (Find requires real device buffers).
miopenStatus_t buildEntryDescriptors(CausalConvCacheEntry &e,
                                     miopenDataType_t dt, int64_t B, int64_t C,
                                     int64_t virtual_len, int64_t seq_len,
                                     int64_t kernel_size, bool has_bias,
                                     int64_t activation) {
  miopenStatus_t st;
#define CC_TRY(call)                                                           \
  do {                                                                         \
    st = (call);                                                               \
    if (st != miopenStatusSuccess)                                             \
      return st;                                                               \
  } while (0)

  CC_TRY(miopenCreateTensorDescriptor(&e.inDesc));
  CC_TRY(miopenCreateTensorDescriptor(&e.wDesc));
  CC_TRY(miopenCreateTensorDescriptor(&e.outDesc));
  CC_TRY(miopenCreateConvolutionDescriptor(&e.convDesc));

  CC_TRY(miopenSet4dTensorDescriptor(e.inDesc, dt, static_cast<int>(B),
                                     static_cast<int>(C), 1,
                                     static_cast<int>(virtual_len)));
  CC_TRY(miopenSet4dTensorDescriptor(e.wDesc, dt, static_cast<int>(C), 1, 1,
                                     static_cast<int>(kernel_size)));
  CC_TRY(miopenSet4dTensorDescriptor(e.outDesc, dt, static_cast<int>(B),
                                     static_cast<int>(C), 1,
                                     static_cast<int>(seq_len)));

  CC_TRY(miopenInitConvolutionDescriptor(e.convDesc, miopenConvolution,
                                         /*pad_h=*/0, /*pad_w=*/0,
                                         /*stride_h=*/1, /*stride_w=*/1,
                                         /*dilation_h=*/1, /*dilation_w=*/1));
  CC_TRY(miopenSetConvolutionGroupCount(e.convDesc, static_cast<int>(C)));

  if (has_bias) {
    CC_TRY(miopenCreateTensorDescriptor(&e.biasDesc));
    CC_TRY(miopenSet4dTensorDescriptor(e.biasDesc, dt, 1, static_cast<int>(C),
                                       1, 1));
  }

  if (activation == 1) {
    CC_TRY(miopenCreateActivationDescriptor(&e.actDesc));
    CC_TRY(miopenSetActivationDescriptor(e.actDesc, miopenActivationLOGISTIC,
                                         0.0, 0.0, 0.0));
  }

#undef CC_TRY
  return miopenStatusSuccess;
}

void destroyEntry(CausalConvCacheEntry &e) {
  if (e.actDesc)
    miopenDestroyActivationDescriptor(e.actDesc);
  if (e.biasDesc)
    miopenDestroyTensorDescriptor(e.biasDesc);
  if (e.convDesc)
    miopenDestroyConvolutionDescriptor(e.convDesc);
  if (e.outDesc)
    miopenDestroyTensorDescriptor(e.outDesc);
  if (e.wDesc)
    miopenDestroyTensorDescriptor(e.wDesc);
  if (e.inDesc)
    miopenDestroyTensorDescriptor(e.inDesc);
  e = CausalConvCacheEntry{};
}

CausalConvCache::~CausalConvCache() {
  for (auto &kv : entries)
    destroyEntry(kv.second);
}

// Per-instance CausalConvWithState op-state (see op-state-slots-design.md):
// owns this instance's per-shape MIOpen descriptor + algorithm cache. Replaces
// the former shared RuntimeState::causal_conv_cache, so concurrent sessions no
// longer share it.
struct CausalConvState : OpStateT<CausalConvState> {
  CausalConvCache cache;
};

} // namespace

extern "C" int8_t
hipdnn_ep_op_state_construct_causal_conv_with_state(RuntimeState *state,
                                                    int32_t slot) {
  return CausalConvState::create(state, slot);
}

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
template <typename T> static inline T silu(T x) {
  return x / (static_cast<T>(1) + std::exp(-x));
}

int wrap_causal_conv_with_state(RuntimeState *state, int op_state_slot,
                                const void *input, const void *weight,
                                const void *bias, const void *past_state,
                                void *output, void *present_state,
                                int64_t batch_size, int64_t channels,
                                int64_t seq_len, int64_t kernel_size,
                                int64_t ndim, int64_t activation,
                                int64_t element_size_bytes) {
  // ---- Cheap, configuration-level validation FIRST. None of these touch the
  // device, so do them before any allocation or descriptor work to avoid
  // ad-hoc cleanup paths (and to keep the OP_PROFILE scope tight around the
  // actual GPU work).
  if (!state || !input || !weight || !output || !present_state) {
    fprintf(stderr, "wrap_causal_conv_with_state: null required argument\n");
    return -1;
  }

  if (ndim != 1) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: ndim=%lld not yet supported "
            "(only ndim=1)\n",
            (long long)ndim);
    return -1;
  }

  miopenDataType_t dt;
  if (element_size_bytes == 4)
    dt = miopenFloat;
  else if (element_size_bytes == 2)
    dt = miopenHalf;
  else {
    fprintf(stderr,
            "wrap_causal_conv_with_state: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_causal_conv_with_state: null stream\n");
    return -1;
  }

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_causal_conv_with_state: null MIOpen handle\n");
    return -1;
  }

  OP_PROFILE(
      "causal_conv",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lld,k=%lld", (long long)batch_size,
                 (long long)channels, (long long)seq_len,
                 (long long)kernel_size);
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG("[REAL] wrap_causal_conv_with_state: batch=%lld, "
                    "channels=%lld, seq_len=%lld, kernel=%lld, ndim=%lld, "
                    "activation=%lld, elem_size=%lld\n",
                    (long long)batch_size, (long long)channels,
                    (long long)seq_len, (long long)kernel_size, (long long)ndim,
                    (long long)activation, (long long)element_size_bytes);

  // ---- Fast path: single-step decode (seq_len==1, k<=8) -------------------
  // The MIOpen path requires building a "virtual" buffer of shape
  // (B, C, k-1+L) by concatenating past_state and input, then running a
  // depthwise conv on it. For decode (L=1) the concat alone is dominated by
  // the per-row hipMemcpy2DAsync overhead -- thousands of rows × ~k bytes
  // each is a degenerate DMA shape and runs ~7 ms on this iGPU instead of
  // microseconds (see commit history / profiling notes).
  //
  // The single-step custom kernel skips the virtual buffer, the conv API,
  // and the bias / activation chain entirely: one block-grid that reads
  // past_state and input directly, computes the dot product in registers,
  // applies optional SiLU, and writes output and present_state. Empirical
  // result: ~17 µs/call vs ~7 ms/call.
  //
  // Only enabled for shapes the kernel actually supports (k in [1,8],
  // activation in {0=none, 1=SiLU}, fp16/fp32). Anything outside falls
  // through to the MIOpen path below, which handles prefill and any odd
  // hybrid shapes correctly.
  if (seq_len == 1 && kernel_size >= 1 && kernel_size <= 8 &&
      (activation == 0 || activation == 1) &&
      (element_size_bytes == 2 || element_size_bytes == 4)) {
    int rc = hip_causal_conv_step_decode(
        stream, input, weight, bias, past_state, output, present_state,
        batch_size, channels, kernel_size, activation, element_size_bytes);
    if (rc != 0) {
      fprintf(stderr,
              "wrap_causal_conv_with_state: hip_causal_conv_step_decode "
              "failed (%d)\n",
              rc);
      return -1;
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_causal_conv_with_state: completed via fast decode\n");
    return 0;
  }

  // ---- Fast path: prefill (seq_len>1) via a single fused custom kernel ----
  // Replaces the MIOpen path (Find + 3 pitched D2D memcpys to build the
  // virtual buffer + conv + bias optensor + activation + mul) -- the single
  // largest text-prefill op -- with one launch. Same supported envelope as
  // the decode fast path (k in [1,8], activation none/SiLU, fp16/fp32);
  // anything else falls through to MIOpen below.
  if (seq_len > 1 && kernel_size >= 1 && kernel_size <= 8 &&
      (activation == 0 || activation == 1) &&
      (element_size_bytes == 2 || element_size_bytes == 4)) {
    int rc =
        hip_causal_conv_prefill(stream, input, weight, bias, past_state, output,
                                present_state, batch_size, channels, seq_len,
                                kernel_size, activation, element_size_bytes);
    if (rc != 0) {
      fprintf(stderr,
              "wrap_causal_conv_with_state: hip_causal_conv_prefill failed "
              "(%d)\n",
              rc);
      return -1;
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_causal_conv_with_state: completed via fast prefill\n");
    return 0;
  }

  const int64_t state_len = kernel_size - 1; // k-1
  const int64_t virtual_len = state_len + seq_len;
  const int64_t bc_rows = batch_size * channels;

  // ---- Look up (or lazily create) descriptors and algorithm for this shape.
  CausalConvKey key{static_cast<int64_t>(dt),
                    batch_size,
                    channels,
                    virtual_len,
                    seq_len,
                    kernel_size,
                    bias ? 1 : 0,
                    activation};

  CausalConvState *ccs = CausalConvState::get_slot(state, op_state_slot);
  if (!ccs) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: no CausalConvState at slot %d\n",
            op_state_slot);
    return -1;
  }
  auto *cache = &ccs->cache;
  CausalConvCacheEntry *entry = nullptr;
  {
    auto it = cache->entries.find(key);
    if (it == cache->entries.end()) {
      CausalConvCacheEntry fresh;
      miopenStatus_t st = buildEntryDescriptors(
          fresh, dt, batch_size, channels, virtual_len, seq_len, kernel_size,
          bias != nullptr, activation);
      if (st != miopenStatusSuccess) {
        fprintf(stderr,
                "wrap_causal_conv_with_state: MIOpen descriptor "
                "creation failed (%d)\n",
                static_cast<int>(st));
        destroyEntry(fresh);
        return -1;
      }
      entry = &cache->entries.emplace(key, std::move(fresh)).first->second;
    } else {
      entry = &it->second;
    }
  }

  // ---- Pack all per-call temporary buffers into the shared workspace, so
  // there is no per-call hipMalloc/hipFree (those serialize with the device).
  //
  // Layout:
  //   [virtual_buf : virtual_size][sigmoid_buf : sigmoid_size]
  // sigmoid_buf only exists when activation == 1 (SiLU) and is the same size
  // as the output tensor.
  const size_t virtual_size =
      static_cast<size_t>(bc_rows * virtual_len * element_size_bytes);
  const size_t sigmoid_size =
      (activation == 1)
          ? static_cast<size_t>(bc_rows * seq_len * element_size_bytes)
          : 0;

  // Convolution workspace size: on a cache hit we already know the exact size
  // the chosen algorithm needs (perfResult.memory from the original Find).
  // On a cache miss we ask MIOpen for an upper bound so Find has space to
  // benchmark candidate algorithms; it then reports the actual algo memory
  // back to us, which we cache for subsequent calls.
  size_t conv_workspace_size;
  if (entry->algo_valid) {
    conv_workspace_size = entry->algo_workspace_size;
  } else {
    miopenStatus_t st = miopenConvolutionForwardGetWorkSpaceSize(
        handle, entry->wDesc, entry->inDesc, entry->convDesc, entry->outDesc,
        &conv_workspace_size);
    if (st != miopenStatusSuccess) {
      fprintf(stderr,
              "wrap_causal_conv_with_state: GetWorkSpaceSize failed (%d)\n",
              static_cast<int>(st));
      return -1;
    }
  }

  const size_t total_ws = virtual_size + sigmoid_size + conv_workspace_size;
  if (total_ws > 0 && hipdnn_ep_state_ensure_workspace(state, total_ws) != 0) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: ensure_workspace(%zu) failed\n",
            total_ws);
    return -1;
  }
  char *ws_base = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
  void *virtual_buf = ws_base;
  void *sigmoid_buf =
      (activation == 1) ? static_cast<void *>(ws_base + virtual_size) : nullptr;
  void *conv_workspace =
      (conv_workspace_size > 0)
          ? static_cast<void *>(ws_base + virtual_size + sigmoid_size)
          : nullptr;

  // ---- Build the virtual buffer using *pitched* 2D copies, replacing the
  // O(B*C) host launch loops. Each plane (b,c) occupies one row of the
  // virtual buffer, of length virtual_len; we copy past_state (or zeros) into
  // the first state_len columns and input into the trailing seq_len columns.
  //
  // height = B * C   (one row per (batch, channel) plane)
  // dpitch = virtual_len * elem_size  (stride between rows in virtual_buf)
  hipError_t herr;
  const size_t es = static_cast<size_t>(element_size_bytes);
  const size_t dpitch = static_cast<size_t>(virtual_len) * es;
  const size_t height = static_cast<size_t>(bc_rows);

  // First state_len columns of every row: past_state, or zeros if absent.
  if (past_state) {
    herr = hipMemcpy2DAsync(virtual_buf, dpitch, past_state,
                            static_cast<size_t>(state_len) * es,
                            static_cast<size_t>(state_len) * es, height,
                            hipMemcpyDeviceToDevice, stream);
    if (herr != hipSuccess) {
      fprintf(stderr,
              "wrap_causal_conv_with_state: hipMemcpy2DAsync(past_state) "
              "failed (%d)\n",
              static_cast<int>(herr));
      return -1;
    }
  } else {
    // hipMemset2DAsync zeros a (width × height) sub-rectangle of a pitched
    // buffer, leaving the trailing seq_len columns untouched (those are
    // overwritten by the input copy below).
    herr =
        hipMemset2DAsync(virtual_buf, dpitch, 0,
                         static_cast<size_t>(state_len) * es, height, stream);
    if (herr != hipSuccess) {
      fprintf(stderr,
              "wrap_causal_conv_with_state: hipMemset2DAsync(past_state) "
              "failed (%d)\n",
              static_cast<int>(herr));
      return -1;
    }
  }

  // Trailing seq_len columns of every row: input.
  herr = hipMemcpy2DAsync(static_cast<char *>(virtual_buf) +
                              static_cast<size_t>(state_len) * es,
                          dpitch, input, static_cast<size_t>(seq_len) * es,
                          static_cast<size_t>(seq_len) * es, height,
                          hipMemcpyDeviceToDevice, stream);
  if (herr != hipSuccess) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: hipMemcpy2DAsync(input) "
            "failed (%d)\n",
            static_cast<int>(herr));
    return -1;
  }

  // Extract present_state: the last state_len columns of every row of
  // virtual_buf (positions [seq_len, seq_len+state_len)). Single 2D copy.
  herr = hipMemcpy2DAsync(present_state, static_cast<size_t>(state_len) * es,
                          static_cast<char *>(virtual_buf) +
                              static_cast<size_t>(seq_len) * es,
                          dpitch, static_cast<size_t>(state_len) * es, height,
                          hipMemcpyDeviceToDevice, stream);
  if (herr != hipSuccess) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: hipMemcpy2DAsync(present_state) "
            "failed (%d)\n",
            static_cast<int>(herr));
    return -1;
  }

  // ---- Convolution. On the first call for this shape, run Find once and
  // remember the algorithm + its workspace size in the cache; subsequent
  // calls go straight to miopenConvolutionForward with the cached algorithm.
  miopenStatus_t mst;

#define CAUSAL_MIOPEN_CHECK(call)                                              \
  do {                                                                         \
    mst = (call);                                                              \
    if (mst != miopenStatusSuccess) {                                          \
      fprintf(stderr,                                                          \
              "wrap_causal_conv_with_state: MIOpen error %d at "               \
              "%s:%d\n",                                                       \
              mst, __FILE__, __LINE__);                                        \
      return -1;                                                               \
    }                                                                          \
  } while (0)

  if (!entry->algo_valid) {
    miopenConvAlgoPerf_t perfResult;
    int returnedAlgoCount = 0;
    CAUSAL_MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        handle, entry->inDesc, virtual_buf, entry->wDesc, weight,
        entry->convDesc, entry->outDesc, output,
        /*requestAlgoCount=*/1, &returnedAlgoCount, &perfResult, conv_workspace,
        conv_workspace_size, /*exhaustiveSearch=*/false));
    if (returnedAlgoCount <= 0) {
      fprintf(stderr,
              "wrap_causal_conv_with_state: Find returned no algorithms\n");
      return -1;
    }
    entry->algo = perfResult.fwd_algo;
    // perfResult.memory is the actual workspace the chosen algo needs (often
    // smaller than the upper bound from GetWorkSpaceSize).
    entry->algo_workspace_size = perfResult.memory;
    entry->algo_valid = true;
    // Find already executed the chosen algorithm with the live tensors as a
    // benchmarking pass, but its outputs are not guaranteed to be the final
    // result we want -- re-run the forward once below using the cached algo.
  }

  {
    float alpha = 1.0f, beta = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenConvolutionForward(
        handle, &alpha, entry->inDesc, virtual_buf, entry->wDesc, weight,
        entry->convDesc, entry->algo, &beta, entry->outDesc, output,
        conv_workspace, entry->algo_workspace_size));
  }

  if (bias) {
    float alpha_bias = 1.0f, beta_bias = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha_bias,
                                       entry->outDesc, output, &alpha_bias,
                                       entry->biasDesc, bias, &beta_bias,
                                       entry->outDesc, output));
  }

  // SiLU(x) = x * sigmoid(x). MIOpen has no fused SILU in this build, so we
  // compute sigmoid into the workspace-resident sigmoid_buf and then multiply
  // in place. (The temp buffer is reused across calls -- no hipMalloc.)
  if (activation == 1) {
    float alpha_act = 1.0f, beta_act = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenActivationForward(
        handle, entry->actDesc, &alpha_act, entry->outDesc, output, &beta_act,
        entry->outDesc, sigmoid_buf));

    float alpha_mul = 1.0f, beta_mul = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpMul, &alpha_mul,
                                       entry->outDesc, output, &alpha_mul,
                                       entry->outDesc, sigmoid_buf, &beta_mul,
                                       entry->outDesc, output));
  }

#undef CAUSAL_MIOPEN_CHECK

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_causal_conv_with_state: completed successfully\n");
  return 0;
}
