/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <string>

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
//
// The custom kernels never materialize that virtual buffer -- they read
// past_state and input directly and fuse steps 2-4 into a single launch -- so
// this wrapper is only validation and a two-way dispatch on sequence length.
//
// Kernel widths past the templated kernels' k=8 are carried by a dynamic-K
// path, so there is no per-shape cache, no scratch workspace and no state to
// hold on to.
//===----------------------------------------------------------------------===//

int wrap_causal_conv_with_state(RuntimeState *state, const void *input,
                                const void *weight, const void *bias,
                                const void *past_state, void *output,
                                void *present_state, int64_t batch_size,
                                int64_t channels, int64_t seq_len,
                                int64_t kernel_size, int64_t ndim,
                                int64_t activation, int64_t element_size_bytes,
                                int64_t channels_last) {
  // ---- Cheap, configuration-level validation FIRST. None of these touch the
  // device, so do them before any device work to keep the OP_PROFILE scope
  // tight around the actual GPU work.
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

  if (element_size_bytes != 2 && element_size_bytes != 4) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: unsupported element_size %lld "
            "(expect 2 for fp16 or 4 for fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  if (seq_len < 1) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: seq_len=%lld must be at least 1\n",
            (long long)seq_len);
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_causal_conv_with_state: null stream\n");
    return -1;
  }

  OP_PROFILE(
      "causal_conv",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lldx%lld,k=%lld%s", (long long)batch_size,
                 (long long)channels, (long long)seq_len,
                 (long long)kernel_size, channels_last ? ",nlc" : "");
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG("[REAL] wrap_causal_conv_with_state: batch=%lld, "
                    "channels=%lld, seq_len=%lld, kernel=%lld, ndim=%lld, "
                    "activation=%lld, elem_size=%lld, channels_last=%lld\n",
                    (long long)batch_size, (long long)channels,
                    (long long)seq_len, (long long)kernel_size, (long long)ndim,
                    (long long)activation, (long long)element_size_bytes,
                    (long long)channels_last);

  // ---- Single-step decode (seq_len == 1) ---------------------------------
  // The decode kernel skips the virtual buffer and the bias / activation chain
  // entirely: one grid that reads past_state and input directly, computes the
  // dot product in registers, applies optional SiLU, and writes output and
  // present_state. Materializing the concatenation instead would cost ~7 ms per
  // call against this kernel's ~17 us, because thousands of rows of ~k bytes
  // each is a degenerate DMA shape.
  //
  // channels_last needs no separate kernel here: at seq_len == 1 the flat
  // index of element (b, c) is b*channels + c under both layouts, since the
  // permuted axis has extent 1. The decode kernel is therefore layout-agnostic
  // by construction rather than by accident.
  if (seq_len == 1) {
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
        "[REAL] wrap_causal_conv_with_state: completed via decode\n");
    return 0;
  }

  // ---- Prefill (seq_len > 1) ---------------------------------------------
  // One fused launch for what is the single largest text-prefill op.
  //
  // channels_last takes the _nlc kernel, which reads and writes (B, L, C)
  // directly instead of paying the Transpose pair the channels-first kernel
  // needs to be spliced into a channels-last graph.
  int rc =
      channels_last
          ? hip_causal_conv_prefill_nlc(stream, input, weight, bias, past_state,
                                        output, present_state, batch_size,
                                        channels, seq_len, kernel_size,
                                        activation, element_size_bytes)
          : hip_causal_conv_prefill(stream, input, weight, bias, past_state,
                                    output, present_state, batch_size, channels,
                                    seq_len, kernel_size, activation,
                                    element_size_bytes);
  if (rc != 0) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: hip_causal_conv_prefill%s failed "
            "(%d)\n",
            channels_last ? "_nlc" : "", rc);
    return -1;
  }
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_causal_conv_with_state: completed via prefill%s\n",
      channels_last ? " (nlc)" : "");
  return 0;
}
