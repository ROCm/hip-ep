/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../profiler.h"
#include "runtime_types.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
template <typename T> static inline T silu(T x) {
  return x / (static_cast<T>(1) + std::exp(-x));
}

int wrap_causal_conv_with_state(
    RuntimeState *state, const void *input, const void *weight,
    const void *bias, const void *past_state, void *output, void *present_state,
    int64_t batch_size, int64_t channels, int64_t seq_len, int64_t kernel_size,
    int64_t ndim, int64_t activation, int64_t element_size_bytes) {
  PERF_TIMER(state, "CausalConv",
             "b=%lld,ch=%lld,seq=%lld,k=%lld",
             (long long)batch_size, (long long)channels,
             (long long)seq_len, (long long)kernel_size);

  if (!state || !input || !weight || !output || !present_state) {
    fprintf(stderr, "wrap_causal_conv_with_state: null required argument\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_causal_conv_with_state: batch=%lld, "
                    "channels=%lld, seq_len=%lld, kernel=%lld, ndim=%lld, "
                    "activation=%lld, elem_size=%lld\n",
                    (long long)batch_size, (long long)channels,
                    (long long)seq_len, (long long)kernel_size, (long long)ndim,
                    (long long)activation, (long long)element_size_bytes);

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_causal_conv_with_state: null stream\n");
    return -1;
  }

  // Currently only ndim=1 is implemented
  if (ndim != 1) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: ndim=%lld not yet supported "
            "(only ndim=1)\n",
            (long long)ndim);
    return -1;
  }

  int64_t state_len = kernel_size - 1; // k-1
  int64_t virtual_len = state_len + seq_len;

  // Allocate temporary buffer for the virtual input on device:
  // shape (batch, channels, state_len + seq_len) = (B, C, k-1+L)
  int64_t virtual_size =
      batch_size * channels * virtual_len * element_size_bytes;
  void *virtual_buf = nullptr;
  hipError_t err = hipMalloc(&virtual_buf, virtual_size);
  if (err != hipSuccess || !virtual_buf) {
    fprintf(stderr,
            "wrap_causal_conv_with_state: hipMalloc failed for "
            "virtual buffer (%lld bytes)\n",
            (long long)virtual_size);
    return -1;
  }

  // Fill virtual buffer: [past_state | input] per (batch, channel)
  // past_state shape: (B, C, k-1) or nullptr (zero-fill)
  // input shape: (B, C, L)
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t c = 0; c < channels; ++c) {
      int64_t bc = b * channels + c;
      char *dst_base = static_cast<char *>(virtual_buf) +
                       bc * virtual_len * element_size_bytes;

      // Copy past_state portion (first k-1 elements)
      if (past_state) {
        const char *ps_src = static_cast<const char *>(past_state) +
                             bc * state_len * element_size_bytes;
        hipMemcpyAsync(dst_base, ps_src, state_len * element_size_bytes,
                       hipMemcpyDeviceToDevice, stream);
      } else {
        hipMemsetAsync(dst_base, 0, state_len * element_size_bytes, stream);
      }

      // Copy input portion (last L elements)
      const char *in_src =
          static_cast<const char *>(input) + bc * seq_len * element_size_bytes;
      hipMemcpyAsync(dst_base + state_len * element_size_bytes, in_src,
                     seq_len * element_size_bytes, hipMemcpyDeviceToDevice,
                     stream);
    }
  }

  // Extract present_state: last (k-1) positions from the virtual input
  // present_state[b,c,:] = virtual[b,c, seq_len : seq_len + k-1]
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t c = 0; c < channels; ++c) {
      int64_t bc = b * channels + c;
      const char *src = static_cast<const char *>(virtual_buf) +
                        (bc * virtual_len + seq_len) * element_size_bytes;
      char *dst = static_cast<char *>(present_state) +
                  bc * state_len * element_size_bytes;
      hipMemcpyAsync(dst, src, state_len * element_size_bytes,
                     hipMemcpyDeviceToDevice, stream);
    }
  }

  // Perform depthwise causal convolution on the GPU.
  // We use MIOpen convolution with group=channels for depthwise conv.
  // Input to MIOpen: virtual_buf as (B, C, 1, virtual_len) [4D for MIOpen]
  // Weight: (C, 1, 1, k) [depthwise: group=C, each filter sees 1 channel]
  // Output: (B, C, 1, L)
  //
  // We use the existing miopenConvolutionForward for this.
  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    hipFree(virtual_buf);
    fprintf(stderr, "wrap_causal_conv_with_state: null MIOpen handle\n");
    return -1;
  }

  // Determine MIOpen data type
  miopenDataType_t dt;
  if (element_size_bytes == 4)
    dt = miopenFloat;
  else if (element_size_bytes == 2)
    dt = miopenHalf;
  else {
    hipFree(virtual_buf);
    fprintf(stderr,
            "wrap_causal_conv_with_state: unsupported element_size "
            "%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  // Create tensor descriptors (4D: N, C, H, W)
  miopenTensorDescriptor_t inDesc = nullptr, wDesc = nullptr, outDesc = nullptr;
  miopenTensorDescriptor_t biasDesc = nullptr;
  miopenConvolutionDescriptor_t convDesc = nullptr;
  miopenActivationDescriptor_t actDesc = nullptr;
  void *sigmoid_buf = nullptr;
  miopenStatus_t mst;
  int ret = 0;

#define CAUSAL_MIOPEN_CHECK(call)                                              \
  do {                                                                         \
    mst = (call);                                                              \
    if (mst != miopenStatusSuccess) {                                          \
      fprintf(stderr,                                                          \
              "wrap_causal_conv_with_state: MIOpen error %d at "               \
              "%s:%d\n",                                                       \
              mst, __FILE__, __LINE__);                                        \
      ret = -1;                                                                \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

  CAUSAL_MIOPEN_CHECK(miopenCreateTensorDescriptor(&inDesc));
  CAUSAL_MIOPEN_CHECK(miopenCreateTensorDescriptor(&wDesc));
  CAUSAL_MIOPEN_CHECK(miopenCreateTensorDescriptor(&outDesc));
  CAUSAL_MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&convDesc));

  // Input: (B, C, 1, virtual_len)
  CAUSAL_MIOPEN_CHECK(miopenSet4dTensorDescriptor(
      inDesc, dt, static_cast<int>(batch_size), static_cast<int>(channels), 1,
      static_cast<int>(virtual_len)));

  // Weight: (C, 1, 1, kernel_size) for depthwise (group=C)
  CAUSAL_MIOPEN_CHECK(
      miopenSet4dTensorDescriptor(wDesc, dt, static_cast<int>(channels), 1, 1,
                                  static_cast<int>(kernel_size)));

  // Output: (B, C, 1, seq_len)
  CAUSAL_MIOPEN_CHECK(miopenSet4dTensorDescriptor(
      outDesc, dt, static_cast<int>(batch_size), static_cast<int>(channels), 1,
      static_cast<int>(seq_len)));

  // Convolution: pad=(0,0), stride=(1,1), dilation=(1,1), group=channels
  CAUSAL_MIOPEN_CHECK(
      miopenInitConvolutionDescriptor(convDesc, miopenConvolution,
                                      /*pad_h=*/0, /*pad_w=*/0,
                                      /*stride_h=*/1, /*stride_w=*/1,
                                      /*dilation_h=*/1, /*dilation_w=*/1));
  CAUSAL_MIOPEN_CHECK(
      miopenSetConvolutionGroupCount(convDesc, static_cast<int>(channels)));

  {
    // Find algorithm
    miopenConvAlgoPerf_t perfResult;
    int returnedAlgoCount = 0;
    size_t wsSize = 0;
    CAUSAL_MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        handle, wDesc, inDesc, convDesc, outDesc, &wsSize));

    void *workspace = nullptr;
    if (wsSize > 0) {
      if (hipdnn_ep_state_ensure_workspace(state, wsSize) != 0) {
        ret = -1;
        goto cleanup;
      }
      workspace = hipdnn_ep_state_get_workspace(state);
    }

    CAUSAL_MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        handle, inDesc, virtual_buf, wDesc, weight, convDesc, outDesc, output,
        /*requestAlgoCount=*/1, &returnedAlgoCount, &perfResult, workspace,
        wsSize, /*exhaustiveSearch=*/false));

    float alpha = 1.0f, beta = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenConvolutionForward(
        handle, &alpha, inDesc, virtual_buf, wDesc, weight, convDesc,
        perfResult.fwd_algo, &beta, outDesc, output, workspace, wsSize));
  }

  // Add bias if present
  if (bias) {
    CAUSAL_MIOPEN_CHECK(miopenCreateTensorDescriptor(&biasDesc));
    CAUSAL_MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        biasDesc, dt, 1, static_cast<int>(channels), 1, 1));
    float alpha_bias = 1.0f, beta_bias = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha_bias,
                                       outDesc, output, &alpha_bias, biasDesc,
                                       bias, &beta_bias, outDesc, output));
  }

  // Apply activation (SiLU/Swish)
  if (activation == 1) {
    // SiLU(x) = x * sigmoid(x)
    int64_t output_size = batch_size * channels * seq_len * element_size_bytes;
    err = hipMalloc(&sigmoid_buf, output_size);
    if (err != hipSuccess || !sigmoid_buf) {
      fprintf(stderr, "wrap_causal_conv_with_state: hipMalloc failed for "
                      "sigmoid buffer\n");
      ret = -1;
      goto cleanup;
    }

    CAUSAL_MIOPEN_CHECK(miopenCreateActivationDescriptor(&actDesc));
    CAUSAL_MIOPEN_CHECK(miopenSetActivationDescriptor(
        actDesc, miopenActivationLOGISTIC, 0.0, 0.0, 0.0));

    float alpha_act = 1.0f, beta_act = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenActivationForward(handle, actDesc, &alpha_act,
                                                outDesc, output, &beta_act,
                                                outDesc, sigmoid_buf));

    float alpha_mul = 1.0f, beta_mul = 0.0f;
    CAUSAL_MIOPEN_CHECK(miopenOpTensor(
        handle, miopenTensorOpMul, &alpha_mul, outDesc, output, &alpha_mul,
        outDesc, sigmoid_buf, &beta_mul, outDesc, output));
  }

cleanup:
  if (actDesc)
    miopenDestroyActivationDescriptor(actDesc);
  if (sigmoid_buf)
    hipFree(sigmoid_buf);
  if (biasDesc)
    miopenDestroyTensorDescriptor(biasDesc);
  if (convDesc)
    miopenDestroyConvolutionDescriptor(convDesc);
  if (outDesc)
    miopenDestroyTensorDescriptor(outDesc);
  if (wDesc)
    miopenDestroyTensorDescriptor(wDesc);
  if (inDesc)
    miopenDestroyTensorDescriptor(inDesc);
  if (virtual_buf)
    hipFree(virtual_buf);

#undef CAUSAL_MIOPEN_CHECK

  if (ret == 0) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_causal_conv_with_state: completed successfully\n");
  }
  return ret;
}
