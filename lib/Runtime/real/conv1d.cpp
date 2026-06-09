/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "conv1d.h"

#include "../debug_log.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>

// Convenience wrappers for the goto cleanup pattern below.
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)
#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// wrap_conv1d -- 1D convolution via MIOpen (NCL treated as NCHW H=1).
//===----------------------------------------------------------------------===//
//
// Design / structural notes:
//
// * MIOpen has no 1D forward-conv API; we reinterpret NCL as NC[H=1]L and
//   call the same miopenConvolutionForward path the existing 2D wrapper
//   (real/miopen.cpp) uses. Padding / stride along H are pinned to 0/1.
//
// * Workspace lives on the per-state `conv_scratch` pool rather than a
//   per-call hipMalloc/hipFree (the 2D path still does per-call malloc;
//   wrap_conv1d adopts the scratch-pool pattern that qmoe_scratch
//   established because Whisper's two encoder Conv layers are part of
//   the hot prefill path).
//
// * Bias add uses `miopenOpTensor` with `miopenTensorOpAdd` against a
//   per-channel bias tensor of shape (1, Cout, 1, 1) broadcast over
//   (N, H=1, Lout). We do NOT use `miopenConvolutionForwardBias` because
//   per the MIOpen docs that API is fixed at `alpha=1, beta=0`, which
//   would zero the conv output before adding bias (i.e. y = bias, not
//   y = conv + bias). `miopenOpTensor` with `alpha1=alpha2=1, beta=0`
//   computes `y = conv + bias` as required by ONNX Conv with rank-1 bias.
//
// * Algorithm selection uses the Find API (matches the existing 2D wrapper).
//   The 2D wrapper has a TODO to cache algorithm results across calls; for
//   Whisper there are only two Conv shapes per inference so the per-call
//   Find cost is amortised away after the first call. A future change can
//   share an algorithm cache between the 2D and 1D paths.
//
//===----------------------------------------------------------------------===//

extern "C" int wrap_conv1d(RuntimeState *state, const void *input,
                           const void *weights, const void *bias, void *output,
                           int64_t N, int64_t Cin, int64_t Lin, int64_t Cout,
                           int64_t K, int64_t stride, int64_t pad,
                           int64_t element_size_bytes) {
  OP_PROFILE(
      "conv1d",
      [&] {
        char buf[80];
        snprintf(buf, sizeof(buf),
                 "n=%lld,cin=%lld,lin=%lld,cout=%lld,k=%lld,s=%lld",
                 (long long)N, (long long)Cin, (long long)Lin, (long long)Cout,
                 (long long)K, (long long)stride);
        return std::string(buf);
      },
      state);

  if (!state || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_conv1d\n");
    return -1;
  }
  // Dtype: fp16 (element_size_bytes=2) and fp32 (element_size_bytes=4) are
  // wired through. The MIOpen dtype enum selected here is threaded into all
  // four tensor descriptors (input/weights/output/bias) below. The
  // alpha/beta scalars and the miopenOpTensor bias-add are fp32 scalars and
  // dtype-agnostic — MIOpen reads the tensor element type from the
  // descriptors. bf16 is not yet covered (would add a third branch).
  if (element_size_bytes != 2 && element_size_bytes != 4) {
    fprintf(stderr,
            "wrap_conv1d: only fp16 (element_size_bytes=2) and fp32 "
            "(element_size_bytes=4) are implemented, got %lld\n",
            (long long)element_size_bytes);
    return -1;
  }
  const miopenDataType_t mio_dt =
      (element_size_bytes == 2) ? miopenHalf : miopenFloat;

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_conv1d N=%lld Cin=%lld Lin=%lld Cout=%lld K=%lld s=%lld "
      "p=%lld bias=%s\n",
      (long long)N, (long long)Cin, (long long)Lin, (long long)Cout,
      (long long)K, (long long)stride, (long long)pad, bias ? "yes" : "null");

  const int64_t Lout = (Lin + 2 * pad - K) / stride + 1;

  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  // Note: handle's stream is set by RuntimeState init; no per-call
  // miopenSetStream is needed because the stream never changes.

  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t weights_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  miopenTensorDescriptor_t bias_desc = nullptr;
  miopenConvolutionDescriptor_t conv_desc = nullptr;
  void *workspace = nullptr;
  size_t workspace_size = 0;
  miopenConvAlgoPerf_t perf_results[1];
  int returned_algo_count = 0;
  miopenConvFwdAlgorithm_t algo;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int result = 0;

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));

  // Set tensor descriptors as 4D NCHW with H=1 (NCL -> NC1L). Using the
  // NCHW layout enum (not the legacy untyped Set4dTensorDescriptor) avoids
  // MIOpen 7.12+ "unknown layout" warnings.
  {
    int in_dims[] = {(int)N, (int)Cin, 1, (int)Lin};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        input_desc, mio_dt, miopenTensorNCHW, in_dims, 4));

    int w_dims[] = {(int)Cout, (int)Cin, 1, (int)K};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        weights_desc, mio_dt, miopenTensorNCHW, w_dims, 4));

    int out_dims[] = {(int)N, (int)Cout, 1, (int)Lout};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        output_desc, mio_dt, miopenTensorNCHW, out_dims, 4));
  }

  // Conv descriptor: pad/stride/dilation along H pinned to 0/1/1 (H=1).
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(
      conv_desc, miopenConvolution, /*pad_h*/ 0, /*pad_w*/ (int)pad,
      /*stride_h*/ 1, /*stride_w*/ (int)stride,
      /*dilation_h*/ 1, /*dilation_w*/ 1));

  // Sizing: MIOpen Find needs a workspace large enough for any algorithm it
  // tries. We grow the conv_scratch pool to the size the chosen algorithm
  // ends up needing (after Find returns perf_results[0].memory below).
  // The 2D wrapper allocates a fixed 10 MB find workspace per call; for
  // Whisper's two shapes the scratch pool quickly settles to a stable size.
  // Ask MIOpen what the worst-case algorithm needs:
  MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
      miopen_handle, weights_desc, input_desc, conv_desc, output_desc,
      &workspace_size));

  if (workspace_size > 0) {
    if (hipdnn_ep_state_ensure_conv_scratch(state, workspace_size) != 0) {
      fprintf(stderr, "wrap_conv1d: failed to grow conv_scratch to %zu bytes\n",
              workspace_size);
      result = -1;
      goto cleanup;
    }
    workspace = hipdnn_ep_state_get_conv_scratch(state);
  }

  MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
      miopen_handle, input_desc, input, weights_desc, weights, conv_desc,
      output_desc, output,
      /*requestAlgoCount*/ 1, &returned_algo_count, perf_results, workspace,
      workspace_size, /*exhaustiveSearch*/ false));
  if (returned_algo_count < 1) {
    fprintf(stderr, "wrap_conv1d: MIOpen Find returned no algorithms\n");
    result = -1;
    goto cleanup;
  }
  algo = perf_results[0].fwd_algo;

  MIOPEN_CHECK(miopenConvolutionForward(
      miopen_handle, &alpha, input_desc, input, weights_desc, weights,
      conv_desc, algo, &beta, output_desc, output, workspace, workspace_size));

  if (bias) {
    // Add per-channel bias via miopenOpTensor (TensorOpAdd):
    //   C = alpha1*A + alpha2*B + beta*C  with A=C=output, B=bias, beta=0
    //   -> output = output + bias  (broadcast from [1, Cout, 1, 1] over N,H,W)
    //
    // We do NOT use miopenConvolutionForwardBias here: the MIOpen header
    // explicitly states that miopenConvolutionForwardBias's alpha/beta are
    // "only supported for alpha = 1 and beta = 0" — passing beta=1.0 to fuse
    // the add gave silently wrong results (output magnitudes ~2x correct on
    // these whisper-encoder shapes). miopenOpTensor + TensorOpAdd is what
    // causal_conv_with_state.cpp uses for the same bias-after-conv pattern.
    const float alpha_bias = 1.0f, beta_zero = 0.0f;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
    int b_dims[] = {1, (int)Cout, 1, 1};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        bias_desc, mio_dt, miopenTensorNCHW, b_dims, 4));
    MIOPEN_CHECK(miopenOpTensor(miopen_handle, miopenTensorOpAdd, &alpha_bias,
                                output_desc, output, &alpha_bias, bias_desc,
                                bias, &beta_zero, output_desc, output));
  }

cleanup:
  if (bias_desc) {
    miopenDestroyTensorDescriptor(bias_desc);
  }
  if (input_desc) {
    miopenDestroyTensorDescriptor(input_desc);
  }
  if (weights_desc) {
    miopenDestroyTensorDescriptor(weights_desc);
  }
  if (output_desc) {
    miopenDestroyTensorDescriptor(output_desc);
  }
  if (conv_desc) {
    miopenDestroyConvolutionDescriptor(conv_desc);
  }
  // Workspace is owned by RuntimeState->conv_scratch; do NOT hipFree here.
  return result;
}
