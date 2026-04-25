/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>

#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)
#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// MIOpen Convolution Backward-Data Wrapper (a.k.a. ONNX ConvTranspose)
//===----------------------------------------------------------------------===//
//
// ONNX ConvTranspose is the gradient of a forward convolution with respect
// to its input.  For the forward conv
//     y(N, K_fwd, OH, OW) = x(N, C_fwd, IH, IW) * w(K_fwd, C_fwd/grp, kH, kW)
// MIOpen's backward-data computes
//     dx(N, C_fwd, IH, IW) = dy(N, K_fwd, OH, OW) * w(K_fwd, C_fwd/grp, kH, kW)
//
// Mapping ONNX ConvTranspose -> MIOpen backward-data:
//   ConvTranspose input  -> dy   shape (N, weights_k, IH, IW)
//   ConvTranspose weights-> w    shape (weights_k, output_c/group, kH, kW)
//   ConvTranspose output -> dx   shape (N, output_c, OH, OW)
//
// `output_padding` is encoded purely through the dx descriptor shape: the
// caller has already computed the correct OH/OW including output_padding,
// and MIOpen treats the descriptors as authoritative.  The relationship
//     OH = stride*(IH - 1) - pad_top - pad_bottom + (kH-1)*dilH + 1 + opH
// means the "forward" of dx with the given conv_desc lands back on dy
// regardless of opH (so long as opH < stride, which ONNX requires).
//
// Bias is added as a separate fused op via miopenOpTensor (broadcast across
// N/H/W on a [1, output_c, 1, 1] descriptor), matching the pattern used by
// causal_conv_with_state.cpp.
//
// We use mode = `miopenConvolution`; backward-data is the standard way to
// realise ConvTranspose -- the `miopenTranspose` mode is for callers that
// want to invoke `miopenConvolutionForward` and have it run as a transposed
// convolution under the hood.
//
// Spec note: the wrapper signature in the design doc lists no explicit
// `output_c` or `data_type` parameter; both are needed to construct the
// dx / w descriptors and to support fp16 / bf16, so we add them at the end
// of the parameter list (LLVM lowering passes them too).
//===----------------------------------------------------------------------===//

static miopenDataType_t conv_transpose_to_miopen_type(int64_t dt, bool &ok) {
  ok = true;
  switch (dt) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return miopenFloat;
  case HIPDNN_EP_DATATYPE_HALF:
    return miopenHalf;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return miopenBFloat16;
  default:
    fprintf(stderr,
            "wrap_miopenConvolutionBackwardData: unsupported data_type %lld\n",
            (long long)dt);
    ok = false;
    return miopenFloat;
  }
}

extern "C" int wrap_miopenConvolutionBackwardData(
    RuntimeState *state,
    void *input,    // ConvTranspose input  -> backward dy
    int64_t input_n, int64_t input_c, int64_t input_h, int64_t input_w,
    void *weights,
    int64_t weights_k, // ConvTranspose input channels (== fwd K)
    void *bias,        // nullable, length == output_c
    void *output,      // ConvTranspose output -> backward dx
    int64_t output_h, int64_t output_w,
    int64_t output_c, // ConvTranspose output channels (== fwd C)
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left,
    int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w,
    int64_t output_padding_h, int64_t output_padding_w,
    int64_t group,
    int64_t data_type) {
  if (!state || !input || !weights || !output) {
    fprintf(stderr,
            "wrap_miopenConvolutionBackwardData: null required argument\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenConvolutionBackwardData: in=[%lld,%lld,%lld,%lld] "
      "w=[%lld,%lld/g,%lld,%lld] out=[%lld,%lld,%lld,%lld] s=[%lld,%lld] "
      "p=[%lld,%lld,%lld,%lld] d=[%lld,%lld] op=[%lld,%lld] g=%lld dt=%lld\n",
      (long long)input_n, (long long)input_c, (long long)input_h,
      (long long)input_w, (long long)weights_k, (long long)output_c,
      (long long)kernel_h, (long long)kernel_w, (long long)input_n,
      (long long)output_c, (long long)output_h, (long long)output_w,
      (long long)stride_h, (long long)stride_w, (long long)pad_top,
      (long long)pad_left, (long long)pad_bottom, (long long)pad_right,
      (long long)dilation_h, (long long)dilation_w,
      (long long)output_padding_h, (long long)output_padding_w,
      (long long)group, (long long)data_type);

  // MIOpen takes a single per-axis pad; ONNX may emit asymmetric pads when
  // `auto_pad = SAME_*`.  In practice, ONNX ConvTranspose with asymmetric
  // pads is rare; we use the begin-side value and warn if they differ
  // (mirrors the existing wrap_miopenConvolutionForward behaviour).
  if (pad_top != pad_bottom || pad_left != pad_right) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_miopenConvolutionBackwardData: asymmetric pads "
        "(top=%lld bottom=%lld, left=%lld right=%lld); using begin side\n",
        (long long)pad_top, (long long)pad_bottom, (long long)pad_left,
        (long long)pad_right);
  }
  if (output_padding_h != 0 || output_padding_w != 0) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_miopenConvolutionBackwardData: output_padding=%lld,%lld "
        "encoded via dx descriptor shape\n",
        (long long)output_padding_h, (long long)output_padding_w);
  }
  if (input_c != weights_k) {
    fprintf(stderr,
            "wrap_miopenConvolutionBackwardData: input_c (%lld) must equal "
            "weights_k (%lld)\n",
            (long long)input_c, (long long)weights_k);
    return -1;
  }

  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!miopen_handle) {
    fprintf(stderr,
            "wrap_miopenConvolutionBackwardData: null MIOpen handle\n");
    return -1;
  }

  bool ok = false;
  miopenDataType_t mio_dt = conv_transpose_to_miopen_type(data_type, ok);
  if (!ok)
    return -1;

  miopenTensorDescriptor_t dy_desc = nullptr;   // ConvTranspose input
  miopenTensorDescriptor_t w_desc = nullptr;    // weights
  miopenTensorDescriptor_t dx_desc = nullptr;   // ConvTranspose output
  miopenTensorDescriptor_t bias_desc = nullptr; // bias (optional)
  miopenConvolutionDescriptor_t conv_desc = nullptr;
  void *find_workspace = nullptr;
  void *workspace = nullptr;
  int result = 0;
  miopenConvAlgoPerf_t perf_results[1];
  int returned_algo_count = 0;
  miopenConvBwdDataAlgorithm_t algo;
  size_t workspace_size = 0;
  const size_t find_workspace_size = 64 * 1024 * 1024; // 64MB
  float alpha = 1.0f;
  float beta = 0.0f;

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&dy_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&w_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&dx_desc));

  // dy (input) shape: [N, K, IH, IW] = [N, weights_k, input_h, input_w].
  {
    int dy_dims[] = {(int)input_n, (int)weights_k, (int)input_h,
                     (int)input_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        dy_desc, mio_dt, miopenTensorNCHW, dy_dims, 4));
  }
  // weights shape: [K, C_out/group, kH, kW].  MIOpen lays out filters as
  // (filters, channels_per_group, ...) so we divide by `group` here.
  {
    if (output_c % group != 0) {
      fprintf(stderr,
              "wrap_miopenConvolutionBackwardData: output_c (%lld) must be "
              "divisible by group (%lld)\n",
              (long long)output_c, (long long)group);
      result = -1;
      goto cleanup;
    }
    int w_dims[] = {(int)weights_k, (int)(output_c / group), (int)kernel_h,
                    (int)kernel_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        w_desc, mio_dt, miopenTensorNCHW, w_dims, 4));
  }
  // dx (output) shape: [N, C_out, OH, OW].
  {
    int dx_dims[] = {(int)input_n, (int)output_c, (int)output_h,
                     (int)output_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        dx_desc, mio_dt, miopenTensorNCHW, dx_dims, 4));
  }

  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(
      conv_desc, miopenConvolution, (int)pad_top, (int)pad_left,
      (int)stride_h, (int)stride_w, (int)dilation_h, (int)dilation_w));
  if (group > 1) {
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc, (int)group));
  }

  // Workspace for algorithm search.  We follow the same pattern as
  // wrap_miopenConvolutionForward: allocate 10MB scratch, ask MIOpen to
  // benchmark candidates within it, then resize if the chosen algo needs
  // more.  TODO: switch to the shared workspace pool once we add
  // per-call sizing on the bwd path.
  HIP_CHECK(hipMalloc(&find_workspace, find_workspace_size));

  MIOPEN_CHECK(miopenFindConvolutionBackwardDataAlgorithm(
      miopen_handle, dy_desc, input, w_desc, weights, conv_desc, dx_desc,
      output,
      /*requestAlgoCount=*/1, &returned_algo_count, perf_results,
      find_workspace, find_workspace_size, /*exhaustiveSearch=*/false));
  if (returned_algo_count < 1) {
    fprintf(stderr,
            "wrap_miopenConvolutionBackwardData: no algorithm returned\n");
    result = -1;
    goto cleanup;
  }
  algo = perf_results[0].bwd_data_algo;

  MIOPEN_CHECK(miopenConvolutionBackwardDataGetWorkSpaceSize(
      miopen_handle, dy_desc, w_desc, conv_desc, dx_desc, &workspace_size));

  workspace = find_workspace;
  if (workspace_size > find_workspace_size) {
    HIP_CLEANUP(hipFree(find_workspace));
    find_workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, workspace_size));
  }

  MIOPEN_CHECK(miopenConvolutionBackwardData(
      miopen_handle, &alpha, dy_desc, input, w_desc, weights, conv_desc, algo,
      &beta, dx_desc, output, workspace, workspace_size));

  // Add bias broadcast across N/H/W.  miopenOpTensor with a
  // [1, output_c, 1, 1] bias descriptor performs the per-channel add.
  if (bias) {
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
    int bias_dims[] = {1, (int)output_c, 1, 1};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        bias_desc, mio_dt, miopenTensorNCHW, bias_dims, 4));
    float alpha_a = 1.0f, alpha_b = 1.0f, beta_c = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(miopen_handle, miopenTensorOpAdd, &alpha_a,
                                dx_desc, output, &alpha_b, bias_desc, bias,
                                &beta_c, dx_desc, output));
  }

cleanup:
  if (workspace && workspace != find_workspace)
    HIP_CLEANUP(hipFree(workspace));
  if (find_workspace)
    HIP_CLEANUP(hipFree(find_workspace));
  if (bias_desc)
    miopenDestroyTensorDescriptor(bias_desc);
  if (dx_desc)
    miopenDestroyTensorDescriptor(dx_desc);
  if (w_desc)
    miopenDestroyTensorDescriptor(w_desc);
  if (dy_desc)
    miopenDestroyTensorDescriptor(dy_desc);
  if (conv_desc)
    miopenDestroyConvolutionDescriptor(conv_desc);
  return result;
}
