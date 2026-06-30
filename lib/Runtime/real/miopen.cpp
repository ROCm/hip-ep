/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <unordered_map>
// Convenience wrappers for goto cleanup pattern (all functions use 'cleanup'
// label)
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)
#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// MIOpen Convolution Forward Wrapper
//===----------------------------------------------------------------------===//
//
// DESIGN DECISIONS:
//
// 1. GENERIC WRAPPER (Not Specialized)
//    Why use one wrapper for all convolution configurations instead of
//    specialized wrappers per kernel size (1x1, 3x3, 7x7)?
//
//    - MIOpen handles specialization internally via algorithm selection
//    - miopenFindConvolutionForwardAlgorithm() automatically selects optimal
//      kernel (Winograd for 3×3, GEMM for 1×1, FFT for certain configs)
//    - Industry standard: PyTorch, TensorFlow use generic wrappers
//    - No performance benefit from wrapper-level specialization
//    - Avoids complexity explosion (would need 50+ specialized wrappers)
//
//    Sources:
//    -
//    https://rocm.docs.amd.com/projects/MIOpen/en/develop/doxygen/html/group__convolutions.html
//    -
//    https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/
//
// 2. PER-SHAPE ALGORITHM CACHE (implemented; descriptors NOT cached)
//    The (dtype, input/weights/output shape, stride/pad/dilation/group,
//    has_bias) tuple is invariant per conv layer for a static-shape model, so
//    the Find-selected algorithm + its actual workspace size are cached once
//    per shape in RuntimeState::conv_fwd_cache (see ConvFwdCache below). On a
//    cache hit the call skips miopenConvolutionForwardGetWorkSpaceSize AND
//    miopenFindConvolutionForwardAlgorithm. MIOpen docs: "miopenFindConvolution*()
//    is expensive in terms of run time and required workspace, so it's highly
//    recommended to reserve the required algorithm and workspace to reuse them
//    later." (PyTorch's torch.backends.cudnn.benchmark does the same.) For
//    dynamic shapes the hit rate drops with shape variation, but the cache is
//    never wrong: a new shape simply Finds + caches a new entry.
//
//    The MIOpen descriptors are deliberately NOT cached: Find/forward make
//    MIOpen attach per-problem GPU-resident state (compiled solvers, pre-
//    transformed weights, solver scratch) to the live descriptors, and keeping
//    one descriptor set alive per distinct shape made all of those coexist for
//    the whole session -- inflating peak GPU memory by hundreds of MB on
//    conv-heavy models. Descriptors are rebuilt per call (cheap CPU structs)
//    and destroyed at function exit so only one shape's resident state is live
//    at a time. The expensive part (Find) is still amortized via the cache.
//
//    Sources:
//    - https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
//    - https://docs.pytorch.org/docs/stable/notes/cuda.html
//
// 3. WORKSPACE POOLING (implemented)
//    The conv workspace is drawn from the per-RuntimeState conv_scratch pool
//    (grow-on-demand, never shrinks), not a per-call hipMalloc/hipFree.
//
//===----------------------------------------------------------------------===//

// Map HIPDNN_EP_DATATYPE_* to miopenDataType_t for the conv tensor
// descriptors. Mirrors hipdnn_ep_to_miopen_type in elementwise.cpp /
// activation.cpp — copy kept local so this TU has no link-time dependency on
// those siblings (each runtime .cpp is compiled to its own bitcode and the
// shared bitcode link only resolves wrap_* entry points).
static miopenDataType_t conv_to_miopen_type(int64_t data_type, bool &ok) {
  ok = true;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return miopenFloat;
  case HIPDNN_EP_DATATYPE_HALF:
    return miopenHalf;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return miopenBFloat16;
  default:
    ok = false;
    return miopenFloat;
  }
}

// MIOpen convolution forward implementation
// Follows opaque RuntimeState pattern - extracts handle/stream from state.
//
// `data_type` is a HIPDNN_EP_DATATYPE_* enum value applied uniformly to the
// input, weights, and output tensor descriptors. All three buffers MUST share
// the same element type — MIOpen's miopenConvolutionForward does not support
// mixed-precision descriptors. This matches the host-side ConvConversion
// invariant that all three operands use `resultType.getElementType()`.
//
// Historical note: prior to the data_type parameter the descriptors were
// hardcoded to miopenFloat (fp32), which silently passed wrong strides to
// MIOpen when the actual buffers were fp16 (e.g. SigLIP / ViT patch
// embedding). MIOpen then read out-of-bounds bytes as the second-half stride
// for every row, producing NaN/Inf at fp16-max in the output and cascading
// NaN through the rest of the network.

//===----------------------------------------------------------------------===//
// Per-shape MIOpen forward-conv ALGORITHM cache (descriptors NOT cached).
//
// The (dt, NCHW input, weights, output, stride/pad/dilation/group, has_bias)
// tuple is invariant per conv layer across all Compute() calls of a static-
// shape model. miopenFindConvolutionForwardAlgorithm is documented-expensive
// (it launches candidate kernels on the GPU and synchronizes), so we cache its
// result (the chosen algorithm + that algorithm's actual workspace size) once
// per shape. On a hit the call skips miopenConvolutionForwardGetWorkSpaceSize
// AND Find.
//
// We deliberately do NOT cache the MIOpen tensor/convolution descriptors. Find
// and forward make MIOpen attach per-problem GPU-resident state (compiled
// solver programs, pre-transformed/packed weight buffers, solver scratch) to
// the live descriptors; keeping a descriptor set alive per distinct shape made
// all of those coexist for the whole session, inflating peak GPU memory by
// (num_distinct_conv_shapes * per-solution-resident-bytes) -- hundreds of MB on
// conv-heavy models. Rebuilding the descriptors per call (cheap CPU structs)
// and destroying them at function exit lets MIOpen release that per-problem
// state between calls, so only one shape's resident state is live at a time.
// The expensive part (Find) is still amortized via the algorithm cache.
//===----------------------------------------------------------------------===//
namespace {

struct ConvFwdKey {
  int64_t dt; // miopenDataType_t value
  int64_t input_n, input_c, input_h, input_w;
  int64_t weights_k, kernel_h, kernel_w;
  int64_t output_h, output_w;
  int64_t stride_h, stride_w;
  int64_t pad_top, pad_left, pad_bottom, pad_right;
  int64_t dilation_h, dilation_w, group;
  int64_t has_bias;

  bool operator==(const ConvFwdKey &o) const {
    return dt == o.dt && input_n == o.input_n && input_c == o.input_c &&
           input_h == o.input_h && input_w == o.input_w &&
           weights_k == o.weights_k && kernel_h == o.kernel_h &&
           kernel_w == o.kernel_w && output_h == o.output_h &&
           output_w == o.output_w && stride_h == o.stride_h &&
           stride_w == o.stride_w && pad_top == o.pad_top &&
           pad_left == o.pad_left && pad_bottom == o.pad_bottom &&
           pad_right == o.pad_right && dilation_h == o.dilation_h &&
           dilation_w == o.dilation_w && group == o.group &&
           has_bias == o.has_bias;
  }
};

struct ConvFwdKeyHash {
  size_t operator()(const ConvFwdKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.dt);
    hash_combine_val(h, k.input_n);
    hash_combine_val(h, k.input_c);
    hash_combine_val(h, k.input_h);
    hash_combine_val(h, k.input_w);
    hash_combine_val(h, k.weights_k);
    hash_combine_val(h, k.kernel_h);
    hash_combine_val(h, k.kernel_w);
    hash_combine_val(h, k.output_h);
    hash_combine_val(h, k.output_w);
    hash_combine_val(h, k.stride_h);
    hash_combine_val(h, k.stride_w);
    hash_combine_val(h, k.pad_top);
    hash_combine_val(h, k.pad_left);
    hash_combine_val(h, k.pad_bottom);
    hash_combine_val(h, k.pad_right);
    hash_combine_val(h, k.dilation_h);
    hash_combine_val(h, k.dilation_w);
    hash_combine_val(h, k.group);
    hash_combine_val(h, k.has_bias);
    return h;
  }
};

// Only the Find result is cached -- no MIOpen descriptors (see header above).
// Presence of the entry in the map == "Find has run for this shape".
struct ConvFwdCacheEntry {
  miopenConvFwdAlgorithm_t algo;
  size_t algo_workspace_size; // perf_results[0].memory from the original Find
};

struct ConvFwdCache {
  std::unordered_map<ConvFwdKey, ConvFwdCacheEntry, ConvFwdKeyHash> entries;
};

} // namespace

extern "C" void hipdnn_ep_conv_fwd_cache_destroy(void *cache_ptr) {
  // Entries hold only POD (algo + size) -- no MIOpen handles to destroy.
  delete static_cast<ConvFwdCache *>(cache_ptr);
}

int wrap_miopenConvolutionForward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group, int64_t data_type) {
  OP_PROFILE(
      "conv",
      [&] {
        char b[80];
        const char *dt = (data_type == HIPDNN_EP_DATATYPE_HALF)       ? "f16"
                         : (data_type == HIPDNN_EP_DATATYPE_BFLOAT16) ? "bf16"
                                                                      : "f32";
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld,k=%lldx%lldx%lld,%s",
                 (long long)input_n, (long long)input_c, (long long)input_h,
                 (long long)input_w, (long long)weights_k, (long long)kernel_h,
                 (long long)kernel_w, dt);
        return std::string(b);
      },
      state);
  if (!state || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionForward\n");
    return -1;
  }
  bool dt_ok;
  miopenDataType_t miopen_dt = conv_to_miopen_type(data_type, dt_ok);
  if (!dt_ok) {
    fprintf(
        stderr,
        "[REAL] wrap_miopenConvolutionForward: unsupported data_type %lld\n",
        (long long)data_type);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenConvolutionForward N=%lld Cin=%lld H=%lld W=%lld "
      "Cout=%lld kHxkW=%lldx%lld s=%lldx%lld bias=%s dtype=%lld\n",
      (long long)input_n, (long long)input_c, (long long)input_h,
      (long long)input_w, (long long)weights_k, (long long)kernel_h,
      (long long)kernel_w, (long long)stride_h, (long long)stride_w,
      bias ? "yes" : "null", (long long)data_type);

  // Extract handle and stream from opaque RuntimeState via accessor functions
  // (Maintains abstraction barrier - no direct field access)
  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  hipStream_t hip_stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  // Per-call locals. The input/weights/output/conv descriptors are rebuilt
  // every call and destroyed at `cleanup` so MIOpen releases the per-problem
  // GPU-resident state it attaches to them between calls (see cache header
  // above). Only the Find result (algorithm + workspace size) is cached.
  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t weights_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  miopenTensorDescriptor_t bias_desc = nullptr;
  miopenConvolutionDescriptor_t conv_desc = nullptr;
  // Workspace is owned by RuntimeState->conv_scratch (grow-on-demand pool);
  // do NOT hipFree it here.
  void *workspace = nullptr;
  int result = 0;
  miopenConvAlgoPerf_t perf_results[1];
  int returned_algo_count = 0;
  miopenConvFwdAlgorithm_t algo;
  size_t workspace_size = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  // Look up the per-shape Find result FIRST, before creating any descriptors.
  // This must precede the descriptor-creation MIOPEN_CHECKs below: those
  // `goto cleanup` on failure, and C++ forbids a goto from bypassing the
  // initialization of an in-scope variable (conv_cache/conv_key/cache_it). Only
  // the (algo, workspace_size) pair is cached -- the descriptors are NOT (see
  // cache header).
  if (!state->conv_fwd_cache)
    state->conv_fwd_cache = new ConvFwdCache();
  ConvFwdCache *conv_cache = static_cast<ConvFwdCache *>(state->conv_fwd_cache);
  ConvFwdKey conv_key{(int64_t)miopen_dt,
                      input_n,
                      input_c,
                      input_h,
                      input_w,
                      weights_k,
                      kernel_h,
                      kernel_w,
                      output_h,
                      output_w,
                      stride_h,
                      stride_w,
                      pad_top,
                      pad_left,
                      pad_bottom,
                      pad_right,
                      dilation_h,
                      dilation_w,
                      group,
                      bias ? 1 : 0};
  auto cache_it = conv_cache->entries.find(conv_key);
  bool cache_hit = (cache_it != conv_cache->entries.end());

  // Create + set the tensor descriptors with explicit NCHW layout. The dtype is
  // taken from the caller (data_type) — the previous hardcoded miopenFloat
  // produced silent fp16-stride-as-fp32 corruption on fp16 models.
  // miopenSet4dTensorDescriptor leaves layout UNKNOWN which warns in MIOpen 7.12+.
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));
  {
    int in_dims[] = {(int)input_n, (int)input_c, (int)input_h, (int)input_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        input_desc, miopen_dt, miopenTensorNCHW, in_dims, 4));

    // For grouped/depthwise convolution the weight tensor's input-channel dim
    // is input_c / group (e.g. depthwise conv: weights [C,1,kh,kw], group=C).
    // Using input_c here would describe the wrong filter shape to MIOpen and
    // produce silently incorrect results. group=1 reduces to input_c; the
    // `group ? group : 1` guard only defends against a malformed group==0 so we
    // never divide by zero.
    int w_dims[] = {(int)weights_k, (int)(input_c / (group ? group : 1)),
                    (int)kernel_h, (int)kernel_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        weights_desc, miopen_dt, miopenTensorNCHW, w_dims, 4));

    int out_dims[] = {(int)input_n, (int)weights_k, (int)output_h,
                      (int)output_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        output_desc, miopen_dt, miopenTensorNCHW, out_dims, 4));
  }

  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(
      conv_desc, miopenConvolution, pad_top, pad_left, stride_h, stride_w,
      dilation_h, dilation_w));
  // group=1 for standard convolution, group=C for depthwise convolution.
  if (group > 1) {
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc, group));
  }

  if (cache_hit) {
    // HIT: reuse the cached algorithm + its actual workspace size; skip both
    // GetWorkSpaceSize and the expensive Find.
    algo = cache_it->second.algo;
    workspace_size = cache_it->second.algo_workspace_size;
    if (workspace_size > 0) {
      if (hipdnn_ep_state_ensure_conv_scratch(state, workspace_size) != 0) {
        fprintf(stderr,
                "wrap_miopenConvolutionForward: failed to grow conv_scratch to "
                "%zu bytes\n",
                workspace_size);
        result = -1;
        goto cleanup;
      }
      workspace = hipdnn_ep_state_get_conv_scratch(state);
    }
  } else {
    // MISS: query the worst-case workspace (upper bound) so Find has room to
    // benchmark candidate algorithms, run Find once, then cache the chosen algo
    // and its ACTUAL workspace size (perf_results[0].memory, usually smaller
    // than the upper bound). The conv_scratch pool is reused across all conv
    // calls in the session (replaces the old per-call hipMalloc/hipFree).
    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        miopen_handle, weights_desc, input_desc, conv_desc, output_desc,
        &workspace_size));
    if (workspace_size > 0) {
      if (hipdnn_ep_state_ensure_conv_scratch(state, workspace_size) != 0) {
        fprintf(stderr,
                "wrap_miopenConvolutionForward: failed to grow conv_scratch to "
                "%zu bytes\n",
                workspace_size);
        result = -1;
        goto cleanup;
      }
      workspace = hipdnn_ep_state_get_conv_scratch(state);
    }
    // MIOpen 3.x API: returns an array of performance results.
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        miopen_handle, input_desc, input, weights_desc, weights, conv_desc,
        output_desc, output,
        1,                    // requestAlgoCount - ask for 1 algorithm
        &returned_algo_count, // returnedAlgoCount - how many actually returned
        perf_results,         // perfResults - array to receive results
        workspace,            // workspace for algorithm testing
        workspace_size,       // workspaceSize (upper bound)
        false));
    if (returned_algo_count < 1) {
      fprintf(stderr, "wrap_miopenConvolutionForward: MIOpen Find returned no "
                      "algorithms\n");
      result = -1;
      goto cleanup;
    }
    algo = perf_results[0].fwd_algo;
    workspace_size = perf_results[0].memory; // actual workspace for forward
    conv_cache->entries.emplace(conv_key,
                                ConvFwdCacheEntry{algo, workspace_size});
  }
  // Perform convolution using the (cached) algorithm + its workspace size.
  MIOPEN_CHECK(miopenConvolutionForward(
      miopen_handle, &alpha, input_desc, input, weights_desc, weights,
      conv_desc, algo, &beta, output_desc, output, workspace, workspace_size));
  // Add per-channel bias via miopenOpTensor (TensorOpAdd):
  //   C = alpha1*A + alpha2*B + beta*C  with A=C=output, B=bias, beta=0
  //   -> output = output + bias  (broadcast from [1, weights_k, 1, 1])
  //
  // We do NOT use miopenConvolutionForwardBias: the MIOpen header states its
  // alpha/beta are "only supported for alpha = 1 and beta = 0", so it cannot
  // fuse y = conv + bias (it would compute y = bias). miopenOpTensor with
  // alpha1=alpha2=1, beta=0 computes y = conv + bias as ONNX Conv requires.
  if (bias) {
    const float alpha_bias = 1.0f, beta_zero = 0.0f;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
    int b_dims[] = {1, (int)weights_k, 1, 1};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        bias_desc, miopen_dt, miopenTensorNCHW, b_dims, 4));
    MIOPEN_CHECK(miopenOpTensor(miopen_handle, miopenTensorOpAdd, &alpha_bias,
                                output_desc, output, &alpha_bias, bias_desc,
                                bias, &beta_zero, output_desc, output));
  }

cleanup:
  // Workspace is owned by RuntimeState->conv_scratch; do NOT hipFree here.
  // Destroy all per-call descriptors so MIOpen releases the per-problem
  // GPU-resident state attached to them (the whole point of NOT caching them --
  // see cache header). Only the (algo, workspace_size) pair persists, in
  // RuntimeState->conv_fwd_cache.
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
  return result;
}

//===----------------------------------------------------------------------===//
// MIOpen Transposed Convolution (Deconvolution)
//===----------------------------------------------------------------------===//
//
// Implements ONNX ConvTranspose via MIOpen's miopenTranspose convolution mode.
// In transpose mode, miopenConvolutionForward computes the deconvolution: the
// descriptors mirror the "equivalent forward conv" whose forward output is our
// input. That makes the weight descriptor [C_in, M/group, kH, kW] match the
// ONNX ConvTranspose weight layout directly (input channels first, unlike the
// forward Conv layout [M, C/group, ...]).
//
// output_padding (ONNX "adjs") disambiguates the output size when a stride > 1
// maps multiple input sizes to overlapping output ranges; it is applied via
// miopenSetTransposeConvOutputPadding. The final output dims are already fixed
// by the caller (output_h/output_w from compile-time shape inference); the
// output padding only affects which computed cells MIOpen fills.
int wrap_miopenConvolutionTranspose(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, const void *bias,
    void *output, int64_t output_c, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t output_padding_h,
    int64_t output_padding_w, int64_t group, int64_t data_type) {
  OP_PROFILE(
      "conv_transpose",
      [&] {
        char b[80];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld,m=%lld,k=%lldx%lld,s=%lld",
                 (long long)input_n, (long long)input_c, (long long)input_h,
                 (long long)input_w, (long long)output_c, (long long)kernel_h,
                 (long long)kernel_w, (long long)stride_h);
        return std::string(b);
      },
      state);
  (void)pad_bottom;
  (void)pad_right;
  if (!state || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionTranspose\n");
    return -1;
  }

  miopenDataType_t miopen_dt;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    miopen_dt = miopenFloat;
    break;
  case HIPDNN_EP_DATATYPE_HALF:
    miopen_dt = miopenHalf;
    break;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    miopen_dt = miopenBFloat16;
    break;
  default:
    fprintf(stderr,
            "wrap_miopenConvolutionTranspose: unsupported data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));

  // All releasable resources declared at function scope for goto cleanup.
  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t weights_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  miopenTensorDescriptor_t bias_desc = nullptr;
  miopenConvolutionDescriptor_t conv_desc = nullptr;
  void *find_workspace = nullptr;
  void *workspace = nullptr;
  int result = 0;
  miopenConvAlgoPerf_t perf_results[1];
  int returned_algo_count = 0;
  miopenConvFwdAlgorithm_t algo;
  size_t workspace_size = 0;
  const size_t find_workspace_size = 10 * 1024 * 1024; // 10MB
  float alpha = 1.0f;
  float beta = 0.0f;

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&output_desc));

  {
    int in_dims[] = {(int)input_n, (int)input_c, (int)input_h, (int)input_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        input_desc, miopen_dt, miopenTensorNCHW, in_dims, 4));

    // Transpose-mode weights: [C_in, M/group, kH, kW] (ONNX ConvTranspose W).
    int w_dims[] = {(int)input_c, (int)(output_c / group), (int)kernel_h,
                    (int)kernel_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        weights_desc, miopen_dt, miopenTensorNCHW, w_dims, 4));

    int out_dims[] = {(int)input_n, (int)output_c, (int)output_h,
                      (int)output_w};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        output_desc, miopen_dt, miopenTensorNCHW, out_dims, 4));
  }

  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(
      conv_desc, miopenTranspose, pad_top, pad_left, stride_h, stride_w,
      dilation_h, dilation_w));

  // Output padding is optional in MIOpen; only set it when non-zero so the
  // common case takes the default (zero adjustment) path.
  if (output_padding_h > 0 || output_padding_w > 0) {
    MIOPEN_CHECK(miopenSetTransposeConvOutputPadding(
        conv_desc, (int)output_padding_h, (int)output_padding_w));
  }

  if (group > 1) {
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc, group));
  }

  HIP_CHECK(hipMalloc(&find_workspace, find_workspace_size));

  MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
      miopen_handle, input_desc, input, weights_desc, weights, conv_desc,
      output_desc, output, 1, &returned_algo_count, perf_results,
      find_workspace, find_workspace_size, false));

  algo = perf_results[0].fwd_algo;

  MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
      miopen_handle, weights_desc, input_desc, conv_desc, output_desc,
      &workspace_size));

  workspace = find_workspace;
  if (workspace_size > find_workspace_size) {
    hipError_t err = hipFree(find_workspace);
    if (err != hipSuccess)
      fprintf(stderr, "Warning: hipFree failed for find_workspace: %d\n", err);
    find_workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, workspace_size));
  }

  MIOPEN_CHECK(miopenConvolutionForward(
      miopen_handle, &alpha, input_desc, input, weights_desc, weights,
      conv_desc, algo, &beta, output_desc, output, workspace, workspace_size));

  // Bias is [M]; broadcast-add over the [N, M, H', W'] output. Use
  // miopenOpTensor (the same op the forward conv uses) rather than
  // miopenConvolutionForwardBias: the latter is observed to double the
  // deconvolution result here (output came out ~2x). miopenOpTensor with
  // beta=0, alpha1=alpha2=1 computes C = A + B in-place (A == C), adding the
  // [1,M,1,1] bias broadcast over [N,M,H',W'] without re-touching the conv.
  if (bias) {
    int b_dims[] = {1, (int)output_c, 1, 1};
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        bias_desc, miopen_dt, miopenTensorNCHW, b_dims, 4));
    // miopenOpTensor computes C = alpha1*A + alpha2*B + beta*C. With
    // alpha1=alpha2=1, beta=0 and A==C==output, B==bias this is
    // output = output + bias (in place); A==C is required by MIOpen.
    const float alpha1 = 1.0f, alpha2 = 1.0f, beta_zero = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(miopen_handle, miopenTensorOpAdd, &alpha1,
                                output_desc, output, &alpha2, bias_desc, bias,
                                &beta_zero, output_desc, output));
  }

cleanup:
  if (workspace) {
    hipError_t err = hipFree(workspace);
    if (err != hipSuccess)
      fprintf(stderr, "Warning: hipFree failed for workspace: %d\n", err);
  }
  if (find_workspace && find_workspace != workspace) {
    hipError_t err = hipFree(find_workspace);
    if (err != hipSuccess)
      fprintf(stderr, "Warning: hipFree failed for find_workspace: %d\n", err);
  }
  if (input_desc)
    miopenDestroyTensorDescriptor(input_desc);
  if (weights_desc)
    miopenDestroyTensorDescriptor(weights_desc);
  if (output_desc)
    miopenDestroyTensorDescriptor(output_desc);
  if (bias_desc)
    miopenDestroyTensorDescriptor(bias_desc);
  if (conv_desc)
    miopenDestroyConvolutionDescriptor(conv_desc);

  return result;
}
