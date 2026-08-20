/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

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
//===----------------------------------------------------------------------===//

// TODO: Pool workspace memory instead of malloc/free every call
//
// RATIONALE: GPU memory allocation (hipMalloc) is expensive - involves kernel
// launch, synchronization, and memory manager overhead. Current code allocates
// and frees workspace on every inference call (lines 95, 107).
//
// IMPLEMENTATION: Add WorkspacePool to RuntimeState that:
//   - Pre-allocates workspace of maximum required size
//   - Reuses across multiple calls
//   - Grows dynamically if larger workspace needed
//
// BENEFIT: Eliminates malloc/free from hot path.

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

struct ConvTableKey {
  int64_t input_n;
  int64_t input_c;
  int64_t input_h;
  int64_t input_w;
  int64_t weights_k;
  int64_t output_h;
  int64_t output_w;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t pad_top;
  int64_t pad_left;
  int64_t pad_bottom;
  int64_t pad_right;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t group;
  miopenDataType_t data_type;
  miopenConvolutionMode_t conv_mode;
  int64_t output_padding_h;
  int64_t output_padding_w;

  bool operator==(const ConvTableKey &other) const {
    return (input_n == other.input_n) && (input_c == other.input_c) &&
           (input_h == other.input_h) && (input_w == other.input_w) &&
           (weights_k == other.weights_k) && (output_h == other.output_h) &&
           (output_w == other.output_w) && (kernel_h == other.kernel_h) &&
           (kernel_w == other.kernel_w) && (stride_h == other.stride_h) &&
           (stride_w == other.stride_w) && (pad_top == other.pad_top) &&
           (pad_left == other.pad_left) && (pad_bottom == other.pad_bottom) &&
           (pad_right == other.pad_right) && (dilation_h == other.dilation_h) &&
           (dilation_w == other.dilation_w) && (group == other.group) &&
           (data_type == other.data_type) && (conv_mode == other.conv_mode) &&
           (output_padding_h == other.output_padding_h) &&
           (output_padding_w == other.output_padding_w);
  }
};

struct ConvTableEntry {
  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t weights_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  miopenConvolutionDescriptor_t conv_desc = nullptr;
  miopenProblem_t problem = nullptr;
  miopenSolution_t solution = nullptr;
  void *workspace = nullptr;
  size_t workspaceSize = 0;

  ConvTableEntry() = default;

  // Disable copy
  ConvTableEntry(const ConvTableEntry &other) = delete;
  ConvTableEntry &operator=(const ConvTableEntry &other) = delete;

  // Enable move
  ConvTableEntry(ConvTableEntry &&other)
      : input_desc(other.input_desc), weights_desc(other.weights_desc),
        output_desc(other.output_desc), conv_desc(other.conv_desc),
        problem(other.problem), solution(other.solution),
        workspace(other.workspace), workspaceSize(other.workspaceSize) {
    other.input_desc = nullptr;
    other.weights_desc = nullptr;
    other.output_desc = nullptr;
    other.conv_desc = nullptr;
    other.problem = nullptr;
    other.solution = nullptr;
    other.workspace = nullptr;
    other.workspaceSize = 0;
  }
  ConvTableEntry &operator=(ConvTableEntry &&other) = delete;

  ~ConvTableEntry() {
    if (input_desc)
      miopenDestroyTensorDescriptor(input_desc);
    if (weights_desc)
      miopenDestroyTensorDescriptor(weights_desc);
    if (output_desc)
      miopenDestroyTensorDescriptor(output_desc);
    if (conv_desc)
      miopenDestroyConvolutionDescriptor(conv_desc);
    if (problem)
      miopenDestroyProblem(problem);
    if (solution)
      miopenDestroySolution(solution);
    if (workspace)
      hipFree(workspace);
  }
};

struct ConvKeyHash {
  size_t operator()(const ConvTableKey &key) const {
    size_t h = 0;
    hash_combine_val(h, key.input_n);
    hash_combine_val(h, key.input_c);
    hash_combine_val(h, key.input_h);
    hash_combine_val(h, key.input_w);
    hash_combine_val(h, key.weights_k);
    hash_combine_val(h, key.output_h);
    hash_combine_val(h, key.output_w);
    hash_combine_val(h, key.kernel_h);
    hash_combine_val(h, key.kernel_w);
    hash_combine_val(h, key.stride_h);
    hash_combine_val(h, key.stride_w);
    hash_combine_val(h, key.pad_top);
    hash_combine_val(h, key.pad_left);
    hash_combine_val(h, key.pad_bottom);
    hash_combine_val(h, key.pad_right);
    hash_combine_val(h, key.dilation_h);
    hash_combine_val(h, key.dilation_w);
    hash_combine_val(h, key.group);
    hash_combine_val(h, key.data_type);
    hash_combine_val(h, key.conv_mode);
    hash_combine_val(h, key.output_padding_h);
    hash_combine_val(h, key.output_padding_w);
    return h;
  }
};

struct ConvTable {
  std::mutex mutex;
  std::unordered_map<ConvTableKey, ConvTableEntry, ConvKeyHash> map;
};

struct ConvFusionTableEntry {
  miopenFusionPlanDescriptor_t fuse_plan = nullptr;
  miopenFusionOpDescriptor_t conv_op = nullptr;
  miopenFusionOpDescriptor_t bias_op = nullptr;
  miopenFusionOpDescriptor_t activ_op = nullptr;
  bool has_bias = false;
  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t weights_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  miopenTensorDescriptor_t bias_desc = nullptr;
  miopenConvolutionDescriptor_t conv_desc = nullptr;
  miopenConvFwdAlgorithm_t conv_algo = miopenConvolutionFwdAlgoDirect;
  void *fusion_workspace = nullptr;
  size_t fusion_workspace_size = 0;

  ConvFusionTableEntry() = default;
  ConvFusionTableEntry(const ConvFusionTableEntry &) = delete;
  ConvFusionTableEntry &operator=(const ConvFusionTableEntry &) = delete;

  ConvFusionTableEntry(ConvFusionTableEntry &&other)
      : fuse_plan(other.fuse_plan), conv_op(other.conv_op),
        bias_op(other.bias_op), activ_op(other.activ_op),
        has_bias(other.has_bias), input_desc(other.input_desc),
        weights_desc(other.weights_desc), output_desc(other.output_desc),
        bias_desc(other.bias_desc), conv_desc(other.conv_desc),
        conv_algo(other.conv_algo), fusion_workspace(other.fusion_workspace),
        fusion_workspace_size(other.fusion_workspace_size) {
    other.fuse_plan = nullptr;
    other.conv_op = nullptr;
    other.bias_op = nullptr;
    other.activ_op = nullptr;
    other.input_desc = nullptr;
    other.weights_desc = nullptr;
    other.output_desc = nullptr;
    other.bias_desc = nullptr;
    other.conv_desc = nullptr;
    other.fusion_workspace = nullptr;
    other.fusion_workspace_size = 0;
  }

  ~ConvFusionTableEntry() {
    if (fuse_plan)
      miopenDestroyFusionPlan(fuse_plan);
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
    if (fusion_workspace)
      hipFree(fusion_workspace);
  }
};

struct ConvFusionTable {
  std::mutex mutex;
  std::unordered_map<ConvTableKey, ConvFusionTableEntry, ConvKeyHash> map;
  // Keys whose FusionPlan compile/execute failed; skip on later inferences.
  std::unordered_set<ConvTableKey, ConvKeyHash> disabled_keys;
};

struct ClippedRelu6ActivCache {
  miopenActivationDescriptor_t desc = nullptr;

  ClippedRelu6ActivCache() {
    if (miopenCreateActivationDescriptor(&desc) != miopenStatusSuccess)
      desc = nullptr;
    else if (miopenSetActivationDescriptor(desc, miopenActivationCLIPPEDRELU,
                                           6.0, 0.0,
                                           0.0) != miopenStatusSuccess) {
      miopenDestroyActivationDescriptor(desc);
      desc = nullptr;
    }
  }

  ~ClippedRelu6ActivCache() {
    if (desc)
      miopenDestroyActivationDescriptor(desc);
  }
};

struct ConvState : public OpStateT<ConvState> {
  std::shared_ptr<ConvTable> table;
  std::shared_ptr<ConvFusionTable> fusion_table;
  std::shared_ptr<ClippedRelu6ActivCache> relu6_activ;
  ConvState() {
    int dev = 0;
    hipGetDevice(&dev);
    table = WeakStore<int, ConvTable>::get_or_create(
        dev, [] { return std::make_shared<ConvTable>(); });
    fusion_table = WeakStore<int, ConvFusionTable>::get_or_create(
        dev, [] { return std::make_shared<ConvFusionTable>(); });
    relu6_activ = WeakStore<int, ClippedRelu6ActivCache>::get_or_create(
        0, [] { return std::make_shared<ClippedRelu6ActivCache>(); });
  }
};

extern "C" int8_t hipdnn_ep_op_state_construct_conv(RuntimeState *state,
                                                    int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, ConvState::create().release());
  return 0;
}

//===----------------------------------------------------------------------===//
// MIOpen FusionPlan cache (Conv + Bias + ClippedReLU for ReLU6)
//===----------------------------------------------------------------------===//

static int applyClippedRelu6Fallback(ConvState *os, miopenHandle_t handle,
                                     miopenTensorDescriptor_t desc,
                                     void *data) {
  if (!os || !os->relu6_activ || !os->relu6_activ->desc)
    return -1;
  const float alpha = 1.f, beta = 0.f;
  if (miopenActivationForward(handle, os->relu6_activ->desc, &alpha, desc, data,
                              &beta, desc, data) != miopenStatusSuccess)
    return -1;
  return 0;
}

static void disableConvFusion(ConvFusionTable &table, const ConvTableKey &key) {
  table.disabled_keys.insert(key);
}

// MIOpen Conv+Bias+Act FusionPlan is unreliable on gfx115x (Strix Halo).
// Default: skip FusionPlan there and use Find API conv + ClippedReLU instead.
// Override: HIPDNN_EP_CONV_FUSION=1 force enable, =0 force disable.
static bool convFusionPlanEnabled() {
  char buf[8];
  unsigned long n =
      hipdnn_ep::read_env("HIPDNN_EP_CONV_FUSION", buf, sizeof(buf));
  if (n > 0) {
    if (buf[0] == '0')
      return false;
    if (buf[0] >= '1')
      return true;
  }

  static int cached = -1; // 0 = disabled, 1 = enabled
  if (cached >= 0)
    return cached != 0;

  hipDeviceProp_t prop{};
  int dev = 0;
  hipGetDevice(&dev);
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess ||
      prop.gcnArchName[0] == '\0') {
    cached = 1;
    return true;
  }
  cached = (strncmp(prop.gcnArchName, "gfx115", 6) == 0) ? 0 : 1;
  return cached != 0;
}

static void logConvFusionPlanSkipOnce() {
  static bool logged = false;
  if (logged)
    return;
  logged = true;
  RUNTIME_DEBUG_LOG(
      "[CONV] FusionPlan skipped (gfx115x or HIPDNN_EP_CONV_FUSION=0), "
      "using Find+activ for ReLU6\n");
}

static const ConvFusionTableEntry *
queryOrCreateConvFusion(ConvFusionTable &table, const ConvTableKey &key,
                        miopenHandle_t miopen_handle, bool has_bias,
                        int64_t weights_k) {
  std::lock_guard<std::mutex> guard(table.mutex);
  auto it = table.map.find(key);
  if (it != table.map.end())
    return &it->second;

  int result = 0;
  ConvFusionTableEntry entry;

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&entry.input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&entry.weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&entry.output_desc));

  {
    int in_dims[] = {
        static_cast<int>(key.input_n), static_cast<int>(key.input_c),
        static_cast<int>(key.input_h), static_cast<int>(key.input_w)};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        entry.input_desc, key.data_type, miopenTensorNCHW, in_dims, 4));

    int w_dims[] = {
        static_cast<int>(key.weights_k),
        static_cast<int>(key.input_c / key.group),
        static_cast<int>(key.kernel_h),
        static_cast<int>(key.kernel_w),
    };
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        entry.weights_desc, key.data_type, miopenTensorNCHW, w_dims, 4));

    int out_dims[] = {
        static_cast<int>(key.input_n), static_cast<int>(key.weights_k),
        static_cast<int>(key.output_h), static_cast<int>(key.output_w)};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        entry.output_desc, key.data_type, miopenTensorNCHW, out_dims, 4));
  }

  if (has_bias) {
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&entry.bias_desc));
    int b_dims[] = {1, static_cast<int>(weights_k), 1, 1};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        entry.bias_desc, key.data_type, miopenTensorNCHW, b_dims, 4));
    entry.has_bias = true;
  }

  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&entry.conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(
      entry.conv_desc, miopenConvolution, key.pad_top, key.pad_left,
      key.stride_h, key.stride_w, key.dilation_h, key.dilation_w));
  if (key.group > 1)
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(entry.conv_desc, key.group));

  MIOPEN_CHECK(miopenCreateFusionPlan(&entry.fuse_plan, miopenVerticalFusion,
                                      entry.input_desc));
  MIOPEN_CHECK(miopenCreateOpConvForward(entry.fuse_plan, &entry.conv_op,
                                         entry.conv_desc, entry.weights_desc));
  if (has_bias) {
    MIOPEN_CHECK(miopenCreateOpBiasForward(entry.fuse_plan, &entry.bias_op,
                                           entry.bias_desc));
  }
  MIOPEN_CHECK(miopenCreateOpActivationForward(entry.fuse_plan, &entry.activ_op,
                                               miopenActivationCLIPPEDRELU));

  {
    int returned = 0;
    miopenConvFwdAlgorithm_t algos[8] = {};
    if (miopenFusionPlanConvolutionGetAlgo(entry.fuse_plan, 8, &returned,
                                           algos) == miopenStatusSuccess &&
        returned > 0) {
      entry.conv_algo = algos[0];
    }
  }

  if (miopenCompileFusionPlan(miopen_handle, entry.fuse_plan) !=
      miopenStatusSuccess) {
    goto cleanup;
  }

  MIOPEN_CHECK(miopenFusionPlanGetWorkSpaceSize(miopen_handle, entry.fuse_plan,
                                                &entry.fusion_workspace_size,
                                                entry.conv_algo));
  if (entry.fusion_workspace_size > 0)
    HIP_CHECK(hipMalloc(&entry.fusion_workspace, entry.fusion_workspace_size));

  {
    auto [ins, _] = table.map.emplace(key, std::move(entry));
    return &ins->second;
  }

cleanup:
  return nullptr;
}

static int runConvFusionPlan(const ConvFusionTableEntry *entry,
                             miopenHandle_t miopen_handle, const void *input,
                             const void *weights, const void *bias,
                             void *output) {
  miopenOperatorArgs_t args = nullptr;
  const float alpha = 1.f, beta = 0.f;
  int result = 0;

  MIOPEN_CHECK(miopenCreateOperatorArgs(&args));
  MIOPEN_CHECK(
      miopenSetOpArgsConvForward(args, entry->conv_op, &alpha, &beta, weights));
  if (entry->has_bias && bias) {
    MIOPEN_CHECK(
        miopenSetOpArgsBiasForward(args, entry->bias_op, &alpha, &beta, bias));
  }
  MIOPEN_CHECK(miopenSetOpArgsActivForward(args, entry->activ_op, &alpha, &beta,
                                           6.0, 0.0, 0.0));
  MIOPEN_CHECK(miopenExecuteFusionPlan_v2(
      miopen_handle, entry->fuse_plan, entry->input_desc, input,
      entry->output_desc, output, args, entry->fusion_workspace,
      entry->fusion_workspace_size));
  MIOPEN_CHECK(miopenDestroyOperatorArgs(args));
  return 0;

cleanup:
  if (args)
    miopenDestroyOperatorArgs(args);
  return result;
}

static const ConvTableEntry *
queryOrCreateConv(ConvTable &table, const ConvTableKey &key,
                  miopenHandle_t miopen_handle, const void *input,
                  const void *weights, void *output) {
  std::lock_guard<std::mutex> guard(table.mutex);
  auto it = table.map.find(key);
  if (it != table.map.end())
    return &it->second;

  int result = 0;

  ConvTableEntry entry;
  miopenFindOptions_t options;
  size_t foundSolutions;

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&entry.input_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&entry.weights_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&entry.output_desc));

  // Set tensor descriptors with explicit NCHW layout. The dtype is taken
  {
    int in_dims[] = {
        static_cast<int>(key.input_n),
        static_cast<int>(key.input_c),
        static_cast<int>(key.input_h),
        static_cast<int>(key.input_w),
    };
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        entry.input_desc, key.data_type, miopenTensorNCHW, in_dims, 4));

    int w_dims[] = {
        static_cast<int>(key.conv_mode == miopenConvolution ? key.weights_k
                                                            : key.input_c),
        static_cast<int>(
            (key.conv_mode == miopenConvolution ? key.input_c : key.weights_k) /
            key.group),
        static_cast<int>(key.kernel_h),
        static_cast<int>(key.kernel_w),
    };
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        entry.weights_desc, key.data_type, miopenTensorNCHW, w_dims, 4));

    int out_dims[] = {
        static_cast<int>(key.input_n),
        static_cast<int>(key.weights_k),
        static_cast<int>(key.output_h),
        static_cast<int>(key.output_w),
    };
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        entry.output_desc, key.data_type, miopenTensorNCHW, out_dims, 4));
  }

  // Create convolution descriptor
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&entry.conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(
      entry.conv_desc, key.conv_mode, key.pad_top, key.pad_left, key.stride_h,
      key.stride_w, key.dilation_h, key.dilation_w));
  if (key.group > 1) {
    MIOPEN_CHECK(miopenSetConvolutionGroupCount(entry.conv_desc, key.group));
  }
  if (key.output_padding_h > 0 || key.output_padding_w > 0) {
    MIOPEN_CHECK(miopenSetTransposeConvOutputPadding(
        entry.conv_desc, static_cast<int>(key.output_padding_h),
        static_cast<int>(key.output_padding_w)));
  }

  // Find API
  MIOPEN_CHECK(miopenCreateConvProblem(&entry.problem, entry.conv_desc,
                                       miopenProblemDirectionForward));
  MIOPEN_CHECK(miopenSetProblemTensorDescriptor(
      entry.problem, miopenTensorConvolutionX, entry.input_desc));
  MIOPEN_CHECK(miopenSetProblemTensorDescriptor(
      entry.problem, miopenTensorConvolutionW, entry.weights_desc));
  MIOPEN_CHECK(miopenSetProblemTensorDescriptor(
      entry.problem, miopenTensorConvolutionY, entry.output_desc));

  MIOPEN_CHECK(miopenCreateFindOptions(&options));
  MIOPEN_CHECK(miopenSetFindOptionTuning(options, 1));
  MIOPEN_CHECK(miopenSetFindOptionAttachBinaries(options, 1));

  MIOPEN_CHECK(miopenSetFindOptionPreallocatedTensor(
      options, miopenTensorConvolutionX, const_cast<void *>(input)));
  MIOPEN_CHECK(miopenSetFindOptionPreallocatedTensor(
      options, miopenTensorConvolutionW, const_cast<void *>(weights)));
  MIOPEN_CHECK(miopenSetFindOptionPreallocatedTensor(
      options, miopenTensorConvolutionY, output));

  MIOPEN_CHECK(miopenFindSolutions(miopen_handle, entry.problem, options,
                                   &entry.solution, &foundSolutions, 1));
  if (foundSolutions == 0 || entry.solution == nullptr) {
    return nullptr;
  }

  MIOPEN_CHECK(
      miopenGetSolutionWorkspaceSize(entry.solution, &entry.workspaceSize));
  if (entry.workspaceSize > 0)
    HIP_CHECK(hipMalloc(&entry.workspace, entry.workspaceSize));

  MIOPEN_CHECK(miopenDestroyFindOptions(options));

  {
    auto [ins, _] = table.map.emplace(key, std::move(entry));
    return &ins->second;
  }

cleanup:
  return nullptr;
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
int wrap_miopenConvolutionForward(
    RuntimeState *state, int32_t op_state_slot, const void *input,
    int64_t input_n, int64_t input_c, int64_t input_h, int64_t input_w,
    const void *weights, int64_t weights_k, const void *bias, void *output,
    int64_t output_h, int64_t output_w, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_top, int64_t pad_left,
    int64_t pad_bottom, int64_t pad_right, int64_t dilation_h,
    int64_t dilation_w, int64_t group, int64_t data_type, int64_t activation) {
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
      "Cout=%lld kHxkW=%lldx%lld s=%lldx%lld bias=%s dtype=%lld "
      "activation=%lld\n",
      (long long)input_n, (long long)input_c, (long long)input_h,
      (long long)input_w, (long long)weights_k, (long long)kernel_h,
      (long long)kernel_w, (long long)stride_h, (long long)stride_w,
      bias ? "yes" : "null", (long long)data_type, (long long)activation);

  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));

  ConvState *os = ConvState::get_op_state(state, op_state_slot);
  if (!os || !os->table) {
    fprintf(stderr,
            "wrap_miopenConvolutionForward: missing op-state for slot %d\n",
            op_state_slot);
    return -1;
  }

  ConvTableKey key{
      input_n,   input_c,           input_h,   input_w,    weights_k,  output_h,
      output_w,  kernel_h,          kernel_w,  stride_h,   stride_w,   pad_top,
      pad_left,  pad_bottom,        pad_right, dilation_h, dilation_w, group,
      miopen_dt, miopenConvolution, 0,         0,
  };

  if (activation == HIPDNN_EP_CONV_ACTIVATION_RELU6) {
    if (!convFusionPlanEnabled()) {
      logConvFusionPlanSkipOnce();
    } else if (os->fusion_table) {
      bool fusion_disabled = false;
      {
        std::lock_guard<std::mutex> guard(os->fusion_table->mutex);
        fusion_disabled = os->fusion_table->disabled_keys.count(key) != 0;
      }
      if (!fusion_disabled) {
        const ConvFusionTableEntry *fusion_entry = queryOrCreateConvFusion(
            *os->fusion_table, key, miopen_handle, bias != nullptr, weights_k);
        if (fusion_entry && fusion_entry->fuse_plan) {
          int fusion_rc = runConvFusionPlan(fusion_entry, miopen_handle, input,
                                            weights, bias, output);
          if (fusion_rc == 0) {
            RUNTIME_DEBUG_LOG("[CONV] FusionPlan ReLU6 ok N=%lld Cout=%lld\n",
                              (long long)input_n, (long long)weights_k);
            return 0;
          }
          RUNTIME_DEBUG_LOG("[CONV] FusionPlan ReLU6 failed (rc=%d), falling "
                            "back to Find+activ\n",
                            fusion_rc);
          std::lock_guard<std::mutex> guard(os->fusion_table->mutex);
          disableConvFusion(*os->fusion_table, key);
        } else {
          RUNTIME_DEBUG_LOG(
              "[CONV] FusionPlan compile unsupported, fallback\n");
          std::lock_guard<std::mutex> guard(os->fusion_table->mutex);
          disableConvFusion(*os->fusion_table, key);
        }
      }
    }
  }

  miopenTensorDescriptor_t bias_desc = nullptr;
  int result = 0;

  const ConvTableEntry *entry =
      queryOrCreateConv(*os->table, key, miopen_handle, input, weights, output);
  if (!entry || !entry->solution) {
    fprintf(stderr,
            "wrap_miopenConvolutionForward: missing solution for problem\n");
    return -1;
  }

  const miopenTensorArgument_t tensorArgs[3] = {
      {miopenTensorConvolutionX,
       const_cast<miopenTensorDescriptor_t *>(&entry->input_desc),
       const_cast<void *>(input)},
      {miopenTensorConvolutionW,
       const_cast<miopenTensorDescriptor_t *>(&entry->weights_desc),
       const_cast<void *>(weights)},
      {miopenTensorConvolutionY,
       const_cast<miopenTensorDescriptor_t *>(&entry->output_desc), output},
  };
  MIOPEN_CHECK(miopenRunSolution(miopen_handle, entry->solution, 3, tensorArgs,
                                 entry->workspace, entry->workspaceSize));

  if (bias) {
    const float alpha_bias = 1.0f, beta_zero = 0.0f;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
    int b_dims[] = {1, (int)weights_k, 1, 1};
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        bias_desc, miopen_dt, miopenTensorNCHW, b_dims, 4));
    MIOPEN_CHECK(miopenOpTensor(miopen_handle, miopenTensorOpAdd, &alpha_bias,
                                entry->output_desc, output, &alpha_bias,
                                bias_desc, bias, &beta_zero, entry->output_desc,
                                output));
  }

  if (activation == HIPDNN_EP_CONV_ACTIVATION_RELU6) {
    if (applyClippedRelu6Fallback(os, miopen_handle, entry->output_desc,
                                  output) != 0) {
      result = -1;
      goto cleanup;
    }
  }

cleanup:
  if (bias_desc) {
    miopenDestroyTensorDescriptor(bias_desc);
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
    RuntimeState *state, int32_t op_state_slot, const void *input,
    int64_t input_n, int64_t input_c, int64_t input_h, int64_t input_w,
    const void *weights, const void *bias, void *output, int64_t output_c,
    int64_t output_h, int64_t output_w, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w, int64_t pad_top, int64_t pad_left,
    int64_t pad_bottom, int64_t pad_right, int64_t dilation_h,
    int64_t dilation_w, int64_t output_padding_h, int64_t output_padding_w,
    int64_t group, int64_t data_type) {
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

  bool dt_ok;
  miopenDataType_t miopen_dt = conv_to_miopen_type(data_type, dt_ok);
  if (!dt_ok) {
    fprintf(
        stderr,
        "[REAL] wrap_miopenConvolutionTranspose: unsupported data_type %lld\n",
        (long long)data_type);
    return -1;
  }

  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));

  ConvState *os = ConvState::get_op_state(state, op_state_slot);
  if (!os || !os->table) {
    fprintf(stderr,
            "wrap_miopenConvolutionTranspose: missing op-state for slot %d\n",
            op_state_slot);
    return -1;
  }

  ConvTableKey key{
      input_n,          input_c,          input_h,  input_w,    output_c,
      output_h,         output_w,         kernel_h, kernel_w,   stride_h,
      stride_w,         pad_top,          pad_left, pad_bottom, pad_right,
      dilation_h,       dilation_w,       group,    miopen_dt,  miopenTranspose,
      output_padding_h, output_padding_w,
  };
  miopenTensorDescriptor_t bias_desc = nullptr;
  int result = 0;

  const ConvTableEntry *entry =
      queryOrCreateConv(*os->table, key, miopen_handle, input, weights, output);
  if (!entry || !entry->solution) {
    fprintf(stderr,
            "wrap_miopenConvolutionTranspose: missing solution for problem\n");
    return -1;
  }

  const miopenTensorArgument_t tensorArgs[3] = {
      {miopenTensorConvolutionX,
       const_cast<miopenTensorDescriptor_t *>(&entry->input_desc),
       const_cast<void *>(input)},
      {miopenTensorConvolutionW,
       const_cast<miopenTensorDescriptor_t *>(&entry->weights_desc),
       const_cast<void *>(weights)},
      {miopenTensorConvolutionY,
       const_cast<miopenTensorDescriptor_t *>(&entry->output_desc), output},
  };
  MIOPEN_CHECK(miopenRunSolution(miopen_handle, entry->solution, 3, tensorArgs,
                                 entry->workspace, entry->workspaceSize));

  // Bias is [M]; broadcast-add over the [N, M, H', W'] output. Use
  // miopenOpTensor (the same op the forward conv uses) rather than
  // miopenConvolutionForwardBias: the latter is observed to double the
  // deconvolution result here (output came out ~2x). miopenOpTensor with
  // beta=0, alpha1=alpha2=1 computes C = A + B in-place (A == C), adding the
  // [1,M,1,1] bias broadcast over [N,M,H',W'] without re-touching the conv.
  if (bias) {
    float alpha = 1.0f;
    float beta = 0.0f;
    int b_dims[] = {1, (int)output_c, 1, 1};
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
    MIOPEN_CHECK(miopenSetNdTensorDescriptorWithLayout(
        bias_desc, miopen_dt, miopenTensorNCHW, b_dims, 4));
    // miopenOpTensor computes C = alpha1*A + alpha2*B + beta*C. With
    // alpha1=alpha2=1, beta=0 and A==C==output, B==bias this is
    // output = output + bias (in place); A==C is required by MIOpen.
    const float alpha1 = 1.0f, alpha2 = 1.0f, beta_zero = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(miopen_handle, miopenTensorOpAdd, &alpha1,
                                entry->output_desc, output, &alpha2, bias_desc,
                                bias, &beta_zero, entry->output_desc, output));
  }

cleanup:
  if (bias_desc)
    miopenDestroyTensorDescriptor(bias_desc);

  return result;
}
