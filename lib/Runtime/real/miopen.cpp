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
#include <mutex>
#include <unordered_map>

// Convenience wrappers for goto cleanup pattern (all functions use 'cleanup'
// label)
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)
#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// MIOpen Convolution Wrapper — ConvTranspose only
//===----------------------------------------------------------------------===//
//
// Forward Conv does not live here. It goes through the in-tree `hip_conv`
// kernel via `wrap_conv` (lib/Runtime/real/conv.cpp), so nothing on the
// forward path calls MIOpen. ConvTranspose stays on MIOpen: no supported
// model uses it, so it is not worth a hand-written kernel.
//
// The descriptor/solution cache below (`ConvTable`, `queryOrCreateConv`) is
// therefore reached only from `wrap_miopenConvolutionTranspose`. It is still
// keyed on the full forward-conv problem — including `conv_mode` and the
// transpose-only output padding — because in transpose mode MIOpen describes
// the deconvolution as the equivalent forward conv whose output is our input.
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

struct ConvState : public OpStateT<ConvState> {
  std::shared_ptr<ConvTable> table;
  ConvState() {
    int dev = 0;
    hipGetDevice(&dev);
    table = WeakStore<int, ConvTable>::get_or_create(
        dev, [] { return std::make_shared<ConvTable>(); });
  }
};

extern "C" int8_t hipdnn_ep_op_state_construct_conv(RuntimeState *state,
                                                    int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, ConvState::create().release());
  return 0;
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
