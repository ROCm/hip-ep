/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
// On Windows with static CRT, each DLL has its own stdout
// Use OutputDebugString so output appears in DebugView/debugger
// and fprintf(stderr) to try to reach the parent process
#define MOCK_PRINT(...)                                                        \
  do {                                                                         \
    char buf[512];                                                             \
    snprintf(buf, sizeof(buf), __VA_ARGS__);                                   \
    OutputDebugStringA(buf);                                                   \
    fprintf(stderr, "%s", buf);                                                \
    fflush(stderr);                                                            \
  } while (0)
#else
#define MOCK_PRINT(...)                                                        \
  do {                                                                         \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  } while (0)
#endif

// Mock type definitions are now in mock_types.h (included via runtime_types.h)
// No need to redefine them here

// Comprehensive mock implementations for all GPU functions
// Prints all operations for debugging and verification

// Mock HIP device functions
extern "C" hipError_t hipGetDeviceCount(int *count) {
  MOCK_PRINT("[MOCK] hipGetDeviceCount\n");
  *count = 1; // Pretend we have one device
  return hipSuccess;
}

extern "C" hipError_t hipSetDevice(int device) {
  MOCK_PRINT("[MOCK] hipSetDevice(%d)\n", device);
  return hipSuccess;
}

extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop,
                                             int device) {
  MOCK_PRINT("[MOCK] hipGetDeviceProperties(device=%d)\n", device);
  if (prop) {
    strncpy(prop->name, "Mock GPU Device", sizeof(prop->name));
    strncpy(prop->gcnArchName, "gfx1100", sizeof(prop->gcnArchName));
    prop->integrated = 0;
  }
  return hipSuccess;
}

// Mock HIP stream functions (non-static so test can link against them)
extern "C" hipError_t hipStreamCreate(hipStream_t *stream) {
  *stream = malloc(sizeof(void *)); // Fake handle
  MOCK_PRINT("[MOCK] hipStreamCreate() -> %p\n", *stream);
  return hipSuccess;
}

extern "C" hipError_t hipStreamDestroy(hipStream_t stream) {
  MOCK_PRINT("[MOCK] hipStreamDestroy(%p)\n", stream);
  free(stream);
  return hipSuccess;
}

extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
  MOCK_PRINT("[MOCK] hipStreamSynchronize(%p)\n", stream);
  return hipSuccess;
}

extern "C" hipError_t hipEventCreate(hipEvent_t *event) {
  *event = malloc(sizeof(void *));
  return hipSuccess;
}

extern "C" hipError_t hipEventDestroy(hipEvent_t event) {
  free(event);
  return hipSuccess;
}

extern "C" hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  (void)event;
  (void)stream;
  return hipSuccess;
}

extern "C" hipError_t hipEventElapsedTime(float *ms, hipEvent_t start,
                                          hipEvent_t stop) {
  (void)start;
  (void)stop;
  *ms = 0.0f;
  return hipSuccess;
}

extern "C" const char *hipGetErrorString(hipError_t error) {
  (void)error;
  return "mock_error";
}

// Mock HIP memory functions (non-static for cross-module linking)
extern "C" hipError_t hipMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  MOCK_PRINT("[MOCK] hipMalloc(%zu bytes) -> %p\n", size, *ptr);
  return *ptr ? hipSuccess : -1;
}

extern "C" hipError_t hipFree(void *ptr) {
  MOCK_PRINT("[MOCK] hipFree(%p)\n", ptr);
  free(ptr);
  return hipSuccess;
}

extern "C" hipError_t hipHostMalloc(void **ptr, size_t size,
                                    unsigned int flags) {
  (void)flags;
  *ptr = malloc(size);
  MOCK_PRINT("[MOCK] hipHostMalloc(%zu bytes) -> %p\n", size, *ptr);
  return *ptr ? hipSuccess : -1;
}

extern "C" hipError_t hipHostFree(void *ptr) {
  MOCK_PRINT("[MOCK] hipHostFree(%p)\n", ptr);
  free(ptr);
  return hipSuccess;
}

extern "C" hipError_t hipMemcpy(void *dst, const void *src, size_t size,
                                int kind) {
  const char *kind_str = (kind == hipMemcpyHostToDevice)   ? "H2D"
                         : (kind == hipMemcpyDeviceToHost) ? "D2H"
                                                           : "D2D";
  MOCK_PRINT("[MOCK] hipMemcpy(dst=%p, src=%p, size=%zu, %s)\n", dst, src, size,
             kind_str);
  memcpy(dst, src, size);
  return hipSuccess;
}

extern "C" hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size,
                                     int kind, hipStream_t stream) {
  const char *kind_str = (kind == hipMemcpyHostToDevice)   ? "H2D"
                         : (kind == hipMemcpyDeviceToHost) ? "D2H"
                                                           : "D2D";
  MOCK_PRINT("[MOCK] hipMemcpyAsync(dst=%p, src=%p, size=%zu, %s, stream=%p)\n",
             dst, src, size, kind_str, stream);
  memcpy(dst, src, size);
  return hipSuccess;
}

// Mock MIOpen types and constants
typedef void *miopenTensorDescriptor_t;
typedef void *miopenConvolutionDescriptor_t;
typedef enum { miopenFloat = 0 } miopenDataType_t;
typedef enum { miopenConvolution = 0 } miopenConvolutionMode_t;
typedef enum { miopenConvolutionFwdAlgoGEMM = 0 } miopenConvFwdAlgorithm_t;

// Mock MIOpen handle functions (non-static so test can link against them)
extern "C" miopenStatus_t miopenCreate(miopenHandle_t *handle) {
  *handle = malloc(sizeof(void *)); // Fake handle
  MOCK_PRINT("[MOCK] miopenCreate() -> %p\n", *handle);
  return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle) {
  MOCK_PRINT("[MOCK] miopenDestroy(%p)\n", handle);
  free(handle);
  return miopenStatusSuccess;
}

extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle,
                                          hipStream_t stream) {
  MOCK_PRINT("[MOCK] miopenSetStream(handle=%p, stream=%p)\n", handle, stream);
  return miopenStatusSuccess;
}

// Mock MIOpen tensor descriptor functions
static miopenStatus_t
miopenCreateTensorDescriptor(miopenTensorDescriptor_t *desc) {
  *desc = malloc(sizeof(void *)); // Fake descriptor
  return miopenStatusSuccess;
}

static miopenStatus_t
miopenDestroyTensorDescriptor(miopenTensorDescriptor_t desc) {
  free(desc);
  return miopenStatusSuccess;
}

static miopenStatus_t miopenSetNdTensorDescriptorWithLayout(
    miopenTensorDescriptor_t desc, miopenDataType_t dataType,
    miopenTensorLayout_t layout, const int *lens, int num_lens) {
  (void)desc;
  (void)dataType;
  (void)layout;
  if (num_lens == 4) {
    MOCK_PRINT("[MOCK]   Tensor descriptor set: [%d, %d, %d, %d]\n", lens[0],
               lens[1], lens[2], lens[3]);
  } else {
    MOCK_PRINT("[MOCK]   Tensor descriptor set: %d dims\n", num_lens);
  }
  return miopenStatusSuccess;
}

// Mock MIOpen convolution descriptor functions
static miopenStatus_t
miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t *desc) {
  *desc = malloc(sizeof(void *)); // Fake descriptor
  return miopenStatusSuccess;
}

static miopenStatus_t
miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t desc) {
  free(desc);
  return miopenStatusSuccess;
}

static miopenStatus_t miopenInitConvolutionDescriptor(
    miopenConvolutionDescriptor_t desc, miopenConvolutionMode_t mode, int pad_h,
    int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
  (void)desc;
  (void)mode;
  MOCK_PRINT("[MOCK]   Convolution params: pad=[%d,%d], stride=[%d,%d], "
             "dilation=[%d,%d]\n",
             pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  return miopenStatusSuccess;
}

// Mock MIOpen convolution algorithm finding
static miopenStatus_t miopenFindConvolutionForwardAlgorithm(
    miopenHandle_t handle, miopenTensorDescriptor_t input_desc,
    const void *input, miopenTensorDescriptor_t weights_desc,
    const void *weights, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t output_desc, const void *output,
    int requestAlgoCount, miopenConvFwdAlgorithm_t *algo,
    int *returnedAlgoCount, void *workspace, size_t workspaceSize,
    bool exhaustiveSearch) {
  (void)handle;
  (void)input_desc;
  (void)input;
  (void)weights_desc;
  (void)weights;
  (void)conv_desc;
  (void)output_desc;
  (void)output;
  (void)requestAlgoCount;
  (void)returnedAlgoCount;
  (void)workspace;
  (void)workspaceSize;
  (void)exhaustiveSearch;

  MOCK_PRINT("[MOCK]   Finding convolution algorithm...\n");
  if (algo)
    *algo = miopenConvolutionFwdAlgoGEMM;
  return miopenStatusSuccess;
}

static miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
    miopenHandle_t handle, miopenTensorDescriptor_t weights_desc,
    miopenTensorDescriptor_t input_desc,
    miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t output_desc, size_t *workspaceSize) {
  (void)handle;
  (void)weights_desc;
  (void)input_desc;
  (void)conv_desc;
  (void)output_desc;
  *workspaceSize = 0; // No workspace needed in mock
  return miopenStatusSuccess;
}

static miopenStatus_t miopenConvolutionForward(
    miopenHandle_t handle, const void *alpha,
    miopenTensorDescriptor_t input_desc, const void *input,
    miopenTensorDescriptor_t weights_desc, const void *weights,
    miopenConvolutionDescriptor_t conv_desc, miopenConvFwdAlgorithm_t algo,
    const void *beta, miopenTensorDescriptor_t output_desc, void *output,
    void *workspace, size_t workspaceSize) {
  (void)handle;
  (void)alpha;
  (void)input_desc;
  (void)input;
  (void)weights_desc;
  (void)weights;
  (void)conv_desc;
  (void)algo;
  (void)beta;
  (void)output_desc;
  (void)output;
  (void)workspace;
  (void)workspaceSize;

  MOCK_PRINT("[MOCK]   Executing convolution forward pass\n");
  return miopenStatusSuccess;
}

// Mock hipBLASLt types and constants
typedef void *hipblasLtMatrixLayout_t;
typedef void *hipblasLtMatmulDesc_t;
typedef enum { HIPBLAS_R_32F = 0 } hipblasDatatype_t;
typedef enum { HIPBLAS_COMPUTE_32F = 0 } hipblasComputeType_t;

// Mock hipBLASLt handle functions (non-static so test can link against them)
extern "C" hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle) {
  *handle = malloc(sizeof(void *)); // Fake handle
  MOCK_PRINT("[MOCK] hipblasLtCreate() -> %p\n", *handle);
  return HIPBLAS_STATUS_SUCCESS;
}

extern "C" hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle) {
  MOCK_PRINT("[MOCK] hipblasLtDestroy(%p)\n", handle);
  free(handle);
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock hipBLASLt matrix layout functions
static hipblasStatus_t
hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t *layout,
                            hipblasDatatype_t type, uint64_t rows,
                            uint64_t cols, int64_t ld) {
  (void)type;
  (void)ld;
  *layout = malloc(sizeof(void *)); // Fake layout
  MOCK_PRINT("[MOCK]   Matrix layout: [%llu x %llu]\n",
             (unsigned long long)rows, (unsigned long long)cols);
  return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t
hipblasLtMatrixLayoutDestroy(hipblasLtMatrixLayout_t layout) {
  free(layout);
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock hipBLASLt matmul descriptor functions
static hipblasStatus_t
hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t *desc,
                          hipblasComputeType_t computeType,
                          hipblasDatatype_t dataType) {
  (void)computeType;
  (void)dataType;
  *desc = malloc(sizeof(void *)); // Fake descriptor
  return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t desc) {
  free(desc);
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock hipBLASLt matmul function
static hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmul_desc,
                const void *alpha, const void *A, hipblasLtMatrixLayout_t matA,
                const void *B, hipblasLtMatrixLayout_t matB, const void *beta,
                const void *C, hipblasLtMatrixLayout_t matC, void *D,
                hipblasLtMatrixLayout_t matD, const void *algo, void *workspace,
                size_t workspaceSize, hipStream_t stream) {
  (void)handle;
  (void)matmul_desc;
  (void)alpha;
  (void)A;
  (void)matA;
  (void)B;
  (void)matB;
  (void)beta;
  (void)C;
  (void)matC;
  (void)D;
  (void)matD;
  (void)algo;
  (void)workspace;
  (void)workspaceSize;
  (void)stream;

  MOCK_PRINT("[MOCK]   Executing GEMM operation\n");
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock error checking macros
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    (void)(cmd);                                                               \
  } while (0)
#define MIOPEN_CHECK(cmd)                                                      \
  do {                                                                         \
    (void)(cmd);                                                               \
  } while (0)
#define HIPBLAS_CHECK(cmd)                                                     \
  do {                                                                         \
    (void)(cmd);                                                               \
  } while (0)
// hipGetErrorString is now a real mock function declared above

// Mock wrapper implementations (called from generated MLIR code)

int wrap_miopenConvolutionForward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group) {
  if (!state || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionForward\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_miopenConvolutionForward(\n");
  MOCK_PRINT("[MOCK]   input=[%lld,%lld,%lld,%lld],\n", (long long)input_n,
             (long long)input_c, (long long)input_h, (long long)input_w);
  MOCK_PRINT("[MOCK]   weights=[%lld,%lld,%lld,%lld],\n", (long long)weights_k,
             (long long)input_c, (long long)kernel_h, (long long)kernel_w);
  MOCK_PRINT("[MOCK]   output=[%lld,%lld,%lld,%lld],\n", (long long)input_n,
             (long long)weights_k, (long long)output_h, (long long)output_w);
  MOCK_PRINT("[MOCK]   stride=[%lld,%lld], pad=[%lld,%lld,%lld,%lld], "
             "dilation=[%lld,%lld], group=%lld)\n",
             (long long)stride_h, (long long)stride_w, (long long)pad_top,
             (long long)pad_left, (long long)pad_bottom, (long long)pad_right,
             (long long)dilation_h, (long long)dilation_w, (long long)group);

  // Mock: Fill output with dummy data (zeros in this case)
  // In a real implementation, this would call MIOpen
  size_t output_size =
      input_n * weights_k * output_h * output_w * sizeof(float);
  memset(output, 0, output_size);

  return 0;
}

int wrap_hipblasLtGemm(void *handle, void *stream, int64_t m, int64_t n,
                       int64_t k, const void *alpha, const void *A,
                       const void *B, const void *beta, void *C) {
  if (!handle || !stream || !alpha || !A || !B || !beta || !C) {
    fprintf(stderr, "Invalid arguments to wrap_hipblasLtGemm\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_hipblasLtGemm(M=%lld, N=%lld, K=%lld)\n",
             (long long)m, (long long)n, (long long)k);

  hipblasLtHandle_t hipblas_handle = static_cast<hipblasLtHandle_t>(handle);
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // Create matrix descriptors (assuming float32, column-major)
  hipblasLtMatrixLayout_t matA, matB, matC;
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIPBLAS_R_32F, m, k, m));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIPBLAS_R_32F, k, n, k));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIPBLAS_R_32F, m, n, m));

  // Create operation descriptor
  hipblasLtMatmulDesc_t matmul_desc;
  HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F,
                                          HIPBLAS_R_32F));

  // Perform GEMM
  HIPBLAS_CHECK(hipblasLtMatmul(hipblas_handle, matmul_desc, alpha, A, matA, B,
                                matB, beta, C, matC, C, matC,
                                nullptr, // algo
                                nullptr, // workspace
                                0,       // workspaceSize
                                hip_stream));

  // Cleanup
  hipblasLtMatrixLayoutDestroy(matA);
  hipblasLtMatrixLayoutDestroy(matB);
  hipblasLtMatrixLayoutDestroy(matC);
  hipblasLtMatmulDescDestroy(matmul_desc);

  return 0;
}

int wrap_hipblasLtMatmul(RuntimeState *state, const void *A, const void *B,
                         void *output, int64_t M, int64_t N, int64_t K,
                         int64_t batch_count, int64_t elem_size) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_hipblasLtMatmul\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_hipblasLtMatmul(M=%lld, N=%lld, K=%lld, "
             "batch=%lld, elem_size=%lld)\n",
             (long long)M, (long long)N, (long long)K, (long long)batch_count,
             (long long)elem_size);

  return 0;
}

int wrap_group_query_attention(
    RuntimeState *state,
    // Inputs 1-7 (core GQA)
    void *query, void *key, void *value, void *past_key, void *past_value,
    void *seqlens_k, void *total_seq_len,
    // Inputs 8-10 (RoPE)
    void *cos_cache, void *sin_cache, void *position_ids,
    // Inputs 11-14 (advanced features)
    void *attention_bias, void *head_sink, void *k_scale, void *v_scale,
    // Outputs
    void *output, void *present_key, void *present_value, void *output_qk,
    // Attributes (12)
    int64_t num_heads, int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width,
    // Shape values (6)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t past_buf_seq, int64_t head_dim, int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_group_query_attention\n");
    return -1;
  }

  (void)position_ids;
  (void)attention_bias;
  (void)head_sink;
  (void)k_scale;
  (void)v_scale;
  (void)output_qk;
  (void)local_window_size;
  (void)smooth_softmax;
  (void)qk_output;
  (void)k_quant_type;
  (void)v_quant_type;
  (void)kv_cache_bit_width;
  (void)past_buf_seq;
  (void)present_key;
  (void)present_value;

  MOCK_PRINT("[MOCK] wrap_group_query_attention(\n");
  MOCK_PRINT("[MOCK]   num_heads=%lld, kv_num_heads=%lld,\n",
             (long long)num_heads, (long long)kv_num_heads);
  MOCK_PRINT("[MOCK]   scale=%f, softcap=%f,\n", (double)scale,
             (double)softcap);
  MOCK_PRINT("[MOCK]   do_rotary=%lld, rotary_interleaved=%lld,\n",
             (long long)do_rotary, (long long)rotary_interleaved);
  MOCK_PRINT("[MOCK]   batch=%lld, seq_q=%lld, seq_kv=%lld, "
             "past_buf_seq=%lld, head_dim=%lld, elem_size=%lld)\n",
             (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
             (long long)past_buf_seq, (long long)head_dim,
             (long long)element_size_bytes);

  return 0;
}

int wrap_miopenOpTensor(RuntimeState *state, void *lhs, void *rhs, void *output,
                        int64_t lhs_n, int64_t lhs_c, int64_t lhs_h,
                        int64_t lhs_w, int64_t rhs_n, int64_t rhs_c,
                        int64_t rhs_h, int64_t rhs_w, int64_t out_n,
                        int64_t out_c, int64_t out_h, int64_t out_w,
                        int64_t data_type, int64_t tensor_op) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_miopenOpTensor\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_miopenOpTensor(op=%s, "
             "lhs=[%lld,%lld,%lld,%lld], "
             "rhs=[%lld,%lld,%lld,%lld], "
             "out=[%lld,%lld,%lld,%lld], "
             "data_type=%s(%lld))\n",
             hipdnn_ep_tensor_op_name(tensor_op), (long long)lhs_n,
             (long long)lhs_c, (long long)lhs_h, (long long)lhs_w,
             (long long)rhs_n, (long long)rhs_c, (long long)rhs_h,
             (long long)rhs_w, (long long)out_n, (long long)out_c,
             (long long)out_h, (long long)out_w,
             hipdnn_ep_datatype_name(data_type), (long long)data_type);

  return 0;
}

int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                         void *output, int64_t num_elements,
                         int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_elementwise_sub\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_elementwise_sub(num_elements=%lld, "
             "element_size=%lld)\n",
             (long long)num_elements, (long long)element_size_bytes);

  return 0;
}

int wrap_gather(RuntimeState *state, void *data, void *indices, void *output,
                int64_t axis, int64_t data_num_elements,
                int64_t indices_num_elements, int64_t output_num_elements,
                int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_gather\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_gather(axis=%lld, data_num_elements=%lld, "
             "indices_num_elements=%lld, output_num_elements=%lld, "
             "element_size=%lld)\n",
             (long long)axis, (long long)data_num_elements,
             (long long)indices_num_elements, (long long)output_num_elements,
             (long long)element_size_bytes);

  return 0;
}

int wrap_reduce_sum(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t element_size_bytes,
                    int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_sum\n");
    return -1;
  }

  MOCK_PRINT(
      "[MOCK] wrap_reduce_sum(data_num_elements=%lld, "
      "output_num_elements=%lld, axes_num_elements=%lld, element_size=%lld, "
      "keepdims=%lld, noop_with_empty_axes=%lld)\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, (long long)element_size_bytes,
      (long long)keepdims, (long long)noop_with_empty_axes);

  return 0;
}

int wrap_cast(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t src_data_type,
              int64_t dst_data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_cast\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_cast(num_elements=%lld, src_dtype=%lld, "
             "dst_dtype=%lld)\n",
             (long long)num_elements, (long long)src_data_type,
             (long long)dst_data_type);

  return 0;
}

int wrap_miopenActivationForward(RuntimeState *state, void *input, void *output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_miopenActivationForward\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_miopenActivationForward(activation=%s, "
             "num_elements=%lld, data_type=%s(%lld), element_size=%lld)\n",
             hipdnn_ep_activation_name(activation_mode),
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type,
             (long long)hipdnn_ep_datatype_size(data_type));

  return 0;
}

int wrap_rotary_embedding(RuntimeState *state, void *input, void *position_ids,
                          void *cos_cache, void *sin_cache, void *output,
                          int64_t interleaved, int64_t num_heads,
                          int64_t rotary_dim, int64_t input_num_elements,
                          int64_t cos_cache_num_elements,
                          int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_rotary_embedding\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_rotary_embedding(interleaved=%lld, num_heads=%lld, "
             "rotary_dim=%lld, input_num_elements=%lld, "
             "cos_cache_num_elements=%lld, element_size=%lld)\n",
             (long long)interleaved, (long long)num_heads,
             (long long)rotary_dim, (long long)input_num_elements,
             (long long)cos_cache_num_elements, (long long)element_size_bytes);

  return 0;
}

int wrap_miopenT5LayerNormForward(RuntimeState *state, void *input, void *scale,
                                  void *output, int64_t input_num_elements,
                                  int64_t scale_num_elements,
                                  int64_t element_size_bytes, int64_t axis,
                                  float epsilon, int64_t stash_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_miopenT5LayerNormForward\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_miopenT5LayerNormForward(input_num_elements=%lld, "
             "scale_num_elements=%lld, element_size=%lld, axis=%lld, "
             "epsilon=%f, stash_type=%lld)\n",
             (long long)input_num_elements, (long long)scale_num_elements,
             (long long)element_size_bytes, (long long)axis, (double)epsilon,
             (long long)stash_type);

  return 0;
}

int wrap_skip_simplified_layer_norm(RuntimeState *state, void *input,
                                    void *skip, void *gamma, void *bias,
                                    void *output, void *input_skip_bias_sum,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes, float epsilon) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_skip_simplified_layer_norm\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_skip_simplified_layer_norm(input_num_elements=%lld, "
             "gamma_num_elements=%lld, element_size=%lld, epsilon=%f, "
             "bias=%s, input_skip_bias_sum=%s)\n",
             (long long)input_num_elements, (long long)gamma_num_elements,
             (long long)element_size_bytes, (double)epsilon,
             bias ? "yes" : "no", input_skip_bias_sum ? "yes" : "no");

  return 0;
}

int wrap_matmul_nbits(RuntimeState *state, const void *A, const void *B,
                      const void *scales, const void *zero_points,
                      const void *g_idx, const void *bias, void *output,
                      int64_t M, int64_t N, int64_t K, int64_t batch_count,
                      int64_t bits, int64_t block_size, int64_t elem_size) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_matmul_nbits\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_matmul_nbits(M=%lld, N=%lld, K=%lld, "
             "batch=%lld, bits=%lld, block_size=%lld, elem_size=%lld, "
             "zero_points=%s, g_idx=%s, bias=%s)\n",
             (long long)M, (long long)N, (long long)K, (long long)batch_count,
             (long long)bits, (long long)block_size, (long long)elem_size,
             zero_points ? "yes" : "null", g_idx ? "yes" : "null",
             bias ? "yes" : "null");

  return 0;
}

int wrap_qmoe(RuntimeState *state, const void *input, const void *router_probs,
              const void *fc1_weights, const void *fc1_scales,
              const void *fc1_bias, const void *fc2_weights,
              const void *fc2_scales, const void *fc2_bias,
              const void *fc3_weights, const void *fc3_scales,
              const void *fc3_bias, const void *fc1_zero_points,
              const void *fc2_zero_points, const void *fc3_zero_points,
              void *output, int64_t num_tokens, int64_t hidden_size,
              int64_t inter_size, int64_t num_experts, int64_t k,
              int64_t expert_weight_bits, int64_t block_size,
              int64_t swiglu_fusion, int64_t activation_type,
              float activation_alpha, float activation_beta, float swiglu_limit,
              int64_t normalize_routing_weights, int64_t elem_size) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_qmoe\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_qmoe(\n");
  MOCK_PRINT("[MOCK]   num_tokens=%lld, hidden_size=%lld, inter_size=%lld,\n",
             (long long)num_tokens, (long long)hidden_size,
             (long long)inter_size);
  MOCK_PRINT("[MOCK]   num_experts=%lld, k=%lld, bits=%lld,\n",
             (long long)num_experts, (long long)k,
             (long long)expert_weight_bits);
  MOCK_PRINT("[MOCK]   block_size=%lld, swiglu_fusion=%lld, "
             "activation_type=%lld,\n",
             (long long)block_size, (long long)swiglu_fusion,
             (long long)activation_type);
  MOCK_PRINT("[MOCK]   alpha=%f, beta=%f, limit=%f, normalize=%lld, "
             "elem_size=%lld,\n",
             (double)activation_alpha, (double)activation_beta,
             (double)swiglu_limit, (long long)normalize_routing_weights,
             (long long)elem_size);
  MOCK_PRINT("[MOCK]   fc1_bias=%s, fc2_bias=%s, fc3_weights=%s, "
             "fc1_zp=%s, fc2_zp=%s)\n",
             fc1_bias ? "yes" : "null", fc2_bias ? "yes" : "null",
             fc3_weights ? "yes" : "null", fc1_zero_points ? "yes" : "null",
             fc2_zero_points ? "yes" : "null");

  return 0;
}

int wrap_hipMalloc(void **ptr, int64_t size) {
  HIP_CHECK(hipMalloc(ptr, size));
  return 0;
}

int wrap_hipFree(void *ptr) {
  HIP_CHECK(hipFree(ptr));
  return 0;
}

int wrap_hipMemcpyH2D(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipMemcpyD2H(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipStreamSynchronize(void *stream) {
  HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  return 0;
}
