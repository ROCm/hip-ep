/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#  include <windows.h>
// On Windows with static CRT, each DLL has its own stdout
// Use OutputDebugString so output appears in DebugView/debugger
// and fprintf(stderr) to try to reach the parent process
#  define MOCK_PRINT(...)                                                      \
    do {                                                                       \
      char buf[512];                                                           \
      snprintf(buf, sizeof(buf), __VA_ARGS__);                                 \
      OutputDebugStringA(buf);                                                 \
      fprintf(stderr, "%s", buf);                                              \
      fflush(stderr);                                                          \
    } while (0)
#else
#  define MOCK_PRINT(...)                                                      \
    do {                                                                       \
      printf(__VA_ARGS__);                                                     \
      fflush(stdout);                                                          \
    } while (0)
#endif

// Mock type definitions are now in mock_types.h (included via runtime_types.h)
// No need to redefine them here

// Comprehensive mock implementations for all GPU functions
// Prints all operations for debugging and verification

// Mock HIP device functions
extern "C" hipError_t hipGetDeviceCount(int* count) {
  MOCK_PRINT("[MOCK] hipGetDeviceCount\n");
  *count = 1; // Pretend we have one device
  return hipSuccess;
}

extern "C" hipError_t hipSetDevice(int device) {
  MOCK_PRINT("[MOCK] hipSetDevice(%d)\n", device);
  return hipSuccess;
}

extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop,
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
extern "C" hipError_t hipStreamCreate(hipStream_t* stream) {
  *stream = malloc(8); // Fake handle
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

// Mock HIP memory functions (non-static for cross-module linking)
extern "C" hipError_t hipMalloc(void** ptr, size_t size) {
  *ptr = malloc(size);
  MOCK_PRINT("[MOCK] hipMalloc(%zu bytes) -> %p\n", size, *ptr);
  return *ptr ? hipSuccess : -1;
}

extern "C" hipError_t hipFree(void* ptr) {
  MOCK_PRINT("[MOCK] hipFree(%p)\n", ptr);
  free(ptr);
  return hipSuccess;
}

extern "C" hipError_t hipHostMalloc(void** ptr, size_t size,
                                    unsigned int flags) {
  (void)flags;
  *ptr = malloc(size);
  MOCK_PRINT("[MOCK] hipHostMalloc(%zu bytes) -> %p\n", size, *ptr);
  return *ptr ? hipSuccess : -1;
}

extern "C" hipError_t hipHostFree(void* ptr) {
  MOCK_PRINT("[MOCK] hipHostFree(%p)\n", ptr);
  free(ptr);
  return hipSuccess;
}

extern "C" hipError_t hipMemcpy(void* dst, const void* src, size_t size,
                                int kind) {
  const char* kind_str = (kind == hipMemcpyHostToDevice)   ? "H2D"
                         : (kind == hipMemcpyDeviceToHost) ? "D2H"
                                                           : "D2D";
  MOCK_PRINT("[MOCK] hipMemcpy(dst=%p, src=%p, size=%zu, %s)\n", dst, src, size,
             kind_str);
  memcpy(dst, src, size);
  return hipSuccess;
}

extern "C" hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                                     int kind, hipStream_t stream) {
  const char* kind_str = (kind == hipMemcpyHostToDevice)   ? "H2D"
                         : (kind == hipMemcpyDeviceToHost) ? "D2H"
                                                           : "D2D";
  MOCK_PRINT("[MOCK] hipMemcpyAsync(dst=%p, src=%p, size=%zu, %s, stream=%p)\n",
             dst, src, size, kind_str, stream);
  memcpy(dst, src, size);
  return hipSuccess;
}

// Mock MIOpen types and constants
typedef void* miopenTensorDescriptor_t;
typedef void* miopenConvolutionDescriptor_t;
typedef enum { miopenFloat = 0 } miopenDataType_t;
typedef enum { miopenConvolution = 0 } miopenConvolutionMode_t;
typedef enum { miopenConvolutionFwdAlgoGEMM = 0 } miopenConvFwdAlgorithm_t;

// Mock MIOpen handle functions (non-static so test can link against them)
extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle) {
  *handle = malloc(8); // Fake handle
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
miopenCreateTensorDescriptor(miopenTensorDescriptor_t* desc) {
  *desc = malloc(8); // Fake descriptor
  return miopenStatusSuccess;
}

static miopenStatus_t
miopenDestroyTensorDescriptor(miopenTensorDescriptor_t desc) {
  free(desc);
  return miopenStatusSuccess;
}

static miopenStatus_t miopenSet4dTensorDescriptor(miopenTensorDescriptor_t desc,
                                                  miopenDataType_t dataType,
                                                  int n, int c, int h, int w) {
  (void)desc;
  (void)dataType;
  // Print tensor shape for verification
  MOCK_PRINT("[MOCK]   Tensor descriptor set: [%d, %d, %d, %d]\n", n, c, h, w);
  return miopenStatusSuccess;
}

// Mock MIOpen convolution descriptor functions
static miopenStatus_t
miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* desc) {
  *desc = malloc(8); // Fake descriptor
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
    const void* input, miopenTensorDescriptor_t weights_desc,
    const void* weights, miopenConvolutionDescriptor_t conv_desc,
    miopenTensorDescriptor_t output_desc, const void* output,
    int requestAlgoCount, miopenConvFwdAlgorithm_t* algo,
    int* returnedAlgoCount, void* workspace, size_t workspaceSize,
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
    miopenTensorDescriptor_t output_desc, size_t* workspaceSize) {
  (void)handle;
  (void)weights_desc;
  (void)input_desc;
  (void)conv_desc;
  (void)output_desc;
  *workspaceSize = 0; // No workspace needed in mock
  return miopenStatusSuccess;
}

static miopenStatus_t miopenConvolutionForward(
    miopenHandle_t handle, const void* alpha,
    miopenTensorDescriptor_t input_desc, const void* input,
    miopenTensorDescriptor_t weights_desc, const void* weights,
    miopenConvolutionDescriptor_t conv_desc, miopenConvFwdAlgorithm_t algo,
    const void* beta, miopenTensorDescriptor_t output_desc, void* output,
    void* workspace, size_t workspaceSize) {
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
typedef void* hipblasLtMatrixLayout_t;
typedef void* hipblasLtMatmulDesc_t;
typedef enum { HIPBLAS_R_32F = 0 } hipblasDatatype_t;
typedef enum { HIPBLAS_COMPUTE_32F = 0 } hipblasComputeType_t;

// Mock hipBLASLt handle functions (non-static so test can link against them)
extern "C" hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle) {
  *handle = malloc(8); // Fake handle
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
hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* layout,
                            hipblasDatatype_t type, uint64_t rows,
                            uint64_t cols, int64_t ld) {
  (void)type;
  (void)ld;
  *layout = malloc(8); // Fake layout
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
hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t* desc,
                          hipblasComputeType_t computeType,
                          hipblasDatatype_t dataType) {
  (void)computeType;
  (void)dataType;
  *desc = malloc(8); // Fake descriptor
  return HIPBLAS_STATUS_SUCCESS;
}

static hipblasStatus_t hipblasLtMatmulDescDestroy(hipblasLtMatmulDesc_t desc) {
  free(desc);
  return HIPBLAS_STATUS_SUCCESS;
}

// Mock hipBLASLt matmul function
static hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmul_desc,
                const void* alpha, const void* A, hipblasLtMatrixLayout_t matA,
                const void* B, hipblasLtMatrixLayout_t matB, const void* beta,
                const void* C, hipblasLtMatrixLayout_t matC, void* D,
                hipblasLtMatrixLayout_t matD, const void* algo, void* workspace,
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
#define hipGetErrorString(e) "mock_error"

// Mock wrapper implementations (called from generated MLIR code)

int wrap_miopenConvolutionForward(
    RuntimeState* state, const void* input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void* weights, int64_t weights_k,
    const void* bias, void* output, int64_t output_h, int64_t output_w,
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

// Mock implementation for ReLU activation
extern "C" int wrap_miopenActivationForward_relu(
    RuntimeState* state, void* input_gpu_ptr, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, void* output_gpu_ptr, int64_t output_n,
    int64_t output_c, int64_t output_h, int64_t output_w) {
  if (!state || !input_gpu_ptr || !output_gpu_ptr) {
    fprintf(stderr, "Invalid arguments to wrap_miopenActivationForward_relu\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_miopenActivationForward_relu: "
             "input=[%lldx%lldx%lldx%lld] output=[%lldx%lldx%lldx%lld]\n",
             input_n, input_c, input_h, input_w, output_n, output_c, output_h,
             output_w);

  // Mock: In a real implementation, this would call MIOpen to compute ReLU
  // For now, just log and return success
  return 0;
}

int wrap_hipblasLtGemm(void* handle, void* stream, int64_t m, int64_t n,
                       int64_t k, const void* alpha, const void* A,
                       const void* B, const void* beta, void* C) {
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

int wrap_matmul(RuntimeState* state,
                const void* A, const void* B, void* output,
                int64_t M, int64_t N, int64_t K,
                int64_t batch_count, int64_t elem_size) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_matmul\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_matmul(M=%lld, N=%lld, K=%lld, "
             "batch=%lld, elem_size=%lld)\n",
             (long long)M, (long long)N, (long long)K,
             (long long)batch_count, (long long)elem_size);

  return 0;
}

int wrap_group_query_attention(
    RuntimeState* state,
    void* query, void* key, void* value,
    void* past_key, void* past_value,
    void* seqlens_k, void* total_seq_len,
    void* output, void* present_key, void* present_value,
    int64_t num_heads, int64_t kv_num_heads,
    float scale, float softcap,
    int64_t do_rotary, int64_t rotary_interleaved) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_group_query_attention\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_group_query_attention(\n");
  MOCK_PRINT("[MOCK]   num_heads=%lld, kv_num_heads=%lld,\n",
             (long long)num_heads, (long long)kv_num_heads);
  MOCK_PRINT("[MOCK]   scale=%f, softcap=%f,\n", scale, softcap);
  MOCK_PRINT("[MOCK]   do_rotary=%lld, rotary_interleaved=%lld)\n",
             (long long)do_rotary, (long long)rotary_interleaved);

  return 0;
}

int wrap_elementwise_mul(RuntimeState* state, void* lhs, void* rhs,
                         void* output, int64_t num_elements,
                         int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_elementwise_mul\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_elementwise_mul(num_elements=%lld, "
             "element_size=%lld)\n",
             (long long)num_elements, (long long)element_size_bytes);

  return 0;
}

int wrap_elementwise_sub(RuntimeState* state, void* lhs, void* rhs,
                         void* output, int64_t num_elements,
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

int wrap_gather(RuntimeState* state, void* data, void* indices,
                void* output, int64_t axis, int64_t data_num_elements,
                int64_t output_num_elements, int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_gather\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_gather(axis=%lld, data_num_elements=%lld, "
             "output_num_elements=%lld, element_size=%lld)\n",
             (long long)axis, (long long)data_num_elements,
             (long long)output_num_elements, (long long)element_size_bytes);

  return 0;
}

int wrap_reduce_sum(RuntimeState* state, void* data, void* axes,
                    void* output, int64_t data_num_elements,
                    int64_t output_num_elements, int64_t element_size_bytes,
                    int64_t keepdims) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_sum\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_reduce_sum(data_num_elements=%lld, "
             "output_num_elements=%lld, element_size=%lld, keepdims=%lld)\n",
             (long long)data_num_elements, (long long)output_num_elements,
             (long long)element_size_bytes, (long long)keepdims);

  return 0;
}

int wrap_cast(RuntimeState* state, void* input, void* output,
              int64_t num_elements, int64_t input_element_size,
              int64_t output_element_size, int64_t to) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_cast\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_cast(num_elements=%lld, input_elem_size=%lld, "
             "output_elem_size=%lld, to=%lld)\n",
             (long long)num_elements, (long long)input_element_size,
             (long long)output_element_size, (long long)to);

  return 0;
}

int wrap_sigmoid(RuntimeState* state, void* input, void* output,
                 int64_t num_elements, int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_sigmoid\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_sigmoid(num_elements=%lld, element_size=%lld)\n",
             (long long)num_elements, (long long)element_size_bytes);

  return 0;
}

int wrap_rotary_embedding(RuntimeState* state,
                          void* input, void* position_ids,
                          void* cos_cache, void* sin_cache,
                          void* output,
                          int64_t interleaved, int64_t num_heads,
                          int64_t rotary_dim,
                          int64_t input_num_elements,
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
             (long long)cos_cache_num_elements,
             (long long)element_size_bytes);

  return 0;
}

int wrap_simplified_layer_norm(RuntimeState* state,
                               void* input, void* scale, void* output,
                               int64_t input_num_elements,
                               int64_t scale_num_elements,
                               int64_t element_size_bytes,
                               int64_t axis, float epsilon,
                               int64_t stash_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_simplified_layer_norm\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_simplified_layer_norm(input_num_elements=%lld, "
             "scale_num_elements=%lld, element_size=%lld, axis=%lld, "
             "epsilon=%f, stash_type=%lld)\n",
             (long long)input_num_elements, (long long)scale_num_elements,
             (long long)element_size_bytes, (long long)axis,
             (double)epsilon, (long long)stash_type);

  return 0;
}

int wrap_skip_simplified_layer_norm(RuntimeState* state,
                                    void* input, void* skip, void* gamma,
                                    void* output, void* skip_output,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes,
                                    float epsilon) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_skip_simplified_layer_norm\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_skip_simplified_layer_norm(input_num_elements=%lld, "
             "gamma_num_elements=%lld, element_size=%lld, epsilon=%f)\n",
             (long long)input_num_elements, (long long)gamma_num_elements,
             (long long)element_size_bytes, (double)epsilon);

  return 0;
}

int wrap_hipMalloc(void** ptr, int64_t size) {
  HIP_CHECK(hipMalloc(ptr, size));
  return 0;
}

int wrap_hipFree(void* ptr) {
  HIP_CHECK(hipFree(ptr));
  return 0;
}

int wrap_hipMemcpyH2D(void* dst, const void* src, int64_t size, void* stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipMemcpyD2H(void* dst, const void* src, int64_t size, void* stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipStreamSynchronize(void* stream) {
  HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  return 0;
}
