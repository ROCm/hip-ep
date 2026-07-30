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

extern "C" hipError_t hipEventCreateWithFlags(hipEvent_t *event,
                                              unsigned int flags) {
  (void)flags;
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

extern "C" hipError_t hipEventSynchronize(hipEvent_t event) {
  (void)event;
  return hipSuccess;
}

extern "C" hipError_t hipEventElapsedTime(float *ms, hipEvent_t start,
                                          hipEvent_t stop) {
  (void)start;
  (void)stop;
  *ms = 0.0f;
  return hipSuccess;
}

// Mock hipHostGetDevicePointer: in real HIP this returns the device-mapped
// address corresponding to a hipHostMallocMapped host pointer. The mock
// allocator returns plain malloc, which is already host-addressable; just
// aliasing the same pointer is sufficient for the runtime drivers that
// expect a "device" pointer to write into and a host pointer to read from.
extern "C" hipError_t hipHostGetDevicePointer(void **devPtr, void *hstPtr,
                                              unsigned int flags) {
  (void)flags;
  *devPtr = hstPtr;
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

extern "C" hipError_t hipMemsetAsync(void *dst, int value, size_t size,
                                     hipStream_t stream) {
  MOCK_PRINT("[MOCK] hipMemsetAsync(dst=%p, value=%d, size=%zu, stream=%p)\n",
             dst, value, size, stream);
  memset(dst, value, size);
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

static miopenStatus_t miopenSet4dTensorDescriptor(miopenTensorDescriptor_t desc,
                                                  miopenDataType_t dataType,
                                                  int n, int c, int h, int w) {
  (void)desc;
  (void)dataType;
  MOCK_PRINT("[MOCK]   Tensor descriptor set: [%d, %d, %d, %d]\n", n, c, h, w);
  return miopenStatusSuccess;
}

static miopenStatus_t miopenSetNdTensorDescriptorWithLayout(
    miopenTensorDescriptor_t desc, miopenDataType_t dataType,
    miopenTensorLayout_t layout, const int *lens, int num_lens) {
  (void)desc;
  (void)dataType;
  (void)layout;
  if (num_lens == 4) {
    MOCK_PRINT("[MOCK]   Tensor descriptor set (NCHW): [%d, %d, %d, %d]\n",
               lens[0], lens[1], lens[2], lens[3]);
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

int wrap_im2d2col(RuntimeState *state, const void *input, int64_t data_type,
                  int64_t C, int64_t H, int64_t W, int64_t kh, int64_t kw,
                  int64_t pad_top, int64_t pad_bottom, int64_t pad_left,
                  int64_t pad_right, int64_t stride_h, int64_t stride_w,
                  int64_t dilation_h, int64_t dilation_w, void *output,
                  int64_t out_h, int64_t out_w) {

  MOCK_PRINT("[MOCK] wrap_im2d2col(\n");
  MOCK_PRINT("[MOCK]   input=[%lld,%lld,%lld],\n", (long long)C, (long long)H,
             (long long)W);
  MOCK_PRINT("[MOCK]   output=[%lld*%lld*%lld,%lld*%lld],\n", (long long)C,
             (long long)kh, (long long)kw, (long long)out_h, (long long)out_w);
  MOCK_PRINT("[MOCK]   stride=[%lld,%lld], pad=[%lld,%lld,%lld,%lld], "
             "dilation=[%lld,%lld], group=%lld)\n",
             (long long)stride_h, (long long)stride_w, (long long)pad_top,
             (long long)pad_bottom, (long long)pad_left, (long long)pad_right,
             (long long)dilation_h, (long long)dilation_w, (long long)1);

  int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
  size_t elem = (elem_bytes > 0) ? (size_t)elem_bytes : sizeof(float);
  size_t output_size = (size_t)C * kh * kw * out_h * out_w * elem;
  memset(output, 0, output_size);
  return 0;
}

// Mock wrapper implementations (called from generated MLIR code)

int wrap_miopenConvolutionForward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group, int64_t data_type) {
  if (!state || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionForward\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_miopenConvolutionForward(\n");
  MOCK_PRINT("[MOCK]   input=[%lld,%lld,%lld,%lld],\n", (long long)input_n,
             (long long)input_c, (long long)input_h, (long long)input_w);
  MOCK_PRINT("[MOCK]   weights=[%lld,%lld,%lld,%lld],\n", (long long)weights_k,
             (long long)(input_c / (group ? group : 1)), (long long)kernel_h,
             (long long)kernel_w);
  MOCK_PRINT("[MOCK]   output=[%lld,%lld,%lld,%lld],\n", (long long)input_n,
             (long long)weights_k, (long long)output_h, (long long)output_w);
  MOCK_PRINT("[MOCK]   stride=[%lld,%lld], pad=[%lld,%lld,%lld,%lld], "
             "dilation=[%lld,%lld], group=%lld)\n",
             (long long)stride_h, (long long)stride_w, (long long)pad_top,
             (long long)pad_left, (long long)pad_bottom, (long long)pad_right,
             (long long)dilation_h, (long long)dilation_w, (long long)group);

  // Mock: Fill output with dummy data (zeros in this case)
  // In a real implementation, this would call MIOpen. Use the actual element
  // size so fp16 buffers aren't overrun.
  int64_t elem_bytes = hipdnn_ep_datatype_size(data_type);
  size_t elem = (elem_bytes > 0) ? (size_t)elem_bytes : sizeof(float);
  size_t output_size = (size_t)input_n * weights_k * output_h * output_w * elem;
  memset(output, 0, output_size);

  return 0;
}

int wrap_miopenConvolutionTranspose(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, const void *bias,
    void *output, int64_t output_c, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t output_padding_h,
    int64_t output_padding_w, int64_t group, int64_t data_type) {
  if (!state || !input || !weights || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenConvolutionTranspose\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_miopenConvolutionTranspose(\n");
  MOCK_PRINT("[MOCK]   input=[%lld,%lld,%lld,%lld],\n", (long long)input_n,
             (long long)input_c, (long long)input_h, (long long)input_w);
  MOCK_PRINT("[MOCK]   weights=[%lld,%lld,%lld,%lld],\n", (long long)input_c,
             (long long)(output_c / (group ? group : 1)), (long long)kernel_h,
             (long long)kernel_w);
  MOCK_PRINT("[MOCK]   output=[%lld,%lld,%lld,%lld], bias=%s,\n",
             (long long)input_n, (long long)output_c, (long long)output_h,
             (long long)output_w, bias ? "yes" : "null");
  MOCK_PRINT("[MOCK]   stride=[%lld,%lld], pad=[%lld,%lld,%lld,%lld], "
             "dilation=[%lld,%lld], output_padding=[%lld,%lld], group=%lld, "
             "dtype=%s)\n",
             (long long)stride_h, (long long)stride_w, (long long)pad_top,
             (long long)pad_left, (long long)pad_bottom, (long long)pad_right,
             (long long)dilation_h, (long long)dilation_w,
             (long long)output_padding_h, (long long)output_padding_w,
             (long long)group, hipdnn_ep_datatype_name(data_type));

  // Mock: zero the output (real implementation calls MIOpen transpose conv).
  size_t output_size = input_n * output_c * output_h * output_w * sizeof(float);
  memset(output, 0, output_size);

  return 0;
}

int wrap_causal_conv_with_state(RuntimeState *state, int op_state_slot,
                                const void *input, const void *weight,
                                const void *bias, const void *past_state,
                                void *output, void *present_state,
                                int64_t batch_size, int64_t channels,
                                int64_t seq_len, int64_t kernel_size,
                                int64_t ndim, int64_t activation,
                                int64_t element_size_bytes) {
  (void)op_state_slot;
  if (!state || !input || !weight || !output || !present_state) {
    fprintf(stderr,
            "Invalid required argument in wrap_causal_conv_with_state\n");
    return -1;
  }

  // CausalConvWithState fused activation enum: 0=none, 1=silu/swish.
  // This is independent of hipdnn_ep_activation_name() which maps generic
  // miopen activations (sigmoid/relu/tanh).
  const char *act_name = (activation == 1) ? "silu" : "none";

  MOCK_PRINT("[MOCK] wrap_causal_conv_with_state(\n");
  MOCK_PRINT("[MOCK]   batch=%lld, channels=%lld, seq_len=%lld, kernel=%lld,\n",
             (long long)batch_size, (long long)channels, (long long)seq_len,
             (long long)kernel_size);
  MOCK_PRINT("[MOCK]   ndim=%lld, activation=%s(%lld), elem_size=%lld,\n",
             (long long)ndim, act_name, (long long)activation,
             (long long)element_size_bytes);
  MOCK_PRINT("[MOCK]   bias=%s, past_state=%s)\n", bias ? "yes" : "null",
             past_state ? "yes" : "null");

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

int wrap_hipblasLtMatmul(RuntimeState *state, int op_state_slot, const void *A,
                         const void *B, void *output, int64_t M, int64_t N,
                         int64_t K, int64_t batch_count, int64_t elem_size,
                         int64_t b_batch_stride) {
  (void)b_batch_stride;
  (void)op_state_slot;
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
    RuntimeState *state, int op_state_slot,
    // Inputs 1-7 (core GQA)
    void *query, void *key, void *value, void *past_key, void *past_value,
    void *seqlens_k, void *total_seq_len,
    // Inputs 8-10 (RoPE)
    void *cos_cache, void *sin_cache, void *position_ids,
    // Inputs 11-14 (advanced features)
    void *attention_bias, void *head_sink, void *k_scale, void *v_scale,
    // Outputs
    void *output, void *present_key, void *present_value, void *output_qk,
    // Attributes (13)
    int64_t num_heads, int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width,
    // Whisper bidirectional-attention flag (mock stub ignores it).
    int32_t no_causal,
    // Shape values (6)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t past_buf_seq, int64_t head_dim, int64_t element_size_bytes,
    int64_t attn_bias_batch, int64_t attn_bias_num_heads) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_group_query_attention\n");
    return -1;
  }
  (void)op_state_slot;

  (void)position_ids;
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
  (void)no_causal;
  (void)past_buf_seq;
  (void)present_key;
  (void)present_value;
  (void)attn_bias_batch;
  (void)attn_bias_num_heads;

  MOCK_PRINT("[MOCK] wrap_group_query_attention(\n");
  MOCK_PRINT("[MOCK]   num_heads=%lld, kv_num_heads=%lld,\n",
             (long long)num_heads, (long long)kv_num_heads);
  MOCK_PRINT("[MOCK]   scale=%f, softcap=%f,\n", (double)scale,
             (double)softcap);
  MOCK_PRINT("[MOCK]   do_rotary=%lld, rotary_interleaved=%lld,\n",
             (long long)do_rotary, (long long)rotary_interleaved);
  MOCK_PRINT("[MOCK]   attention_bias=%p, attn_bias_batch=%lld, "
             "attn_bias_num_heads=%lld,\n",
             attention_bias, (long long)attn_bias_batch,
             (long long)attn_bias_num_heads);
  MOCK_PRINT("[MOCK]   batch=%lld, seq_q=%lld, seq_kv=%lld, "
             "past_buf_seq=%lld, head_dim=%lld, elem_size=%lld)\n",
             (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
             (long long)past_buf_seq, (long long)head_dim,
             (long long)element_size_bytes);

  return 0;
}

int wrap_multi_head_attention(
    RuntimeState *state, int op_state_slot,
    // Inputs (10)
    void *query, void *key, void *value, void *bias, void *key_padding_mask,
    void *attention_bias, void *past_key, void *past_value,
    void *past_sequence_length, void *cache_indirection,
    // Outputs (4)
    void *output, void *present_key, void *present_value, void *qk,
    // Attributes (4)
    int64_t num_heads, float mask_filter_value, float scale,
    int64_t unidirectional,
    // Shape info (8)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t query_hidden, int64_t v_hidden, int64_t head_size,
    int64_t query_rank, int64_t element_size_bytes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_multi_head_attention\n");
    return -1;
  }
  (void)op_state_slot;

  (void)query;
  (void)bias;
  (void)key_padding_mask;
  (void)attention_bias;
  (void)past_key;
  (void)past_value;
  (void)past_sequence_length;
  (void)cache_indirection;
  (void)output;
  (void)present_key;
  (void)present_value;
  (void)qk;

  MOCK_PRINT("[MOCK] wrap_multi_head_attention(\n");
  MOCK_PRINT("[MOCK]   num_heads=%lld, mask_filter_value=%f, scale=%f, "
             "unidirectional=%lld,\n",
             (long long)num_heads, (double)mask_filter_value, (double)scale,
             (long long)unidirectional);
  MOCK_PRINT("[MOCK]   key=%s, value=%s,\n", key ? "yes" : "null",
             value ? "yes" : "null");
  MOCK_PRINT("[MOCK]   batch=%lld, seq_q=%lld, seq_kv=%lld,\n",
             (long long)batch_size, (long long)seq_len_q,
             (long long)seq_len_kv);
  MOCK_PRINT("[MOCK]   query_hidden=%lld, v_hidden=%lld, head_size=%lld, "
             "query_rank=%lld, elem_size=%lld)\n",
             (long long)query_hidden, (long long)v_hidden, (long long)head_size,
             (long long)query_rank, (long long)element_size_bytes);

  return 0;
}

int wrap_linear_attention(RuntimeState *state, const void *query,
                          const void *key, const void *value,
                          const void *past_state, const void *decay,
                          const void *beta, void *output, void *present_state,
                          int64_t Hq, int64_t Hkv, int64_t Nk,
                          int64_t decay_per_key_dim, int64_t beta_per_head,
                          float scale, int64_t chunk_size, int64_t update_rule,
                          int64_t B, int64_t seq_len, int64_t dk, int64_t dv,
                          int64_t type) {
  if (!state || !query || !key || !value || !output || !present_state) {
    fprintf(stderr, "Invalid required argument in wrap_linear_attention\n");
    return -1;
  }

  // LinearAttention update_rule enum: 0=linear, 1=gated, 2=delta,
  // 3=gated_delta. Kept inline here because it is op-specific and does not
  // belong to the generic hipdnn_ep_* enum helpers.
  const char *rule_name = "unknown";
  switch (update_rule) {
  case 0:
    rule_name = "linear";
    break;
  case 1:
    rule_name = "gated";
    break;
  case 2:
    rule_name = "delta";
    break;
  case 3:
    rule_name = "gated_delta";
    break;
  }

  MOCK_PRINT("[MOCK] wrap_linear_attention(\n");
  MOCK_PRINT("[MOCK]   B=%lld, seq_len=%lld, dk=%lld, dv=%lld,\n", (long long)B,
             (long long)seq_len, (long long)dk, (long long)dv);
  MOCK_PRINT("[MOCK]   Hq=%lld, Hkv=%lld, Nk=%lld,\n", (long long)Hq,
             (long long)Hkv, (long long)Nk);
  MOCK_PRINT("[MOCK]   decay_per_key_dim=%lld, beta_per_head=%lld,\n",
             (long long)decay_per_key_dim, (long long)beta_per_head);
  MOCK_PRINT("[MOCK]   scale=%f, chunk_size=%lld, update_rule=%s(%lld),\n",
             (double)scale, (long long)chunk_size, rule_name,
             (long long)update_rule);
  MOCK_PRINT("[MOCK]   type=%lld,\n", (long long)type);
  MOCK_PRINT("[MOCK]   past_state=%s, decay=%s, beta=%s)\n",
             past_state ? "yes" : "null", decay ? "yes" : "null",
             beta ? "yes" : "null");

  return 0;
}

int wrap_miopenOpTensor(RuntimeState *state, int op_state_slot, void *lhs,
                        void *rhs, void *output, int64_t lhs_n, int64_t lhs_c,
                        int64_t lhs_h, int64_t lhs_w, int64_t rhs_n,
                        int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
                        int64_t out_n, int64_t out_c, int64_t out_h,
                        int64_t out_w, int64_t data_type, int64_t tensor_op) {
  (void)op_state_slot;
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
                         void *output, int64_t lhs_n, int64_t lhs_c,
                         int64_t lhs_h, int64_t lhs_w, int64_t rhs_n,
                         int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
                         int64_t out_n, int64_t out_c, int64_t out_h,
                         int64_t out_w, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_elementwise_sub\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_elementwise_sub lhs=[%lld,%lld,%lld,%lld] "
             "rhs=[%lld,%lld,%lld,%lld] out=[%lld,%lld,%lld,%lld] dtype=%s\n",
             (long long)lhs_n, (long long)lhs_c, (long long)lhs_h,
             (long long)lhs_w, (long long)rhs_n, (long long)rhs_c,
             (long long)rhs_h, (long long)rhs_w, (long long)out_n,
             (long long)out_c, (long long)out_h, (long long)out_w,
             hipdnn_ep_datatype_name(data_type));

  return 0;
}

int wrap_gather(RuntimeState *state, void *data, void *indices, void *output,
                int64_t axis, int64_t data_num_elements,
                int64_t indices_num_elements, int64_t output_num_elements,
                int64_t axis_size, int64_t inner_size,
                int64_t element_size_bytes,
                int64_t indices_element_size_bytes) {
  (void)data;
  (void)indices;
  (void)output;
  (void)axis_size;
  (void)inner_size;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_gather\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_gather(axis=%lld, data_num_elements=%lld, "
             "indices_num_elements=%lld, output_num_elements=%lld, "
             "axis_size=%lld, inner_size=%lld, element_size=%lld, "
             "indices_element_size=%lld)\n",
             (long long)axis, (long long)data_num_elements,
             (long long)indices_num_elements, (long long)output_num_elements,
             (long long)axis_size, (long long)inner_size,
             (long long)element_size_bytes,
             (long long)indices_element_size_bytes);

  return 0;
}

int wrap_one_hot(RuntimeState *state, void *indices, void *depth, void *values,
                 void *output, int64_t axis, int64_t indices_rank,
                 int64_t output_rank, const int64_t *indices_shape,
                 const int64_t *output_shape, int64_t num_indices,
                 int64_t num_output_elements, int64_t element_size_bytes,
                 int64_t indices_element_size_bytes,
                 int64_t depth_element_size_bytes) {
  (void)indices;
  (void)depth;
  (void)values;
  (void)output;
  (void)indices_shape;
  (void)output_shape;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_one_hot\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_one_hot(axis=%lld, idx_rank=%lld, out_rank=%lld, "
             "num_idx=%lld, num_out=%lld, elem=%lld, idx_elem=%lld, "
             "depth_elem=%lld)\n",
             (long long)axis, (long long)indices_rank, (long long)output_rank,
             (long long)num_indices, (long long)num_output_elements,
             (long long)element_size_bytes,
             (long long)indices_element_size_bytes,
             (long long)depth_element_size_bytes);
  return 0;
}

int wrap_compress(RuntimeState *state, void *input, void *condition,
                  void *output, int64_t flatten, int64_t axis,
                  int64_t input_rank, int64_t output_rank,
                  const int64_t *input_shape, const int64_t *output_shape,
                  int64_t condition_len, int64_t num_output_elements,
                  int64_t element_size_bytes) {
  (void)input;
  (void)condition;
  (void)output;
  (void)input_shape;
  (void)output_shape;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_compress\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_compress(flatten=%lld, axis=%lld, in_rank=%lld, "
             "out_rank=%lld, cond_len=%lld, num_out=%lld, elem=%lld)\n",
             (long long)flatten, (long long)axis, (long long)input_rank,
             (long long)output_rank, (long long)condition_len,
             (long long)num_output_elements, (long long)element_size_bytes);
  return 0;
}

int wrap_scatter_elements(RuntimeState *state, void *data, void *indices,
                          void *updates, void *output, int64_t axis,
                          int64_t reduction_id, int64_t rank,
                          const int64_t *data_shape,
                          const int64_t *indices_shape, int64_t num_updates,
                          int64_t element_size_bytes,
                          int64_t indices_element_size_bytes) {
  (void)data;
  (void)indices;
  (void)updates;
  (void)output;
  (void)data_shape;
  (void)indices_shape;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_scatter_elements\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_scatter_elements(axis=%lld, reduction=%lld, "
             "rank=%lld, num_updates=%lld, element_size=%lld, "
             "indices_element_size=%lld)\n",
             (long long)axis, (long long)reduction_id, (long long)rank,
             (long long)num_updates, (long long)element_size_bytes,
             (long long)indices_element_size_bytes);
  return 0;
}

int wrap_gather_elements(RuntimeState *state, void *data, void *indices,
                         void *output, int64_t axis, int64_t rank,
                         const int64_t *data_shape,
                         const int64_t *indices_shape, int64_t num_elements,
                         int64_t element_size_bytes,
                         int64_t indices_element_size_bytes) {
  (void)data;
  (void)indices;
  (void)output;
  (void)data_shape;
  (void)indices_shape;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_gather_elements\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_gather_elements(axis=%lld, rank=%lld, "
             "num_elements=%lld, element_size=%lld, "
             "indices_element_size=%lld)\n",
             (long long)axis, (long long)rank, (long long)num_elements,
             (long long)element_size_bytes,
             (long long)indices_element_size_bytes);
  return 0;
}

int wrap_top_k(RuntimeState *state, void *x, void *k, void *values,
               void *indices, int64_t axis, int64_t largest, int64_t sorted,
               int64_t rank, const int64_t *x_shape, int64_t num_elements,
               int64_t element_size_bytes) {
  (void)x;
  (void)k;
  (void)values;
  (void)indices;
  (void)x_shape;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_top_k\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_top_k(axis=%lld, largest=%lld, sorted=%lld, "
             "rank=%lld, num_elements=%lld, element_size=%lld)\n",
             (long long)axis, (long long)largest, (long long)sorted,
             (long long)rank, (long long)num_elements,
             (long long)element_size_bytes);
  return 0;
}

int wrap_range(RuntimeState *state, void *start, void *limit, void *delta,
               void *output, int64_t output_num_elements, int64_t hip_dtype) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_range\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_range(output_num_elements=%lld, hip_dtype=%lld)\n",
             (long long)output_num_elements, (long long)hip_dtype);
  return 0;
}

int wrap_reduce_max(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_max\n");
    return -1;
  }

  MOCK_PRINT(
      "[MOCK] wrap_reduce_max(data_num_elements=%lld, "
      "output_num_elements=%lld, axes_num_elements=%lld, data_type=%s(%lld), "
      "keepdims=%lld, noop_with_empty_axes=%lld)\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, hipdnn_ep_datatype_name(data_type),
      (long long)data_type, (long long)keepdims,
      (long long)noop_with_empty_axes);

  return 0;
}

int wrap_reduce_min(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_min\n");
    return -1;
  }

  MOCK_PRINT(
      "[MOCK] wrap_reduce_min(data_num_elements=%lld, "
      "output_num_elements=%lld, axes_num_elements=%lld, data_type=%s(%lld), "
      "keepdims=%lld, noop_with_empty_axes=%lld)\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, hipdnn_ep_datatype_name(data_type),
      (long long)data_type, (long long)keepdims,
      (long long)noop_with_empty_axes);

  return 0;
}

int wrap_reduce_sum(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_sum\n");
    return -1;
  }

  MOCK_PRINT(
      "[MOCK] wrap_reduce_sum(data_num_elements=%lld, "
      "output_num_elements=%lld, axes_num_elements=%lld, data_type=%s(%lld), "
      "keepdims=%lld, noop_with_empty_axes=%lld)\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, hipdnn_ep_datatype_name(data_type),
      (long long)data_type, (long long)keepdims,
      (long long)noop_with_empty_axes);

  return 0;
}

int wrap_reduce_mean(RuntimeState *state, void *data, void *axes, void *output,
                     int64_t data_num_elements, int64_t output_num_elements,
                     int64_t axes_num_elements, int64_t data_type,
                     int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_mean\n");
    return -1;
  }

  MOCK_PRINT(
      "[MOCK] wrap_reduce_mean(data_num_elements=%lld, "
      "output_num_elements=%lld, axes_num_elements=%lld, data_type=%s(%lld), "
      "keepdims=%lld, noop_with_empty_axes=%lld)\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, hipdnn_ep_datatype_name(data_type),
      (long long)data_type, (long long)keepdims,
      (long long)noop_with_empty_axes);

  return 0;
}

int wrap_reduce_l2(RuntimeState *state, void *data, void *axes, void *output,
                   int64_t data_num_elements, int64_t output_num_elements,
                   int64_t axes_num_elements, int64_t data_type,
                   int64_t keepdims, int64_t noop_with_empty_axes,
                   int64_t inner_size) {
  (void)axes;
  (void)inner_size;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_l2\n");
    return -1;
  }

  MOCK_PRINT(
      "[MOCK] wrap_reduce_l2(data_num_elements=%lld, "
      "output_num_elements=%lld, axes_num_elements=%lld, data_type=%s(%lld), "
      "keepdims=%lld, noop_with_empty_axes=%lld, inner_size=%lld)\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, hipdnn_ep_datatype_name(data_type),
      (long long)data_type, (long long)keepdims,
      (long long)noop_with_empty_axes, (long long)inner_size);

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

int wrap_pool(RuntimeState *state, void *input, void *output, void *indices,
              int64_t data_type, int64_t pool_mode, int64_t spatial_rank,
              int64_t N, int64_t C, int64_t in0, int64_t in1, int64_t in2,
              int64_t out0, int64_t out1, int64_t out2, int64_t k0, int64_t k1,
              int64_t k2, int64_t s0, int64_t s1, int64_t s2, int64_t p0,
              int64_t p1, int64_t p2, int64_t dil0, int64_t dil1, int64_t dil2,
              int64_t storage_order, int64_t ceil_mode, int64_t has_indices,
              int64_t count_include_pad, int64_t p) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_pool\n");
    return -1;
  }
  (void)k0;
  (void)k1;
  (void)k2;
  (void)s0;
  (void)s1;
  (void)s2;
  (void)p0;
  (void)p1;
  (void)p2;
  (void)dil0;
  (void)dil1;
  (void)dil2;
  MOCK_PRINT("[MOCK] wrap_pool(mode=%s, dtype=%s(%lld), rank=%lld, N=%lld, "
             "C=%lld, in=[%lld,%lld,%lld], out=[%lld,%lld,%lld], "
             "storage_order=%lld, ceil_mode=%lld, has_indices=%lld, "
             "count_include_pad=%lld, p=%lld, in=%s, out=%s, idx=%s)\n",
             hipdnn_ep_pool_mode_name(pool_mode),
             hipdnn_ep_datatype_name(data_type), (long long)data_type,
             (long long)spatial_rank, (long long)N, (long long)C,
             (long long)in0, (long long)in1, (long long)in2, (long long)out0,
             (long long)out1, (long long)out2, (long long)storage_order,
             (long long)ceil_mode, (long long)has_indices,
             (long long)count_include_pad, (long long)p, input ? "yes" : "null",
             output ? "yes" : "null", indices ? "yes" : "null");
  return 0;
}

int wrap_resize(RuntimeState *state, void *input, void *output,
                int64_t data_type, int64_t spatial_rank, int64_t N, int64_t C,
                int64_t in0, int64_t in1, int64_t in2, int64_t out0,
                int64_t out1, int64_t out2, int64_t mode,
                int64_t coord_transform, int64_t nearest_mode) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_resize\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_resize(dtype=%s(%lld), rank=%lld, N=%lld, C=%lld, "
             "in=[%lld,%lld,%lld], out=[%lld,%lld,%lld], mode=%lld, "
             "coord_transform=%lld, nearest_mode=%lld, in=%s, out=%s)\n",
             hipdnn_ep_datatype_name(data_type), (long long)data_type,
             (long long)spatial_rank, (long long)N, (long long)C,
             (long long)in0, (long long)in1, (long long)in2, (long long)out0,
             (long long)out1, (long long)out2, (long long)mode,
             (long long)coord_transform, (long long)nearest_mode,
             input ? "yes" : "null", output ? "yes" : "null");
  return 0;
}

int wrap_global_pool(RuntimeState *state, void *input, void *output,
                     int64_t outer, int64_t reduce_size, int64_t data_type,
                     int64_t mode, int64_t p) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_global_pool\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_global_pool(mode=%s(%lld), p=%lld, outer=%lld, "
             "reduce_size=%lld, data_type=%s(%lld), elem_size=%lld, in=%s, "
             "out=%s)\n",
             hipdnn_ep_global_pool_mode_name(mode), (long long)mode,
             (long long)p, (long long)outer, (long long)reduce_size,
             hipdnn_ep_datatype_name(data_type), (long long)data_type,
             (long long)hipdnn_ep_datatype_size(data_type),
             input ? "yes" : "null", output ? "yes" : "null");

  return 0;
}

int wrap_miopenActivationForward(RuntimeState *state, int op_state_slot,
                                 void *input, void *output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode) {
  (void)op_state_slot;
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

int wrap_bias_gelu(RuntimeState *state, void *data, void *bias, void *output,
                   int64_t num_elements, int64_t bias_len, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_bias_gelu\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_bias_gelu(num_elements=%lld, bias_len=%lld, "
             "data_type=%s(%lld))\n",
             (long long)num_elements, (long long)bias_len,
             hipdnn_ep_datatype_name(data_type), (long long)data_type);

  return 0;
}

int wrap_fast_gelu(RuntimeState *state, void *input, void *bias, void *output,
                   int64_t num_elements, int64_t bias_len, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_fast_gelu\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_fast_gelu(num_elements=%lld, bias_len=%lld, "
             "data_type=%s(%lld), bias=%s)\n",
             (long long)num_elements, (long long)bias_len,
             hipdnn_ep_datatype_name(data_type), (long long)data_type,
             bias ? "yes" : "null");

  return 0;
}

int wrap_leaky_relu(RuntimeState *state, void *input, void *output,
                    int64_t num_elements, int64_t data_type, double alpha) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_leaky_relu\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_leaky_relu(num_elements=%lld, data_type=%s(%lld), "
             "alpha=%f)\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type, alpha);

  return 0;
}

// Mock impl of the runtime symbol referenced by the hip.miopen.softmax
// lowering. Signature must match lib/Runtime/real/miopen.cpp.
extern "C" int hip_miopen_softmax(RuntimeState *state, const void *input,
                                  void *output, int64_t rows, int64_t cols,
                                  int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in hip_miopen_softmax\n");
    return -1;
  }
  (void)input;
  (void)output;
  MOCK_PRINT("[MOCK] hip_miopen_softmax(rows=%lld, cols=%lld, "
             "data_type=%s(%lld))\n",
             (long long)rows, (long long)cols,
             hipdnn_ep_datatype_name(data_type), (long long)data_type);
  return 0;
}

int wrap_rotary_embedding(RuntimeState *state, void *input, void *position_ids,
                          void *cos_cache, void *sin_cache, void *output,
                          int64_t interleaved, int64_t batch_size,
                          int64_t seq_len, int64_t num_heads, int64_t head_dim,
                          int64_t rotary_dim, int64_t cos_cache_num_elements,
                          int64_t element_size_bytes, int64_t is_bnsh) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_rotary_embedding\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_rotary_embedding(interleaved=%lld, batch=%lld, "
             "seq_len=%lld, num_heads=%lld, head_dim=%lld, rotary_dim=%lld, "
             "cos_cache_num_elements=%lld, element_size=%lld, is_bnsh=%lld)\n",
             (long long)interleaved, (long long)batch_size, (long long)seq_len,
             (long long)num_heads, (long long)head_dim, (long long)rotary_dim,
             (long long)cos_cache_num_elements, (long long)element_size_bytes,
             (long long)is_bnsh);

  return 0;
}

int wrap_miopenT5LayerNormForward(RuntimeState *state, int op_state_slot,
                                  void *input, void *scale, void *output,
                                  int64_t input_num_elements,
                                  int64_t scale_num_elements,
                                  int64_t element_size_bytes, int64_t axis,
                                  float epsilon, int64_t stash_type) {
  (void)op_state_slot;
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

int wrap_skip_simplified_layer_norm(RuntimeState *state, int op_state_slot,
                                    void *input, void *skip, void *gamma,
                                    void *bias, void *output,
                                    void *input_skip_bias_sum,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes, float epsilon) {
  (void)op_state_slot;
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

int wrap_gemm(RuntimeState *state, int op_state_slot, const void *A,
              const void *B, const void *C, void *output, int64_t M, int64_t N,
              int64_t K, float alpha, float beta, int64_t transA,
              int64_t transB, int64_t typeCode, int64_t cDim0, int64_t cDim1) {
  MOCK_PRINT("[MOCK] wrap_gemm\n");
  return 0;
}

int wrap_matmul_nbits(RuntimeState *state, int op_state_slot, const void *A,
                      const void *B, const void *scales,
                      const void *zero_points, const void *g_idx,
                      const void *bias, void *output, int64_t M, int64_t N,
                      int64_t K, int64_t batch_count, int64_t bits,
                      int64_t block_size, int64_t elem_size,
                      int64_t zp_elem_size) {
  (void)op_state_slot;
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

int wrap_gather_block_quantized(
    RuntimeState *state, const void *data, const void *indices,
    const void *scales, const void *zero_points, void *output,
    const int64_t *data_shape, int64_t data_rank, const int64_t *indices_shape,
    int64_t indices_rank, const int64_t *scales_shape, int64_t scales_rank,
    const int64_t *output_shape, int64_t output_rank, int64_t bits,
    int64_t block_size, int64_t gather_axis, int64_t quantize_axis,
    int64_t data_dtype, int64_t indices_dtype, int64_t scales_dtype) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_gather_block_quantized\n");
    return -1;
  }
  (void)data;
  (void)indices;
  (void)scales;
  (void)zero_points;
  (void)output;
  (void)data_shape;
  (void)indices_shape;
  (void)scales_shape;
  (void)output_shape;

  MOCK_PRINT(
      "[MOCK] wrap_gather_block_quantized(data_rank=%lld, indices_rank=%lld, "
      "scales_rank=%lld, output_rank=%lld, bits=%lld, block_size=%lld, "
      "gather_axis=%lld, quantize_axis=%lld, data_dtype=%s(%lld), "
      "indices_dtype=%s(%lld), scales_dtype=%s(%lld), zero_points=%s)\n",
      (long long)data_rank, (long long)indices_rank, (long long)scales_rank,
      (long long)output_rank, (long long)bits, (long long)block_size,
      (long long)gather_axis, (long long)quantize_axis,
      hipdnn_ep_datatype_name(data_dtype), (long long)data_dtype,
      hipdnn_ep_datatype_name(indices_dtype), (long long)indices_dtype,
      hipdnn_ep_datatype_name(scales_dtype), (long long)scales_dtype,
      zero_points ? "yes" : "null");
  return 0;
}

int wrap_qmoe(RuntimeState *state, const void *input, const void *router_probs,
              const void *router_weights, const void *fc1_weights,
              const void *fc1_scales, const void *fc1_bias,
              const void *fc2_weights, const void *fc2_scales,
              const void *fc2_bias, const void *fc3_weights,
              const void *fc3_scales, const void *fc3_bias,
              const void *fc1_zero_points, const void *fc2_zero_points,
              const void *fc3_zero_points, void *output, int64_t num_tokens,
              int64_t hidden_size, int64_t inter_size, int64_t num_experts,
              int64_t k, int64_t expert_weight_bits, int64_t block_size,
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
             "fc1_zp=%s, fc2_zp=%s, router_weights=%s)\n",
             fc1_bias ? "yes" : "null", fc2_bias ? "yes" : "null",
             fc3_weights ? "yes" : "null", fc1_zero_points ? "yes" : "null",
             fc2_zero_points ? "yes" : "null", router_weights ? "yes" : "null");

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

// hip.reciprocal / hip.sqrt lower to wrap_power(..., alpha, beta, gamma).
int wrap_power(RuntimeState *state, void *input, void *output,
               int64_t num_elements, int64_t data_type, double alpha,
               double beta, double gamma) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_power\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_power(num_elements=%lld, data_type=%s(%lld), "
             "alpha=%g, beta=%g, gamma=%g)\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type, alpha, beta, gamma);

  return 0;
}

// ONNX Where: output[i] = condition[i] ? x[i] : y[i] with multidirectional
// broadcasting. Mirrors the validation in real/where.cpp so the same bad
// inputs are rejected in both builds.
int wrap_where(RuntimeState *state, void *condition, void *x, void *y,
               void *output, const int64_t *cond_shape, int64_t cond_rank,
               const int64_t *x_shape, int64_t x_rank, const int64_t *y_shape,
               int64_t y_rank, const int64_t *out_shape, int64_t out_rank,
               int64_t data_type) {
  if (!state || !condition || !x || !y || !output) {
    fprintf(stderr, "wrap_where: null tensor argument\n");
    return -1;
  }
  if ((cond_rank > 0 && !cond_shape) || (x_rank > 0 && !x_shape) ||
      (y_rank > 0 && !y_shape) || (out_rank > 0 && !out_shape)) {
    fprintf(stderr, "wrap_where: null shape argument with non-zero rank\n");
    return -1;
  }

  auto dump_shape = [](const char *name, const int64_t *shape, int64_t rank) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf), "[MOCK]   %s rank=%lld shape=[", name,
                     (long long)rank);
    for (int64_t i = 0; i < rank && n < (int)sizeof(buf); ++i) {
      n += snprintf(buf + n, sizeof(buf) - n, "%s%lld", i ? "," : "",
                    (long long)shape[i]);
    }
    if (n < (int)sizeof(buf))
      snprintf(buf + n, sizeof(buf) - n, "]\n");
    MOCK_PRINT("%s", buf);
  };

  MOCK_PRINT("[MOCK] wrap_where(data_type=%s(%lld))\n",
             hipdnn_ep_datatype_name(data_type), (long long)data_type);
  dump_shape("condition", cond_shape, cond_rank);
  dump_shape("x", x_shape, x_rank);
  dump_shape("y", y_shape, y_rank);
  dump_shape("output", out_shape, out_rank);

  return 0;
}

int wrap_equal(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
               int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
               int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
               int64_t out_h, int64_t out_w, int64_t data_type) {
  (void)a;
  (void)b;
  (void)output;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_equal\n");
    return -1;
  }
  MOCK_PRINT(
      "[MOCK] wrap_equal a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld] "
      "out=[%lld,%lld,%lld,%lld] dtype=%s\n",
      (long long)a_n, (long long)a_c, (long long)a_h, (long long)a_w,
      (long long)b_n, (long long)b_c, (long long)b_h, (long long)b_w,
      (long long)out_n, (long long)out_c, (long long)out_h, (long long)out_w,
      hipdnn_ep_datatype_name(data_type));
  return 0;
}

int wrap_or(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
            int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
            int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
            int64_t out_h, int64_t out_w, int64_t data_type) {
  (void)a;
  (void)b;
  (void)output;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_or\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_or a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld] "
             "out=[%lld,%lld,%lld,%lld] dtype=%s\n",
             (long long)a_n, (long long)a_c, (long long)a_h, (long long)a_w,
             (long long)b_n, (long long)b_c, (long long)b_h, (long long)b_w,
             (long long)out_n, (long long)out_c, (long long)out_h,
             (long long)out_w, hipdnn_ep_datatype_name(data_type));
  return 0;
}

int wrap_and(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
             int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
             int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
             int64_t out_h, int64_t out_w, int64_t data_type) {
  (void)a;
  (void)b;
  (void)output;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_and\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_and a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld] "
             "out=[%lld,%lld,%lld,%lld] dtype=%s\n",
             (long long)a_n, (long long)a_c, (long long)a_h, (long long)a_w,
             (long long)b_n, (long long)b_c, (long long)b_h, (long long)b_w,
             (long long)out_n, (long long)out_c, (long long)out_h,
             (long long)out_w, hipdnn_ep_datatype_name(data_type));
  return 0;
}

int wrap_div(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t lhs_n, int64_t lhs_c, int64_t lhs_h, int64_t lhs_w,
             int64_t rhs_n, int64_t rhs_c, int64_t rhs_h, int64_t rhs_w,
             int64_t out_n, int64_t out_c, int64_t out_h, int64_t out_w,
             int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_div\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_div lhs=[%lld,%lld,%lld,%lld] "
             "rhs=[%lld,%lld,%lld,%lld] out=[%lld,%lld,%lld,%lld] dtype=%s\n",
             (long long)lhs_n, (long long)lhs_c, (long long)lhs_h,
             (long long)lhs_w, (long long)rhs_n, (long long)rhs_c,
             (long long)rhs_h, (long long)rhs_w, (long long)out_n,
             (long long)out_c, (long long)out_h, (long long)out_w,
             hipdnn_ep_datatype_name(data_type));
  return 0;
}

int wrap_abs(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_abs\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_abs(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_neg(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_neg\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_neg(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_not(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_not\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_not(num_elements=%lld, data_type=%lld)\n",
             (long long)num_elements, (long long)data_type);
  return 0;
}

// Mock memory is plain host memory, so we can compute the true non-zero count
// directly from `input`. This makes the host-readback path (hip.readback_dim
// -> hipdnn_ep_readback_i32) return a meaningful dynamic dim under mock builds.
template <typename T>
static int32_t mock_count_nonzero(const void *data, int64_t n) {
  const T *p = static_cast<const T *>(data);
  int32_t count = 0;
  for (int64_t i = 0; i < n; ++i)
    if (p[i] != static_cast<T>(0))
      ++count;
  return count;
}

int wrap_nonzero(RuntimeState *state, void *input, void *output,
                 int32_t *count_ptr, int64_t input_num_elements,
                 int64_t input_rank, const int64_t *input_dims,
                 int64_t output_capacity, int64_t input_data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_nonzero\n");
    return -1;
  }
  (void)output;
  (void)input_dims;

  int32_t count = 0;
  if (input && count_ptr) {
    switch (input_data_type) {
    case HIPDNN_EP_DATATYPE_FLOAT:
      count = mock_count_nonzero<float>(input, input_num_elements);
      break;
    case HIPDNN_EP_DATATYPE_INT32:
      count = mock_count_nonzero<int32_t>(input, input_num_elements);
      break;
    case HIPDNN_EP_DATATYPE_INT64:
      count = mock_count_nonzero<int64_t>(input, input_num_elements);
      break;
    case HIPDNN_EP_DATATYPE_INT8:
    case HIPDNN_EP_DATATYPE_UINT8:
      count = mock_count_nonzero<int8_t>(input, input_num_elements);
      break;
    default:
      // f16 and any other types: leave count at 0 (mock has no fp16 type).
      break;
    }
  }
  if (count_ptr)
    *count_ptr = count;

  MOCK_PRINT("[MOCK] wrap_nonzero(input_num_elements=%lld, input_rank=%lld, "
             "output_capacity=%lld, input_data_type=%s(%lld)) -> count=%d\n",
             (long long)input_num_elements, (long long)input_rank,
             (long long)output_capacity,
             hipdnn_ep_datatype_name(input_data_type),
             (long long)input_data_type, count);
  return 0;
}

int32_t hipdnn_ep_readback_i32(RuntimeState *state, const void *device_scalar) {
  (void)state;
  // Mock "device" memory is host memory: read the scalar directly.
  if (!device_scalar)
    return 0;
  return *static_cast<const int32_t *>(device_scalar);
}

void hipdnn_ep_readback_scalar(RuntimeState *state, void *host_dst,
                               const void *device_scalar, int64_t num_bytes) {
  (void)state;
  // Mock "device" memory is host memory: copy the scalar directly.
  if (!host_dst || !device_scalar || num_bytes <= 0)
    return;
  memcpy(host_dst, device_scalar, static_cast<size_t>(num_bytes));
}

int wrap_cos(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_cos\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_cos(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_sin(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_sin\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_sin(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_ceil(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_ceil\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_ceil(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_exp(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_exp\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_exp(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_log(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_log\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_log(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_cumsum(RuntimeState *state, void *x, void *axis, void *y,
                const int64_t *data_shape, int64_t data_rank,
                int64_t num_elements, int64_t data_type, int64_t axis_dtype,
                int64_t exclusive, int64_t reverse) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_cumsum\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_cumsum(data_rank=%lld, num_elements=%lld, "
             "data_type=%s(%lld), axis_dtype=%s(%lld), exclusive=%lld, "
             "reverse=%lld)\n",
             (long long)data_rank, (long long)num_elements,
             hipdnn_ep_datatype_name(data_type), (long long)data_type,
             hipdnn_ep_datatype_name(axis_dtype), (long long)axis_dtype,
             (long long)exclusive, (long long)reverse);
  return 0;
}

int wrap_pad(RuntimeState *state, void *data, void *pads, void *constant_value,
             void *axes, void *output, const int64_t *data_shape,
             int64_t data_rank, const int64_t *output_shape,
             int64_t output_rank, int64_t pads_num_elements,
             int64_t axes_num_elements, int64_t data_type, int64_t mode_id) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_pad\n");
    return -1;
  }
  static const char *kPadModes[] = {"constant", "reflect", "edge", "wrap"};
  const char *mode_name =
      (mode_id >= 0 && mode_id < 4) ? kPadModes[mode_id] : "unknown";
  MOCK_PRINT("[MOCK] wrap_pad(data_rank=%lld, output_rank=%lld, "
             "pads_num=%lld, axes_num=%lld, data_type=%s(%lld), mode=%s(%lld), "
             "cval=%s)\n",
             (long long)data_rank, (long long)output_rank,
             (long long)pads_num_elements, (long long)axes_num_elements,
             hipdnn_ep_datatype_name(data_type), (long long)data_type,
             mode_name, (long long)mode_id, constant_value ? "yes" : "null");
  return 0;
}

int wrap_tile(RuntimeState *state, void *input, void *repeats, void *output,
              const int64_t *input_shape, int64_t input_rank,
              const int64_t *output_shape, int64_t output_rank,
              int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_tile\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_tile(input_rank=%lld, output_rank=%lld, "
             "data_type=%s(%lld))\n",
             (long long)input_rank, (long long)output_rank,
             hipdnn_ep_datatype_name(data_type), (long long)data_type);
  return 0;
}

int wrap_expand(RuntimeState *state, void *input, void *shape, void *output,
                const int64_t *input_shape, int64_t input_rank,
                const int64_t *output_shape, int64_t output_rank,
                int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_expand\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_expand(input_rank=%lld, output_rank=%lld, "
             "data_type=%s(%lld))\n",
             (long long)input_rank, (long long)output_rank,
             hipdnn_ep_datatype_name(data_type), (long long)data_type);
  return 0;
}

int wrap_reduce_prod(RuntimeState *state, void *data, void *axes, void *output,
                     int64_t data_num_elements, int64_t output_num_elements,
                     int64_t axes_num_elements, int64_t data_type,
                     int64_t keepdims, int64_t noop_with_empty_axes) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_reduce_prod\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_reduce_prod(data_num_elements=%lld, "
             "output_num_elements=%lld, axes_num_elements=%lld, "
             "data_type=%s(%lld), keepdims=%lld, noop_with_empty_axes=%lld)\n",
             (long long)data_num_elements, (long long)output_num_elements,
             (long long)axes_num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type, (long long)keepdims,
             (long long)noop_with_empty_axes);
  return 0;
}

int wrap_less(RuntimeState *state, void *a, void *b, void *output, int64_t a_n,
              int64_t a_c, int64_t a_h, int64_t a_w, int64_t b_n, int64_t b_c,
              int64_t b_h, int64_t b_w, int64_t out_n, int64_t out_c,
              int64_t out_h, int64_t out_w, int64_t data_type) {
  (void)a;
  (void)b;
  (void)output;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_less\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_less a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld] "
             "out=[%lld,%lld,%lld,%lld] dtype=%s\n",
             (long long)a_n, (long long)a_c, (long long)a_h, (long long)a_w,
             (long long)b_n, (long long)b_c, (long long)b_h, (long long)b_w,
             (long long)out_n, (long long)out_c, (long long)out_h,
             (long long)out_w, hipdnn_ep_datatype_name(data_type));
  return 0;
}

int wrap_gather_nd(RuntimeState *state, void *data, void *indices, void *output,
                   const int64_t *data_shape, int64_t data_rank,
                   const int64_t *indices_shape, int64_t indices_rank,
                   const int64_t *output_shape, int64_t output_rank,
                   int64_t batch_dims, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_gather_nd\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_gather_nd(data_rank=%lld, indices_rank=%lld, "
             "output_rank=%lld, batch_dims=%lld, data_type=%s(%lld))\n",
             (long long)data_rank, (long long)indices_rank,
             (long long)output_rank, (long long)batch_dims,
             hipdnn_ep_datatype_name(data_type), (long long)data_type);
  return 0;
}

int wrap_slice(RuntimeState *state, void *data, void *starts, void *ends,
               void *axes, void *steps, void *output, const int64_t *data_shape,
               int64_t data_rank, const int64_t *output_shape,
               int64_t output_rank, int64_t starts_num_elements,
               int64_t axes_num_elements, int64_t steps_num_elements,
               int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_slice\n");
    return -1;
  }
  (void)data;
  (void)starts;
  (void)ends;
  (void)output;
  (void)data_shape;
  (void)output_shape;
  MOCK_PRINT("[MOCK] wrap_slice(data_rank=%lld, output_rank=%lld, "
             "starts_n=%lld, axes_n=%lld (%s), steps_n=%lld (%s), "
             "data_type=%s(%lld))\n",
             (long long)data_rank, (long long)output_rank,
             (long long)starts_num_elements, (long long)axes_num_elements,
             axes ? "yes" : "null", (long long)steps_num_elements,
             steps ? "yes" : "null", hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_scatter_nd(RuntimeState *state, void *data, void *indices,
                    void *updates, void *output, const int32_t *count_ptr,
                    const int64_t *data_shape, int64_t data_rank,
                    const int64_t *indices_shape, int64_t indices_rank,
                    const int64_t *updates_shape, int64_t updates_rank,
                    const int64_t *output_shape, int64_t output_rank,
                    int64_t reduction_id, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_scatter_nd\n");
    return -1;
  }
  (void)data;
  (void)indices;
  (void)updates;
  (void)output;
  (void)count_ptr;
  (void)data_shape;
  (void)indices_shape;
  (void)updates_shape;
  (void)output_shape;
  MOCK_PRINT("[MOCK] wrap_scatter_nd(data_rank=%lld, indices_rank=%lld, "
             "updates_rank=%lld, output_rank=%lld, reduction_id=%lld, "
             "data_type=%s(%lld), has_count=%d)\n",
             (long long)data_rank, (long long)indices_rank,
             (long long)updates_rank, (long long)output_rank,
             (long long)reduction_id, hipdnn_ep_datatype_name(data_type),
             (long long)data_type, count_ptr != nullptr);
  return 0;
}

int wrap_sign(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_sign\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_sign(num_elements=%lld, data_type=%s(%lld))\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type);
  return 0;
}

int wrap_mod(RuntimeState *state, void *lhs, void *rhs, void *output,
             int64_t num_elements, int64_t data_type, int64_t fmod) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_mod\n");
    return -1;
  }
  MOCK_PRINT("[MOCK] wrap_mod(num_elements=%lld, data_type=%s(%lld), "
             "fmod=%lld)\n",
             (long long)num_elements, hipdnn_ep_datatype_name(data_type),
             (long long)data_type, (long long)fmod);
  return 0;
}

int wrap_layer_normalization(RuntimeState *state, void *input, void *scale,
                             void *bias, void *output, void *mean,
                             void *inv_std, int64_t input_num_elements,
                             int64_t scale_num_elements,
                             int64_t element_size_bytes, int64_t axis,
                             float epsilon, int64_t stash_type) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_layer_normalization\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_layer_normalization(input_num_elements=%lld, "
             "scale_num_elements=%lld, element_size_bytes=%lld, "
             "axis=%lld, epsilon=%f, stash_type=%lld, "
             "bias=%s, mean=%s, inv_std=%s)\n",
             (long long)input_num_elements, (long long)scale_num_elements,
             (long long)element_size_bytes, (long long)axis, epsilon,
             (long long)stash_type, bias ? "yes" : "null",
             mean ? "yes" : "null", inv_std ? "yes" : "null");

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
