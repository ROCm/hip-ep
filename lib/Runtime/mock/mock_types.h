/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_RUNTIME_MOCK_TYPES_H
#define HIPDNN_EP_RUNTIME_MOCK_TYPES_H

// Mock type definitions for testing without GPU
typedef void *hipStream_t;
typedef void *hipEvent_t;
typedef void *miopenHandle_t;
typedef void *hipblasLtHandle_t;
typedef int hipError_t;
typedef int miopenStatus_t;
typedef int hipblasStatus_t;

struct hipDeviceProp_t {
  char name[256];
  char gcnArchName[256];
  int integrated; // 0 = discrete GPU, 1 = integrated GPU
};

#define hipSuccess 0
#define miopenStatusSuccess 0
#define HIPBLAS_STATUS_SUCCESS 0
#define hipMemcpyHostToDevice 0
#define hipMemcpyDeviceToHost 1
#define hipHostMallocDefault 0
#define hipHostMallocMapped 0
#define hipHostMallocNonCoherent 0
#define hipEventDisableTiming 0
// Mock stub: the real flag drops hipEventRecord's system-scope fence (see
// op_profile.cpp). The mock hipEventCreateWithFlags ignores flags, so the
// value is irrelevant -- it only needs to be a declared identifier.
#define hipEventDisableSystemFence 0

// MIOpen tensor layout enum (subset used by the runtime)
typedef int miopenTensorLayout_t;
#define miopenTensorNCHW 0

// Forward declarations for mock GPU functions (defined in mock_gpu.cpp)
extern "C" hipError_t hipGetDeviceCount(int *count);
extern "C" hipError_t hipSetDevice(int device);
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t *prop, int device);
extern "C" hipError_t hipStreamCreate(hipStream_t *stream);
extern "C" hipError_t hipStreamDestroy(hipStream_t stream);
extern "C" hipError_t hipStreamSynchronize(hipStream_t stream);
extern "C" hipError_t hipMalloc(void **ptr, size_t size);
extern "C" hipError_t hipFree(void *ptr);
extern "C" hipError_t hipHostMalloc(void **ptr, size_t size,
                                    unsigned int flags);
extern "C" hipError_t hipHostFree(void *ptr);
extern "C" hipError_t hipMemcpy(void *dst, const void *src, size_t size,
                                int kind);
extern "C" hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size,
                                     int kind, hipStream_t stream);
extern "C" hipError_t hipMemsetAsync(void *dst, int value, size_t size,
                                     hipStream_t stream);
extern "C" hipError_t hipEventCreate(hipEvent_t *event);
extern "C" hipError_t hipEventCreateWithFlags(hipEvent_t *event,
                                              unsigned int flags);
extern "C" hipError_t hipEventDestroy(hipEvent_t event);
extern "C" hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
extern "C" hipError_t hipEventSynchronize(hipEvent_t event);
extern "C" hipError_t hipEventElapsedTime(float *ms, hipEvent_t start,
                                          hipEvent_t stop);
extern "C" hipError_t hipHostGetDevicePointer(void **devPtr, void *hstPtr,
                                              unsigned int flags);
extern "C" const char *hipGetErrorString(hipError_t error);
extern "C" miopenStatus_t miopenCreate(miopenHandle_t *handle);
extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle);
extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle,
                                          hipStream_t stream);
extern "C" hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle);
extern "C" hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle);

#endif // HIPDNN_EP_RUNTIME_MOCK_TYPES_H
