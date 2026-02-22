/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_RUNTIME_MOCK_TYPES_H
#define HIPDNN_EP_RUNTIME_MOCK_TYPES_H

// Mock type definitions for testing without GPU
typedef void* hipStream_t;
typedef void* miopenHandle_t;
typedef void* hipblasLtHandle_t;
typedef int hipError_t;
typedef int miopenStatus_t;
typedef int hipblasStatus_t;

struct hipDeviceProp_t {
  char name[256];
  char gcnArchName[256];
};

#define hipSuccess 0
#define miopenStatusSuccess 0
#define HIPBLAS_STATUS_SUCCESS 0
#define hipMemcpyHostToDevice 0
#define hipMemcpyDeviceToHost 1

// Forward declarations for mock GPU functions (defined in mock_gpu.cpp)
extern "C" hipError_t hipGetDeviceCount(int* count);
extern "C" hipError_t hipSetDevice(int device);
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device);
extern "C" hipError_t hipStreamCreate(hipStream_t* stream);
extern "C" hipError_t hipStreamDestroy(hipStream_t stream);
extern "C" hipError_t hipStreamSynchronize(hipStream_t stream);
extern "C" hipError_t hipMalloc(void** ptr, size_t size);
extern "C" hipError_t hipFree(void* ptr);
extern "C" hipError_t hipMemcpy(void* dst, const void* src, size_t size,
                                int kind);
extern "C" hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                                     int kind, hipStream_t stream);
extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle);
extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle);
extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle,
                                          hipStream_t stream);
extern "C" hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle);
extern "C" hipblasStatus_t hipblasLtDestroy(hipblasLtHandle_t handle);

#endif // HIPDNN_EP_RUNTIME_MOCK_TYPES_H
