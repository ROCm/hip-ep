/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Vendor-symbol shim for HIPDNN_EP_DISABLE_VENDOR_BLAS. Runtime files call
// hipBLASLt directly, but that library is not linked in this mode. Defining
// each referenced entry point as an error-returning stub keeps the model link
// resolvable on any arch; a vendor GEMM that actually runs fails at runtime,
// not at link. Compiled into runtime.bc only when the flag is set. The vendor
// headers still declare these symbols, so the stubs match the exact ABI.
#ifdef HIPDNN_EP_DISABLE_VENDOR_BLAS

#include <hipblaslt/hipblaslt.h>

extern "C" {

hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle) {
  (void)handle;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle) {
  (void)handle;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc,
                const void *alpha, const void *A, hipblasLtMatrixLayout_t Adesc,
                const void *B, hipblasLtMatrixLayout_t Bdesc, const void *beta,
                const void *C, hipblasLtMatrixLayout_t Cdesc, void *D,
                hipblasLtMatrixLayout_t Ddesc,
                const hipblasLtMatmulAlgo_t *algo, void *workspace,
                size_t workspaceSizeInBytes, hipStream_t stream) {
  (void)handle;
  (void)matmulDesc;
  (void)alpha;
  (void)A;
  (void)Adesc;
  (void)B;
  (void)Bdesc;
  (void)beta;
  (void)C;
  (void)Cdesc;
  (void)D;
  (void)Ddesc;
  (void)algo;
  (void)workspace;
  (void)workspaceSizeInBytes;
  (void)stream;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc,
    hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc,
    hipblasLtMatmulPreference_t pref, int requestedAlgoCount,
    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount) {
  (void)handle;
  (void)matmulDesc;
  (void)Adesc;
  (void)Bdesc;
  (void)Cdesc;
  (void)Ddesc;
  (void)pref;
  (void)requestedAlgoCount;
  (void)heuristicResultsArray;
  (void)returnAlgoCount;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t *matmulDesc,
                                          hipblasComputeType_t computeType,
                                          hipDataType scaleType) {
  (void)matmulDesc;
  (void)computeType;
  (void)scaleType;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t
hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t matmulDesc) {
  (void)matmulDesc;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t
hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc,
                                hipblasLtMatmulDescAttributes_t attr,
                                const void *buf, size_t sizeInBytes) {
  (void)matmulDesc;
  (void)attr;
  (void)buf;
  (void)sizeInBytes;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t
hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t *pref) {
  (void)pref;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t
hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref) {
  (void)pref;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attr, const void *buf,
    size_t sizeInBytes) {
  (void)pref;
  (void)attr;
  (void)buf;
  (void)sizeInBytes;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t *matLayout,
                                            hipDataType type, uint64_t rows,
                                            uint64_t cols, int64_t ld) {
  (void)matLayout;
  (void)type;
  (void)rows;
  (void)cols;
  (void)ld;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t
hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t matLayout) {
  (void)matLayout;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}
hipblasStatus_t
hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t matLayout,
                                  hipblasLtMatrixLayoutAttribute_t attr,
                                  const void *buf, size_t sizeInBytes) {
  (void)matLayout;
  (void)attr;
  (void)buf;
  (void)sizeInBytes;
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

} // extern "C"

#endif // HIPDNN_EP_DISABLE_VENDOR_BLAS
