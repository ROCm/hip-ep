/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Vendor-symbol shim for HIPDNN_EP_DISABLE_VENDOR_BLAS. Many runtime files call
// MIOpen / hipBLASLt directly, but those libraries are not linked in this mode.
// Defining each referenced entry point as an error-returning stub keeps the
// model link resolvable on any arch; a vendor op that actually runs fails at
// runtime, not at link. Compiled into runtime.bc only when the flag is set. The
// vendor headers still declare these symbols, so the stubs match the exact ABI.
#ifdef HIPDNN_EP_DISABLE_VENDOR_BLAS

// MIOPEN_BETA_API unlocks the layernorm types (miopenNormMode_t /
// miopenT5LayerNormForward), matching real/simplified_layer_norm.cpp.
#define MIOPEN_BETA_API

#include <hipblaslt/hipblaslt.h>
#include <miopen/miopen.h>

extern "C" {

miopenStatus_t miopenActivationForward(
    miopenHandle_t handle, const miopenActivationDescriptor_t activDesc,
    const void *alpha, const miopenTensorDescriptor_t xDesc, const void *x,
    const void *beta, const miopenTensorDescriptor_t yDesc, void *y) {
  (void)handle;
  (void)activDesc;
  (void)alpha;
  (void)xDesc;
  (void)x;
  (void)beta;
  (void)yDesc;
  (void)y;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenConvolutionForward(miopenHandle_t handle, const void *alpha,
                         const miopenTensorDescriptor_t xDesc, const void *x,
                         const miopenTensorDescriptor_t wDesc, const void *w,
                         const miopenConvolutionDescriptor_t convDesc,
                         miopenConvFwdAlgorithm_t algo, const void *beta,
                         const miopenTensorDescriptor_t yDesc, void *y,
                         void *workSpace, size_t workSpaceSize) {
  (void)handle;
  (void)alpha;
  (void)xDesc;
  (void)x;
  (void)wDesc;
  (void)w;
  (void)convDesc;
  (void)algo;
  (void)beta;
  (void)yDesc;
  (void)y;
  (void)workSpace;
  (void)workSpaceSize;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t wDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, size_t *workSpaceSize) {
  (void)handle;
  (void)wDesc;
  (void)xDesc;
  (void)convDesc;
  (void)yDesc;
  (void)workSpaceSize;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenCreate(miopenHandle_t *handle) {
  (void)handle;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenCreateActivationDescriptor(miopenActivationDescriptor_t *activDesc) {
  (void)activDesc;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t *convDesc) {
  (void)convDesc;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenCreateTensorDescriptor(miopenTensorDescriptor_t *tensorDesc) {
  (void)tensorDesc;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenDestroy(miopenHandle_t handle) {
  (void)handle;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc) {
  (void)activDesc;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc) {
  (void)convDesc;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc) {
  (void)tensorDesc;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenFindConvolutionForwardAlgorithm(
    miopenHandle_t handle, const miopenTensorDescriptor_t xDesc, const void *x,
    const miopenTensorDescriptor_t wDesc, const void *w,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, void *y, const int requestAlgoCount,
    int *returnedAlgoCount, miopenConvAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSize, bool exhaustiveSearch) {
  (void)handle;
  (void)xDesc;
  (void)x;
  (void)wDesc;
  (void)w;
  (void)convDesc;
  (void)yDesc;
  (void)y;
  (void)requestAlgoCount;
  (void)returnedAlgoCount;
  (void)perfResults;
  (void)workSpace;
  (void)workSpaceSize;
  (void)exhaustiveSearch;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenInitConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc,
                                miopenConvolutionMode_t c_mode, int pad_h,
                                int pad_w, int stride_h, int stride_w,
                                int dilation_h, int dilation_w) {
  (void)convDesc;
  (void)c_mode;
  (void)pad_h;
  (void)pad_w;
  (void)stride_h;
  (void)stride_w;
  (void)dilation_h;
  (void)dilation_w;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenOpTensor(miopenHandle_t handle, miopenTensorOp_t tensorOp,
                              const void *alpha1,
                              const miopenTensorDescriptor_t aDesc,
                              const void *A, const void *alpha2,
                              const miopenTensorDescriptor_t bDesc,
                              const void *B, const void *beta,
                              const miopenTensorDescriptor_t cDesc, void *C) {
  (void)handle;
  (void)tensorOp;
  (void)alpha1;
  (void)aDesc;
  (void)A;
  (void)alpha2;
  (void)bDesc;
  (void)B;
  (void)beta;
  (void)cDesc;
  (void)C;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenSet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                           miopenDataType_t dataType, int n,
                                           int c, int h, int w) {
  (void)tensorDesc;
  (void)dataType;
  (void)n;
  (void)c;
  (void)h;
  (void)w;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenSetActivationDescriptor(const miopenActivationDescriptor_t activDesc,
                              miopenActivationMode_t mode, double activAlpha,
                              double activBeta, double activGamma) {
  (void)activDesc;
  (void)mode;
  (void)activAlpha;
  (void)activBeta;
  (void)activGamma;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc,
                               int groupCount) {
  (void)convDesc;
  (void)groupCount;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenSetTransposeConvOutputPadding(miopenConvolutionDescriptor_t convDesc,
                                    int adj_h, int adj_w) {
  (void)convDesc;
  (void)adj_h;
  (void)adj_w;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenSetNdTensorDescriptorWithLayout(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType,
    miopenTensorLayout_t tensorLayout, const int *lens, int num_lens) {
  (void)tensorDesc;
  (void)dataType;
  (void)tensorLayout;
  (void)lens;
  (void)num_lens;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenSetStream(miopenHandle_t handle,
                               miopenAcceleratorQueue_t streamID) {
  (void)handle;
  (void)streamID;
  return miopenStatusNotImplemented;
}
miopenStatus_t miopenSetTensorDescriptor(miopenTensorDescriptor_t tensorDesc,
                                         miopenDataType_t dataType, int nbDims,
                                         const int *dimsA,
                                         const int *stridesA) {
  (void)tensorDesc;
  (void)dataType;
  (void)nbDims;
  (void)dimsA;
  (void)stridesA;
  return miopenStatusNotImplemented;
}
miopenStatus_t
miopenT5LayerNormForward(miopenHandle_t handle, miopenNormMode_t mode,
                         const miopenTensorDescriptor_t xDesc, const void *x,
                         const miopenTensorDescriptor_t weightDesc,
                         const void *weight, const float epsilon,
                         const miopenTensorDescriptor_t yDesc, void *y,
                         const miopenTensorDescriptor_t rstdDesc, void *rstd) {
  (void)handle;
  (void)mode;
  (void)xDesc;
  (void)x;
  (void)weightDesc;
  (void)weight;
  (void)epsilon;
  (void)yDesc;
  (void)y;
  (void)rstdDesc;
  (void)rstd;
  return miopenStatusNotImplemented;
}
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
