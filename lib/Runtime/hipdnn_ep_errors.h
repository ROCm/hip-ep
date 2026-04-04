/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_ERRORS_H
#define HIPDNN_EP_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Unified Error Codes
//===----------------------------------------------------------------------===//
//
// All runtime functions return int status codes following this scheme:
//   0 = success (HIPDNN_EP_SUCCESS)
//   negative values = specific errors
//
// Error code ranges:
//   -1 to -99:    General errors (invalid arguments, out of bounds, etc.)
//   -100 to -199: GPU resource errors (allocation, transfer, etc.)
//   -200 to -299: Library errors (MIOpen, hipBLAS, etc.)
//===----------------------------------------------------------------------===//

// Success
#define HIPDNN_EP_SUCCESS 0

// General errors (-1 to -99)
#define HIPDNN_EP_ERR_INDEX_OUT_OF_BOUNDS -1
#define HIPDNN_EP_ERR_RANK_MISMATCH -2
#define HIPDNN_EP_ERR_NULL_POINTER -3
#define HIPDNN_EP_ERR_INVALID_DIMENSION -4
#define HIPDNN_EP_ERR_SIZE_OVERFLOW -5

// GPU resource errors (-100 to -199)
#define HIPDNN_EP_ERR_GPU_ALLOC_FAILED -100
#define HIPDNN_EP_ERR_H2D_TRANSFER_FAILED -101
#define HIPDNN_EP_ERR_D2H_TRANSFER_FAILED -102
#define HIPDNN_EP_ERR_STREAM_SYNC_FAILED -103
#define HIPDNN_EP_ERR_GPU_FREE_FAILED -104

// Library errors (-200 to -299)
#define HIPDNN_EP_ERR_MIOPEN_FAILED -200
#define HIPDNN_EP_ERR_HIPBLAS_FAILED -201

#ifdef __cplusplus
}
#endif

#endif // HIPDNN_EP_ERRORS_H
