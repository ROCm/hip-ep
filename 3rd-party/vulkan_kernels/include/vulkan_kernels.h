/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Vulkan compute-shader kernel library — public C API.
 *
 * Function signatures intentionally mirror hip_custom_kernels.h so that
 * the runtime wrapper (wrap_matmul_nbits) can call either backend with
 * identical arguments.
 */

#ifndef VULKAN_CUSTOM_KERNELS_H
#define VULKAN_CUSTOM_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the global Vulkan compute context (instance, device, queue).
 * Safe to call multiple times — subsequent calls are no-ops.
 * Returns 0 on success, non-zero on error.
 */
int vulkan_context_init(void);

/*
 * Get the GPU-only kernel execution time of the most recent
 * vulkan_matmul_nbits() call, in milliseconds.
 * Measured via Vulkan timestamp queries (hardware nanosecond precision).
 * Returns 0.0 if no call has been made yet.
 */
double vulkan_get_last_kernel_time_ms(void);

/*
 * Destroy the global Vulkan compute context.
 * Safe to call if not initialized (no-op).
 */
void vulkan_context_destroy(void);

/*
 * MatMulNBits — quantized int4 matrix-vector/matrix multiply.
 *
 * Computes: output[batch, m, n] = A[batch,m,:] @ dequant(B[n,:])^T [+ bias[n]]
 *
 * HOST-POINTER variant: copies data to/from GPU per call.
 * Use vulkan_matmul_nbits_gpu() for zero-copy GPU-resident pointers.
 *
 * Returns 0 on success, non-zero on error.
 */
int vulkan_matmul_nbits(
    void *stream,
    const void *A,
    const void *B,
    const void *scales,
    const void *zero_points,
    const void *bias,
    void *output,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,
    int64_t bits,
    int64_t block_size,
    int64_t element_size_bytes);

/*
 * MatMulNBits — GPU-RESIDENT pointer variant (zero-overhead).
 *
 * Same computation as vulkan_matmul_nbits(), but all pointers are assumed
 * to be GPU-accessible memory (e.g. from hipMalloc on UMA, or Vulkan
 * device memory). Uses VK_EXT_external_memory_host to import pointers
 * directly — no host↔device copies.
 *
 * Use this in the real EP inference flow where buffers are already on GPU.
 *
 * Returns 0 on success, non-zero on error.
 */
int vulkan_matmul_nbits_gpu(
    const void *A_gpu,
    const void *B_gpu,
    const void *scales_gpu,
    const void *zp_gpu,
    const void *bias_gpu,
    void *output_gpu,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,
    int64_t bits,
    int64_t block_size,
    int64_t element_size_bytes);

/*
 * Get the minimum alignment required for GPU pointer import.
 * Pointers passed to vulkan_matmul_nbits_gpu() must be aligned to this.
 * Typically 4096 (page size). Returns 0 if not yet initialized.
 */
size_t vulkan_get_host_pointer_alignment(void);

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_CUSTOM_KERNELS_H */
