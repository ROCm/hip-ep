/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic N-D transpose: swaps two dimensions of a contiguous row-major
// tensor.  Uses host-side index shuffle with blocking hipMemcpy.
//
// NOTE: We deliberately avoid launching HIP device kernels here because
// the hip_custom_kernels.lib is statically linked into the JIT'd model
// DLL, and the HIP fat-binary registration (__hipRegisterFunction) for
// kernels in this .lib doesn't take effect inside the DLL context.
// hipLaunchKernel then returns hipErrorInvalidValue because it can't
// find the registered device code.  All other custom kernels (slice,
// gather, elementwise, etc.) work because their host wrappers are
// called from the runtime bitcode which links into the LLVM IR module
// -- their device kernel launches happen in the EP DLL's context where
// registration DID succeed.  Transpose is the only kernel called
// directly from hip_custom_kernels.lib via the model DLL.
//
// TODO: Move transpose to a runtime wrapper (wrap_transpose in
// runtime.bc) that calls through the EP DLL, matching how all other
// custom ops work.  That would let us use a proper GPU kernel.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define HIP_TRANSPOSE_MAX_RANK 8

extern "C" int hip_transpose_nd(void *stream, const void *input, void *output,
                                int rank, const int64_t *in_shape, int dim0,
                                int dim1, int hip_dtype) {
    if (!input || !output || !in_shape) {
        fprintf(stderr, "[custom_kernels] hip_transpose_nd: null pointer\n");
        return -1;
    }
    if (rank < 1 || rank > HIP_TRANSPOSE_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd: rank %d out of range\n",
                rank);
        return -1;
    }
    if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd: bad dims (%d,%d)\n",
                dim0, dim1);
        return -1;
    }

    size_t elem_size = 0;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32: case HIP_DTYPE_INT32: elem_size = 4; break;
    case HIP_DTYPE_FLOAT16: case HIP_DTYPE_BFLOAT16: elem_size = 2; break;
    case HIP_DTYPE_INT64: case HIP_DTYPE_FLOAT64: elem_size = 8; break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd: unsupported dtype %d\n",
                hip_dtype);
        return -1;
    }

    int64_t total = 1;
    for (int i = 0; i < rank; ++i)
        total *= in_shape[i];
    size_t total_bytes = static_cast<size_t>(total) * elem_size;

    if (dim0 == dim1 || total <= 0) {
        if (total > 0)
            return static_cast<int>(hipMemcpyAsync(
                output, input, total_bytes, hipMemcpyDeviceToDevice,
                static_cast<hipStream_t>(stream)));
        return 0;
    }

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_transpose_nd: dtype=%d, rank=%d, dim0=%d, "
        "dim1=%d, total=%lld (host fallback)\n",
        hip_dtype, rank, dim0, dim1, (long long)total);

    (void)hipGetLastError();

    // hipDeviceSynchronize waits for ALL streams on the device.
    // This ensures the input data (written by prior ops on the EP's
    // compute stream) is ready for the D2H copy.
    hipDeviceSynchronize();
    (void)hipGetLastError();

    std::vector<uint8_t> h_in(total_bytes), h_out(total_bytes);
    hipMemcpy(h_in.data(), input, total_bytes, hipMemcpyDeviceToHost);

    // Debug: print first few values to verify D2H worked.
    static int dbg_count = 0;
    if (elem_size == 4) {
        float *fp = reinterpret_cast<float *>(h_in.data());
        bool all_zero = true;
        for (int64_t i = 0; i < total && i < 1000; ++i)
            if (fp[i] != 0.f) { all_zero = false; break; }
        if (dbg_count < 5 || !all_zero) {
            fprintf(stderr,
                    "[transpose_nd] #%d rank=%d d0=%d d1=%d total=%lld "
                    "in[0..3]=%.4f %.4f %.4f %.4f %s\n",
                    dbg_count, rank, dim0, dim1, (long long)total,
                    fp[0], fp[1], total > 2 ? fp[2] : 0.f,
                    total > 3 ? fp[3] : 0.f,
                    all_zero ? "ALL_ZERO" : "has_data");
            fflush(stderr);
        }
        dbg_count++;
    }

    // Compute output shape and strides.
    int64_t out_shape[HIP_TRANSPOSE_MAX_RANK];
    for (int i = 0; i < rank; ++i) out_shape[i] = in_shape[i];
    { int64_t t = out_shape[dim0]; out_shape[dim0] = out_shape[dim1];
      out_shape[dim1] = t; }

    int64_t in_strides[HIP_TRANSPOSE_MAX_RANK];
    int64_t out_strides[HIP_TRANSPOSE_MAX_RANK];
    in_strides[rank - 1] = 1;
    out_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    for (int64_t flat = 0; flat < total; ++flat) {
        int64_t coord[HIP_TRANSPOSE_MAX_RANK];
        int64_t rem = flat;
        for (int i = 0; i < rank; ++i) {
            coord[i] = rem / out_strides[i];
            rem -= coord[i] * out_strides[i];
        }
        { int64_t t = coord[dim0]; coord[dim0] = coord[dim1];
          coord[dim1] = t; }
        int64_t in_off = 0;
        for (int i = 0; i < rank; ++i)
            in_off += coord[i] * in_strides[i];
        memcpy(&h_out[flat * elem_size],
               &h_in[in_off * elem_size], elem_size);
    }

    hipMemcpy(output, h_out.data(), total_bytes, hipMemcpyHostToDevice);
    // Blocking hipMemcpy syncs the default stream.  The EP's compute
    // stream may or may not have implicit ordering with the default
    // stream depending on hipDeviceSetFlags.  Since we called
    // hipDeviceSynchronize above (which waits for ALL streams), and
    // now we've done the H2D on the default stream (also blocking),
    // the device memory is valid.  The EP stream's next dispatch will
    // pick up the written data because hipDeviceSynchronize already
    // drained it.

    return 0;
}
