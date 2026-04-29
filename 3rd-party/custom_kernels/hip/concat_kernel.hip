/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic Concat kernel.  ONNX Concat with N inputs along axis A is reshaped
// at the host into:
//
//   for each input i:                                                    \\
//     output[outer, base[i] + inner] = input[outer, inner]               \\
//
// where:
//   - outer  = product(output.shape[:axis])
//   - inner  = product(input[i].shape[axis:])
//   - base[i] = sum_{j<i} input[j].shape[axis]  (in element units)
//
// We launch one HIP kernel per input slice (so the launch overhead scales
// with input count, not output size).  Each launch copies one input's
// elements at the right place in the output via the per-element address
// formula above.  This avoids any need to materialise per-element offset
// tables in GPU memory.
//
// Element-size dispatch (1/2/4/8 bytes) follows the slice kernel pattern,
// so f16/bf16/f32/i64 all share the same compiled code paths.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

template <typename Word>
__global__ void concat_slice_kernel(const Word *__restrict__ in,
                                    Word *__restrict__ out, int64_t outer,
                                    int64_t input_inner,
                                    int64_t output_inner,
                                    int64_t base_inner) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = outer * input_inner;
    if (flat >= total)
        return;
    int64_t o = flat / input_inner;
    int64_t i = flat % input_inner;
    out[o * output_inner + base_inner + i] = in[flat];
}

template <typename Word>
int launch_concat_slice(hipStream_t stream, const Word *in, Word *out,
                        int64_t outer, int64_t input_inner,
                        int64_t output_inner, int64_t base_inner) {
    int64_t total = outer * input_inner;
    if (total <= 0)
        return 0;
    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(concat_slice_kernel<Word>, dim3(grid), dim3(kBlockSize),
                       0, stream, in, out, outer, input_inner, output_inner,
                       base_inner);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_concat(void *stream, void *output, int element_size_bytes,
                          int64_t outer, int64_t output_inner,
                          int num_inputs, const void *const *inputs,
                          const int64_t *input_inner_sizes) {
    if (!output || !inputs || !input_inner_sizes) {
        fprintf(stderr, "[custom_kernels] hip_concat: null pointer\n");
        return -1;
    }
    if (num_inputs <= 0)
        return 0;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);


    int64_t base_inner = 0;
    int rc = 0;
    for (int i = 0; i < num_inputs; ++i) {
        int64_t inner = input_inner_sizes[i];
        const void *in = inputs[i];
        switch (element_size_bytes) {
        case 1:
            rc = launch_concat_slice<uint8_t>(
                hip_stream, static_cast<const uint8_t *>(in),
                static_cast<uint8_t *>(output), outer, inner, output_inner,
                base_inner);
            break;
        case 2:
            rc = launch_concat_slice<uint16_t>(
                hip_stream, static_cast<const uint16_t *>(in),
                static_cast<uint16_t *>(output), outer, inner, output_inner,
                base_inner);
            break;
        case 4:
            rc = launch_concat_slice<uint32_t>(
                hip_stream, static_cast<const uint32_t *>(in),
                static_cast<uint32_t *>(output), outer, inner, output_inner,
                base_inner);
            break;
        case 8:
            rc = launch_concat_slice<uint64_t>(
                hip_stream, static_cast<const uint64_t *>(in),
                static_cast<uint64_t *>(output), outer, inner, output_inner,
                base_inner);
            break;
        default:
            fprintf(stderr,
                    "[custom_kernels] hip_concat: unsupported element_size=%d\n",
                    element_size_bytes);
            return -1;
        }
        if (rc != 0)
            return rc;
        base_inner += inner;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_concat launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}
