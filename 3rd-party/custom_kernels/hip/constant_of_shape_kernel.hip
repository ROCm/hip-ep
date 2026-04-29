/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic ConstantOfShape kernel.  ONNX ConstantOfShape takes a shape and
// produces a tensor of that shape filled with a single scalar.  We handle
// the fill on the GPU so the output buffer never has to leave device memory.
//
// The host-side wrapper passes the scalar value as a 64-bit blob (filled
// with the appropriate bit pattern) plus the element size.  This lets one
// kernel cover f16/bf16/f32/i32/i64.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

template <typename Word>
__global__ void constant_of_shape_kernel(Word *__restrict__ output,
                                         int64_t num_elements, Word value) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    output[flat] = value;
}

template <typename Word>
int launch_fill(hipStream_t stream, void *out, int64_t n, uint64_t bits) {
    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    Word value;
    static_assert(sizeof(Word) <= sizeof(uint64_t), "Word too wide");
    // Use the low bits of `bits` for the value - the host side packs the
    // scalar into the low end of the 64-bit blob, padding the upper bits
    // with zeros.
    if constexpr (sizeof(Word) == 1) {
        uint8_t v = static_cast<uint8_t>(bits & 0xff);
        std::memcpy(&value, &v, sizeof(Word));
    } else if constexpr (sizeof(Word) == 2) {
        uint16_t v = static_cast<uint16_t>(bits & 0xffff);
        std::memcpy(&value, &v, sizeof(Word));
    } else if constexpr (sizeof(Word) == 4) {
        uint32_t v = static_cast<uint32_t>(bits & 0xffffffff);
        std::memcpy(&value, &v, sizeof(Word));
    } else {
        std::memcpy(&value, &bits, sizeof(Word));
    }
    hipLaunchKernelGGL(constant_of_shape_kernel<Word>, dim3(grid),
                       dim3(kBlockSize), 0, stream,
                       static_cast<Word *>(out), n, value);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_constant_of_shape(void *stream, void *output,
                                     int64_t num_elements,
                                     int element_size_bytes,
                                     uint64_t scalar_bits) {
    if (num_elements <= 0)
        return 0;
    if (!output) {
        fprintf(stderr, "[custom_kernels] hip_constant_of_shape: null out\n");
        return -1;
    }
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);


    int rc = -1;
    switch (element_size_bytes) {
    case 1:
        rc = launch_fill<uint8_t>(hip_stream, output, num_elements,
                                   scalar_bits);
        break;
    case 2:
        rc = launch_fill<uint16_t>(hip_stream, output, num_elements,
                                    scalar_bits);
        break;
    case 4:
        rc = launch_fill<uint32_t>(hip_stream, output, num_elements,
                                    scalar_bits);
        break;
    case 8:
        rc = launch_fill<uint64_t>(hip_stream, output, num_elements,
                                    scalar_bits);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_constant_of_shape: unsupported "
                "element_size=%d\n",
                element_size_bytes);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_constant_of_shape launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
