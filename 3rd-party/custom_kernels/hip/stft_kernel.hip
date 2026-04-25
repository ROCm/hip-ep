/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Pre/post helpers for ONNX STFT (opset 17), real-to-complex onesided.
//
// The runtime wrapper (real/stft.cpp) drives rocFFT for the actual DFT.
// rocFFT works best on a contiguous (B * n_frames, frame_length) fp32
// buffer per batch.  We provide two HIP kernels:
//
//   1. hip_stft_frame_window:
//        signal[batch, signal_len]  (fp32)
//        window[frame_len]          (fp32, optional)
//      -> frames[batch * n_frames, frame_len] (fp32)
//      Each output frame is a windowed slice of the signal at hop
//      `frame_step`.  When `window` is null we copy the slice verbatim.
//
//   2. hip_stft_split_complex:
//        complex_in[batch * n_frames, n_freqs] (interleaved real/imag)
//      -> output[batch, n_frames, n_freqs, 2]
//      rocFFT R2C produces an array of `hipfftComplex` (= struct of two
//      f32) per frame with `n_freqs = frame_len / 2 + 1`.  We don't even
//      need a separate copy when the source is already an array of
//      `{re, im}` pairs in row-major order -- the layout is identical to
//      the (..., 2) destination.  We still go through this kernel so we
//      can convert between data types in the future and so the runtime
//      wrapper sees a single uniform interface.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

__global__ void stft_frame_window_kernel(const float *__restrict__ signal,
                                         const float *__restrict__ window,
                                         float *__restrict__ frames,
                                         int64_t batch, int64_t signal_len,
                                         int64_t frame_len,
                                         int64_t frame_step,
                                         int64_t n_frames) {
    int64_t bx =
        static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    int64_t flat = bx * blockDim.x + threadIdx.x;
    int64_t total = batch * n_frames * frame_len;
    if (flat >= total)
        return;

    int64_t k = flat % frame_len;
    int64_t fi = (flat / frame_len) % n_frames;
    int64_t b = flat / (frame_len * n_frames);

    int64_t in_idx = b * signal_len + fi * frame_step + k;
    float v = (in_idx < (b + 1) * signal_len) ? signal[in_idx] : 0.0f;
    if (window)
        v *= window[k];
    frames[flat] = v;
}

__global__ void stft_split_complex_kernel(const float *__restrict__ complex_in,
                                          float *__restrict__ output,
                                          int64_t total_pairs) {
    int64_t bx =
        static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    int64_t flat = bx * blockDim.x + threadIdx.x;
    if (flat >= total_pairs)
        return;
    // complex_in is laid out as [re0, im0, re1, im1, ...]; output uses
    // the same byte order, so we just copy the two floats.
    output[2 * flat]     = complex_in[2 * flat];
    output[2 * flat + 1] = complex_in[2 * flat + 1];
}

inline void compute_grid(int64_t total, int kBlockSize, dim3 &grid,
                         dim3 &block) {
    int64_t blocks = (total + kBlockSize - 1) / kBlockSize;
    constexpr int64_t kMaxGridX = 65535;
    int64_t grid_x = blocks;
    int64_t grid_y = 1;
    if (blocks > kMaxGridX) {
        grid_y = (blocks + kMaxGridX - 1) / kMaxGridX;
        grid_x = (blocks + grid_y - 1) / grid_y;
    }
    grid = dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                1);
    block = dim3(static_cast<unsigned>(kBlockSize), 1, 1);
}

} // namespace

extern "C" int hip_stft_frame_window(void *stream, const void *signal,
                                     const void *window, void *frames,
                                     int64_t batch, int64_t signal_len,
                                     int64_t frame_len, int64_t frame_step,
                                     int64_t n_frames, int hip_dtype) {
    if (!signal || !frames) {
        fprintf(stderr,
                "[custom_kernels] hip_stft_frame_window: null pointer\n");
        return -1;
    }
    if (hip_dtype != HIP_DTYPE_FLOAT32) {
        fprintf(stderr,
                "[custom_kernels] hip_stft_frame_window: unsupported dtype "
                "%d (only FLOAT32 supported today)\n",
                hip_dtype);
        return -1;
    }
    if (batch <= 0 || signal_len <= 0 || frame_len <= 0 || frame_step <= 0 ||
        n_frames <= 0)
        return 0;

    int64_t total = batch * n_frames * frame_len;
    constexpr int kBlockSize = 256;
    dim3 grid, block;
    compute_grid(total, kBlockSize, grid, block);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_stft_frame_window: batch=%lld, signal_len=%lld, "
        "frame_len=%lld, frame_step=%lld, n_frames=%lld\n",
        (long long)batch, (long long)signal_len, (long long)frame_len,
        (long long)frame_step, (long long)n_frames);

    hipLaunchKernelGGL(stft_frame_window_kernel, grid, block, 0, hip_stream,
                       static_cast<const float *>(signal),
                       static_cast<const float *>(window),
                       static_cast<float *>(frames), batch, signal_len,
                       frame_len, frame_step, n_frames);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_stft_frame_window launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}

extern "C" int hip_stft_split_complex(void *stream, const void *complex_in,
                                      void *output, int64_t batch,
                                      int64_t n_frames, int64_t n_freqs,
                                      int hip_dtype) {
    if (!complex_in || !output) {
        fprintf(stderr,
                "[custom_kernels] hip_stft_split_complex: null pointer\n");
        return -1;
    }
    if (hip_dtype != HIP_DTYPE_FLOAT32) {
        fprintf(stderr,
                "[custom_kernels] hip_stft_split_complex: unsupported dtype "
                "%d (only FLOAT32 supported today)\n",
                hip_dtype);
        return -1;
    }
    if (batch <= 0 || n_frames <= 0 || n_freqs <= 0)
        return 0;

    int64_t total_pairs = batch * n_frames * n_freqs;
    constexpr int kBlockSize = 256;
    dim3 grid, block;
    compute_grid(total_pairs, kBlockSize, grid, block);

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_stft_split_complex: batch=%lld, n_frames=%lld, "
        "n_freqs=%lld, total_pairs=%lld\n",
        (long long)batch, (long long)n_frames, (long long)n_freqs,
        (long long)total_pairs);

    hipLaunchKernelGGL(stft_split_complex_kernel, grid, block, 0, hip_stream,
                       static_cast<const float *>(complex_in),
                       static_cast<float *>(output), total_pairs);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_stft_split_complex launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}
