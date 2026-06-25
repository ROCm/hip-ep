/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <type_traits>

#include "hip_custom_kernels.h"

//===----------------------------------------------------------------------===//
// hip_causal_conv_step_decode -- single-step fused causal depthwise 1D conv.
//
// Replaces the MIOpen "build virtual buffer with hipMemcpy2DAsync, run
// depthwise conv, run bias + SiLU as separate ops" chain for the seq_len == 1
// decode path. Single launch, one block per (batch, channel), each thread is
// responsible for one channel.
//
// At decode the math is laughably small:
//   per channel: read k-1 past-state elements + 1 input element + k weights,
//   compute k FMAs, optional bias, optional SiLU, write 1 output + k-1 state.
//
// On a Mamba/Gated DeltaNet block with C=8192, k=4 this is ~96 KB of VRAM
// reads and ~16 KB of VRAM writes. Runs in ~1-20 µs end-to-end.
//
// We unroll on kernel_size at the call site (template <int K>) so the inner
// loops collapse to a fixed small number of FMAs and the per-thread state
// register array is statically sized.
//===----------------------------------------------------------------------===//

namespace {

template <typename T> __device__ __forceinline__ float to_f32(T x);
template <> __device__ __forceinline__ float to_f32(float x) { return x; }
template <> __device__ __forceinline__ float to_f32(__half x) {
    return __half2float(x);
}

template <typename T> __device__ __forceinline__ T from_f32(float x);
template <> __device__ __forceinline__ float from_f32(float x) { return x; }
template <> __device__ __forceinline__ __half from_f32(float x) {
    return __float2half(x);
}

// Per-(b,c) thread:
//   1) Loads past_state[b,c,0..K-2] into registers (or zeros if past is null).
//   2) Loads input[b,c,0], weight[c,0,0..K-1], optional bias[c].
//   3) Computes the conv dot product in fp32.
//   4) Applies optional SiLU.
//   5) Writes output[b,c,0] and present_state[b,c,0..K-2].
//
// Memory layout (matches wrap_causal_conv_with_state):
//   input,  output         flat [b*C + c]                 (one element/plane)
//   past_state, present_state  flat [(b*C + c)*(K-1) + j]
//   weight                     flat [c*K + j]
//   bias                       flat [c]
template <typename T, int K, int HAS_BIAS, int ACT_SILU>
__global__ void causal_conv_step_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ past_state,
    T* __restrict__ output,
    T* __restrict__ present_state,
    int batch_size,
    int channels) {

    // One thread per (b, c) plane. Grid = ceil(B*C / blockDim.x).
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels;
    if (idx >= total)
        return;

    // (b, c) for this thread
    const int c = idx % channels;
    // const int b = idx / channels;  // not needed beyond `idx`

    // Load past_state[b,c,0..K-2] into a small register array. K is a template
    // constant so this unrolls fully and `state[]` lives in registers.
    constexpr int STATE_LEN = K - 1;
    float state[STATE_LEN > 0 ? STATE_LEN : 1];
    if (past_state != nullptr) {
        const T* ps = past_state + static_cast<int64_t>(idx) * STATE_LEN;
        #pragma unroll
        for (int j = 0; j < STATE_LEN; ++j)
            state[j] = to_f32<T>(ps[j]);
    } else {
        #pragma unroll
        for (int j = 0; j < STATE_LEN; ++j)
            state[j] = 0.0f;
    }

    // Load input[b,c,0] and weight[c,0,0..K-1] (weight is per-channel only).
    const float x = to_f32<T>(input[idx]);
    float w[K];
    {
        const T* wp = weight + static_cast<int64_t>(c) * K;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            w[j] = to_f32<T>(wp[j]);
    }

    // Conv dot product: virtual = [past_state[0..K-2], x]
    //   y = sum_{j=0..K-2} w[j] * state[j] + w[K-1] * x
    float y = w[K - 1] * x;
    #pragma unroll
    for (int j = 0; j < STATE_LEN; ++j)
        y += w[j] * state[j];

    if (HAS_BIAS)
        y += to_f32<T>(bias[c]);

    if (ACT_SILU) {
        // SiLU(y) = y * sigmoid(y) = y / (1 + exp(-y))
        const float s = 1.0f / (1.0f + expf(-y));
        y *= s;
    }

    output[idx] = from_f32<T>(y);

    // Shift state forward by one timestep:
    //   present_state[b,c,0..K-3] = past_state[b,c,1..K-2]
    //   present_state[b,c,K-2]    = x
    if (STATE_LEN > 0) {
        T* nps = present_state + static_cast<int64_t>(idx) * STATE_LEN;
        #pragma unroll
        for (int j = 0; j < STATE_LEN - 1; ++j)
            nps[j] = from_f32<T>(state[j + 1]);
        nps[STATE_LEN - 1] = from_f32<T>(x);
    }
}

// Dispatcher: stamps out the small cross-product of (T, K, HAS_BIAS,
// ACT_SILU) template instantiations we actually use at runtime. Only the K
// values currently observed in supported models (Mamba uses k=4) need fast
// paths; we provide k in {2,3,4,5,6,7,8} for future hybrids.
template <typename T>
hipError_t launch_for_K(
    int K, hipStream_t stream,
    const T* input, const T* weight, const T* bias, const T* past_state,
    T* output, T* present_state,
    int batch_size, int channels, int has_bias, int act_silu) {

    const int total = batch_size * channels;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    auto launch = [&](auto k_const, auto bias_const, auto act_const) {
        constexpr int Kc = decltype(k_const)::value;
        constexpr int Bc = decltype(bias_const)::value;
        constexpr int Ac = decltype(act_const)::value;
        hipLaunchKernelGGL(
            (causal_conv_step_kernel<T, Kc, Bc, Ac>),
            dim3(grid), dim3(block), 0, stream,
            input, weight, bias, past_state, output, present_state,
            batch_size, channels);
        return hipGetLastError();
    };

    // Compile-time switch on K so we get fully-unrolled inner loops.
#define DISPATCH_K_BIAS_ACT(KVAL)                                             \
    case KVAL: {                                                              \
        using K_t = std::integral_constant<int, KVAL>;                        \
        if (has_bias && act_silu)                                             \
            return launch(K_t{},                                              \
                          std::integral_constant<int, 1>{},                   \
                          std::integral_constant<int, 1>{});                  \
        if (has_bias)                                                         \
            return launch(K_t{},                                              \
                          std::integral_constant<int, 1>{},                   \
                          std::integral_constant<int, 0>{});                  \
        if (act_silu)                                                         \
            return launch(K_t{},                                              \
                          std::integral_constant<int, 0>{},                   \
                          std::integral_constant<int, 1>{});                  \
        return launch(K_t{},                                                  \
                      std::integral_constant<int, 0>{},                       \
                      std::integral_constant<int, 0>{});                      \
    }

    switch (K) {
        DISPATCH_K_BIAS_ACT(1)
        DISPATCH_K_BIAS_ACT(2)
        DISPATCH_K_BIAS_ACT(3)
        DISPATCH_K_BIAS_ACT(4)
        DISPATCH_K_BIAS_ACT(5)
        DISPATCH_K_BIAS_ACT(6)
        DISPATCH_K_BIAS_ACT(7)
        DISPATCH_K_BIAS_ACT(8)
        default:
            return hipErrorInvalidValue;
    }
#undef DISPATCH_K_BIAS_ACT
}

// Prefill (seq_len > 1) fused causal depthwise 1D conv + bias + SiLU.
// One thread per (b, c) plane slides a length-K window over the virtual
// sequence [past_state[0..K-2], input[0..L-1]], writing L outputs and the
// trailing K-1 present_state. fp32 accumulate -- numerically matches
// causal_conv_step_kernel at L=1. Replaces the MIOpen prefill path (Find +
// 3 pitched memcpys + conv + bias + activation + mul) with a single launch.
// Layout matches wrap_causal_conv_with_state:
//   input, output            flat [(b*C + c)*L + t]
//   past_state, present_state flat [(b*C + c)*(K-1) + j]
//   weight                    flat [c*K + j];   bias flat [c]
template <typename T, int K, int HAS_BIAS, int ACT_SILU>
__global__ void causal_conv_prefill_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ past_state,
    T* __restrict__ output,
    T* __restrict__ present_state,
    int batch_size,
    int channels,
    int seq_len) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels;
    if (idx >= total)
        return;

    const int c = idx % channels;
    constexpr int STATE_LEN = K - 1;

    // Per-channel weights (K taps) and optional bias.
    float w[K];
    {
        const T* wp = weight + static_cast<int64_t>(c) * K;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            w[j] = to_f32<T>(wp[j]);
    }
    const float bias_v = HAS_BIAS ? to_f32<T>(bias[c]) : 0.0f;

    // window[0..K-1] holds virtual[t .. t+K-1] for the current output t.
    // Init for t=0: [past_state[0..K-2], input[0]].
    float window[K];
    {
        const T* ps = (STATE_LEN > 0 && past_state != nullptr)
                          ? past_state + static_cast<int64_t>(idx) * STATE_LEN
                          : nullptr;
        #pragma unroll
        for (int j = 0; j < STATE_LEN; ++j)
            window[j] = ps ? to_f32<T>(ps[j]) : 0.0f;
    }
    const T* in_row = input + static_cast<int64_t>(idx) * seq_len;
    T* out_row = output + static_cast<int64_t>(idx) * seq_len;
    window[K - 1] = to_f32<T>(in_row[0]);

    for (int t = 0; t < seq_len; ++t) {
        if (t > 0) {
            // Slide the window: drop oldest, append virtual[t+K-1] = input[t].
            #pragma unroll
            for (int j = 0; j < K - 1; ++j)
                window[j] = window[j + 1];
            window[K - 1] = to_f32<T>(in_row[t]);
        }
        float y = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; ++j)
            y += w[j] * window[j];
        if (HAS_BIAS)
            y += bias_v;
        if (ACT_SILU) {
            const float s = 1.0f / (1.0f + expf(-y));
            y *= s;
        }
        out_row[t] = from_f32<T>(y);
    }

    // present_state = last K-1 of virtual = window[1..K-1] after the last step.
    if (STATE_LEN > 0) {
        T* nps = present_state + static_cast<int64_t>(idx) * STATE_LEN;
        #pragma unroll
        for (int j = 0; j < STATE_LEN; ++j)
            nps[j] = from_f32<T>(window[j + 1]);
    }
}

template <typename T>
hipError_t launch_prefill_for_K(
    int K, hipStream_t stream,
    const T* input, const T* weight, const T* bias, const T* past_state,
    T* output, T* present_state,
    int batch_size, int channels, int seq_len, int has_bias, int act_silu) {

    const int total = batch_size * channels;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    auto launch = [&](auto k_const, auto bias_const, auto act_const) {
        constexpr int Kc = decltype(k_const)::value;
        constexpr int Bc = decltype(bias_const)::value;
        constexpr int Ac = decltype(act_const)::value;
        hipLaunchKernelGGL(
            (causal_conv_prefill_kernel<T, Kc, Bc, Ac>),
            dim3(grid), dim3(block), 0, stream,
            input, weight, bias, past_state, output, present_state,
            batch_size, channels, seq_len);
        return hipGetLastError();
    };

#define DISPATCH_PREFILL_K(KVAL)                                              \
    case KVAL: {                                                              \
        using K_t = std::integral_constant<int, KVAL>;                        \
        if (has_bias && act_silu)                                             \
            return launch(K_t{}, std::integral_constant<int, 1>{},            \
                          std::integral_constant<int, 1>{});                  \
        if (has_bias)                                                         \
            return launch(K_t{}, std::integral_constant<int, 1>{},            \
                          std::integral_constant<int, 0>{});                  \
        if (act_silu)                                                         \
            return launch(K_t{}, std::integral_constant<int, 0>{},            \
                          std::integral_constant<int, 1>{});                  \
        return launch(K_t{}, std::integral_constant<int, 0>{},                \
                      std::integral_constant<int, 0>{});                      \
    }

    switch (K) {
        DISPATCH_PREFILL_K(1)
        DISPATCH_PREFILL_K(2)
        DISPATCH_PREFILL_K(3)
        DISPATCH_PREFILL_K(4)
        DISPATCH_PREFILL_K(5)
        DISPATCH_PREFILL_K(6)
        DISPATCH_PREFILL_K(7)
        DISPATCH_PREFILL_K(8)
        default:
            return hipErrorInvalidValue;
    }
#undef DISPATCH_PREFILL_K
}

} // namespace

extern "C" int hip_causal_conv_step_decode(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t kernel_size,
    int64_t activation,
    int64_t element_size_bytes) {

    if (!input || !weight || !output || !present_state) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: null required "
                "pointer (input=%p weight=%p output=%p present_state=%p)\n",
                input, weight, output, present_state);
        return -1;
    }
    if (batch_size <= 0 || channels <= 0) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: invalid "
                "batch=%lld channels=%lld\n",
                (long long)batch_size, (long long)channels);
        return -1;
    }
    if (kernel_size < 1 || kernel_size > 8) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: kernel_size=%lld "
                "outside supported range [1,8]\n",
                (long long)kernel_size);
        return -1;
    }
    if (activation != 0 && activation != 1) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: unsupported "
                "activation=%lld (0=none, 1=SiLU)\n",
                (long long)activation);
        return -1;
    }

    hipStream_t s = static_cast<hipStream_t>(stream);
    const int K = static_cast<int>(kernel_size);
    const int B = static_cast<int>(batch_size);
    const int C = static_cast<int>(channels);
    const int has_bias = bias ? 1 : 0;
    const int act_silu = (activation == 1) ? 1 : 0;

    hipError_t err;
    if (element_size_bytes == 4) {
        err = launch_for_K<float>(
            K, s,
            static_cast<const float*>(input),
            static_cast<const float*>(weight),
            static_cast<const float*>(bias),
            static_cast<const float*>(past_state),
            static_cast<float*>(output),
            static_cast<float*>(present_state),
            B, C, has_bias, act_silu);
    } else if (element_size_bytes == 2) {
        err = launch_for_K<__half>(
            K, s,
            static_cast<const __half*>(input),
            static_cast<const __half*>(weight),
            static_cast<const __half*>(bias),
            static_cast<const __half*>(past_state),
            static_cast<__half*>(output),
            static_cast<__half*>(present_state),
            B, C, has_bias, act_silu);
    } else {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: unsupported "
                "element_size_bytes=%lld (expect 2 or 4)\n",
                (long long)element_size_bytes);
        return -1;
    }

    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_step_decode: launch failed "
                "(%s)\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}

extern "C" int hip_causal_conv_prefill(
    void* stream,
    const void* input,
    const void* weight,
    const void* bias,
    const void* past_state,
    void* output,
    void* present_state,
    int64_t batch_size,
    int64_t channels,
    int64_t seq_len,
    int64_t kernel_size,
    int64_t activation,
    int64_t element_size_bytes) {

    if (!input || !weight || !output || !present_state) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_prefill: null required "
                "pointer (input=%p weight=%p output=%p present_state=%p)\n",
                input, weight, output, present_state);
        return -1;
    }
    if (batch_size <= 0 || channels <= 0 || seq_len <= 0) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_prefill: invalid "
                "batch=%lld channels=%lld seq_len=%lld\n",
                (long long)batch_size, (long long)channels, (long long)seq_len);
        return -1;
    }
    if (kernel_size < 1 || kernel_size > 8) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_prefill: kernel_size=%lld "
                "outside supported range [1,8]\n",
                (long long)kernel_size);
        return -1;
    }
    if (activation != 0 && activation != 1) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_prefill: unsupported "
                "activation=%lld (0=none, 1=SiLU)\n",
                (long long)activation);
        return -1;
    }

    hipStream_t s = static_cast<hipStream_t>(stream);
    const int K = static_cast<int>(kernel_size);
    const int B = static_cast<int>(batch_size);
    const int C = static_cast<int>(channels);
    const int L = static_cast<int>(seq_len);
    const int has_bias = bias ? 1 : 0;
    const int act_silu = (activation == 1) ? 1 : 0;

    hipError_t err;
    if (element_size_bytes == 4) {
        err = launch_prefill_for_K<float>(
            K, s,
            static_cast<const float*>(input),
            static_cast<const float*>(weight),
            static_cast<const float*>(bias),
            static_cast<const float*>(past_state),
            static_cast<float*>(output),
            static_cast<float*>(present_state),
            B, C, L, has_bias, act_silu);
    } else if (element_size_bytes == 2) {
        err = launch_prefill_for_K<__half>(
            K, s,
            static_cast<const __half*>(input),
            static_cast<const __half*>(weight),
            static_cast<const __half*>(bias),
            static_cast<const __half*>(past_state),
            static_cast<__half*>(output),
            static_cast<__half*>(present_state),
            B, C, L, has_bias, act_silu);
    } else {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_prefill: unsupported "
                "element_size_bytes=%lld (expect 2 or 4)\n",
                (long long)element_size_bytes);
        return -1;
    }

    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_causal_conv_prefill: launch failed (%s)\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}
