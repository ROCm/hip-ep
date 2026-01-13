/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/
#pragma once

#include <functional>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                                                                     \
    if(error != hipSuccess)                                                                        \
    {                                                                                              \
        fprintf(stderr,                                                                            \
                "Hip error: '%s'(%d) at %s:%d\n",                                                  \
                hipGetErrorString(error),                                                          \
                error,                                                                             \
                __FILE__,                                                                          \
                __LINE__);                                                                         \
        exit(EXIT_FAILURE);                                                                        \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                               \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                            \
    {                                                                                              \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__);          \
        fprintf(stderr, "\n");                                                                     \
        exit(EXIT_FAILURE);                                                                        \
    }
#endif

template <typename InTypeA,
          typename InTypeB,
          typename OutType,
          typename AlphaType,
          typename BetaType,
          typename BiasType = OutType>
struct Runner
{
    Runner(int64_t   m,
           int64_t   n,
           int64_t   k,
           int64_t   batch_count,
           AlphaType alpha,
           BetaType  beta,
           int64_t   max_workspace_size_in_bytes)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size_in_bytes)
    {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

        CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
        CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
        CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipMalloc(&d_alphaVec, m * batch_count * sizeof(float)));

        CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InTypeA)));
        CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InTypeB)));
        CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipHostMalloc(&alphaVec, m * batch_count * sizeof(float)));

        if(max_workspace_size > 0)
            CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

        for(int i = 0; i < m * k * batch_count; i++) {
            __half h = __float2half(static_cast<float>((rand() % 7) - 3));
            ((InTypeA*)a)[i].data = *reinterpret_cast<uint16_t*>(&h);
        }
        for(int i = 0; i < n * k * batch_count; i++) {
            __half h = __float2half(static_cast<float>((rand() % 7) - 3));
            ((InTypeB*)b)[i].data = *reinterpret_cast<uint16_t*>(&h);
        }
        for(int i = 0; i < m * n * batch_count; i++) {
            __half h = __float2half(static_cast<float>((rand() % 7) - 3));
            ((OutType*)c)[i].data = *reinterpret_cast<uint16_t*>(&h);
        }
        for(int i = 0; i < m * batch_count; ++i)
            ((float*)alphaVec)[i] = static_cast<float>((rand() % 7) - 3);
    }

    ~Runner()
    {
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIP_ERROR(hipFree(a));
        CHECK_HIP_ERROR(hipFree(b));
        CHECK_HIP_ERROR(hipFree(c));
        CHECK_HIP_ERROR(hipFree(d));
        CHECK_HIP_ERROR(hipFree(alphaVec));

        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_c));
        CHECK_HIP_ERROR(hipFree(d_d));
        CHECK_HIP_ERROR(hipFree(d_alphaVec));

        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));

        if(biasVec)
        {
            CHECK_HIP_ERROR(hipFree(biasVec));
            CHECK_HIP_ERROR(hipFree(d_biasVec));
        }
    }

    void setBiasInfo(bool useBias, char biasSrc)
    {
        biasElems = 0;
        if(useBias)
        {
            if(biasSrc == 'B' || biasSrc == 'b')
                biasElems = n;
            else if(biasSrc == 'A' || biasSrc == 'a' || biasSrc == 'D' || biasSrc == 'd')
                biasElems = m;
            // else, biasElems = 0
        }

        // alloc bias if use bias
        if(biasElems > 0)
        {
            if(biasVec)
            {
                CHECK_HIP_ERROR(hipFree(biasVec));
                CHECK_HIP_ERROR(hipFree(d_biasVec));
            }

            CHECK_HIP_ERROR(hipMalloc(&d_biasVec, biasElems * sizeof(BiasType)));
            CHECK_HIP_ERROR(hipHostMalloc(&biasVec, biasElems * sizeof(BiasType)));

            for(int i = 0; i < biasElems; ++i)
                ((BiasType*)biasVec)[i] = static_cast<BiasType>((rand() % 7) - 3);
        }
    }

    void hostToDevice()
    {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_alphaVec, alphaVec, m * batch_count * sizeof(float), hipMemcpyHostToDevice, stream));

        // cpy bias if needed
        if(biasVec)
            CHECK_HIP_ERROR(hipMemcpyAsync(
                d_biasVec, biasVec, biasElems * sizeof(BiasType), hipMemcpyHostToDevice, stream));
    }

    void deviceToHost()
    {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d, d_d, m * n * batch_count * sizeof(OutType), hipMemcpyDeviceToHost, stream));
    }

    void run(const std::function<void()>& func)
    {
        hostToDevice();
        static_cast<void>(func());
        deviceToHost();
        static_cast<void>(hipStreamSynchronize(stream));
    }

    int64_t   m;
    int64_t   n;
    int64_t   k;
    int64_t   batch_count;
    AlphaType alpha;
    BetaType  beta;

    void *a, *b, *c, *d, *alphaVec;           // host
    void *d_a, *d_b, *d_c, *d_d, *d_alphaVec; // device
    void* d_workspace;
    int64_t max_workspace_size;

    int64_t biasElems    = 0;
    void*   biasVec      = nullptr; // host
    void*   d_biasVec    = nullptr; // device

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};
