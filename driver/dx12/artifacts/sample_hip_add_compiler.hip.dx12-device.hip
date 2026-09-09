#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void hip_ep_add_f32(const float* __restrict__ lhs,
                                           const float* __restrict__ rhs,
                                           float* __restrict__ output,
                                           int64_t num_elements)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < num_elements)
        output[index] = lhs[index] + rhs[index];
}