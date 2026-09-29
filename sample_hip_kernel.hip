#include <hip/hip_runtime.h>

__global__ void add_one(float* x) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  x[idx] += 1.0f;
}

int main() {
  float* d_x = nullptr;
  hipMalloc(&d_x, 16 * sizeof(float));
  add_one<<<1, 16>>>(d_x);
  hipFree(d_x);
  return 0;
}
