/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ONNX TensorScatter (opset 24) row writes into a KV-cache buffer:
//   present[b, o, w(b, t), :] = update[b, o, t, :]
// with w(b, t) = write_indices[b] + t (linear) or that modulo max_seq
// (circular). The write positions are read on the device so a decode step
// needs no host sync. Copying past into present is the caller's job and is
// skipped when the two share a buffer.

#include <hip/hip_runtime.h>
#include <cstdint>

#include "hip_custom_kernels.h"

namespace {

template <typename T>
__global__ void tensor_scatter_kernel(T *__restrict__ present,
                                      const T *__restrict__ update,
                                      const int64_t *__restrict__ write_indices,
                                      int64_t outer, int64_t max_seq,
                                      int64_t seq, int64_t row_units,
                                      int circular) {
  const int64_t per_batch = outer * seq * row_units;
  const int64_t b = blockIdx.y;
  const int64_t start = write_indices ? write_indices[b] : 0;
  for (int64_t u = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       u < per_batch; u += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t col = u % row_units;
    const int64_t r = u / row_units;
    const int64_t t = r % seq;
    const int64_t o = r / seq;
    int64_t row = start + t;
    if (circular)
      row = ((row % max_seq) + max_seq) % max_seq;
    else if (row < 0 || row >= max_seq)
      continue;
    present[((b * outer + o) * max_seq + row) * row_units + col] =
        update[((b * outer + o) * seq + t) * row_units + col];
  }
}

template <typename T>
int launch(hipStream_t s, void *present, const void *update,
           const int64_t *write_indices, int64_t batch, int64_t outer,
           int64_t max_seq, int64_t seq, int64_t row_bytes, int circular) {
  const int64_t row_units = row_bytes / static_cast<int64_t>(sizeof(T));
  const int64_t per_batch = outer * seq * row_units;
  const int block = 256;
  int64_t grid_x = (per_batch + block - 1) / block;
  if (grid_x > 4096)
    grid_x = 4096;
  hipLaunchKernelGGL(tensor_scatter_kernel<T>,
                     dim3(static_cast<unsigned>(grid_x),
                          static_cast<unsigned>(batch)),
                     dim3(block), 0, s, static_cast<T *>(present),
                     static_cast<const T *>(update), write_indices, outer,
                     max_seq, seq, row_units, circular);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // namespace

extern "C" int hip_tensor_scatter(void *stream, void *present,
                                  const void *update,
                                  const int64_t *write_indices, int64_t batch,
                                  int64_t outer, int64_t max_seq, int64_t seq,
                                  int64_t row_bytes, int circular) {
  if (batch <= 0 || outer <= 0 || seq <= 0 || row_bytes <= 0)
    return 0;
  if (!present || !update || max_seq <= 0 || batch > 65535)
    return -1;
  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  const uintptr_t align = reinterpret_cast<uintptr_t>(present) |
                          reinterpret_cast<uintptr_t>(update) |
                          static_cast<uintptr_t>(row_bytes);
  if (align % 16 == 0)
    return launch<uint4>(s, present, update, write_indices, batch, outer,
                         max_seq, seq, row_bytes, circular);
  if (align % 8 == 0)
    return launch<uint2>(s, present, update, write_indices, batch, outer,
                         max_seq, seq, row_bytes, circular);
  if (align % 4 == 0)
    return launch<uint32_t>(s, present, update, write_indices, batch, outer,
                            max_seq, seq, row_bytes, circular);
  if (align % 2 == 0)
    return launch<uint16_t>(s, present, update, write_indices, batch, outer,
                            max_seq, seq, row_bytes, circular);
  return launch<uint8_t>(s, present, update, write_indices, batch, outer,
                         max_seq, seq, row_bytes, circular);
}
