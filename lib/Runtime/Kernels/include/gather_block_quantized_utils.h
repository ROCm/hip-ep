/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_RUNTIME_GATHER_BLOCK_QUANTIZED_UTILS_H
#define HIP_RUNTIME_GATHER_BLOCK_QUANTIZED_UTILS_H

#include <cstdint>

#if defined(__HIPCC__)
#define HIP_GBQ_HOST_DEVICE __host__ __device__
#else
#define HIP_GBQ_HOST_DEVICE
#endif

namespace hipdnn_ep::gbq {

HIP_GBQ_HOST_DEVICE constexpr int defaultZeroPoint(int bits,
                                                   bool unsignedStorage) {
  return unsignedStorage ? 1 << (bits - 1) : 0;
}

HIP_GBQ_HOST_DEVICE constexpr int decodeStorageByte(uint8_t byte, int bits,
                                                    bool signedStorage,
                                                    int logicalIndex) {
  int value = byte;
  if (bits == 4)
    value = logicalIndex & 1 ? byte >> 4 : byte & 0x0f;
  if (signedStorage && value >= (1 << (bits - 1)))
    value -= 1 << bits;
  return value;
}

HIP_GBQ_HOST_DEVICE constexpr bool
supportsUint8Axes(int bits, bool unsignedStorage, int dataRank, int gatherAxis,
                  int quantizeAxis) {
  if (bits != 8 || !unsignedStorage)
    return true;
  return gatherAxis == 0 && quantizeAxis == dataRank - 1 &&
         gatherAxis != quantizeAxis;
}

} // namespace hipdnn_ep::gbq

#undef HIP_GBQ_HOST_DEVICE

#endif // HIP_RUNTIME_GATHER_BLOCK_QUANTIZED_UTILS_H
