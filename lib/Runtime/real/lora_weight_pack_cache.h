/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_REAL_LORA_WEIGHT_PACK_CACHE_H
#define HIPDNN_EP_REAL_LORA_WEIGHT_PACK_CACHE_H

// Per MatMulNBits op-state cache of LoRA weight_pack buffers. Fuses
// Transpose + signed_offset(+128) into one GPU buffer at first use per
// op instance, then reuses it for the session. Do not key on raw_i8 pointer:
// hybrid/streaming constant upload may restage weights at a new address every
// Compute() even though the logical weight is unchanged.

#include <cstddef>

namespace hipdnn_ep_real {

struct LoraWeightPackEntry {
  void *packed = nullptr;
  size_t bytes = 0;
  int cached_k = 0;
  int cached_n = 0;

  ~LoraWeightPackEntry();
  LoraWeightPackEntry() = default;
  LoraWeightPackEntry(const LoraWeightPackEntry &) = delete;
  LoraWeightPackEntry &operator=(const LoraWeightPackEntry &) = delete;
};

// Returns cached packed uint8 [N, K]. Packs at most once per op-state entry
// (fixed K,N for a compiled MatMulNBits). Returns nullptr on hipMalloc failure.
const void *lookup_or_pack_lora_weight(LoraWeightPackEntry &entry, void *stream,
                                       const void *raw_i8, int K, int N);

} // namespace hipdnn_ep_real

#endif // HIPDNN_EP_REAL_LORA_WEIGHT_PACK_CACHE_H
