/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_RUNTIME_GQA_CONTRACT_H
#define HIPDNN_EP_RUNTIME_GQA_CONTRACT_H

#include "hipdnn_ep_runtime.h"

#include <cstdint>

enum class GqaKvCacheMode {
  Unquantized,
  Int8PerChannel,
};

enum class GqaRuntimeContractViolation {
  None,
  PositionIds,
  OutputQk,
  Softcap,
  RotaryInterleaved,
  MixedQuantization,
  PerTensorQuantization,
  QuantizedBitWidth,
  ScalePair,
  CacheDataType,
  ScaleDataType,
};

/// Validate the GQA schema features that neither runtime implementation may
/// silently ignore. IEEE positive and negative zero are both accepted;
/// NaN, infinity, subnormal, and every other nonzero softcap are rejected.
inline GqaRuntimeContractViolation validateGqaRuntimeContract(
    bool hasPositionIds, bool hasOutputQk, int64_t qkOutput, float softcap,
    int64_t rotaryInterleaved, int64_t kQuantType, int64_t vQuantType,
    int64_t kvCacheBitWidth, bool hasKScale, bool hasVScale,
    int64_t elementSizeBytes, int64_t kCacheDataType, int64_t vCacheDataType,
    int64_t kScaleDataType, int64_t vScaleDataType, GqaKvCacheMode *mode) {
  if (hasPositionIds)
    return GqaRuntimeContractViolation::PositionIds;
  if (hasOutputQk || qkOutput != 0)
    return GqaRuntimeContractViolation::OutputQk;
  if (softcap != 0.0f)
    return GqaRuntimeContractViolation::Softcap;
  if (rotaryInterleaved != 0)
    return GqaRuntimeContractViolation::RotaryInterleaved;
  if (kQuantType != vQuantType)
    return GqaRuntimeContractViolation::MixedQuantization;

  constexpr int64_t quantNone = 0;
  constexpr int64_t quantPerTensor = 1;
  constexpr int64_t quantPerChannel = 2;
  if (kQuantType == quantNone) {
    if (hasKScale || hasVScale || kScaleDataType != -1 || vScaleDataType != -1)
      return GqaRuntimeContractViolation::ScalePair;
    int64_t expectedCacheType = elementSizeBytes == 2 ? HIPDNN_EP_DATATYPE_HALF
                                : elementSizeBytes == 4
                                    ? HIPDNN_EP_DATATYPE_FLOAT
                                    : -1;
    if (expectedCacheType < 0 || kCacheDataType != expectedCacheType ||
        vCacheDataType != expectedCacheType)
      return GqaRuntimeContractViolation::CacheDataType;
    if (mode)
      *mode = GqaKvCacheMode::Unquantized;
    return GqaRuntimeContractViolation::None;
  }
  if (kQuantType == quantPerTensor)
    return GqaRuntimeContractViolation::PerTensorQuantization;
  if (kQuantType != quantPerChannel)
    return GqaRuntimeContractViolation::MixedQuantization;
  if (kvCacheBitWidth != 8)
    return GqaRuntimeContractViolation::QuantizedBitWidth;
  if (!hasKScale || !hasVScale)
    return GqaRuntimeContractViolation::ScalePair;
  if (elementSizeBytes != 2 || kCacheDataType != HIPDNN_EP_DATATYPE_INT8 ||
      vCacheDataType != HIPDNN_EP_DATATYPE_INT8)
    return GqaRuntimeContractViolation::CacheDataType;
  if (kScaleDataType != HIPDNN_EP_DATATYPE_FLOAT ||
      vScaleDataType != HIPDNN_EP_DATATYPE_FLOAT)
    return GqaRuntimeContractViolation::ScaleDataType;
  if (mode)
    *mode = GqaKvCacheMode::Int8PerChannel;
  return GqaRuntimeContractViolation::None;
}

inline const char *
gqaRuntimeContractMessage(GqaRuntimeContractViolation violation) {
  switch (violation) {
  case GqaRuntimeContractViolation::PositionIds:
    return "position_ids not supported";
  case GqaRuntimeContractViolation::OutputQk:
    return "qk_output not supported";
  case GqaRuntimeContractViolation::Softcap:
    return "nonzero softcap not supported";
  case GqaRuntimeContractViolation::RotaryInterleaved:
    return "rotary_interleaved not supported";
  case GqaRuntimeContractViolation::MixedQuantization:
    return "mixed or unknown K/V quantization schemes not supported";
  case GqaRuntimeContractViolation::PerTensorQuantization:
    return "PER_TENSOR KV quantization not supported";
  case GqaRuntimeContractViolation::QuantizedBitWidth:
    return "only 8-bit PER_CHANNEL KV quantization is supported";
  case GqaRuntimeContractViolation::ScalePair:
    return "K/V quantization scales must be paired with their quantization "
           "scheme";
  case GqaRuntimeContractViolation::CacheDataType:
    return "K/V cache data types do not match the selected quantization mode";
  case GqaRuntimeContractViolation::ScaleDataType:
    return "PER_CHANNEL K/V scale data types must both be float32";
  case GqaRuntimeContractViolation::None:
    return "";
  }
  return "unknown contract violation";
}

#endif // HIPDNN_EP_RUNTIME_GQA_CONTRACT_H
