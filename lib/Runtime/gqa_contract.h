/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_RUNTIME_GQA_CONTRACT_H
#define HIPDNN_EP_RUNTIME_GQA_CONTRACT_H

#include <cstdint>

enum class GqaRuntimeContractViolation {
  None,
  PositionIds,
  OutputQk,
  Softcap,
};

/// Validate the GQA schema features that neither runtime implementation may
/// silently ignore. IEEE positive and negative zero are both accepted;
/// NaN, infinity, subnormal, and every other nonzero softcap are rejected.
inline GqaRuntimeContractViolation
validateGqaRuntimeContract(bool hasPositionIds, bool hasOutputQk,
                           int64_t qkOutput, float softcap) {
  if (hasPositionIds)
    return GqaRuntimeContractViolation::PositionIds;
  if (hasOutputQk || qkOutput != 0)
    return GqaRuntimeContractViolation::OutputQk;
  if (softcap != 0.0f)
    return GqaRuntimeContractViolation::Softcap;
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
  case GqaRuntimeContractViolation::None:
    return "";
  }
  return "unknown contract violation";
}

#endif // HIPDNN_EP_RUNTIME_GQA_CONTRACT_H
