/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// What of an operation's configuration convert-ggml-to-hip lowers, exported
// for what decides which operations reach it. libhipep_backend answers a ggml
// backend's supports_op before a split is compiled; keying it on these same
// predicates keeps it from taking an operation the lowering would then
// refuse, and with it every operation compiled beside it.
//
// The configuration is what GGML keeps in op_params (GgmlOps.td's
// ggmlSources), an enumerator by the dialect's spelling (GgmlEnums.td), and
// which optional operands the operation was given; of MUL_MAT, also the
// weights the native kernels read. Each refusal is why the lowering refuses
// an operation so configured, "" when it lowers it. The other element types
// and the shapes the patterns check on their own.

#ifndef HIP_CONVERSION_GGMLTOHIP_GGMLTOHIPLEGALITY_H
#define HIP_CONVERSION_GGMLTOHIP_GGMLTOHIPLEGALITY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::hip {

/// GLU: SwiGLU of a gate and an up operand (`split`: given two), the gate
/// first, without SwiGLU-OAI's alpha and limit.
inline llvm::StringRef gluRefusal(llvm::StringRef kind, bool split,
                                  bool swapped, double alpha, double limit) {
  if (kind != "swiglu" || !split || swapped || alpha != 0.0 || limit != 0.0)
    return "only unswapped two-input GGML SwiGLU is implemented";
  return "";
}

/// ROPE: a normal or NEOX rotation from position zero, of no sections and
/// without YaRN's extension.
inline llvm::StringRef ropeRefusal(llvm::StringRef mode, int64_t pastTokens,
                                   int64_t legacyContext, int64_t offsets,
                                   llvm::ArrayRef<int64_t> sections,
                                   double extensionFactor) {
  if ((mode != "normal" && mode != "neox") || pastTokens != 0 ||
      legacyContext != 0 || offsets != 0 ||
      llvm::any_of(sections, [](int64_t s) { return s != 0; }) ||
      extensionFactor != 0.0)
    return "supports only normal/NEOX GGML RoPE without YaRN";
  return "";
}

/// MUL_MAT's weight (its ggml type): one the native K-quant kernels read where
/// llama.cpp loaded it.
inline bool isNativeMatmulWeight(llvm::StringRef type) {
  return type == "q4_K" || type == "q5_K" || type == "q6_K";
}

/// MUL_MAT: at the default precision, with no hint.
inline llvm::StringRef mulMatRefusal(llvm::StringRef precision, int64_t hint) {
  if (precision != "default" || hint != 0)
    return "HIP lowering supports only default precision and hint";
  return "";
}

/// FLASH_ATTN_EXT: under a mask, with no sinks, a positive scale, no ALiBi
/// slope or logit softcap, at f32 precision.
inline llvm::StringRef flashAttentionRefusal(bool mask, bool sinks,
                                             double scale, double maxBias,
                                             double logitSoftcap,
                                             llvm::StringRef precision) {
  if (!mask || sinks || scale <= 0.0 || maxBias != 0.0 || logitSoftcap != 0.0 ||
      precision != "f32")
    return "requires explicit-mask f32 GGML flash-attention semantics";
  return "";
}

} // namespace mlir::hip

#endif // HIP_CONVERSION_GGMLTOHIP_GGMLTOHIPLEGALITY_H
