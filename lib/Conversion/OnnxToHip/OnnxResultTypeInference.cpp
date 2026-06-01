/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxResultTypeInference.cpp - rules library + dialect fallback -===//
//
// FallbackModel implementation of `OnnxResultTypeInferenceInterface`.
//
// `OnnxResultTypeInferenceFallback::computeResultType` is the dispatch
// point: it switches on `op->getName()` and delegates to per-op rule
// helpers for each ONNX op the rules library knows about. Returns null
// Type for every op outside the rules library; callers (e.g.
// `OnnxLoopOutlinePass`'s body op refinement) MUST treat null as "leave
// the op alone" -- this is the safety belt against false promotion of
// rank-changing ops we have not yet reasoned about.
//
// Initial landing (this commit): the dispatch switch is intentionally
// empty. Every onnx.* op returns null Type. Subsequent commits in the
// same series add per-op rules (Add, Concat, ...) one by one with
// per-rule LIT coverage. The empty-switch state is regression-tested by
// `test/lit/Conversion/onnx-to-hip/onnx-result-type-inference.mlir`,
// which verifies the dispatch reaches this FallbackModel for every
// onnx.* op (rather than failing at `dyn_cast` time).
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHip/OnnxResultTypeInference.h"
#include "hip/InitAllPasses.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
namespace hip {

namespace {

/// Singleton FallbackModel installed on OnnxStubDialect for the
/// `OnnxResultTypeInferenceInterface`. One instance per process; alive
/// for the process lifetime via a function-local static.
struct OnnxResultTypeInferenceFallback
    : public OnnxResultTypeInferenceInterface::FallbackModel<
          OnnxResultTypeInferenceFallback> {
  ::mlir::Type computeResultType(::mlir::Operation *op,
                                 unsigned resultIdx) const {
    if (resultIdx >= op->getNumResults() || op->getNumOperands() == 0)
      return {};
    // Dispatch table is populated incrementally as rules land. While the
    // table is empty every onnx.* op falls through to the null return,
    // which callers treat as "no rule -- leave the op alone".
    return {};
  }
};

} // namespace

void registerOnnxResultTypeInferenceFallback() {
  // Function-local static keeps the FallbackModel alive for the lifetime
  // of the process; OnnxStubDialect just stores a non-owning void*.
  // `registerInterfaceFallback` is idempotent: re-registering the same
  // (interface TypeID, fallback ptr) pair is a no-op since the registry
  // is a DenseMap that overwrites on insert.
  static OnnxResultTypeInferenceFallback s_fallback;
  ::hip::compiler::detail::OnnxStubDialect::registerInterfaceFallback(
      ::mlir::TypeID::get<OnnxResultTypeInferenceInterface>(), &s_fallback);
}

} // namespace hip
} // namespace mlir
