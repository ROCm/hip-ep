/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxResultTypeInferenceTestPass.cpp - LIT-only test harness -=======//
//
// `--hip-test-onnx-result-type-inference`: walks every `onnx.*` op in a
// module, calls `OnnxResultTypeInferenceInterface::computeResultType(0)`
// on it, and emits an `op->emitRemark()` recording either the inferred
// result type or `<unhandled>` (rules library returned null Type).
//
// Used by `test/lit/Conversion/onnx-to-hip/onnx-result-type-inference.mlir`
// to regression-test (a) that the dialect-level fallback dispatch reaches
// the FallbackModel for every onnx.* op (cast must NOT fail), and (b)
// that the rules library produces the expected type for each handled op
// shape.
//
// Production callers (e.g. `OnnxLoopOutlinePass`) follow the same usage
// pattern internally -- there is no production pass that emits remarks;
// the test pass exists purely so each rule can be unit-tested in
// isolation as it lands.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHip/OnnxResultTypeInference.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_ONNXRESULTTYPEINFERENCETESTPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct OnnxResultTypeInferenceTestPass
    : public impl::OnnxResultTypeInferenceTestPassBase<
          OnnxResultTypeInferenceTestPass> {
  void runOnOperation() override {
    // Idempotent: subsequent invocations within the same process re-bind
    // the same fallback singleton.
    registerOnnxResultTypeInferenceFallback();

    ModuleOp module = getOperation();
    module.walk([](Operation *op) {
      if (!op->getName().getStringRef().starts_with("onnx."))
        return;
      auto iface = dyn_cast<OnnxResultTypeInferenceInterface>(op);
      if (!iface) {
        // The dialect fallback returned nullptr for the requested
        // interface ID. Should not happen once `registerOnnxResultType
        // InferenceFallback()` has run; observable here as a self-test
        // against a regression in the dialect override.
        op->emitRemark() << op->getName() << " no_iface";
        return;
      }
      Type t = iface.computeResultType(0);
      if (!t)
        op->emitRemark() << op->getName() << " computed: <unhandled>";
      else
        op->emitRemark() << op->getName() << " computed: " << t;
    });
  }
};

} // namespace

} // namespace hip
} // namespace mlir
