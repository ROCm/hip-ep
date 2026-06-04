/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxResultTypeInferenceTestPass.cpp - LIT-only test harness ------===//
//
// `--hip-test-onnx-result-type-inference`: walks every `onnx.*` op in a
// module, calls `OnnxResultTypeInferenceInterface::computeResultType(0)`
// on it, and emits a remark recording either the inferred result type
// or `<unhandled>` (rules library returned null Type). A `no_iface`
// remark is emitted when the dialect fallback dispatch fails -- a
// regression in the `OnnxStubDialect` override would surface here.
//
// Consumed by `test/lit/Conversion/onnx-to-hip/onnx-result-type-
// inference.mlir` via `--verify-diagnostics`; production callers (e.g.
// `OnnxLoopOutlinePass`) use the same interface dispatch pattern
// without emitting remarks.
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
    registerOnnxResultTypeInferenceFallback();

    ModuleOp module = getOperation();
    module.walk([](Operation *op) {
      if (!op->getName().getStringRef().starts_with("onnx."))
        return;
      auto iface = dyn_cast<OnnxResultTypeInferenceInterface>(op);
      if (!iface) {
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
