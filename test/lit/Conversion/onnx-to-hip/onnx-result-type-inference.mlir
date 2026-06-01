// RUN: hip-mlir-opt %s --hip-test-onnx-result-type-inference --verify-diagnostics

// Regression test for the dialect-level fallback dispatch of
// `OnnxResultTypeInferenceInterface` on unregistered `onnx.*` ops.
//
// PR #265 commit 3 baseline: the rules library is intentionally empty
// (the per-op dispatch table inside `OnnxResultTypeInferenceFallback::
// computeResultType` is not yet populated). The dispatch path is
// nevertheless live: every `onnx.*` op below succeeds at
// `dyn_cast<OnnxResultTypeInferenceInterface>(op)` and reaches the
// FallbackModel, which returns null Type ("<unhandled>") because no
// per-op rule has landed yet. Subsequent commits extend the dispatch
// table and tighten the expected remark for each rule.
//
// What this case PINS at this stage:
//   1. `OnnxStubDialect::getRegisteredInterfaceForOp` resolves
//      `OnnxResultTypeInferenceInterface`'s TypeID to the registered
//      FallbackModel pointer.
//   2. The OpInterface CRTP layer correctly forwards the
//      `computeResultType(...)` call through the FallbackModel's
//      Concept dispatch table to the implementation.
//   3. Distinct onnx.* op names ("onnx.Add", "onnx.Identity",
//      "onnx.UnknownOp") all hit the same dispatch path.
//   4. `dyn_cast` does NOT return null, so the test pass does not emit
//      the "no_iface" remark for any onnx.* op.
//
// All assertions are via `expected-remark@+1` directives consumed by
// `--verify-diagnostics`; no FileCheck pipe is needed (and would not
// match anything because verify-diagnostics consumes the remarks before
// they reach stderr).
func.func @dispatch_baseline(%a: tensor<2x3xf16>, %b: tensor<2x3xf16>)
    -> tensor<2x3xf16> {
  // expected-remark@+1 {{onnx.Add computed: <unhandled>}}
  %0 = "onnx.Add"(%a, %b) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  // expected-remark@+1 {{onnx.Identity computed: <unhandled>}}
  %1 = "onnx.Identity"(%0) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  // expected-remark@+1 {{onnx.UnknownOp computed: <unhandled>}}
  %2 = "onnx.UnknownOp"(%1) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return %2 : tensor<2x3xf16>
}
