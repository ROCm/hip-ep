/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_ONNXTOHIP_ONNX_RESULT_TYPE_INFERENCE_H
#define HIP_CONVERSION_ONNXTOHIP_ONNX_RESULT_TYPE_INFERENCE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace hip {

namespace detail {

/// Hand-coded OpInterface scaffolding for
/// `OnnxResultTypeInferenceInterface`. Mirrors the shape that
/// `mlir-tblgen -gen-op-interface-decls` emits for a TableGen interface
/// with a single instance method, so callers can `dyn_cast<...>(op)`
/// exactly like a generated interface.
///
/// We hand-code rather than wire up tablegen because the `onnx.*`
/// dialect is unregistered (no .td definitions for individual ops, so
/// there is no `OpDefinition` source where a `let interfaces = [...]`
/// clause could attach this interface) and we want a single
/// FallbackModel implementation -- registered against `OnnxStubDialect`
/// -- to dispatch interface calls for ALL onnx.* ops via a name-keyed
/// switch internally (see the rules library in
/// `lib/Conversion/OnnxToHip/OnnxResultTypeInference.cpp`).
struct OnnxResultTypeInferenceInterfaceTraits {
  struct Concept {
    /// FallbackModel forwarder signature: (this-equivalent, op, result idx).
    Type (*computeResultType)(const Concept *impl, Operation *, unsigned);
  };

  template <typename ConcreteOp> class Model : public Concept {
  public:
    Model() : Concept{computeResultType} {}

    /// Forwarder for ops that implement the interface natively. Not
    /// exercised today (dispatch is via FallbackModel on the unregistered
    /// onnx.* ops); the symbol exists for OpInterface CRTP-base parity.
    static Type computeResultType(const Concept *, Operation *op,
                                  unsigned resultIdx) {
      return llvm::cast<ConcreteOp>(op).computeResultType(resultIdx);
    }
  };

  template <typename ConcreteModel> class FallbackModel : public Concept {
  public:
    FallbackModel() : Concept{computeResultType} {}

    static Type computeResultType(const Concept *impl, Operation *op,
                                  unsigned resultIdx) {
      return static_cast<const ConcreteModel *>(impl)->computeResultType(
          op, resultIdx);
    }
  };

  /// Required by the OpInterface CRTP base; mirrors the tablegen stub.
  template <typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};

} // namespace detail

/// OpInterface that returns the inferred type of a result given the
/// current operand types and op attributes. Returns null Type when the
/// rules library has no rule for this op -- callers MUST treat null as
/// "leave the op alone" (the safety belt against false promotion of
/// rank-changing ops we have not yet reasoned about).
///
/// Dispatched on `OnnxStubDialect` for any `onnx.*` op via the
/// dialect-level fallback registry (see `OnnxStubDialect` in
/// `hip/InitAllPasses.h` and the FallbackModel singleton in
/// `lib/Conversion/OnnxToHip/OnnxResultTypeInference.cpp`).
class OnnxResultTypeInferenceInterface
    : public OpInterface<OnnxResultTypeInferenceInterface,
                         detail::OnnxResultTypeInferenceInterfaceTraits> {
public:
  using OpInterface<
      OnnxResultTypeInferenceInterface,
      detail::OnnxResultTypeInferenceInterfaceTraits>::OpInterface;

  Type computeResultType(unsigned resultIdx) {
    return getImpl()->computeResultType(getImpl(), getOperation(), resultIdx);
  }
};

/// Install the `OnnxResultTypeInferenceInterface` FallbackModel
/// singleton on `OnnxStubDialect`'s process-global interface registry.
/// Idempotent: subsequent calls re-bind the same function-local-static
/// FallbackModel pointer. Once registered, every `onnx.*` op can be
/// `dyn_cast<OnnxResultTypeInferenceInterface>`-ed and the dispatch
/// reaches the rules library. There is no matching unregister; the
/// FallbackModel lives for the process lifetime.
void registerOnnxResultTypeInferenceFallback();

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_ONNX_RESULT_TYPE_INFERENCE_H
