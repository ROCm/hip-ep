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

class OnnxResultTypeInferenceInterface;

namespace detail {

/// Hand-coded OpInterface scaffolding for
/// `OnnxResultTypeInferenceInterface`. Mirrors the shape that
/// `mlir-tblgen -gen-op-interface-decls` emits for a TableGen interface
/// with a single instance method, so callers can `dyn_cast<...>(op)`
/// exactly like a generated interface.
///
/// We hand-code rather than wire up tablegen for two reasons:
///   1. The `onnx.*` dialect is unregistered (no .td definitions for
///      individual ops), so there is no `OpDefinition` source where a
///      `let interfaces = [...]` clause could attach this interface.
///   2. We want a SINGLE FallbackModel implementation registered against
///      the dialect to dispatch interface calls for ALL onnx.* ops via
///      a name-keyed switch internally (see the rules library in
///      `OnnxResultTypeInference.cpp`). Tablegen-generated ExternalModels
///      would require one model class per op.
struct OnnxResultTypeInferenceInterfaceTraits {
  struct Concept {
    // Function pointer matching the FallbackModel forwarder signature.
    // The first argument is the FallbackModel impl (`this`-equivalent),
    // the second is the operation, the third is the result index whose
    // type we are inferring.
    ::mlir::Type (*computeResultType)(const Concept *impl, ::mlir::Operation *,
                                      unsigned);
  };

  template <typename ConcreteOp> class Model : public Concept {
  public:
    using Interface = ::mlir::hip::OnnxResultTypeInferenceInterface;
    Model() : Concept{computeResultType} {}

    // Forwarder for ops that implement the interface natively. Never
    // exercised in production today (the entire point of the interface
    // is dispatch on UNREGISTERED onnx.* ops via FallbackModel) but the
    // symbol must exist for the OpInterface CRTP base to compile.
    static ::mlir::Type computeResultType(const Concept *,
                                          ::mlir::Operation *op,
                                          unsigned resultIdx) {
      return ::llvm::cast<ConcreteOp>(op).computeResultType(resultIdx);
    }
  };

  template <typename ConcreteModel> class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::hip::OnnxResultTypeInferenceInterface;
    FallbackModel() : Concept{computeResultType} {}

    static ::mlir::Type computeResultType(const Concept *impl,
                                          ::mlir::Operation *op,
                                          unsigned resultIdx) {
      return static_cast<const ConcreteModel *>(impl)->computeResultType(
          op, resultIdx);
    }
  };

  // Required by the OpInterface CRTP base; mirrors the tablegen stub.
  template <typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};

} // namespace detail

/// OpInterface that returns the inferred type of a result given the
/// current operand types and op attributes. Returns null Type when the
/// rules library has no rule for this op (the safety belt: callers
/// MUST treat null as "leave the op alone").
///
/// Dispatched on the `OnnxStubDialect` for any `onnx.*` op via the
/// dialect-level fallback registry (see `OnnxStubDialect` in
/// `hip/InitAllPasses.h` and the FallbackModel singleton in
/// `lib/Conversion/OnnxToHip/OnnxResultTypeInference.cpp`).
class OnnxResultTypeInferenceInterface
    : public ::mlir::OpInterface<
          OnnxResultTypeInferenceInterface,
          detail::OnnxResultTypeInferenceInterfaceTraits> {
public:
  using ::mlir::OpInterface<
      OnnxResultTypeInferenceInterface,
      detail::OnnxResultTypeInferenceInterfaceTraits>::OpInterface;

  ::mlir::Type computeResultType(unsigned resultIdx) {
    return getImpl()->computeResultType(getImpl(), getOperation(), resultIdx);
  }
};

/// Install the `OnnxResultTypeInferenceInterface` FallbackModel singleton
/// on `OnnxStubDialect`'s process-global interface registry. Idempotent;
/// safe to call multiple times because the underlying registry stores a
/// single non-owning pointer per interface TypeID and every call resolves
/// to the same function-local-static FallbackModel.
///
/// After this call, every `onnx.*` op in any subsequently-parsed module
/// can be `dyn_cast<OnnxResultTypeInferenceInterface>`-ed and the
/// dispatch reaches the rules library's FallbackModel.
///
/// Typical caller: a pass that needs to refine cloned onnx body op result
/// types (e.g. `OnnxLoopOutlinePass`) calls this from its constructor /
/// runOnOperation entry point. There is no matching unregister: the
/// FallbackModel is alive for the process lifetime.
void registerOnnxResultTypeInferenceFallback();

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_ONNX_RESULT_TYPE_INFERENCE_H
