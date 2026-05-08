//===- InitAllPasses.h - Register all HIP dialects/passes -------*- C++ -*-===//
//
// Public entry points for setting up an MLIR DialectRegistry / MLIRContext
// for the HIP compiler. The implementations live in
// `lib/Compiler/InitAllPasses.cpp` so that the heavy MLIR transitive includes
// they require do not bleed into every translation unit that needs a HIP
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_INITALLPASSES_H
#define HIP_INITALLPASSES_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeID.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir {
namespace hip {

namespace detail {
/// Minimal ONNX dialect stub that claims the "onnx" namespace and permits
/// unknown operations.  This avoids depending on the full onnx-mlir library
/// when running passes that only need to round-trip ONNX ops textually
/// (e.g. `hip-mlir-opt --convert-onnx-to-hip` on canned IR).
///
/// Defined inline because callers must be able to template-instantiate
/// `registry.insert<OnnxStubDialect>()` from any TU.
class OnnxStubDialect : public mlir::Dialect {
public:
  explicit OnnxStubDialect(mlir::MLIRContext* ctx)
      : Dialect(getDialectNamespace(), ctx,
                mlir::TypeID::get<OnnxStubDialect>()) {
    allowUnknownOperations();
  }
  static constexpr llvm::StringLiteral getDialectNamespace() { return "onnx"; }
};
} // namespace detail

/// Register all required dialects into a DialectRegistry.
void registerAllDialects(mlir::DialectRegistry& registry);

/// Load all required dialects into an MLIRContext.
void loadAllDialects(mlir::MLIRContext& context);

/// Register all passes and pipelines.
void registerAllPasses();

} // namespace hip
} // namespace mlir

#endif // HIP_INITALLPASSES_H
