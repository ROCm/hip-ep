//===- Passes.h - ONNX to HipDNN graph pass declarations --------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIPDNN_PASSES_H
#define HIP_CONVERSION_ONNXTOHIPDNN_PASSES_H

#include "llvm/ADT/StringMap.h"

#include <memory>

struct hipdnnHandle;
typedef hipdnnHandle *hipdnnHandle_t;

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir {
namespace hip {

#define GEN_PASS_DECL_OUTLINEONNXTOHIPDNNPASS
#include "hip/Conversion/Passes.h.inc"

/// Custom deleter for type-erased HipDNNGraph pointers.
/// Implemented in CompileHipDNNGraphs.cpp where the real type is visible.
/// Using void* avoids introducing the global ::hip namespace, which collides
/// with ::mlir::hip when unqualified "hip::" is used with "using namespace
/// mlir".
struct GraphDeleter {
  void operator()(void *ptr) const;
};

using OwnedGraph = std::unique_ptr<void, GraphDeleter>;

/// Map of compiled hipDNN graphs, keyed by graph name.
/// Values are type-erased hip::graph::HipDNNGraph pointers.
/// Shared ownership: the pass populates it, CompilerDriver retrieves it.
using CompiledGraphMap = std::shared_ptr<llvm::StringMap<OwnedGraph>>;

/// Create the CompileHipDNNGraphs pass.
///
/// Walks hip.hipdnn_graph_outline ops, compiles their regions via hipDNN,
/// and replaces them with hip.hipdnn_graph.  Un-outlines on failure.
///
/// This factory is hand-rolled because its constructor takes a runtime
/// `hipdnnHandle_t` and a `CompiledGraphMap` out-parameter, neither of
/// which can be expressed as a TableGen `Option`.
///
/// @param handle        Live hipDNN handle (GPU must be available)
/// @param output_graphs Map to store compiled graphs (populated by the pass)
std::unique_ptr<Pass>
createCompileHipDNNGraphsPass(hipdnnHandle_t handle,
                              CompiledGraphMap output_graphs);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIPDNN_PASSES_H
