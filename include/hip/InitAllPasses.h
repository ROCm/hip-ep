/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_COMPILER_INITALLPASSES_H
#define HIP_COMPILER_INITALLPASSES_H

#include <mutex>

#include "hip/Compiler/PluginRegistry.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/Passes.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/Transforms/BufferizableOpInterfaceImpl.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Pipelines.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"

namespace hip::compiler {

namespace detail {
/// Minimal ONNX dialect stub that claims the "onnx" namespace and
/// permits unknown operations, avoiding a dependency on the full
/// onnx-mlir library. The importer is responsible for emitting
/// `tensor<*xT>` (unranked) for values whose shape it does not know;
/// any unranked tensors that survive into the HIP dialect are refined
/// post-conversion by `--hip-infer-shapes` via
/// `ReifyRankedShapedTypeOpInterface`. See
/// `docs/design/unranked-tensor-handling.md` for the cross-repo
/// contract.
class OnnxStubDialect : public mlir::Dialect {
public:
  explicit OnnxStubDialect(mlir::MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx,
                mlir::TypeID::get<OnnxStubDialect>()) {
    allowUnknownOperations();
  }
  static constexpr llvm::StringLiteral getDialectNamespace() { return "onnx"; }
};
} // namespace detail

/// Register all required dialects into a DialectRegistry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<mlir::hipsr::HipsrDialect>();
  mlir::hipsr::registerConvertHipsrToLLVMInterface(registry);
  registry.insert<detail::OnnxStubDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  // The built-in pipeline omits ownership-based buffer deallocation, but the
  // pass remains available to custom pipelines and characterization tests.
  // It walks arith ops (e.g. arith.select on buffers) and requires this model.
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);
  mlir::hipsr::registerBufferizableOpInterfaceExternalModels(registry);
}

/// Load all required dialects into an MLIRContext.
///
/// After the in-tree dialects are registered, any dialect-registration
/// callbacks contributed by statically-linked plugins
/// (`addDialectRegistration`) are applied to the same registry, so a plugin's
/// vendor dialect -- and its bufferization / HIP->LLVM-lowering interface
/// models attached via DialectExtension -- are present in this context. No-op
/// when no plugin contributes a dialect. Callers (e.g. CompilerDriver) invoke
/// `dispatchPluginRegistrationsOnce()` before this; the accessor is
/// also defensively idempotent.
inline void loadAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  for (auto registerFn : pluginDialectRegistrations())
    registerFn(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

/// Populate MLIR's global pass registry with every pass and pipeline the
/// production ONNX->HIP->LLVM flow composes, plus the standard MLIR passes it
/// interleaves. Like upstream `mlir::registerAllPasses`, this matters only for
/// name-based pass lookup -- a compiler that builds its pipeline
/// programmatically (the default EP path via `buildOnnxToHipPipeline` etc.)
/// does not need it. We register here because two name-based consumers do: the
/// `HIPDNN_EP_PIPELINE` textual-pipeline override (`CompilerDriver`) and the
/// `hip-mlir-opt` command line. Both call this one function, so the set of
/// nameable passes can never drift between the EP and the tool.
///
/// The authoritative name lists live with the passes themselves
/// (`Dialect/Transforms/Passes.td`, each conversion pass's `getArgument()`)
/// and are catalogued in `docs/pipeline_pass_menu.md`; this function only
/// triggers their registration, so it deliberately does not re-enumerate them.
///
/// Override caveat: a few production passes cannot be named individually --
/// `generate-interface` takes a `CompilationOptionsT`, `compile-hipdnn-graphs`
/// takes a runtime handle, and a handful of MLIR utility passes are added only
/// inside the pipeline builders. A textual override that needs the C-ABI entry
/// point therefore composes the registered *pipeline* names
/// (`onnx-to-hip-pipeline`, `hip-to-llvm-pipeline`, `hipdnn-pipeline`), which
/// embed those passes. See docs/design/plugin-interface.md "Pipeline
/// composition".
inline void registerAllPasses() {
  // Register exactly once per process. PassPipelineRegistration (used by
  // registerHipPipelines for `onnx-to-hip-pipeline` etc.) writes into MLIR's
  // GLOBAL pipeline registry and asserts/aborts ("... registered multiple
  // times") if the same name is registered twice. CompilerDriver::compile()
  // / compileFromModule() call this per compile, so multi-subgraph models
  // (e.g. VLM: embedding + text + vision) would register a second time and
  // abort. std::call_once makes the whole registration idempotent across
  // repeated compiles in one process (matching
  // dispatchPluginRegistrationsOnce).
  static std::once_flag registered;
  std::call_once(registered, [] {
    // HIP transform passes (TableGen GEN_PASS_REGISTRATION) and the composable
    // pipeline names. See docs/pipeline_pass_menu.md for the catalogue.
    mlir::hip::registerHipPasses();
    mlir::hip::registerHipPipelines();

    // hipsr dialect transform passes (TableGen GEN_PASS_REGISTRATION):
    // hipsr-populate-shape-region, hipsr-externalize-constants, ...
    mlir::hipsr::registerHipsrPasses();

    // Conversion passes (convert-onnx-to-hip, outline-onnx-to-hipdnn,
    // convert-hip-to-llvm); onnx-loop-outline and its sibling onnx-if-outline
    // are hand-written, not in the .td set, so they are registered separately
    // below.
    registerConversionPasses();
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return mlir::hip::createOnnxLoopOutlinePass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return mlir::hip::createOnnxIfOutlinePass();
    });
    // Standard MLIR passes used by the production flow or supported custom/test
    // pipelines, registered so an override can name them around the hip-*
    // passes. The registrar names
    // differ from the textual pass names they register (e.g.
    // registerSCFToControlFlowPass registers `convert-scf-to-cf`);
    // docs/pipeline_pass_menu.md lists every textual name. Bufferization's
    // registrars cover several passes each (one-shot-bufferize,
    // buffer-results-to-out-params, buffer-deallocation-pipeline, ...).
    mlir::registerCanonicalizerPass();
    mlir::registerPass(
        []() -> std::unique_ptr<mlir::Pass> { return mlir::createCSEPass(); });
    mlir::bufferization::registerBufferizationPasses();
    mlir::bufferization::registerBufferizationPipelines();
    mlir::registerConvertBufferizationToMemRefPass();
    mlir::registerSCFToControlFlowPass();
    mlir::registerReconcileUnrealizedCastsPass();
    mlir::memref::registerResolveShapedTypeResultDimsPass();
  });
}

} // namespace hip::compiler

#endif // HIP_COMPILER_INITALLPASSES_H
