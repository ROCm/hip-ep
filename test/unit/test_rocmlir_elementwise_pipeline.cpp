/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Elementwise versus anchor pipeline selection, compiled for gfx1151 without
// a GPU. main_graph holds a conv + add chain and an independent mul + add
// chain. hip-fuse-rocmlir outlines one kernel for each and marks only the
// pointwise one rock.elementwise. The marker has to survive HIP-to-TOSA and
// the rock high-level pipeline, because rocMLIR picks buildElementwisePipeline
// by it. Both kernels then compile with the default compile, the pointwise one
// with no perfConfig. A marked kernel whose computation folds away goes back
// to its hip ops.

#include "hip/Compiler/RocMlirKernelCompiler.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/InitAllPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"

namespace {

int g_failures = 0;

void check(bool cond, llvm::StringRef what) {
  if (cond) {
    llvm::outs() << "[ OK ] " << what << "\n";
    return;
  }
  llvm::errs() << "[FAIL] " << what << "\n";
  ++g_failures;
}

constexpr llvm::StringLiteral kGraph = R"mlir(
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x8x8xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %other: tensor<1x16x8x8xf16>, %x: tensor<1x16x8x8xf16>,
                      %y: tensor<1x16x8x8xf16>, %z: tensor<1x16x8x8xf16>)
    -> (tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>) {
  %e = tensor.empty() : tensor<1x16x8x8xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x8x8xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x8x8xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x8x8xf16>
  %ea = tensor.empty() : tensor<1x16x8x8xf16>
  %anchored = hip.add(%ctx) ins(%c, %other : tensor<1x16x8x8xf16>,
                                             tensor<1x16x8x8xf16>)
      outs(%ea : tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
  %em = tensor.empty() : tensor<1x16x8x8xf16>
  %m = hip.mul(%ctx) ins(%x, %y : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>)
      outs(%em : tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
  %ep = tensor.empty() : tensor<1x16x8x8xf16>
  %pointwise = hip.add(%ctx) ins(%m, %z : tensor<1x16x8x8xf16>,
                                          tensor<1x16x8x8xf16>)
      outs(%ep : tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
  return %anchored, %pointwise : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>
}
)mlir";

// hip.max(%x, %x) is %x: the kernel has nothing left to compute once lowered.
constexpr llvm::StringLiteral kFolded = R"mlir(
func.func @rocMlir0(%x: tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
    attributes {rock.arch = "gfx1151", rock.elementwise, rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x16x8x8xf16>
  %r = hip.max(%ctx) ins(%x, %x : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>)
      outs(%e : tensor<1x16x8x8xf16>) : tensor<1x16x8x8xf16>
  return %r : tensor<1x16x8x8xf16>
}

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x8x8xf16>)
    -> tensor<1x16x8x8xf16> {
  %e = tensor.empty() : tensor<1x16x8x8xf16>
  %r = hip.rocmlir(%ctx) @rocMlir0 ins(%x : tensor<1x16x8x8xf16>)
      outs(%e : tensor<1x16x8x8xf16>) : tensor<1x16x8x8xf16>
  return %r : tensor<1x16x8x8xf16>
}
)mlir";

bool containsOp(mlir::func::FuncOp func, llvm::StringRef name) {
  bool found = false;
  func.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == name)
      found = true;
  });
  return found;
}

// The outlined kernel holding `marker`, or null.
mlir::func::FuncOp kernelWith(mlir::ModuleOp module, llvm::StringRef marker) {
  for (auto func : module.getOps<mlir::func::FuncOp>())
    if (func->hasAttr("rock.kernel") && containsOp(func, marker))
      return func;
  return {};
}

// High-level pipeline then default compile, on the kernel `name` taken out of
// the TOSA module.
void compileKernel(mlir::ModuleOp tosaKernels, llvm::StringRef name,
                   bool elementwise, llvm::StringRef what) {
  mlir::OwningOpRef<mlir::ModuleOp> single =
      mlir::hip::takeRocMlirKernelModule(tosaKernels, name);
  check(static_cast<bool>(single), (what + ": kernel split out").str());
  if (!single)
    return;
  check(mlir::hip::runRocMlirHighLevelPipeline(*single),
        (what + ": high-level pipeline").str());
  check(mlir::hip::isRocMlirElementwiseKernelModule(*single) == elementwise,
        (what + (elementwise ? ": marker survives high-level"
                             : ": no marker after high-level"))
            .str());
  check(mlir::hip::hasRocMlirComputation(*single),
        (what + ": has computation after high-level").str());
  mlir::hip::CompiledKernel out;
  bool ok = mlir::hip::compileRocMlirDefaultPerfConfig(*single, "gfx1151", out);
  check(ok && !out.binary.empty() && out.gridSize > 0 && out.blockSize > 0,
        (what + ": default compile yields a binary and launch geometry").str());
}

} // namespace

int main() {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  hip::compiler::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  mlir::hip::registerRocMlirDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> graph =
      mlir::parseSourceString<mlir::ModuleOp>(kGraph, &context);
  if (!graph) {
    llvm::errs() << "failed to parse the graph module\n";
    return 1;
  }
  mlir::PassManager fuse(&context);
  fuse.addNestedPass<mlir::func::FuncOp>(mlir::hip::createFuseROCMlirPass());
  if (mlir::failed(fuse.run(*graph))) {
    llvm::errs() << "hip-fuse-rocmlir failed\n";
    return 1;
  }

  mlir::func::FuncOp anchor = kernelWith(*graph, "hip.conv");
  mlir::func::FuncOp pointwise = kernelWith(*graph, "hip.mul");
  check(anchor && pointwise && anchor != pointwise,
        "fuse outlines the conv chain and the pointwise chain separately");
  if (!anchor || !pointwise) {
    graph->print(llvm::errs());
    return 1;
  }
  std::string anchorName = anchor.getSymName().str();
  std::string pointwiseName = pointwise.getSymName().str();
  check(!anchor->hasAttr("rock.elementwise"),
        "conv kernel is not marked rock.elementwise");
  check(pointwise->hasAttr("rock.elementwise"),
        "pointwise kernel is marked rock.elementwise");

  mlir::OwningOpRef<mlir::ModuleOp> embedded = graph->clone();

  mlir::OwningOpRef<mlir::ModuleOp> tosaKernels =
      mlir::hip::takeRocMlirKernelsAsTosaModule(*graph, "gfx1151");
  check(static_cast<bool>(tosaKernels), "HIP-to-TOSA converts both kernels");
  if (!tosaKernels)
    return 1;
  auto tosaAnchor = tosaKernels->lookupSymbol<mlir::func::FuncOp>(anchorName);
  auto tosaPointwise =
      tosaKernels->lookupSymbol<mlir::func::FuncOp>(pointwiseName);
  check(tosaAnchor && !tosaAnchor->hasAttr("rock.elementwise"),
        "conv kernel has no marker after HIP-to-TOSA");
  check(tosaPointwise && tosaPointwise->hasAttr("rock.elementwise"),
        "marker survives HIP-to-TOSA");

  compileKernel(*tosaKernels, anchorName, /*elementwise=*/false, "conv kernel");
  compileKernel(*tosaKernels, pointwiseName, /*elementwise=*/true,
                "pointwise kernel");

  // The same graph through the EP's entry point with the default compile.
  mlir::hip::RocMlirEmbedOptions options;
  options.arch = "gfx1151";
  if (mlir::failed(
          mlir::hip::compileAndEmbedRocMlirKernels(*embedded, options))) {
    llvm::errs() << "compileAndEmbedRocMlirKernels failed\n";
    return 1;
  }
  int withBinary = 0;
  int hipOps = 0;
  embedded->walk([&](mlir::Operation *op) {
    if (auto dispatch = llvm::dyn_cast<mlir::hip::RocMlirOp>(op))
      if (!dispatch.getKernelBinary().empty())
        ++withBinary;
    if (llvm::isa<mlir::hip::ConvOp, mlir::hip::MulOp, mlir::hip::AddOp>(op))
      ++hipOps;
  });
  check(withBinary == 2, "both dispatches carry a rocMLIR binary");
  check(hipOps == 0, "no kernel fell back to its hip ops");

  mlir::OwningOpRef<mlir::ModuleOp> folded =
      mlir::parseSourceString<mlir::ModuleOp>(kFolded, &context);
  if (!folded) {
    llvm::errs() << "failed to parse the folded module\n";
    return 1;
  }
  if (mlir::failed(
          mlir::hip::compileAndEmbedRocMlirKernels(*folded, options))) {
    llvm::errs() << "compileAndEmbedRocMlirKernels failed on the folded "
                    "kernel\n";
    return 1;
  }
  int dispatches = 0;
  int maxes = 0;
  folded->walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::hip::RocMlirOp>(op))
      ++dispatches;
    if (llvm::isa<mlir::hip::MaxOp>(op))
      ++maxes;
  });
  check(dispatches == 0 && maxes == 1,
        "a marked kernel that folds to nothing runs its hip ops");

  if (g_failures) {
    llvm::errs() << g_failures << " check(s) failed\n";
    return 1;
  }
  llvm::outs() << "rocmlir elementwise pipeline selection passed\n";
  return 0;
}
