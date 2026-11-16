/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Elementwise, reduce, and anchor pipeline selection, compiled for gfx1151
// without a GPU. The first module holds two outlined kernels: a conv + add
// chain and a mul + add chain that carries rock.elementwise. A second module
// holds a last-axis sum that carries rock.reduce. Each marker has to survive
// HIP-to-TOSA and the rock high-level pipeline, because rocMLIR picks
// buildElementwisePipeline or buildReducePipeline by it. The kernels then
// compile with the default compile; the pointwise and reduce kernels have no
// perfConfig. A marked kernel whose computation folds away, or that rocMLIR
// cannot compile, goes back to its hip ops; with strictElementwise only the
// folded one does. A reduce over a non-last axis fails the same way.

#include "hip/Compiler/RocMlirKernelCompiler.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/InitAllPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

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
func.func @rocMlir0(%in: tensor<1x8x8x8xf16>, %w: tensor<16x8x3x3xf16>,
                    %b: tensor<16xf16>, %other: tensor<1x16x8x8xf16>)
    -> tensor<1x16x8x8xf16> attributes {rock.arch = "gfx1151", rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x16x8x8xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x8x8xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x8x8xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x8x8xf16>
  %ea = tensor.empty() : tensor<1x16x8x8xf16>
  %r = hip.add(%ctx) ins(%c, %other : tensor<1x16x8x8xf16>,
                                      tensor<1x16x8x8xf16>)
      outs(%ea : tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
  return %r : tensor<1x16x8x8xf16>
}

func.func @rocMlir1(%x: tensor<1x16x8x8xf16>, %y: tensor<1x16x8x8xf16>,
                    %z: tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
    attributes {rock.arch = "gfx1151", rock.elementwise, rock.kernel} {
  %ctx = ub.poison : !hip.context
  %em = tensor.empty() : tensor<1x16x8x8xf16>
  %m = hip.mul(%ctx) ins(%x, %y : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>)
      outs(%em : tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
  %ep = tensor.empty() : tensor<1x16x8x8xf16>
  %r = hip.add(%ctx) ins(%m, %z : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>)
      outs(%ep : tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16>
  return %r : tensor<1x16x8x8xf16>
}

func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x8x8xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %other: tensor<1x16x8x8xf16>, %x: tensor<1x16x8x8xf16>,
                      %y: tensor<1x16x8x8xf16>, %z: tensor<1x16x8x8xf16>)
    -> (tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>) {
  %e0 = tensor.empty() : tensor<1x16x8x8xf16>
  %anchored = hip.rocmlir(%ctx) @rocMlir0
      ins(%in, %w, %b, %other : tensor<1x8x8x8xf16>, tensor<16x8x3x3xf16>,
                                tensor<16xf16>, tensor<1x16x8x8xf16>)
      outs(%e0 : tensor<1x16x8x8xf16>) : tensor<1x16x8x8xf16>
  %e1 = tensor.empty() : tensor<1x16x8x8xf16>
  %pointwise = hip.rocmlir(%ctx) @rocMlir1
      ins(%x, %y, %z : tensor<1x16x8x8xf16>, tensor<1x16x8x8xf16>,
                       tensor<1x16x8x8xf16>)
      outs(%e1 : tensor<1x16x8x8xf16>) : tensor<1x16x8x8xf16>
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

// rock.blockwise_load/store stop at 32-bit integers, so an i64 chain fails
// in the backend.
constexpr llvm::StringLiteral kUncompilable = R"mlir(
func.func @rocMlir0(%x: tensor<1x16x8x8xi64>, %y: tensor<1x16x8x8xi64>)
    -> tensor<1x16x8x8xi64>
    attributes {rock.arch = "gfx1151", rock.elementwise, rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x16x8x8xi64>
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x16x8x8xi64>, tensor<1x16x8x8xi64>)
      outs(%e : tensor<1x16x8x8xi64>) -> tensor<1x16x8x8xi64>
  return %r : tensor<1x16x8x8xi64>
}

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x8x8xi64>,
                      %y: tensor<1x16x8x8xi64>) -> tensor<1x16x8x8xi64> {
  %e = tensor.empty() : tensor<1x16x8x8xi64>
  %r = hip.rocmlir(%ctx) @rocMlir0 ins(%x, %y : tensor<1x16x8x8xi64>,
                                                tensor<1x16x8x8xi64>)
      outs(%e : tensor<1x16x8x8xi64>) : tensor<1x16x8x8xi64>
  return %r : tensor<1x16x8x8xi64>
}
)mlir";

// A last-axis sum. buildReducePipeline compiles it once, with no perfConfig,
// and the block is four waves.
constexpr llvm::StringLiteral kReduce = R"mlir(
func.func @rocMlir0(%data: tensor<8x16xf32>) -> tensor<8x1xf32>
    attributes {rock.arch = "gfx1151", rock.kernel, rock.reduce} {
  %ctx = ub.poison : !hip.context
  %axes = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<8x1xf32>
  %sum = hip.reduce_sum(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%e : tensor<8x1xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
       reduced_axes = array<i64: 1>} : tensor<8x1xf32>
  return %sum : tensor<8x1xf32>
}

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>)
    -> tensor<8x1xf32> {
  %e = tensor.empty() : tensor<8x1xf32>
  %r = hip.rocmlir(%ctx) @rocMlir0 ins(%data : tensor<8x16xf32>)
      outs(%e : tensor<8x1xf32>) : tensor<8x1xf32>
  return %r : tensor<8x1xf32>
}
)mlir";

// rock-lower-reduce-to-blockwise only accepts the last axis, so this kernel
// fails in the backend and runs as hip.reduce_sum outside strict mode.
constexpr llvm::StringLiteral kReduceUncompilable = R"mlir(
func.func @rocMlir0(%data: tensor<8x16xf32>) -> tensor<1x16xf32>
    attributes {rock.arch = "gfx1151", rock.kernel, rock.reduce} {
  %ctx = ub.poison : !hip.context
  %axes = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<1x16xf32>
  %sum = hip.reduce_sum(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%e : tensor<1x16xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
       reduced_axes = array<i64: 0>} : tensor<1x16xf32>
  return %sum : tensor<1x16xf32>
}

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>)
    -> tensor<1x16xf32> {
  %e = tensor.empty() : tensor<1x16xf32>
  %r = hip.rocmlir(%ctx) @rocMlir0 ins(%data : tensor<8x16xf32>)
      outs(%e : tensor<1x16xf32>) : tensor<1x16xf32>
  return %r : tensor<1x16xf32>
}
)mlir";

// The dispatches and the hip ops of type OpT left in `module`.
template <typename OpT> std::pair<int, int> countOps(mlir::ModuleOp module) {
  int dispatches = 0;
  int hipOps = 0;
  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::hip::RocMlirOp>(op))
      ++dispatches;
    if (llvm::isa<OpT>(op))
      ++hipOps;
  });
  return {dispatches, hipOps};
}

enum class Marker { Anchor, Elementwise, Reduce };

// High-level pipeline then default compile, on the kernel `name` taken out of
// the TOSA module.
void compileKernel(mlir::ModuleOp tosaKernels, llvm::StringRef name,
                   Marker marker, llvm::StringRef what) {
  mlir::OwningOpRef<mlir::ModuleOp> single =
      mlir::hip::takeRocMlirKernelModule(tosaKernels, name);
  check(static_cast<bool>(single), (what + ": kernel split out").str());
  if (!single)
    return;
  check(mlir::hip::runRocMlirHighLevelPipeline(*single),
        (what + ": high-level pipeline").str());
  bool elementwise = mlir::hip::isRocMlirElementwiseKernelModule(*single);
  bool reduce = mlir::hip::isRocMlirReduceKernelModule(*single);
  check(elementwise == (marker == Marker::Elementwise),
        (what + (marker == Marker::Elementwise
                     ? ": elementwise marker survives high-level"
                     : ": no elementwise marker after high-level"))
            .str());
  check(reduce == (marker == Marker::Reduce),
        (what + (marker == Marker::Reduce
                     ? ": reduce marker survives high-level"
                     : ": no reduce marker after high-level"))
            .str());
  if (marker == Marker::Reduce) {
    bool sawReduce = false;
    single->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "rock.reduce")
        sawReduce = true;
    });
    check(sawReduce, (what + ": high-level lowering emits rock.reduce").str());
  } else {
    check(mlir::hip::hasRocMlirComputation(*single),
          (what + ": has computation after high-level").str());
  }
  mlir::hip::CompiledKernel out;
  bool ok = mlir::hip::compileRocMlirDefaultPerfConfig(*single, "gfx1151", out);
  check(ok && !out.binary.empty() && out.gridSize > 0 && out.blockSize > 0,
        (what + ": default compile yields a binary and launch geometry").str());
  if (marker == Marker::Reduce)
    check(out.blockSize == 128,
          (what + ": reduce block size is four waves").str());
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
  std::string anchorName = "rocMlir0";
  std::string pointwiseName = "rocMlir1";

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

  compileKernel(*tosaKernels, anchorName, Marker::Anchor, "conv kernel");
  compileKernel(*tosaKernels, pointwiseName, Marker::Elementwise,
                "pointwise kernel");

  // The same graph through the EP's entry point with the default compile.
  llvm::SmallVector<std::string> reasons;
  auto onHipPath = [&](llvm::StringRef, llvm::StringRef reason) {
    reasons.push_back(reason.str());
  };
  mlir::hip::RocMlirEmbedOptions options;
  options.arch = "gfx1151";
  options.onHipPath = onHipPath;
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
  check(hipOps == 0 && reasons.empty(), "no kernel fell back to its hip ops");

  // A kernel that folds away runs its hip ops, also in strict mode.
  mlir::hip::RocMlirEmbedOptions strict = options;
  strict.strictElementwise = true;
  mlir::OwningOpRef<mlir::ModuleOp> folded =
      mlir::parseSourceString<mlir::ModuleOp>(kFolded, &context);
  if (!folded) {
    llvm::errs() << "failed to parse the folded module\n";
    return 1;
  }
  reasons.clear();
  check(mlir::succeeded(
            mlir::hip::compileAndEmbedRocMlirKernels(*folded, strict)),
        "strict: a folded kernel does not fail the module");
  check(countOps<mlir::hip::MaxOp>(*folded) == std::make_pair(0, 1),
        "a marked kernel that folds to nothing runs its hip ops");
  check(reasons.size() == 1 && reasons[0] == "folded",
        "the folded kernel is reported as folded");

  // A kernel that fails in the backend runs its hip ops, except in strict
  // mode.
  mlir::OwningOpRef<mlir::ModuleOp> uncompilable =
      mlir::parseSourceString<mlir::ModuleOp>(kUncompilable, &context);
  if (!uncompilable) {
    llvm::errs() << "failed to parse the uncompilable module\n";
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> uncompilableStrict = uncompilable->clone();
  reasons.clear();
  check(mlir::succeeded(
            mlir::hip::compileAndEmbedRocMlirKernels(*uncompilable, options)),
        "a kernel rocMLIR cannot compile does not fail the module");
  check(countOps<mlir::hip::AddOp>(*uncompilable) == std::make_pair(0, 1),
        "a kernel rocMLIR cannot compile runs its hip ops");
  check(reasons.size() == 1 && reasons[0] != "folded",
        "the failed kernel is reported with its failed stage");
  check(mlir::failed(mlir::hip::compileAndEmbedRocMlirKernels(
            *uncompilableStrict, strict)),
        "strict: a kernel rocMLIR cannot compile fails the module");

  // A last-axis sum selects the reduce pipeline and embeds a binary.
  mlir::OwningOpRef<mlir::ModuleOp> reduceGraph =
      mlir::parseSourceString<mlir::ModuleOp>(kReduce, &context);
  if (!reduceGraph) {
    llvm::errs() << "failed to parse the reduce module\n";
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> reduceEmbedded = reduceGraph->clone();
  mlir::OwningOpRef<mlir::ModuleOp> reduceTosa =
      mlir::hip::takeRocMlirKernelsAsTosaModule(*reduceGraph, "gfx1151");
  check(static_cast<bool>(reduceTosa), "HIP-to-TOSA converts the reduce kernel");
  if (!reduceTosa)
    return 1;
  auto tosaReduce = reduceTosa->lookupSymbol<mlir::func::FuncOp>("rocMlir0");
  check(tosaReduce && tosaReduce->hasAttr("rock.reduce") &&
            !tosaReduce->hasAttr("rock.elementwise"),
        "reduce marker survives HIP-to-TOSA");
  compileKernel(*reduceTosa, "rocMlir0", Marker::Reduce, "reduce kernel");

  reasons.clear();
  if (mlir::failed(mlir::hip::compileAndEmbedRocMlirKernels(*reduceEmbedded,
                                                           options))) {
    llvm::errs() << "compileAndEmbedRocMlirKernels failed for the reduce\n";
    return 1;
  }
  int reduceBinary = 0;
  int64_t reduceGrid = 0;
  int64_t reduceBlock = 0;
  reduceEmbedded->walk([&](mlir::hip::RocMlirOp op) {
    if (!op.getKernelBinary().empty())
      ++reduceBinary;
    reduceGrid = op.getGridSize();
    reduceBlock = op.getBlockSize();
  });
  check(reduceBinary == 1 && reduceGrid > 0 && reduceBlock == 128,
        "the reduce dispatch carries a binary, a grid, and a 128-wide block");
  check(countOps<mlir::hip::ReduceSumOp>(*reduceEmbedded) ==
            std::make_pair(1, 0),
        "the compiled reduce keeps its dispatch and drops the hip op");
  check(reasons.empty(), "the reduce kernel did not fall back");

  // A non-last-axis sum marked rock.reduce fails to compile and runs as HIP,
  // except in strict mode.
  mlir::OwningOpRef<mlir::ModuleOp> badReduce =
      mlir::parseSourceString<mlir::ModuleOp>(kReduceUncompilable, &context);
  if (!badReduce) {
    llvm::errs() << "failed to parse the uncompilable reduce module\n";
    return 1;
  }
  mlir::OwningOpRef<mlir::ModuleOp> badReduceStrict = badReduce->clone();
  reasons.clear();
  check(mlir::succeeded(
            mlir::hip::compileAndEmbedRocMlirKernels(*badReduce, options)),
        "a non-last-axis reduce does not fail the module");
  check(countOps<mlir::hip::ReduceSumOp>(*badReduce) == std::make_pair(0, 1),
        "a non-last-axis reduce runs its hip op");
  check(reasons.size() == 1 && reasons[0] != "folded",
        "the failed reduce is reported with its failed stage");
  check(mlir::failed(mlir::hip::compileAndEmbedRocMlirKernels(*badReduceStrict,
                                                             strict)),
        "strict: a non-last-axis reduce fails the module");

  if (g_failures) {
    llvm::errs() << g_failures << " check(s) failed\n";
    return 1;
  }
  llvm::outs() << "rocmlir elementwise pipeline selection passed\n";
  return 0;
}
