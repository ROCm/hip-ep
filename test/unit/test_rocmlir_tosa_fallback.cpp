/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The plan example, without a GPU. main_graph dispatches two outlined kernels:
//
//   rocMlir0: static conv then leaky_relu. HIP-to-TOSA accepts it, so the
//             dispatch stays and receives the binary compileOne returns.
//   rocMlir1: static conv, then a dynamic sub, then a mul. hip.sub is illegal
//             for a dynamic tensor, so the whole fused function is inlined.
//             conv is not split onto rocMLIR; hip.conv, hip.sub, and hip.mul
//             all come back under main_graph.
//
// compileOne is stubbed. HIP-to-TOSA and the rock high-level pipeline are not.

#include "hip/Compiler/RocMlirKernelCompiler.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/InitAllPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

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

constexpr llvm::StringLiteral kModule = R"mlir(
func.func @rocMlir0(%in: tensor<1x8x4x4xf16>, %w: tensor<16x8x3x3xf16>,
                    %b: tensor<16xf16>) -> tensor<1x16x4x4xf16>
    attributes {rock.arch = "gfx1151", rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %er = tensor.empty() : tensor<1x16x4x4xf16>
  %r = hip.leaky_relu(%ctx) ins(%c : tensor<1x16x4x4xf16>)
      outs(%er : tensor<1x16x4x4xf16>) {alpha = 0.01 : f64}
      : tensor<1x16x4x4xf16>
  return %r : tensor<1x16x4x4xf16>
}

func.func @rocMlir1(%in: tensor<1x8x4x4xf16>, %w: tensor<16x8x3x3xf16>,
                    %b: tensor<16xf16>, %lhs: tensor<2x?xf16>,
                    %rhs: tensor<2x?xf16>) -> tensor<2x?xf16>
    attributes {rock.arch = "gfx1151", rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %dim = arith.constant 4 : index
  %es = tensor.empty(%dim) : tensor<2x?xf16>
  %s = hip.sub(%ctx) ins(%lhs, %rhs : tensor<2x?xf16>, tensor<2x?xf16>)
      outs(%es : tensor<2x?xf16>) : tensor<2x?xf16>
  %em = tensor.empty(%dim) : tensor<2x?xf16>
  %m = hip.mul(%ctx) ins(%s, %rhs : tensor<2x?xf16>, tensor<2x?xf16>)
      outs(%em : tensor<2x?xf16>) -> tensor<2x?xf16>
  return %m : tensor<2x?xf16>
}

func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %lhs: tensor<2x?xf16>, %rhs: tensor<2x?xf16>)
    -> tensor<2x?xf16> {
  %e0 = tensor.empty() : tensor<1x16x4x4xf16>
  %kept = hip.rocmlir(%ctx) @rocMlir0
      ins(%in, %w, %b : tensor<1x8x4x4xf16>, tensor<16x8x3x3xf16>,
                        tensor<16xf16>)
      outs(%e0 : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  %dim = arith.constant 4 : index
  %e1 = tensor.empty(%dim) : tensor<2x?xf16>
  %fell = hip.rocmlir(%ctx) @rocMlir1
      ins(%in, %w, %b, %lhs, %rhs : tensor<1x8x4x4xf16>, tensor<16x8x3x3xf16>,
                                    tensor<16xf16>, tensor<2x?xf16>,
                                    tensor<2x?xf16>)
      outs(%e1 : tensor<2x?xf16>) : tensor<2x?xf16>
  return %fell : tensor<2x?xf16>
}
)mlir";

} // namespace

int main() {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  hip::compiler::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  mlir::hip::registerRocMlirDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &context);
  if (!module) {
    llvm::errs() << "failed to parse the example module\n";
    return 1;
  }

  mlir::hip::RocMlirEmbedOptions options;
  options.arch = "gfx1151";
  options.compileOne = [](mlir::ModuleOp, llvm::StringRef name,
                          mlir::hip::CompiledKernel &out) {
    if (name != "rocMlir0")
      return false;
    out.binary = "fake-elf";
    out.gridSize = 4;
    out.blockSize = 64;
    return true;
  };

  if (mlir::failed(
          mlir::hip::compileAndEmbedRocMlirKernels(*module, options))) {
    llvm::errs() << "compileAndEmbedRocMlirKernels failed\n";
    module->print(llvm::errs());
    return 1;
  }

  int dispatches = 0;
  int convs = 0;
  int subs = 0;
  int muls = 0;
  bool keptBinary = false;
  module->walk([&](mlir::Operation *op) {
    if (auto dispatch = llvm::dyn_cast<mlir::hip::RocMlirOp>(op)) {
      ++dispatches;
      keptBinary = dispatch.getKernel() == "rocMlir0" &&
                   dispatch.getKernelBinary().size() == 8 &&
                   dispatch.getGridSize() == 4 && dispatch.getBlockSize() == 64;
    }
    if (llvm::isa<mlir::hip::ConvOp>(op))
      ++convs;
    if (llvm::isa<mlir::hip::SubOp>(op))
      ++subs;
    if (llvm::isa<mlir::hip::MulOp>(op))
      ++muls;
  });

  check(dispatches == 1, "only rocMlir0 remains a rocMLIR dispatch");
  check(keptBinary, "rocMlir0 carries the stub binary and launch geometry");
  check(convs == 1, "rocMlir1's conv is back as hip.conv");
  check(subs == 1, "rocMlir1's sub is back as hip.sub");
  check(muls == 1, "rocMlir1's mul is back as hip.mul");
  check(!module->lookupSymbol("rocMlir0") && !module->lookupSymbol("rocMlir1"),
        "outlined kernel funcs are gone from the module");

  if (g_failures) {
    llvm::errs() << g_failures << " check(s) failed\n";
    module->print(llvm::errs());
    return 1;
  }
  llvm::outs() << "rocmlir tosa fallback example passed\n";
  return 0;
}
