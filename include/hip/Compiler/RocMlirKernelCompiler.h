/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The kernel-compilation half of the rocMLIR offload path.
//
// fuse-rocmlir outlines eligible op groups into `rock.kernel` funcs and leaves
// a `hip.rocmlir` dispatch behind for each. A dispatch carries its GPU binary
// and launch geometry inline, so something has to run rocMLIR over the kernels
// and stamp the results back before the module is lowered. That step used to
// live only in hip-rocmlir-compiler's main(), which is why forcing
// fuse-rocmlir into the EP's pipeline produced dispatches with an empty
// kernel_binary that failed at hipModuleLoadData. It lives here so both the
// offline tool and the EP's in-process CompilerDriver run the same code.
//
// Only built when ENABLE_ROCMLIRTRITON is on; consumers gate on
// HIP_EP_HAS_ROCMLIR.

#ifndef HIP_COMPILER_ROCMLIRKERNELCOMPILER_H
#define HIP_COMPILER_ROCMLIRKERNELCOMPILER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace hip {

// One compiled kernel: the ELF blob plus the launch geometry rocMLIR picked.
struct CompiledKernel {
  std::string binary;
  int64_t gridSize = 0;
  int64_t blockSize = 0;
  // Only populated when stopping after the high-level pipeline.
  std::string highLevelMlir;
};

// Compile one single-kernel tosa module. Lets a caller substitute its own
// strategy -- hip-rocmlir-compiler passes an autotuning implementation.
using RocMlirKernelCompileFn =
    llvm::function_ref<bool(ModuleOp, StringRef, CompiledKernel &)>;

struct RocMlirEmbedOptions {
  std::string arch;
  // Per-kernel progress. Null keeps the compile silent, which is what the EP
  // wants: its compiles happen inside ORT session creation.
  llvm::raw_ostream *log = nullptr;
  StringRef logPrefix = "[rocmlir]";
  // Empty means runRocMlirOnKernel.
  RocMlirKernelCompileFn compileOne = RocMlirKernelCompileFn();
};

// ROCK_ARCH wins, else a default the rock backend can parse.
std::string resolveRocMlirArch();

// Register rock/migraphx/triton into `context` alongside the hip dialects, so
// the rock pipelines run on our own ModuleOp with no serialization boundary.
// Safe to call more than once on the same context.
void registerRocMlirDialects(MLIRContext &context);

// hip -> tosa on a clone of `module`: the front of buildRocMlirPipeline. The
// clone exists because rocMLIR's high-level pipeline asserts on non-kernel
// funcs, so `main_graph` and its hip.* ops cannot come along. Returns null on
// failure.
OwningOpRef<ModuleOp> buildRocMlirTosaClone(ModuleOp module);

// High-level + backend pipelines over a module holding exactly one
// `rock.kernel` func. With `stopAfterHighLevel` it stops after tosa -> rock
// and returns the printed rock MLIR in `out.highLevelMlir`.
bool runRocMlirOnKernel(ModuleOp kernelModule, StringRef arch,
                        bool stopAfterHighLevel, CompiledKernel &out);

// Backend only, for a caller that already affixed a perfConfig.
bool compileRocMlirBackend(ModuleOp kernelModule, StringRef arch,
                           StringRef perfConfig, CompiledKernel &out);

// Names of the `rock.kernel` funcs in `module`, in declaration order.
SmallVector<std::string> collectRocMlirKernelNames(ModuleOp module);

// Compile every `rock.kernel` func in `tosaModule` and stamp the binary and
// launch geometry onto the matching `hip.rocmlir` dispatches in `module`, then
// erase the compiled kernel funcs from `module`. `tosaModule` must be the
// result of buildRocMlirTosaClone(module).
//
// A module with no kernels succeeds and changes nothing -- fuse-rocmlir
// declining every anchor is the ordinary library-path outcome, not an error.
LogicalResult compileAndEmbedRocMlirKernels(ModuleOp module,
                                            ModuleOp tosaModule,
                                            const RocMlirEmbedOptions &options);

} // namespace hip
} // namespace mlir

#endif // HIP_COMPILER_ROCMLIRKERNELCOMPILER_H
