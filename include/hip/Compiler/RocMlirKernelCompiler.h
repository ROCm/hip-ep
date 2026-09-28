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
// Only built when ENABLE_ROCMLIRTRITON is on, which is also the macro
// consumers gate on.

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

// Compile one single-kernel module, already lowered to rock form by
// runRocMlirHighLevelPipeline. Lets a caller substitute its own strategy --
// hip-rocmlir-compiler passes an autotuning implementation.
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

// hip -> tosa on a clone of the whole `module`, non-kernel funcs included.
// Only for debugging (hip-rocmlir-compiler's --dump-tosa): it deep-copies the
// entire model, which on a large graph is expensive enough that the compile
// path deliberately avoids it. Returns null on failure.
OwningOpRef<ModuleOp> buildRocMlirTosaClone(ModuleOp module);

// Move every `rock.kernel` func out of `module` into one fresh module and
// lower them hip -> tosa in a single pass-manager run. The funcs are unlinked
// from `module`, not copied -- they are dead there once their binaries are
// embedded -- so this costs nothing beyond the kernels themselves. Outlined
// kernels are IsolatedFromAbove and carry `rock.arch` on the func, so nothing
// from the enclosing module needs to come with them. \p arch, when non-empty,
// replaces the fixed `rock.arch` fuse-rocmlir wrote, so the tuning passes see
// the chip the backend is targeting. Returns an empty module when there are
// no kernels, null on failure.
OwningOpRef<ModuleOp> takeRocMlirKernelsAsTosaModule(ModuleOp module,
                                                     StringRef arch = {});

// Split one func back out of the module returned above into its own
// single-func module, which is the shape the rock pipelines want. Another
// re-parent: no pass runs, nothing copied. Returns null if not found.
OwningOpRef<ModuleOp> takeRocMlirKernelModule(ModuleOp kernelsModule,
                                              StringRef kernelName);

// tosa -> rock over a module holding exactly one `rock.kernel` func.
// compileAndEmbedRocMlirKernels runs this before it hands a kernel to
// `compileOne`, so a callback never has to.
bool runRocMlirHighLevelPipeline(ModuleOp kernelModule);

// Pick the first perfConfig out of the tuning search space and compile with
// it. Expects `kernelModule` to already be in rock form. This is what a caller
// gets when it supplies no `compileOne`.
bool compileRocMlirDefaultPerfConfig(ModuleOp kernelModule, StringRef arch,
                                     CompiledKernel &out);

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

// Compile every `rock.kernel` func in `module` and stamp the binary and launch
// geometry onto the matching `hip.rocmlir` dispatches. On return the kernel
// funcs are gone from `module`: each one is consumed in place of being cloned
// and then erased.
//
// A module with no kernels succeeds and changes nothing -- fuse-rocmlir
// declining every anchor is the ordinary library-path outcome, not an error.
LogicalResult compileAndEmbedRocMlirKernels(ModuleOp module,
                                            const RocMlirEmbedOptions &options);

} // namespace hip
} // namespace mlir

#endif // HIP_COMPILER_ROCMLIRKERNELCOMPILER_H
