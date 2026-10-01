/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip-rocmlir-compiler: drive an ONNX-dialect module (or a module already in
// the HIP dialect) through the ONNX->HIP head passes and the rocmlir hip->tosa
// conversion, then run rocMLIR's high-level + backend pipelines on the fused
// GEMM to produce a GPU binary. HIP-dialect input skips the ONNX->HIP head
// passes and enters at fuse-rocmlir.
//
// rocMLIR (rocmlirTriton) is now built in-tree against the SAME LLVM/MLIR as
// this executable (ENABLE_ROCMLIRTRITON), so its dialects/types share one
// TypeID system with ours. That removes the reason the old design serialized
// IR across a dlopen'd librockCompiler.so boundary (two incompatible LLVM
// copies): we register the rocMLIR dialects into our own MLIRContext and call
// the `mlir::rock::*` pipeline builders and tuning API directly on our own
// ModuleOp -- no serialization, no dlopen, no C-API.
//
// The high-level pipeline only rewrites functions marked `rock.kernel` (set by
// fuse-rocmlir) and asserts on non-kernel funcs, so we still hand it a module
// containing ONLY the `rock.kernel` funcs (main_graph and its hip.* ops are
// dropped from the clone we compile).

#include "hip/Compiler/RocMlirAutotune.h"
#include "hip/Compiler/RocMlirKernelCompiler.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/InitAllPasses.h"
#include "hip/Support/DiskFileSystem.h"
#include "hip/Target/LLVM/LLVMBackend.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

// rocMLIR (rocmlirTriton) C++ API -- linked in-tree, same LLVM/MLIR as us.
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "mlir/InitRocMLIRDialects.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include "CrashHandler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// Same convention as MIGraphX `enabled(MIGRAPHX_SKIP_BENCHMARKING)`: unset or
// "0" is off; any other value skips GPU benchmarking and affixes the first
// enumerated perfConfig.
static bool skipBenchmarking() {
  const char *env = std::getenv("HIP_ROCMLIR_SKIP_BENCHMARKING");
  return env && std::strcmp(env, "0") != 0;
}

using mlir::hip::CompiledKernel;

// Wall times accumulated on the main thread. perfConfig compiles for the whole
// model share one pool, so that figure is the pool wait, not the sum of CPU
// times. Benchmarks stay serial. HIP-to-LLVM is the final lowering plus
// bitcode emit.
struct CompilePhaseTimes {
  double perfConfigCompileMs = 0;
  double benchmarkMs = 0;
  double hipToLlvmMs = 0;
};
static CompilePhaseTimes compilePhaseTimes;

static double millisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

static void printPhaseTimes() {
  llvm::errs() << "[hip-rocmlir-compiler] perfConfig compile wall: "
               << llvm::format("%.1f", compilePhaseTimes.perfConfigCompileMs)
               << " ms\n";
  llvm::errs() << "[hip-rocmlir-compiler] benchmark wall: "
               << llvm::format("%.1f", compilePhaseTimes.benchmarkMs)
               << " ms\n";
  llvm::errs() << "[hip-rocmlir-compiler] HIP-to-LLVM: "
               << llvm::format("%.1f", compilePhaseTimes.hipToLlvmMs)
               << " ms\n";
}

// Write a module as MLIR text for --dump-hip / --dump-tosa. Those dumps are
// diagnostics on the way to `-o`, so a failure to write one is reported but
// does not fail the compile.
static void dumpModule(mlir::ModuleOp mod, llvm::StringRef label,
                       llvm::StringRef path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "warning: cannot open '" << path << "' to dump " << label
                 << " MLIR: " << ec.message() << "\n";
    return;
  }
  mod->print(os);
  os << "\n";
  // raw_fd_ostream defers write and flush failures (a full disk, say) rather
  // than reporting them at open time, so check before claiming success --
  // otherwise a truncated dump reads as a good one.
  os.flush();
  if (os.has_error()) {
    llvm::errs() << "warning: failed writing " << label << " MLIR to " << path
                 << ": " << os.error().message() << "\n";
    os.clear_error();
    return;
  }
  llvm::errs() << "[hip-rocmlir-compiler] wrote " << label << " MLIR to "
               << path << "\n";
}

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-rocmlir-compiler");

  std::string inputFilename;
  std::string outputPath;
  std::string dumpHipPath;
  std::string dumpTosaPath;
  bool dumpHighLevel = false;
  bool autotuneFlagSeen = false;
  // On unless HIP_ROCMLIR_SKIP_BENCHMARKING says otherwise: benchmarking is
  // the reason this tool exists, unlike the EP, which opts in.
  bool autotuneEnabled = true;
  mlir::hip::AutotuneOptions autotune;
  autotune.log = &llvm::errs();
  autotune.logPrefix = "[hip-rocmlir-compiler]";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      outputPath = argv[++i];
    } else if (arg == "--dump-hip" && i + 1 < argc) {
      dumpHipPath = argv[++i];
    } else if (arg == "--dump-tosa" && i + 1 < argc) {
      dumpTosaPath = argv[++i];
    } else if (arg == "--dump-high-level") {
      dumpHighLevel = true;
    } else if (arg == "--verbose") {
      autotune.verbose = true;
    } else if (arg == "--autotune") {
      autotuneFlagSeen = true;
      autotuneEnabled = true;
      autotune.space = mlir::hip::AutotuneSpace::Quick;
    } else if (llvm::StringRef(arg).starts_with("--autotune=")) {
      autotuneFlagSeen = true;
      autotuneEnabled = true;
      llvm::StringRef kind = llvm::StringRef(arg).drop_front(11);
      if (!mlir::hip::parseAutotuneSpace(kind, autotune.space)) {
        llvm::errs() << "error: unknown autotune space '" << kind
                     << "' (expected quick, full, or exhaustive)\n";
        return 1;
      }
    } else if (arg == "--autotune-warmup" && i + 1 < argc) {
      if (llvm::StringRef(argv[++i]).getAsInteger(10, autotune.warmupRuns)) {
        llvm::errs() << "error: invalid --autotune-warmup value\n";
        return 1;
      }
    } else if (arg == "--autotune-runs" && i + 1 < argc) {
      if (llvm::StringRef(argv[++i]).getAsInteger(10, autotune.measuredRuns) ||
          autotune.measuredRuns == 0) {
        llvm::errs() << "error: --autotune-runs must be greater than zero\n";
        return 1;
      }
    } else if (argv[i][0] != '-') {
      inputFilename = argv[i];
    }
  }
  if (skipBenchmarking())
    autotuneEnabled = false;
#if !HIP_ROCMLIR_AUTOTUNE
  if (autotuneFlagSeen && autotuneEnabled) {
    llvm::errs() << "error: autotune requires a real HIP build\n";
    return 1;
  }
  autotuneEnabled = false;
#endif
  if (inputFilename.empty() || outputPath.empty()) {
    llvm::errs()
        << "Usage: " << argv[0]
        << " <input.onnx.mlir|input.hip.mlir> -o <output> [options]\n"
        << "  Accepts either an ONNX-dialect module (runs the ONNX->HIP head\n"
        << "  passes) or a module already in the HIP dialect (skips them).\n"
        << "  Then compiles the fused GEMM via the rocMLIR pipeline (linked\n"
        << "  in-tree), embeds the GPU binary into hip.rocmlir, runs the\n"
        << "  ONNX->HIP tail + HIP->LLVM lowering, and emits LLVM bitcode to\n"
        << "  <output>.\n"
        << "\n"
        << "Options:\n"
        << "  --autotune[=quick|full|exhaustive]\n"
        << "                       Override the default quick autotune space "
           "and\n"
        << "                       embed the fastest GPU candidate.\n"
        << "  --autotune-warmup <n>\n"
        << "                       Warmup launches per candidate (default: "
           "20).\n"
        << "  --autotune-runs <n> Timed launches per candidate (default: "
           "100).\n"
        << "                       Lower values get noisy: at 5/20 the same "
           "kernel\n"
        << "                       measured 4.2e-02 to 7.5e-02 ms, wider than "
           "the\n"
        << "                       differences being searched for.\n"
        << "  --verbose            Print per-config compile/benchmark lines "
           "and the\n"
        << "                       selected perfConfig.\n"
        << "  --dump-hip <file>    Write the hip MLIR after the ONNX->HIP head "
           "passes\n"
        << "                       and fuse-rocmlir, then keep going.\n"
        << "  --dump-tosa <file>   Write the TOSA MLIR handed to the rocMLIR "
           "pipeline,\n"
        << "                       then keep going.\n"
        << "  --dump-high-level    Stop after the rocMLIR high-level pipeline "
           "and\n"
        << "                       write the rock MLIR (text) to <output> "
           "instead of\n"
        << "                       the compiled bitcode.\n"
        << "  Set ROCK_ARCH to override the target GPU arch (default: "
        << mlir::hip::resolveRocMlirArch() << ").\n"
        << "  Set HIP_ROCMLIR_SKIP_BENCHMARKING=1 to skip autotune and affix "
           "the\n"
        << "  first enumerated perfConfig (same role as "
           "MIGRAPHX_SKIP_BENCHMARKING).\n"
        << "  Set HIP_ROCMLIR_COMPILE_JOBS to how many perfConfig compiles "
           "run\n"
        << "  at once (default: one per hardware thread). Benchmarks stay\n"
        << "  serial.\n"
        << "  Every search config in the model is compiled in one pool, then\n"
        << "  benchmarked. Repeated rocMLIR problems compile only the winning\n"
        << "  perfConfig, together in a second pool.\n";
    return 1;
  }

  auto bufOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!bufOrErr) {
    llvm::errs() << "error: cannot open '" << inputFilename
                 << "': " << bufOrErr.getError().message() << "\n";
    return 1;
  }

  // Our own context, loaded with the same dialects the EP uses (adds tosa
  // lazily as a dependent dialect of convert-hip-to-tosa). rocMLIR is built
  // against the same LLVM/MLIR (ENABLE_ROCMLIRTRITON), so we additionally
  // register its dialects (rock, migraphx, triton, ...) into this one context
  // and run the rock pipelines directly -- no serialization boundary.
  mlir::MLIRContext context;
  hip::compiler::loadAllDialects(context);
  {
    mlir::DialectRegistry rockRegistry;
    mlir::registerRocMLIRDialects(rockRegistry);
    context.appendDialectRegistry(rockRegistry);
    context.loadAllAvailableDialects();
  }
  hip::compiler::registerAllPasses();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*bufOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR input\n";
    return 1;
  }

  // Detect the input dialect: an `onnx`-dialect module still needs the
  // ONNX->HIP head passes, whereas a module already lowered to the `hip`
  // dialect (e.g. a `--dump-hip` companion or hand-written hip.mlir) skips
  // straight to fuse-rocmlir. Any op in the `onnx` namespace marks the input as
  // ONNX; otherwise it is treated as hip.
  bool isOnnxInput = false;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() == "onnx") {
      isOnnxInput = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  // Stage 1: (ONNX input only) ONNX->HIP head passes, then -- for both input
  // dialects -- fuse-rocmlir + duplicate-function-elimination. The ONNX head
  // mirrors buildOnnxToHipPipeline (simplify-onnx, hip-add-context-arg, loop/if
  // outline, infer-loop-body-shapes, convert-onnx-to-hip; plain path, no hipdnn
  // handle). fuse-rocmlir outlines the fused GEMM into a `rock.kernel` func and
  // creates the `hip.rocmlir` dispatch. `module` is LEFT in this hip form: it
  // is the artifact we mutate at the end.
  llvm::errs() << "[hip-rocmlir-compiler] input dialect: "
               << (isOnnxInput ? "onnx" : "hip") << "\n";
  mlir::PassManager pm(module->getContext());
  if (isOnnxInput) {
    pm.addPass(mlir::hip::createSimplifyOnnxPass());
    pm.addPass(mlir::hip::createHipAddContextArgPass());
    pm.addPass(mlir::hip::createOnnxLoopOutlinePass());
    pm.addPass(mlir::hip::createOnnxIfOutlinePass());
    pm.addPass(mlir::hip::createInferLoopBodyShapesPass());
    pm.addPass(mlir::hip::createConvertOnnxToHipPass());
  }
  // rocMLIR has no transposed-convolution anchor, so split conv_transpose into
  // plain convolutions before fuse-rocmlir outlines a kernel around it.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hip::createDecomposeConvTransposePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hip::createFuseROCMlirPass());
  pm.addPass(mlir::func::createDuplicateFunctionEliminationPass());

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "error: "
                 << (isOnnxInput ? "ONNX->HIP + fuse-rocmlir" : "fuse-rocmlir")
                 << " passes failed\n";
    return 1;
  }

  if (!dumpHipPath.empty())
    dumpModule(*module, "hip", dumpHipPath);

  // --dump-tosa shows the hip->tosa conversion applied to the whole module,
  // non-kernel funcs included, which needs a clone of everything. That is
  // debug-only cost: the compile path below converts one kernel at a time and
  // never copies the model. Note this is the *canonicalized* TOSA, so dead ops
  // that `hip-mlir-opt --convert-hip-to-tosa` alone would print (the orphaned
  // `tensor.empty` feeding a dropped DPS `outs`, for one) are gone.
  if (!dumpTosaPath.empty()) {
    mlir::OwningOpRef<mlir::ModuleOp> tosaModule =
        mlir::hip::buildRocMlirTosaClone(*module);
    if (!tosaModule) {
      llvm::errs() << "error: hip->tosa conversion failed\n";
      return 1;
    }
    dumpModule(*tosaModule, "tosa", dumpTosaPath);
  }

  const std::string arch = mlir::hip::resolveRocMlirArch();

  // --dump-high-level: write the rock MLIR (text) and stop, so nothing is
  // embedded. A single kernel writes <output> verbatim so existing tuning
  // scripts keep working; several get one file each because each dump
  // contains one tunable kernel.
  if (dumpHighLevel) {
    llvm::SmallVector<std::string> kernelNames =
        mlir::hip::collectRocMlirKernelNames(*module);
    mlir::OwningOpRef<mlir::ModuleOp> tosaKernels =
        mlir::hip::takeRocMlirKernelsAsTosaModule(*module, arch);
    if (!tosaKernels) {
      llvm::errs() << "error: hip->tosa conversion failed\n";
      return 1;
    }
    for (const std::string &name : kernelNames) {
      mlir::OwningOpRef<mlir::ModuleOp> single =
          mlir::hip::takeRocMlirKernelModule(*tosaKernels, name);
      if (!single) {
        llvm::errs() << "error: no tosa kernel named '" << name << "'\n";
        return 1;
      }

      mlir::hip::CompiledKernel kernel;
      if (!mlir::hip::runRocMlirOnKernel(*single, arch,
                                         /*stopAfterHighLevel=*/true, kernel)) {
        llvm::errs() << "error: rocMLIR high-level pipeline failed for '"
                     << name << "'\n";
        return 1;
      }

      std::string path = kernelNames.size() == 1
                             ? outputPath
                             : outputPath + "." + name + ".mlir";
      std::error_code ec;
      llvm::raw_fd_ostream os(path, ec);
      if (ec) {
        llvm::errs() << "error: cannot open '" << path << "': " << ec.message()
                     << "\n";
        return 1;
      }
      os << kernel.highLevelMlir << "\n";
      llvm::errs() << "[hip-rocmlir-compiler] wrote rock high-level MLIR to "
                   << path << "\n";
    }
    return 0;
  }

  // Stage 3: compile each `rock.kernel` and embed the artifact back into
  // `module`'s `hip.rocmlir` dispatches, leaving a self-contained hip module
  // carrying the GPU binary inline -- what hip-compiler consumes.
  mlir::hip::RocMlirEmbedOptions embedOpts;
  embedOpts.arch = arch;
  embedOpts.log = &llvm::errs();
  embedOpts.logPrefix = "[hip-rocmlir-compiler]";
  // Autotune replaces the default perfConfig choice -- position 0 of the
  // tuning space, which rocMLIR orders by applicability rather than speed --
  // with a benchmarked winner. It lives in LibHipCompiler so the EP can reach
  // it too; see include/hip/Compiler/RocMlirAutotune.h.
  //
  // Declared in this scope because embedOpts only holds function_refs into it.
  std::optional<mlir::hip::RocMlirAutotuner> autotuner;
#if HIP_ROCMLIR_AUTOTUNE
  if (autotuneEnabled) {
    autotuner.emplace(arch, autotune);
    if (!autotuner->isUsable()) {
      llvm::errs() << "error: autotune could not reach a HIP device\n";
      return 1;
    }
    autotuner->installInto(embedOpts);
  }
#else
  if (autotuneEnabled) {
    llvm::errs() << "error: this build has no autotune support (LibHipCompiler "
                    "compiled RocMlirAutotune's stub; needs a non-mock build "
                    "with HIPDNN_EP_LINK_HIP_HOST=ON). Set "
                    "HIP_ROCMLIR_SKIP_BENCHMARKING=1 to compile with the "
                    "default perfConfig instead.\n";
    return 1;
  }
#endif
  bool embedOk =
      mlir::hip::compileAndEmbedRocMlirKernels(*module, embedOpts).succeeded();
  // Whether or not the embed succeeded: the pool and benchmark figures explain
  // where a slow or failed compile spent its time.
  if (autotuner) {
    compilePhaseTimes.perfConfigCompileMs +=
        autotuner->phaseTimes().perfConfigCompileMs;
    compilePhaseTimes.benchmarkMs += autotuner->phaseTimes().benchmarkMs;
  }
  if (!embedOk)
    return 1;

  // Stage 4: run the standard ONNX-to-HIP tail (shape inference, constant
  // externalization, bufferization, output-allocator rewrite, pooling, extern-
  // constant resolution) -- everything hip-compiler runs after OnnxToHip up to,
  // but not including, the HIP-to-LLVM lowering. The `hip.rocmlir` op now
  // carries its kernel inline and bufferizes like any other DPS op.
  //
  // ExternalizeConstants only accepts `memory_address` constant sources when a
  // FileSystem is injected (the production externalization path). Inject a
  // DiskFileSystem writing to the cwd and skip the constant payload: the
  // in-process `memory_address` pointers are not valid to read here, so emit
  // metadata only (matching the EP live-compile path).
  {
    mlir::hip::DiskFileSystem fs(".");
    mlir::hip::OnnxToHipPipelineOptions tailOpts;
    tailOpts.externalizeMinNumElements =
        mlir::hip::kDefaultExternalizeMinNumElements;
    tailOpts.skipConstantData = true;
    mlir::PassManager tailPm(module->getContext());
    mlir::hip::buildOnnxToHipPipelineTail(tailPm, tailOpts, &fs);
    if (mlir::failed(tailPm.run(*module))) {
      llvm::errs() << "error: ONNX-to-HIP tail passes failed\n";
      return 1;
    }
  }

  // Without -o: stop at the bufferized HIP module and print it.
  if (outputPath.empty()) {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  // Stage 5: HIP->LLVM lowering + interface generation, then translate to LLVM
  // IR, optimize, and emit OS-portable bitcode -- the same artifact
  // hip-compiler produces in its default (LLVM_IR) mode.
  const auto hipToLlvmStart = std::chrono::steady_clock::now();
  mlir::registerLLVMDialectTranslation(context);
  {
    mlir::hip::HipToLLVMPipelineOptions llvmOpts;
    mlir::PassManager llvmPm(module->getContext());
    mlir::hip::buildHipToLLVMPipeline(llvmPm, llvmOpts);
    if (mlir::failed(llvmPm.run(*module))) {
      compilePhaseTimes.hipToLlvmMs += millisecondsSince(hipToLlvmStart);
      printPhaseTimes();
      llvm::errs() << "error: HIP-to-LLVM lowering failed\n";
      return 1;
    }
  }

  hipdnn::LLVMBackend backend;
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      backend.translateMLIRtoLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    compilePhaseTimes.hipToLlvmMs += millisecondsSince(hipToLlvmStart);
    printPhaseTimes();
    llvm::errs() << "error: failed to translate MLIR to LLVM IR\n";
    return 1;
  }
  backend.optimizeLLVMIR(llvmModule.get(), /*optLevel=*/3);
  if (!backend.emitLlvmIr(llvmModule.get(), outputPath)) {
    compilePhaseTimes.hipToLlvmMs += millisecondsSince(hipToLlvmStart);
    printPhaseTimes();
    llvm::errs() << "error: failed to emit LLVM bitcode to '" << outputPath
                 << "'\n";
    return 1;
  }
  compilePhaseTimes.hipToLlvmMs += millisecondsSince(hipToLlvmStart);
  printPhaseTimes();

  llvm::errs() << "[hip-rocmlir-compiler] wrote LLVM bitcode to " << outputPath
               << "\n";
  return 0;
}
