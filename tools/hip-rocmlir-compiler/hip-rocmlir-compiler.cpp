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
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "CrashHandler.h"

#if HIP_ROCMLIR_AUTOTUNE
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

struct AutotuneOptions {
  bool enabled = false;
  bool verbose = false;
  mlir::rock::TuningParamSetKind kind = mlir::rock::TuningParamSetKind::Quick;
  unsigned warmupRuns = 5;
  unsigned measuredRuns = 20;
};

#if HIP_ROCMLIR_AUTOTUNE
static bool reportHipError(hipError_t status, llvm::StringRef operation) {
  if (status == hipSuccess)
    return true;
  llvm::errs() << "error: " << operation
               << " failed: " << hipGetErrorString(status) << "\n";
  return false;
}

static bool getBufferSize(mlir::Type type, size_t &bytes) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return false;

  mlir::Type elementType = shaped.getElementType();
  uint64_t elementBits =
      elementType.isIndex() ? 64 : elementType.getIntOrFloatBitWidth();
  int64_t elements = shaped.getNumElements();
  if (elements <= 0 || elementBits == 0)
    return false;

  uint64_t numElements = static_cast<uint64_t>(elements);
  if (numElements > std::numeric_limits<uint64_t>::max() / elementBits)
    return false;
  uint64_t totalBits = numElements * elementBits;
  uint64_t totalBytes = totalBits / 8 + (totalBits % 8 != 0);
  if (totalBytes > std::numeric_limits<size_t>::max())
    return false;
  bytes = static_cast<size_t>(totalBytes);
  return true;
}

class AutotuneBuffers {
public:
  AutotuneBuffers() = default;
  AutotuneBuffers(const AutotuneBuffers &) = delete;
  AutotuneBuffers &operator=(const AutotuneBuffers &) = delete;

  ~AutotuneBuffers() {
    for (void *buffer : deviceBuffers)
      if (buffer)
        (void)hipFree(buffer);
    if (stream)
      (void)hipStreamDestroy(stream);
  }

  bool initialize(mlir::ModuleOp module) {
    auto func = *module.getOps<mlir::func::FuncOp>().begin();
    llvm::SmallVector<mlir::Type> kernelArgTypes(func.getArgumentTypes());
    llvm::append_range(kernelArgTypes, func.getResultTypes());

    if (kernelArgTypes.empty()) {
      llvm::errs() << "error: autotune kernel has no buffer arguments\n";
      return false;
    }
    if (!reportHipError(hipStreamCreate(&stream), "hipStreamCreate"))
      return false;

    for (mlir::Type type : kernelArgTypes) {
      size_t bytes = 0;
      if (!getBufferSize(type, bytes)) {
        llvm::errs()
            << "error: autotune requires statically-shaped tensor/memref "
               "kernel arguments; unsupported type: "
            << type << "\n";
        return false;
      }
      void *buffer = nullptr;
      if (!reportHipError(hipMalloc(&buffer, bytes), "hipMalloc"))
        return false;
      deviceBuffers.push_back(buffer);
      if (!reportHipError(hipMemsetAsync(buffer, 0, bytes, stream),
                          "hipMemsetAsync"))
        return false;
    }
    return reportHipError(hipStreamSynchronize(stream),
                          "hipStreamSynchronize(buffer initialization)");
  }

  hipStream_t getStream() const { return stream; }
  std::vector<void *> &getDeviceBuffers() { return deviceBuffers; }

private:
  hipStream_t stream = nullptr;
  std::vector<void *> deviceBuffers;
};

static bool launchKernel(hipFunction_t function,
                         const mlir::hip::CompiledKernel &kernel,
                         AutotuneBuffers &buffers) {
  std::vector<void *> &deviceBuffers = buffers.getDeviceBuffers();
  size_t kernargSize = deviceBuffers.size() * sizeof(void *);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, deviceBuffers.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargSize,
                    HIP_LAUNCH_PARAM_END};
  (void)hipGetLastError();
  hipError_t status =
      hipModuleLaunchKernel(function, static_cast<unsigned>(kernel.gridSize), 1,
                            1, static_cast<unsigned>(kernel.blockSize), 1, 1, 0,
                            buffers.getStream(), nullptr, config);
  return reportHipError(status, "hipModuleLaunchKernel") &&
         reportHipError(hipGetLastError(), "kernel launch");
}

static bool benchmarkKernel(const mlir::hip::CompiledKernel &kernel,
                            llvm::StringRef kernelName,
                            const AutotuneOptions &options,
                            AutotuneBuffers &buffers, double &milliseconds) {
  hipModule_t hipModule = nullptr;
  if (!reportHipError(hipModuleLoadData(&hipModule, kernel.binary.data()),
                      "hipModuleLoadData"))
    return false;

  hipFunction_t function = nullptr;
  std::string kernelNameStorage = kernelName.str();
  hipError_t lookupStatus =
      hipModuleGetFunction(&function, hipModule, kernelNameStorage.c_str());
  if (!reportHipError(lookupStatus, "hipModuleGetFunction")) {
    (void)hipModuleUnload(hipModule);
    return false;
  }

  bool ok = true;
  for (unsigned i = 0; ok && i < options.warmupRuns; ++i)
    ok = launchKernel(function, kernel, buffers);
  if (ok)
    ok = reportHipError(hipStreamSynchronize(buffers.getStream()),
                        "hipStreamSynchronize(warmup)");

  std::vector<hipEvent_t> starts(options.measuredRuns, nullptr);
  std::vector<hipEvent_t> stops(options.measuredRuns, nullptr);
  for (unsigned i = 0; ok && i < options.measuredRuns; ++i) {
    ok = reportHipError(hipEventCreate(&starts[i]), "hipEventCreate(start)") &&
         reportHipError(hipEventCreate(&stops[i]), "hipEventCreate(stop)");
  }
  for (unsigned i = 0; ok && i < options.measuredRuns; ++i) {
    ok = reportHipError(hipEventRecord(starts[i], buffers.getStream()),
                        "hipEventRecord(start)") &&
         launchKernel(function, kernel, buffers) &&
         reportHipError(hipEventRecord(stops[i], buffers.getStream()),
                        "hipEventRecord(stop)");
  }
  if (ok)
    ok = reportHipError(hipStreamSynchronize(buffers.getStream()),
                        "hipStreamSynchronize(benchmark)");

  std::vector<float> samples;
  for (unsigned i = 0; ok && i < options.measuredRuns; ++i) {
    float elapsed = 0.0f;
    ok = reportHipError(hipEventElapsedTime(&elapsed, starts[i], stops[i]),
                        "hipEventElapsedTime");
    if (ok)
      samples.push_back(elapsed);
  }
  for (hipEvent_t event : starts)
    if (event)
      (void)hipEventDestroy(event);
  for (hipEvent_t event : stops)
    if (event)
      (void)hipEventDestroy(event);
  (void)hipModuleUnload(hipModule);

  if (!ok || samples.empty())
    return false;
  std::sort(samples.begin(), samples.end());
  size_t trim = samples.size() / 4;
  auto first = samples.begin() + trim;
  auto last = samples.end() - trim;
  milliseconds = std::accumulate(first, last, 0.0) / std::distance(first, last);
  return true;
}

static bool autotuneKernel(mlir::ModuleOp module, llvm::StringRef arch,
                           llvm::StringRef kernelName,
                           const AutotuneOptions &options,
                           mlir::hip::CompiledKernel &winner) {
  hipDeviceProp_t properties{};
  int device = 0;
  if (!reportHipError(hipGetDevice(&device), "hipGetDevice") ||
      !reportHipError(hipGetDeviceProperties(&properties, device),
                      "hipGetDeviceProperties"))
    return false;
  llvm::StringRef requestedArch = arch.split(':').first;
  llvm::StringRef deviceArch(properties.gcnArchName);
  deviceArch = deviceArch.split(':').first;
  if (requestedArch != deviceArch)
    llvm::errs() << "warning: autotuning for " << requestedArch << " on device "
                 << deviceArch << "; compiled candidates may not load\n";

  std::unique_ptr<mlir::rock::TuningParamSet> space(
      mlir::rock::createTunableParamSpace(module, options.kind));
  if (!space || space->tuningRange.empty()) {
    llvm::errs() << "error: autotune perfConfig search space is empty\n";
    return false;
  }

  AutotuneBuffers buffers;
  if (!buffers.initialize(module))
    return false;

  double bestMilliseconds = std::numeric_limits<double>::infinity();
  std::string bestConfig;
  unsigned compiled = 0;
  unsigned benchmarked = 0;
  if (options.verbose)
    llvm::errs() << "[hip-rocmlir-compiler] autotuning kernel '" << kernelName
                 << "' across " << space->tuningRange.size()
                 << " perfConfigs\n";

  for (auto [index, tuningAttr] : llvm::enumerate(space->tuningRange)) {
    llvm::SmallString<1024> perfConfig;
    tuningAttr.getPerfConfigStr(perfConfig);
    mlir::OwningOpRef<mlir::ModuleOp> candidate = module.clone();
    mlir::ModuleOp candidateModule = *candidate;
    if (!mlir::rock::tuningSetStr(candidateModule, perfConfig))
      continue;

    mlir::hip::CompiledKernel compiledKernel;
    if (!mlir::hip::compileRocMlirBackend(candidateModule, arch, perfConfig,
                                          compiledKernel)) {
      if (options.verbose)
        llvm::errs() << "[hip-rocmlir-compiler] autotune " << (index + 1) << "/"
                     << space->tuningRange.size() << ": compile failed\n";
      continue;
    }
    ++compiled;

    double elapsed = 0.0;
    if (!benchmarkKernel(compiledKernel, kernelName, options, buffers,
                         elapsed)) {
      if (options.verbose)
        llvm::errs() << "[hip-rocmlir-compiler] autotune " << (index + 1) << "/"
                     << space->tuningRange.size() << ": benchmark failed\n";
      continue;
    }
    ++benchmarked;
    if (options.verbose)
      llvm::errs() << "[hip-rocmlir-compiler] autotune " << (index + 1) << "/"
                   << space->tuningRange.size() << ": " << elapsed << " ms  "
                   << perfConfig << "\n";
    if (elapsed < bestMilliseconds) {
      bestMilliseconds = elapsed;
      bestConfig = perfConfig.str().str();
      winner = std::move(compiledKernel);
    }
  }

  if (bestConfig.empty()) {
    llvm::errs() << "error: autotune found no runnable perfConfig (compiled "
                 << compiled << ", benchmarked " << benchmarked << ")\n";
    return false;
  }
  if (options.verbose)
    llvm::errs() << "[hip-rocmlir-compiler] autotune winner for '" << kernelName
                 << "': " << bestMilliseconds << " ms  " << bestConfig << "\n";
  return true;
}
#endif

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
  AutotuneOptions autotune;
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
      autotune.enabled = true;
    } else if (llvm::StringRef(arg).starts_with("--autotune=")) {
      autotune.enabled = true;
      llvm::StringRef kind = llvm::StringRef(arg).drop_front(11);
      if (kind == "quick")
        autotune.kind = mlir::rock::TuningParamSetKind::Quick;
      else if (kind == "full")
        autotune.kind = mlir::rock::TuningParamSetKind::Full;
      else if (kind == "exhaustive")
        autotune.kind = mlir::rock::TuningParamSetKind::Exhaustive;
      else {
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
        << "                       Benchmark the tuning space and embed the "
           "fastest\n"
        << "                       GPU candidate (default space: quick).\n"
        << "  --autotune-warmup <n>\n"
        << "                       Warmup launches per candidate (default: "
           "5).\n"
        << "  --autotune-runs <n> Timed launches per candidate (default: 20).\n"
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
        << mlir::hip::resolveRocMlirArch() << ").\n";
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

  // Stage 2: on a CLONE, run the hip->tosa conversion (front of
  // buildRocMlirPipeline, minus its terminal hipEpAddHighLevelPipeline -- that
  // runs in the .so), then compile the `rock.kernel` funcs through the
  // in-tree rocMLIR pipeline. The clone is discarded; we only want the
  // compiled binary + launch geometry back.
  mlir::OwningOpRef<mlir::ModuleOp> tosaModule =
      mlir::hip::buildRocMlirTosaClone(*module);
  if (!tosaModule) {
    llvm::errs() << "error: hip->tosa conversion failed\n";
    return 1;
  }

  // Dumped before the non-kernel funcs are dropped below, so the dump shows the
  // whole module rather than just the kernels that cross into the .so. Note
  // this is the *canonicalized* TOSA: `tpm` ran the canonicalizer above, so
  // dead ops that `hip-mlir-opt --convert-hip-to-tosa` alone would print (the
  // orphaned `tensor.empty` feeding a dropped DPS `outs`, for one) are gone.
  if (!dumpTosaPath.empty())
    dumpModule(*tosaModule, "tosa", dumpTosaPath);

  const std::string arch = mlir::hip::resolveRocMlirArch();

  // --dump-high-level: write the rock MLIR (text) and stop, so nothing is
  // embedded. A single kernel writes <output> verbatim so existing tuning
  // scripts keep working; several get one file each because each dump
  // contains one tunable kernel.
  if (dumpHighLevel) {
    llvm::SmallVector<std::string> kernelNames =
        mlir::hip::collectRocMlirKernelNames(*tosaModule);
    for (const std::string &name : kernelNames) {
      mlir::OwningOpRef<mlir::ModuleOp> single = tosaModule->clone();
      for (auto func :
           llvm::make_early_inc_range(single->getOps<mlir::func::FuncOp>()))
        if (func.getSymName() != name)
          func.erase();

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
  // --autotune replaces the default "first perfConfig" choice with a
  // benchmarked winner. It needs a real HIP device, so it stays in the tool.
#if HIP_ROCMLIR_AUTOTUNE
  auto autotuneOne = [&](mlir::ModuleOp single, llvm::StringRef name,
                         mlir::hip::CompiledKernel &out) {
    return autotuneKernel(single, arch, name, autotune, out);
  };
  if (autotune.enabled)
    embedOpts.compileOne = autotuneOne;
#else
  if (autotune.enabled) {
    llvm::errs() << "error: autotune requires a real HIP build\n";
    return 1;
  }
#endif
  if (mlir::failed(mlir::hip::compileAndEmbedRocMlirKernels(
          *module, *tosaModule, embedOpts)))
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
  mlir::registerLLVMDialectTranslation(context);
  {
    mlir::hip::HipToLLVMPipelineOptions llvmOpts;
    mlir::PassManager llvmPm(module->getContext());
    mlir::hip::buildHipToLLVMPipeline(llvmPm, llvmOpts);
    if (mlir::failed(llvmPm.run(*module))) {
      llvm::errs() << "error: HIP-to-LLVM lowering failed\n";
      return 1;
    }
  }

  hipdnn::LLVMBackend backend;
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      backend.translateMLIRtoLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "error: failed to translate MLIR to LLVM IR\n";
    return 1;
  }
  backend.optimizeLLVMIR(llvmModule.get(), /*optLevel=*/3);
  if (!backend.emitLlvmIr(llvmModule.get(), outputPath)) {
    llvm::errs() << "error: failed to emit LLVM bitcode to '" << outputPath
                 << "'\n";
    return 1;
  }

  llvm::errs() << "[hip-rocmlir-compiler] wrote LLVM bitcode to " << outputPath
               << "\n";
  return 0;
}
