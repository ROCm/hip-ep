/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip-rocmlir-compiler: drive an ONNX-dialect module through the ONNX->HIP
// head passes and the rocmlir hip->tosa conversion, then run rocMLIR's
// high-level + backend pipelines on the fused GEMM to produce a GPU binary.
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

#include <cstdint>
#include <cstdlib>
#include <string>

// Resolve the target GPU arch: ROCK_ARCH wins, else a default. The rock
// backend pipeline validates and parses this (triple/chip/features).
static std::string resolveArch() {
  if (const char *env = std::getenv("ROCK_ARCH"))
    if (env[0] != '\0')
      return env;
  return "gfx1151";
}

// Read one `name=value` field out of a serialized perfConfig string of the form
// `prefix:key=value,key=value,...` (see getPerfConfigStr in RockAttrDefs.td).
// Returns `fallback` when the field is absent or unparseable.
static int64_t perfConfigField(llvm::StringRef perf, llvm::StringRef name,
                               int64_t fallback) {
  // Drop the `gemm:`/`attn:` prefix if present so a prefix substring can't be
  // mistaken for a field.
  size_t colon = perf.find(':');
  if (colon != llvm::StringRef::npos)
    perf = perf.drop_front(colon + 1);

  while (!perf.empty()) {
    auto [field, rest] = perf.split(',');
    auto [key, value] = field.split('=');
    if (key.trim() == name) {
      int64_t parsed = 0;
      if (!value.trim().getAsInteger(10, parsed))
        return parsed;
      return fallback;
    }
    perf = rest;
  }
  return fallback;
}

// Drive the rocMLIR flow directly on the tosa `rock.kernel` module (in our own
// context -- rocMLIR dialects are registered alongside ours):
//   1. high-level pipeline (tosa -> rock.gemm)
//   2. affix a perfConfig to the gemm op (the `perf_config` string attribute):
//      either the caller-supplied `userPerfConfig`, or -- when empty -- the
//      first entry enumerated from the tuning search space.
//   3. backend pipeline (rock -> LLVM / binary)
// When `stopAfterHighLevel` is set, stops after step 1 and returns the printed
// rock MLIR in `out.highLevelMlir` (steps 2-3 are skipped). Otherwise returns
// the compiled GPU binary in `binary` and the kernel launch geometry in
// `gridSize`/`blockSize`.
struct CompiledKernel {
  std::string binary;
  int64_t gridSize = 0;
  int64_t blockSize = 0;
  std::string highLevelMlir;
};

// Map a parsed perfConfig string + arch onto rock's Triton/backend option
// structs. Mirrors the option wiring the fork's C-API entrypoint
// (hipEpAddBackendPipeline) did: the Triton/backend knobs are the tuning
// fields of the chosen perfConfig, read straight out of the string rather than
// re-defaulted (perfConfig field `numWaves` is the backend's `numWarps`); a
// missing field falls back to the rock default and kKnobDefault (-1) leaves a
// bool knob at its arch default.
static bool buildBackendPipelineFor(mlir::OpPassManager &pm,
                                    llvm::StringRef arch,
                                    llvm::StringRef perfConfig) {
  mlir::RocmDeviceName devName;
  if (arch.empty() || mlir::failed(devName.parse(arch))) {
    llvm::errs() << "error: invalid architecture: " << arch << "\n";
    return false;
  }

  mlir::rock::KernelOptions kOpts;
  mlir::rock::buildKernelPipeline(pm, kOpts);

  mlir::rock::TritonOptions tOpts;
  tOpts.arch = devName.getChip().str();
  tOpts.numWarps = static_cast<int>(perfConfigField(perfConfig, "numWaves", 4));
  tOpts.numCTAs = static_cast<int>(perfConfigField(perfConfig, "numCTAs", 1));
  tOpts.numStages =
      static_cast<int>(perfConfigField(perfConfig, "numStages", 1));
  tOpts.matrixInstrNonkdim =
      static_cast<int>(perfConfigField(perfConfig, "matrixInstrNonkdim", 0));
  tOpts.kpack = static_cast<int>(perfConfigField(perfConfig, "kpack", 1));
  tOpts.useAsyncCopy = perfConfigField(perfConfig, "useAsyncCopy", -1);
  tOpts.useBlockPingpong = perfConfigField(perfConfig, "useBlockPingpong", -1);
  tOpts.useInThreadTranspose =
      perfConfigField(perfConfig, "useInThreadTranspose", -1);
  tOpts.useBufferOps = perfConfigField(perfConfig, "useBufferOps", -1);
  tOpts.useBufferAtomics = perfConfigField(perfConfig, "useBufferAtomics", -1);
  tOpts.useReductionLayout =
      perfConfigField(perfConfig, "useReductionLayout", -1);
  tOpts.useOptimizeEpilogue =
      perfConfigField(perfConfig, "useOptimizeEpilogue", -1);
  mlir::rock::buildTritonPipeline(pm, tOpts);

  mlir::rock::BackendOptions bOpts;
  bOpts.triple = devName.getTriple().str();
  bOpts.chip = devName.getChip().str();
  bOpts.features = devName.getFeaturesForBackend();
  bOpts.optLevel = 3;
  bOpts.numWarps = tOpts.numWarps;
  bOpts.numCTAs = tOpts.numCTAs;
  bOpts.wavesPerEU =
      static_cast<int>(perfConfigField(perfConfig, "wavesPerEU", 0));
  mlir::rock::buildBackendPipeline(pm, bOpts);
  return true;
}

// Walk the compiled module for the single gpu.binary object and pull out its
// ELF blob plus the {block, grid} launch geometry (mirrors the fork's
// mlirGetBinary / mlirGetKernelAttrs C-API, in C++).
static bool extractCompiledKernel(mlir::ModuleOp mod, CompiledKernel &out) {
  unsigned count = 0;
  mod.walk([&](mlir::gpu::BinaryOp binary) {
    auto object = llvm::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
    llvm::StringRef blob = object.getObject().getValue();
    out.binary.assign(blob.begin(), blob.end());
    for (auto kernel : object.getKernels()) {
      auto block = kernel.getAttr<mlir::IntegerAttr>(
          mlir::rock::BlockSizeAttr::getMnemonic());
      auto grid = kernel.getAttr<mlir::IntegerAttr>(
          mlir::rock::GridSizeAttr::getMnemonic());
      if (!block || !grid)
        continue;
      out.blockSize = block.getInt();
      out.gridSize = grid.getInt();
      ++count;
    }
  });
  return count == 1 && !out.binary.empty();
}

static bool runRocmlir(mlir::ModuleOp module, const std::string &arch,
                       const std::string &userPerfConfig,
                       bool stopAfterHighLevel, CompiledKernel &out) {
  auto fail = [&](const char *msg) {
    llvm::errs() << "error: " << msg << "\n";
    return false;
  };

  // 1. High-level pipeline: tosa -> rock.gemm.
  {
    mlir::PassManager pm(module.getContext());
    pm.setNesting(mlir::PassManager::Nesting::Implicit);
    mlir::rock::buildHighlevelPipeline(pm);
    if (mlir::failed(pm.run(module)))
      return fail("rocMLIR high-level pipeline failed");
  }

  // Stop after the high-level pipeline: capture the rock MLIR text and return.
  if (stopAfterHighLevel) {
    llvm::raw_string_ostream os(out.highLevelMlir);
    module.print(os);
    return true;
  }

  // 2. Affix a perfConfig to the gemm op (as the `perf_config` string attr).
  //    A caller-supplied config is used verbatim; otherwise take the first
  //    entry enumerated from the tuning search space. rock::tuningSetStr stamps
  //    `perf_config` onto the gemm op.
  llvm::SmallString<1024> perfConfig; // ROCMLIR_TUNING_PARAM_STRING_BUFSZ
  if (!userPerfConfig.empty()) {
    perfConfig.assign(userPerfConfig.begin(), userPerfConfig.end());
    llvm::errs() << "[hip-rocmlir-compiler] affixing supplied perfConfig: "
                 << perfConfig << "\n";
    if (!mlir::rock::tuningSetStr(module, perfConfig))
      return fail("failed to affix supplied perfConfig to the gemm op");
  } else {
    mlir::rock::TuningParamSet *space = mlir::rock::createTunableParamSpace(
        module, mlir::rock::TuningParamSetKind::Full);
    unsigned num = space ? space->tuningRange.size() : 0;
    if (num == 0) {
      delete space;
      return fail("perfConfig search space is empty");
    }
    mlir::rock::ParamEntry entry;
    if (!mlir::rock::tuningGetParam(space, /*pos=*/0, &entry)) {
      delete space;
      return fail("failed to read the first perfConfig entry");
    }
    entry.param.getPerfConfigStr(perfConfig);
    delete space;
    llvm::errs() << "[hip-rocmlir-compiler] perfConfig search space size: "
                 << num << "; affixing first entry: " << perfConfig << "\n";
    if (!mlir::rock::tuningSetStr(module, perfConfig))
      return fail("failed to affix perfConfig to the gemm op");
  }

  // 3. Backend pipeline: rock (with the affixed perf_config) -> LLVM / binary.
  {
    mlir::PassManager pm(module.getContext());
    pm.setNesting(mlir::PassManager::Nesting::Implicit);
    if (!buildBackendPipelineFor(pm, arch, perfConfig))
      return fail("failed to build the backend pipeline");
    if (mlir::failed(pm.run(module)))
      return fail("rocMLIR backend pipeline failed");
  }

  // 4. Extract the compiled artifact: the gpu.binary blob + launch geometry.
  if (!extractCompiledKernel(module, out))
    return fail("failed to extract the compiled binary / kernel attributes");
  return true;
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
  // A bare --perf-config applies to every kernel; `<kernel>=<config>` targets
  // one. A config is always `<anchor>:<field>=<value>,...`, so the text before
  // the first '=' contains a ':' exactly when the argument is unkeyed.
  std::string defaultPerfConfig;
  llvm::StringMap<std::string> perfConfigByKernel;
  std::string dumpHipPath;
  std::string dumpTosaPath;
  bool dumpHighLevel = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      outputPath = argv[++i];
    } else if (arg == "--perf-config" && i + 1 < argc) {
      llvm::StringRef value(argv[++i]);
      auto [head, tail] = value.split('=');
      if (!tail.empty() && !head.contains(':'))
        perfConfigByKernel[head] = tail.str();
      else
        defaultPerfConfig = value.str();
    } else if (arg == "--dump-hip" && i + 1 < argc) {
      dumpHipPath = argv[++i];
    } else if (arg == "--dump-tosa" && i + 1 < argc) {
      dumpTosaPath = argv[++i];
    } else if (arg == "--dump-high-level") {
      dumpHighLevel = true;
    } else if (argv[i][0] != '-') {
      inputFilename = argv[i];
    }
  }
  if (inputFilename.empty() || outputPath.empty()) {
    llvm::errs()
        << "Usage: " << argv[0] << " <input.onnx.mlir> -o <output> [options]\n"
        << "  Runs ONNX->HIP head passes, compiles the fused GEMM via\n"
        << "  the rocMLIR pipeline (linked in-tree), embeds the GPU\n"
        << "  binary into hip.rocmlir, runs the ONNX->HIP tail +\n"
        << "  HIP->LLVM lowering, and emits LLVM bitcode to <output>.\n"
        << "\n"
        << "Options:\n"
        << "  --perf-config <str>  Affix this perfConfig to the gemm op "
           "instead of\n"
        << "                       enumerating the tuning space and taking "
           "the first.\n"
        << "                       Use <kernel>=<config> to target one kernel; "
           "repeat\n"
        << "                       the flag to configure several. A bare "
           "config applies\n"
        << "                       to every kernel without one of its own.\n"
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
        << resolveArch() << ").\n";
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

  // Stage 1: ONNX->HIP head passes + fuse-rocmlir. Mirrors the head of
  // buildOnnxToHipPipeline (simplify-onnx, hip-add-context-arg, loop/if
  // outline, infer-loop-body-shapes, convert-onnx-to-hip; plain path, no hipdnn
  // handle) followed by fuse-rocmlir + duplicate-function-elimination, which
  // outline the fused GEMM into a `rock.kernel` func and create the
  // `hip.rocmlir` dispatch. `module` is LEFT in this hip form: it is the
  // artifact we mutate at the end.
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::hip::createSimplifyOnnxPass());
  pm.addPass(mlir::hip::createHipAddContextArgPass());
  pm.addPass(mlir::hip::createOnnxLoopOutlinePass());
  pm.addPass(mlir::hip::createOnnxIfOutlinePass());
  pm.addPass(mlir::hip::createInferLoopBodyShapesPass());
  pm.addPass(mlir::hip::createConvertOnnxToHipPass());
  // rocMLIR has no transposed-convolution anchor, so split conv_transpose into
  // plain convolutions before fuse-rocmlir outlines a kernel around it.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::hip::createDecomposeConvTransposePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hip::createFuseROCMlirPass());
  pm.addPass(mlir::func::createDuplicateFunctionEliminationPass());

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "error: ONNX->HIP + fuse-rocmlir passes failed\n";
    return 1;
  }

  if (!dumpHipPath.empty())
    dumpModule(*module, "hip", dumpHipPath);

  // Stage 2: on a CLONE, run the hip->tosa conversion (front of
  // buildRocMlirPipeline, minus its terminal hipEpAddHighLevelPipeline -- that
  // runs in the .so), then serialize only the `rock.kernel` funcs and compile
  // them through the in-tree rocMLIR pipeline. The clone is
  // discarded; we only want the compiled binary + launch geometry back.
  mlir::OwningOpRef<mlir::ModuleOp> tosaModule = module->clone();
  {
    mlir::PassManager tpm(tosaModule->getContext());
    tpm.addNestedPass<mlir::func::FuncOp>(
        mlir::hip::createConvertHipToTosaPass());
    tpm.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(tpm.run(*tosaModule))) {
      llvm::errs() << "error: hip->tosa conversion failed\n";
      return 1;
    }
  }

  // Dumped before the non-kernel funcs are dropped below, so the dump shows the
  // whole module rather than just the kernels that cross into the .so. Note
  // this is the *canonicalized* TOSA: `tpm` ran the canonicalizer above, so
  // dead ops that `hip-mlir-opt --convert-hip-to-tosa` alone would print (the
  // orphaned `tensor.empty` feeding a dropped DPS `outs`, for one) are gone.
  if (!dumpTosaPath.empty())
    dumpModule(*tosaModule, "tosa", dumpTosaPath);

  // The rocMLIR high-level pipeline errors on any non-kernel func (its
  // tosa->rock passes assert a `rock.kernel` attribute and walk ops assuming
  // registered dialects). So collect the `rock.kernel` functions; each is
  // compiled from a clone that drops every other func (e.g. `main_graph` and
  // its hip.* ops).
  llvm::SmallVector<std::string> kernelNames;
  for (auto func : tosaModule->getOps<mlir::func::FuncOp>())
    if (func->hasAttr("rock.kernel"))
      kernelNames.push_back(func.getSymName().str());
  if (kernelNames.empty()) {
    llvm::errs() << "error: no rock.kernel func to compile\n";
    return 1;
  }

  // A keyed --perf-config naming a kernel that does not exist would otherwise
  // be dropped on the floor and that kernel compiled with the default config,
  // so a typo or a stale name reads as a successful run of a configuration
  // that was never applied. Tuning decisions get made off those numbers, so
  // refuse the run instead.
  for (const auto &entry : perfConfigByKernel) {
    if (llvm::is_contained(kernelNames, entry.getKey()))
      continue;
    llvm::errs() << "error: --perf-config names unknown kernel '"
                 << entry.getKey() << "'; this module has:\n";
    for (const std::string &name : kernelNames)
      llvm::errs() << "  " << name << "\n";
    return 1;
  }

  // The tuning and backend entry points are module-scoped and assume a single
  // anchor op per module, so a graph with several outlined kernels (a
  // decomposed conv_transpose, say) has to be compiled one kernel at a time.
  // Each single-kernel module keeps the original module-level attributes; only
  // the sibling kernel funcs are dropped, then the linked-in rocMLIR pipelines
  // run on it directly (same context, no serialization).
  llvm::StringMap<CompiledKernel> compiledByKernel;
  const std::string arch = resolveArch();
  for (auto [index, name] : llvm::enumerate(kernelNames)) {
    mlir::OwningOpRef<mlir::ModuleOp> single = tosaModule->clone();
    for (auto func :
         llvm::make_early_inc_range(single->getOps<mlir::func::FuncOp>()))
      if (func.getSymName() != name)
        func.erase();

    auto it = perfConfigByKernel.find(name);
    const std::string &perfConfig =
        it != perfConfigByKernel.end() ? it->second : defaultPerfConfig;
    if (kernelNames.size() > 1)
      llvm::errs() << "[hip-rocmlir-compiler] compiling kernel '" << name
                   << "' (" << (index + 1) << " of " << kernelNames.size()
                   << ")\n";
    if (!runRocmlir(*single, arch, perfConfig, dumpHighLevel,
                    compiledByKernel[name]))
      return 1;
  }

  // --dump-high-level: write the rock MLIR (text) and stop. A single kernel
  // writes <output> verbatim so existing tuning scripts keep working; several
  // get one file each, since rocmlir-tuning-driver takes one kernel at a time.
  if (dumpHighLevel) {
    for (const std::string &name : kernelNames) {
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
      os << compiledByKernel[name].highLevelMlir << "\n";
      llvm::errs() << "[hip-rocmlir-compiler] wrote rock high-level MLIR to "
                   << path << "\n";
    }
    return 0;
  }

  // Stage 3: embed the compiled artifact back into `module`'s `hip.rocmlir`
  // dispatch op (kernel_binary + grid_size + block_size), then delete the
  // now-compiled `rock.kernel` funcs. This matches what hip-compiler consumes:
  // a self-contained hip module carrying the GPU binary inline.
  //
  // Each dispatch names its kernel func, so give it that kernel's binary and
  // launch geometry. hip.rocmlir -> wrap_rocmlir lowering already reads both
  // per op, so several kernels in one module need nothing further downstream.
  mlir::Builder b(&context);
  auto i64 = mlir::IntegerType::get(&context, 64);

  llvm::SmallVector<mlir::func::FuncOp> kernelFuncs;
  for (auto func : module->getOps<mlir::func::FuncOp>())
    if (func->hasAttr("rock.kernel"))
      kernelFuncs.push_back(func);

  unsigned stamped = 0;
  size_t totalBytes = 0;
  mlir::WalkResult walked = module->walk([&](mlir::hip::RocMlirOp op) {
    llvm::StringRef callee = op.getKernel();
    auto it = compiledByKernel.find(callee);
    if (it == compiledByKernel.end()) {
      op.emitError() << "no compiled kernel for '" << callee << "'";
      return mlir::WalkResult::interrupt();
    }
    const CompiledKernel &kernel = it->second;
    op.setKernelBinaryAttr(mlir::StringAttr::get(
        &context, llvm::StringRef(kernel.binary.data(), kernel.binary.size())));
    op.setGridSizeAttr(mlir::IntegerAttr::get(i64, kernel.gridSize));
    op.setBlockSizeAttr(mlir::IntegerAttr::get(i64, kernel.blockSize));
    ++stamped;
    totalBytes += kernel.binary.size();
    return mlir::WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return 1;
  if (stamped == 0) {
    llvm::errs() << "error: no hip.rocmlir op found to embed the binary into\n";
    return 1;
  }

  // Delete the successfully compiled kernel funcs; their body now lives in the
  // embedded binary and the symbol is no longer needed.
  for (auto func : kernelFuncs)
    func.erase();

  // Keep the single-kernel line byte-for-byte what it was; log the geometry
  // per kernel first when there is more than one, since they differ.
  if (kernelNames.size() > 1)
    for (const std::string &name : kernelNames) {
      const CompiledKernel &kernel = compiledByKernel[name];
      llvm::errs() << "[hip-rocmlir-compiler]   " << name << ": "
                   << kernel.binary.size()
                   << "-byte binary, grid_size=" << kernel.gridSize
                   << " block_size=" << kernel.blockSize << "\n";
    }
  const CompiledKernel &first = compiledByKernel[kernelNames.front()];
  llvm::errs() << "[hip-rocmlir-compiler] embedded " << totalBytes
               << "-byte binary into " << stamped << " hip.rocmlir op(s); "
               << "grid_size=" << first.gridSize
               << " block_size=" << first.blockSize << "; deleted "
               << kernelFuncs.size() << " kernel func(s)\n";

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
