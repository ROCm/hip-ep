/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Moved out of tools/hip-rocmlir-compiler/hip-rocmlir-compiler.cpp so the EP's
// in-process CompilerDriver can run the same flow. See the header for why.

#include "hip/Compiler/RocMlirKernelCompiler.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/debug_log.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
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

#include <memory>

namespace mlir {
namespace hip {

namespace {

// Read one `name=value` field out of a serialized perfConfig string of the
// form `prefix:key=value,key=value,...` (see getPerfConfigStr in
// RockAttrDefs.td). Returns `fallback` when the field is absent or
// unparseable.
int64_t perfConfigField(StringRef perf, StringRef name, int64_t fallback) {
  // Drop the `gemm:`/`attn:` prefix if present so a prefix substring can't be
  // mistaken for a field.
  size_t colon = perf.find(':');
  if (colon != StringRef::npos)
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

// Map a parsed perfConfig string + arch onto rock's Triton/backend option
// structs. The Triton/backend knobs are the tuning fields of the chosen
// perfConfig, read straight out of the string rather than re-defaulted
// (perfConfig field `numWaves` is the backend's `numWarps`); a missing field
// falls back to the rock default and kKnobDefault (-1) leaves a bool knob at
// its arch default.
bool buildBackendPipelineFor(OpPassManager &pm, StringRef arch,
                             StringRef perfConfig) {
  RocmDeviceName devName;
  if (arch.empty() || failed(devName.parse(arch)))
    return false;

  rock::KernelOptions kOpts;
  rock::buildKernelPipeline(pm, kOpts);

  rock::TritonOptions tOpts;
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
  rock::buildTritonPipeline(pm, tOpts);

  rock::BackendOptions bOpts;
  bOpts.triple = devName.getTriple().str();
  bOpts.chip = devName.getChip().str();
  bOpts.features = devName.getFeaturesForBackend();
  bOpts.optLevel = 3;
  bOpts.numWarps = tOpts.numWarps;
  bOpts.numCTAs = tOpts.numCTAs;
  bOpts.wavesPerEU =
      static_cast<int>(perfConfigField(perfConfig, "wavesPerEU", 0));
  rock::buildBackendPipeline(pm, bOpts);
  return true;
}

// Walk the compiled module for the single gpu.binary object and pull out its
// ELF blob plus the {block, grid} launch geometry.
bool extractCompiledKernel(ModuleOp mod, CompiledKernel &out) {
  unsigned count = 0;
  mod.walk([&](gpu::BinaryOp binary) {
    auto object = llvm::cast<gpu::ObjectAttr>(binary.getObjects()[0]);
    StringRef blob = object.getObject().getValue();
    out.binary.assign(blob.begin(), blob.end());
    for (auto kernel : object.getKernels()) {
      auto block =
          kernel.getAttr<IntegerAttr>(rock::BlockSizeAttr::getMnemonic());
      auto grid =
          kernel.getAttr<IntegerAttr>(rock::GridSizeAttr::getMnemonic());
      if (!block || !grid)
        continue;
      out.blockSize = block.getInt();
      out.gridSize = grid.getInt();
      ++count;
    }
  });
  return count == 1 && !out.binary.empty();
}

} // namespace

std::string resolveRocMlirArch() {
  // hip_get_env, not std::getenv: this runs inside the static-CRT EP DLL when
  // called from CompilerDriver, and std::getenv there cannot see env vars set
  // by the host process -- a ROCK_ARCH override would be silently dropped and
  // every kernel built for the fallback arch below.
  std::string arch = hip_get_env("ROCK_ARCH");
  if (!arch.empty())
    return arch;
  return "gfx1151";
}

void registerRocMlirDialects(MLIRContext &context) {
  DialectRegistry rockRegistry;
  registerRocMLIRDialects(rockRegistry);
  context.appendDialectRegistry(rockRegistry);
  context.loadAllAvailableDialects();
}

OwningOpRef<ModuleOp> buildRocMlirTosaClone(ModuleOp module) {
  OwningOpRef<ModuleOp> tosaModule = module.clone();
  PassManager pm(tosaModule->getContext());
  pm.addNestedPass<func::FuncOp>(createConvertHipToTosaPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(*tosaModule)))
    return nullptr;
  return tosaModule;
}

OwningOpRef<ModuleOp> takeRocMlirKernelsAsTosaModule(ModuleOp module) {
  OpBuilder builder(module.getContext());
  OwningOpRef<ModuleOp> kernels = ModuleOp::create(builder.getUnknownLoc());

  // Unlink and re-parent rather than clone. Each kernel is dead in `module`
  // from here on -- its body ships as an embedded binary and the symbol goes
  // away -- so there is nothing to preserve, and an outlined kernel is
  // IsolatedFromAbove, so no enclosing SSA value or module attribute has to
  // travel with it (rock.arch rides on the func, put there by fuse-rocmlir).
  for (auto func : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    if (!func->hasAttr("rock.kernel"))
      continue;
    func->remove();
    kernels->getBody()->push_back(func);
  }
  if (kernels->getBody()->empty())
    return kernels;

  // One pass-manager run for every kernel at once. Per-kernel runs would be
  // simpler but measurably slower: the canonicalizer rebuilds its pattern set
  // from every loaded dialect each time it is constructed, and with the rock
  // and triton dialects registered that fixed cost dominates the actual work
  // on a single outlined kernel.
  PassManager pm(kernels->getContext());
  pm.addNestedPass<func::FuncOp>(createConvertHipToTosaPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(*kernels)))
    return nullptr;
  return kernels;
}

OwningOpRef<ModuleOp> takeRocMlirKernelModule(ModuleOp kernelsModule,
                                              StringRef kernelName) {
  func::FuncOp kernel;
  for (auto func : kernelsModule.getOps<func::FuncOp>())
    if (func.getSymName() == kernelName) {
      kernel = func;
      break;
    }
  if (!kernel)
    return nullptr;

  OpBuilder builder(kernelsModule.getContext());
  OwningOpRef<ModuleOp> single = ModuleOp::create(builder.getUnknownLoc());
  kernel->remove();
  single->getBody()->push_back(kernel);
  return single;
}

bool compileRocMlirBackend(ModuleOp kernelModule, StringRef arch,
                           StringRef perfConfig, CompiledKernel &out) {
  PassManager pm(kernelModule.getContext());
  pm.setNesting(PassManager::Nesting::Implicit);
  if (!buildBackendPipelineFor(pm, arch, perfConfig) ||
      failed(pm.run(kernelModule)))
    return false;
  return extractCompiledKernel(kernelModule, out);
}

bool runRocMlirHighLevelPipeline(ModuleOp kernelModule) {
  PassManager pm(kernelModule.getContext());
  pm.setNesting(PassManager::Nesting::Implicit);
  rock::buildHighlevelPipeline(pm);
  return succeeded(pm.run(kernelModule));
}

bool compileRocMlirDefaultPerfConfig(ModuleOp kernelModule, StringRef arch,
                                     CompiledKernel &out) {
  // Affix a perfConfig to the gemm op (the `perf_config` string attr). Take
  // the first entry enumerated from the tuning search space; autotuning
  // callers substitute their own compileOne instead, which is why this is not
  // folded into the backend call.
  llvm::SmallString<1024> perfConfig; // ROCMLIR_TUNING_PARAM_STRING_BUFSZ
  std::unique_ptr<rock::TuningParamSet> space(rock::createTunableParamSpace(
      kernelModule, rock::TuningParamSetKind::Full));
  if (!space || space->tuningRange.empty())
    return false;
  rock::ParamEntry entry;
  if (!rock::tuningGetParam(space.get(), /*pos=*/0, &entry))
    return false;
  entry.param.getPerfConfigStr(perfConfig);
  if (!rock::tuningSetStr(kernelModule, perfConfig))
    return false;

  return compileRocMlirBackend(kernelModule, arch, perfConfig, out);
}

bool runRocMlirOnKernel(ModuleOp kernelModule, StringRef arch,
                        bool stopAfterHighLevel, CompiledKernel &out) {
  if (!runRocMlirHighLevelPipeline(kernelModule))
    return false;

  if (stopAfterHighLevel) {
    llvm::raw_string_ostream os(out.highLevelMlir);
    kernelModule.print(os);
    return true;
  }

  return compileRocMlirDefaultPerfConfig(kernelModule, arch, out);
}

SmallVector<std::string> collectRocMlirKernelNames(ModuleOp module) {
  SmallVector<std::string> names;
  for (auto func : module.getOps<func::FuncOp>())
    if (func->hasAttr("rock.kernel"))
      names.push_back(func.getSymName().str());
  return names;
}

LogicalResult
compileAndEmbedRocMlirKernels(ModuleOp module,
                              const RocMlirEmbedOptions &options) {
  SmallVector<std::string> kernelNames = collectRocMlirKernelNames(module);
  if (kernelNames.empty())
    return success();

  // Lift the kernels out of `module` and convert them all to tosa in one go.
  // This used to clone the whole module for the conversion and then clone it
  // again per kernel, so a graph with N kernels made N+1 deep copies of the
  // entire model and converted `main_graph` to tosa every time only to erase
  // it. Handing the IR over whole was unavoidable when rocMLIR sat behind a
  // dlopen boundary; it is built in-tree now, so the kernels can simply be
  // re-parented.
  OwningOpRef<ModuleOp> kernels = takeRocMlirKernelsAsTosaModule(module);
  if (!kernels)
    return module.emitError() << "hip->tosa conversion failed";

  // The tuning and backend entry points are module-scoped and assume a single
  // anchor op per module, so the kernels still have to be compiled one at a
  // time -- but splitting one off is now just a re-parent, with no pass run
  // and nothing copied.
  llvm::StringMap<CompiledKernel> compiledByKernel;
  for (auto [index, name] : llvm::enumerate(kernelNames)) {
    if (options.log && kernelNames.size() > 1)
      *options.log << options.logPrefix << " compiling kernel '" << name
                   << "' (" << (index + 1) << " of " << kernelNames.size()
                   << ")\n";

    OwningOpRef<ModuleOp> single = takeRocMlirKernelModule(*kernels, name);
    if (!single)
      return module.emitError() << "no tosa kernel named '" << name << "'";

    // tosa -> rock happens here, not inside compileOne: the tuning entry
    // points a custom compileOne uses (createTunableParamSpace, tuningSetStr)
    // only see anything once the kernel is in rock form. Handing the callback
    // raw tosa instead gave --autotune an empty search space.
    if (!runRocMlirHighLevelPipeline(*single))
      return module.emitError()
             << "rocMLIR high-level pipeline failed for kernel '" << name
             << "'";

    CompiledKernel &out = compiledByKernel[name];
    bool ok = options.compileOne
                  ? options.compileOne(*single, name, out)
                  : compileRocMlirDefaultPerfConfig(*single, options.arch, out);
    if (!ok)
      return module.emitError()
             << "rocMLIR failed to compile kernel '" << name << "'";
  }

  // Stamp the compiled artifact onto each `hip.rocmlir` dispatch
  // (kernel_binary + grid_size + block_size). The kernel funcs themselves are
  // already gone from `module` -- they were moved out above -- which is what
  // used to be a separate erase pass here. Each dispatch names its kernel
  // func, so give it that kernel's binary and geometry; the hip.rocmlir ->
  // wrap_rocmlir lowering reads both per op.
  MLIRContext *context = module.getContext();
  auto i64 = IntegerType::get(context, 64);

  size_t totalBytes = 0;
  WalkResult walked = module.walk([&](RocMlirOp op) {
    StringRef callee = op.getKernel();
    auto it = compiledByKernel.find(callee);
    if (it == compiledByKernel.end()) {
      op.emitError() << "no compiled kernel for '" << callee << "'";
      return WalkResult::interrupt();
    }
    const CompiledKernel &kernel = it->second;
    op.setKernelBinaryAttr(StringAttr::get(
        context, StringRef(kernel.binary.data(), kernel.binary.size())));
    op.setGridSizeAttr(IntegerAttr::get(i64, kernel.gridSize));
    op.setBlockSizeAttr(IntegerAttr::get(i64, kernel.blockSize));
    totalBytes += kernel.binary.size();
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();

  if (options.log) {
    for (const std::string &name : kernelNames) {
      const CompiledKernel &kernel = compiledByKernel[name];
      *options.log << options.logPrefix << "   " << name << ": "
                   << kernel.binary.size()
                   << "-byte binary, grid_size=" << kernel.gridSize
                   << " block_size=" << kernel.blockSize << "\n";
    }
    *options.log << options.logPrefix << " embedded " << totalBytes
                 << "-byte binary; deleted " << kernelNames.size()
                 << " kernel func(s)\n";
  }
  return success();
}

} // namespace hip
} // namespace mlir
