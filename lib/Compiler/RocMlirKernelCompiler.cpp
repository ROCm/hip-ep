/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Moved out of tools/hip-rocmlir-compiler/hip-rocmlir-compiler.cpp so the EP's
// in-process CompilerDriver can run the same flow. See the header for why.

#include "hip/Compiler/RocMlirKernelCompiler.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

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

#include <cstdlib>
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
  if (const char *env = std::getenv("ROCK_ARCH"))
    if (env[0] != '\0')
      return env;
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

bool compileRocMlirBackend(ModuleOp kernelModule, StringRef arch,
                           StringRef perfConfig, CompiledKernel &out) {
  PassManager pm(kernelModule.getContext());
  pm.setNesting(PassManager::Nesting::Implicit);
  if (!buildBackendPipelineFor(pm, arch, perfConfig) ||
      failed(pm.run(kernelModule)))
    return false;
  return extractCompiledKernel(kernelModule, out);
}

bool runRocMlirOnKernel(ModuleOp kernelModule, StringRef arch,
                        bool stopAfterHighLevel, CompiledKernel &out) {
  // 1. High-level pipeline: tosa -> rock.gemm.
  {
    PassManager pm(kernelModule.getContext());
    pm.setNesting(PassManager::Nesting::Implicit);
    rock::buildHighlevelPipeline(pm);
    if (failed(pm.run(kernelModule)))
      return false;
  }

  if (stopAfterHighLevel) {
    llvm::raw_string_ostream os(out.highLevelMlir);
    kernelModule.print(os);
    return true;
  }

  // 2. Affix a perfConfig to the gemm op (the `perf_config` string attr).
  //    Take the first entry enumerated from the tuning search space;
  //    autotuning callers substitute their own compileOne instead.
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

  // 3-4. Run the backend and extract the binary + launch geometry.
  return compileRocMlirBackend(kernelModule, arch, perfConfig, out);
}

SmallVector<std::string> collectRocMlirKernelNames(ModuleOp module) {
  SmallVector<std::string> names;
  for (auto func : module.getOps<func::FuncOp>())
    if (func->hasAttr("rock.kernel"))
      names.push_back(func.getSymName().str());
  return names;
}

LogicalResult
compileAndEmbedRocMlirKernels(ModuleOp module, ModuleOp tosaModule,
                              const RocMlirEmbedOptions &options) {
  SmallVector<std::string> kernelNames = collectRocMlirKernelNames(tosaModule);
  if (kernelNames.empty())
    return success();

  // The tuning and backend entry points are module-scoped and assume a single
  // anchor op per module, so a graph with several outlined kernels has to be
  // compiled one kernel at a time. Each single-kernel module keeps the
  // original module-level attributes; only the sibling kernel funcs are
  // dropped.
  llvm::StringMap<CompiledKernel> compiledByKernel;
  for (auto [index, name] : llvm::enumerate(kernelNames)) {
    OwningOpRef<ModuleOp> single = tosaModule.clone();
    for (auto func : llvm::make_early_inc_range(single->getOps<func::FuncOp>()))
      if (func.getSymName() != name)
        func.erase();

    if (options.log && kernelNames.size() > 1)
      *options.log << options.logPrefix << " compiling kernel '" << name
                   << "' (" << (index + 1) << " of " << kernelNames.size()
                   << ")\n";

    CompiledKernel &out = compiledByKernel[name];
    bool ok = options.compileOne
                  ? options.compileOne(*single, name, out)
                  : runRocMlirOnKernel(*single, options.arch,
                                       /*stopAfterHighLevel=*/false, out);
    if (!ok)
      return module.emitError()
             << "rocMLIR failed to compile kernel '" << name << "'";
  }

  // Stamp the compiled artifact onto each `hip.rocmlir` dispatch
  // (kernel_binary + grid_size + block_size), then delete the now-compiled
  // `rock.kernel` funcs: their body lives in the embedded binary and the
  // symbol is no longer needed. Each dispatch names its kernel func, so give
  // it that kernel's binary and geometry; the hip.rocmlir -> wrap_rocmlir
  // lowering reads both per op.
  MLIRContext *context = module.getContext();
  auto i64 = IntegerType::get(context, 64);

  SmallVector<func::FuncOp> kernelFuncs;
  for (auto func : module.getOps<func::FuncOp>())
    if (func->hasAttr("rock.kernel"))
      kernelFuncs.push_back(func);

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

  for (auto func : kernelFuncs)
    func.erase();

  if (options.log) {
    for (const std::string &name : kernelNames) {
      const CompiledKernel &kernel = compiledByKernel[name];
      *options.log << options.logPrefix << "   " << name << ": "
                   << kernel.binary.size()
                   << "-byte binary, grid_size=" << kernel.gridSize
                   << " block_size=" << kernel.blockSize << "\n";
    }
    *options.log << options.logPrefix << " embedded " << totalBytes
                 << "-byte binary; deleted " << kernelFuncs.size()
                 << " kernel func(s)\n";
  }
  return success();
}

} // namespace hip
} // namespace mlir
