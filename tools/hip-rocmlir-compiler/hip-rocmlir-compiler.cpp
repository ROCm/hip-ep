/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip-rocmlir-compiler: drive an ONNX-dialect module through the ONNX->HIP
// head passes and the rocmlir hip->tosa conversion, then hand the tosa IR to
// rocMLIR's high-level pipeline living inside librockCompiler.so.
//
// Why the split / dlopen: librockCompiler.so statically embeds its OWN complete
// copy of LLVM/MLIR. This executable also links its own copy. The two copies
// have incompatible type systems (each dialect/type/op TypeID is the address of
// a per-copy static, so they differ). Passing a live MLIRContext / pass manager
// across the boundary aborts with "Trying to register different dialects for
// the same namespace: tosa". The only safe hand-off is *serialized IR*: we run
// our passes in our context, print the module to text, then dlopen the .so and
// use ITS exported MLIR C-API (mlirContextCreate, mlirModuleCreateParse,
// hipEpAddHighLevelPipeline, ...) to parse + run + print entirely inside the
// .so's own MLIR. No MLIR object ever crosses the boundary -- only bytes.
//
// The .so's high-level pipeline only rewrites functions marked `rock.kernel`
// (set by fuse-rocmlir), so `main_graph` and its unregistered `hip.*` ops ride
// through untouched; we allow unregistered dialects on the .so side and print
// in generic form so those ops survive the round-trip.

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/InitAllPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "CrashHandler.h"

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <string>

// ---------------------------------------------------------------------------
// Minimal mlir-c type/function declarations for the symbols we resolve out of
// librockCompiler.so. We deliberately do NOT include <mlir-c/*.h> here: those
// declare the functions with default (import) linkage and could bind to this
// executable's own MLIR at link time. Everything below is called ONLY through
// dlsym'd pointers so it always hits the .so's copy. The struct layouts match
// mlir-c/Support.h and mlir-c/IR.h (single-pointer opaque handles).
// ---------------------------------------------------------------------------
namespace rockcapi {

struct MlirContext {
  void *ptr;
};
struct MlirDialectRegistry {
  void *ptr;
};
struct MlirModule {
  void *ptr;
};
struct MlirOperation {
  void *ptr;
};
struct MlirPassManager {
  void *ptr;
};
struct MlirOpPassManager {
  void *ptr;
};
struct MlirStringRef {
  const char *data;
  size_t length;
};
using MlirStringCallback = void (*)(MlirStringRef, void *);

// Opaque rock tuning handles (mlir-c/Dialect/Rock.h).
struct MlirRockTuningSpace {
  void *ptr;
};
struct MlirRockTuningParam {
  void *ptr;
};

// RocmlirTuningParamSetKind (mlir-c/Dialect/RockEnums.h).
enum RocmlirTuningParamSetKind {
  RocmlirTuningParamSetKindQuick = 0,
  RocmlirTuningParamSetKindFull = 1,
  RocmlirTuningParamSetKindExhaustive = 2,
};

// HipEpBackendOptions (mlir-c/Dialect/HipEp.h). Layout must match exactly --
// hipEpAddBackendPipeline reads it by value through this struct.
struct HipEpBackendOptions {
  const char *arch;
  int optLevel;
  int numWarps;
  int numCTAs;
  int numStages;
  int matrixInstrNonkdim;
  int kpack;
  int64_t useAsyncCopy;
  int64_t useBlockPingpong;
  int64_t useInThreadTranspose;
  int64_t useBufferOps;
  int64_t useBufferAtomics;
  int64_t useReductionLayout;
  int64_t useOptimizeEpilogue;
  int wavesPerEU;
};

struct Api {
  void *handle = nullptr;

  MlirDialectRegistry (*dialectRegistryCreate)();
  void (*dialectRegistryDestroy)(MlirDialectRegistry);
  void (*registerRocMLIRDialects)(MlirDialectRegistry);
  MlirContext (*contextCreateWithRegistry)(MlirDialectRegistry, bool);
  void (*contextSetAllowUnregisteredDialects)(MlirContext, bool);
  void (*contextLoadAllAvailableDialects)(MlirContext);
  void (*contextDestroy)(MlirContext);
  MlirModule (*moduleCreateParse)(MlirContext, MlirStringRef);
  MlirOperation (*moduleGetOperation)(MlirModule);
  void (*moduleDestroy)(MlirModule);
  MlirPassManager (*passManagerCreate)(MlirContext);
  MlirOpPassManager (*passManagerGetAsOpPassManager)(MlirPassManager);
  int (*passManagerRunOnOp)(MlirPassManager, MlirOperation); // MlirLogicalResult
  void (*passManagerDestroy)(MlirPassManager);
  void (*operationPrint)(MlirOperation, MlirStringCallback, void *);

  void (*hipEpAddHighLevelPipeline)(MlirOpPassManager);
  bool (*hipEpAddBackendPipeline)(MlirOpPassManager,
                                  const HipEpBackendOptions *);

  // Rock perfConfig search-space enumeration (mlir-c/Dialect/Rock.h).
  MlirRockTuningSpace (*rockTuningSpaceCreate)(MlirModule,
                                               RocmlirTuningParamSetKind);
  unsigned (*rockTuningGetNumParams)(MlirRockTuningSpace);
  MlirRockTuningParam (*rockTuningParamCreate)();
  bool (*rockTuningParamGet)(MlirRockTuningSpace, unsigned, MlirRockTuningParam);
  size_t (*rockTuningParamToString)(MlirRockTuningParam, char *, size_t);
  bool (*rockTuningSetFromStr)(MlirModule, MlirStringRef);
  void (*rockTuningParamDestroy)(MlirRockTuningParam);
  void (*rockTuningSpaceDestroy)(MlirRockTuningSpace);

  // Compiled-artifact extraction (mlir-c/Dialect/MIGraphX.h).
  void (*getKernelAttrs)(MlirModule, uint32_t *); // [block, grid, cluster]
  bool (*getBinary)(MlirModule, size_t *, char *);
};

template <typename T>
static bool bind(void *handle, T &fn, const char *name) {
  fn = reinterpret_cast<T>(dlsym(handle, name));
  if (!fn) {
    llvm::errs() << "error: symbol '" << name
                 << "' not found in librockCompiler.so\n";
    return false;
  }
  return true;
}

static bool load(Api &api, const std::string &soPath) {
  // RTLD_LOCAL keeps the .so's ~769 exported mlir* symbols out of the global
  // namespace so they cannot interpose on this executable's own MLIR.
  api.handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!api.handle) {
    llvm::errs() << "error: dlopen('" << soPath << "') failed: " << dlerror()
                 << "\n";
    return false;
  }
  bool ok = true;
  ok &= bind(api.handle, api.dialectRegistryCreate, "mlirDialectRegistryCreate");
  ok &= bind(api.handle, api.dialectRegistryDestroy,
             "mlirDialectRegistryDestroy");
  ok &= bind(api.handle, api.registerRocMLIRDialects,
             "mlirRegisterRocMLIRDialects");
  ok &= bind(api.handle, api.contextCreateWithRegistry,
             "mlirContextCreateWithRegistry");
  ok &= bind(api.handle, api.contextSetAllowUnregisteredDialects,
             "mlirContextSetAllowUnregisteredDialects");
  ok &= bind(api.handle, api.contextLoadAllAvailableDialects,
             "mlirContextLoadAllAvailableDialects");
  ok &= bind(api.handle, api.contextDestroy, "mlirContextDestroy");
  ok &= bind(api.handle, api.moduleCreateParse, "mlirModuleCreateParse");
  ok &= bind(api.handle, api.moduleGetOperation, "mlirModuleGetOperation");
  ok &= bind(api.handle, api.moduleDestroy, "mlirModuleDestroy");
  ok &= bind(api.handle, api.passManagerCreate, "mlirPassManagerCreate");
  ok &= bind(api.handle, api.passManagerGetAsOpPassManager,
             "mlirPassManagerGetAsOpPassManager");
  ok &= bind(api.handle, api.passManagerRunOnOp, "mlirPassManagerRunOnOp");
  ok &= bind(api.handle, api.passManagerDestroy, "mlirPassManagerDestroy");
  ok &= bind(api.handle, api.operationPrint, "mlirOperationPrint");
  ok &= bind(api.handle, api.hipEpAddHighLevelPipeline,
             "hipEpAddHighLevelPipeline");
  ok &= bind(api.handle, api.hipEpAddBackendPipeline,
             "hipEpAddBackendPipeline");
  ok &= bind(api.handle, api.rockTuningSpaceCreate,
             "mlirRockTuningSpaceCreate");
  ok &= bind(api.handle, api.rockTuningGetNumParams,
             "mlirRockTuningGetNumParams");
  ok &= bind(api.handle, api.rockTuningParamCreate,
             "mlirRockTuningParamCreate");
  ok &= bind(api.handle, api.rockTuningParamGet, "mlirRockTuningParamGet");
  ok &= bind(api.handle, api.rockTuningParamToString,
             "mlirRockTuningParamToString");
  ok &= bind(api.handle, api.rockTuningSetFromStr,
             "mlirRockTuningSetFromStr");
  ok &= bind(api.handle, api.rockTuningParamDestroy,
             "mlirRockTuningParamDestroy");
  ok &= bind(api.handle, api.rockTuningSpaceDestroy,
             "mlirRockTuningSpaceDestroy");
  ok &= bind(api.handle, api.getKernelAttrs, "mlirGetKernelAttrs");
  ok &= bind(api.handle, api.getBinary, "mlirGetBinary");
  return ok;
}

} // namespace rockcapi

// Resolve the librockCompiler.so path: ROCK_COMPILER_SO wins, else the
// in-tree build location.
static std::string resolveSoPath() {
  if (const char *env = std::getenv("ROCK_COMPILER_SO"))
    if (env[0] != '\0')
      return env;
  return "build/librockCompiler.so";
}

// Run the rocMLIR high-level pipeline inside the .so on the serialized module,
// printing the result to stdout. `moduleText` is generic-form MLIR text.
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

// Drive the rocMLIR flow inside the .so on the serialized tosa module:
//   1. high-level pipeline (tosa -> rock.gemm)
//   2. enumerate the perfConfig search space; take the first entry and affix it
//      to the gemm op (as the `perf_config` string attribute)
//   3. backend pipeline (rock -> LLVM / binary)
// On success, returns the compiled GPU binary in `binary` and the kernel launch
// geometry in `gridSize`/`blockSize`. `moduleText` is generic-form MLIR text.
struct CompiledKernel {
  std::string binary;
  int64_t gridSize = 0;
  int64_t blockSize = 0;
};

static bool runRocmlirInSo(const std::string &moduleText,
                           const std::string &soPath, const std::string &arch,
                           CompiledKernel &out) {
  using namespace rockcapi;
  Api api;
  if (!load(api, soPath))
    return false;

  MlirDialectRegistry registry = api.dialectRegistryCreate();
  api.registerRocMLIRDialects(registry);
  // Disable threading: keeps diagnostics ordered and avoids the .so spinning up
  // its own thread pool for a single one-shot pipeline run.
  MlirContext ctx = api.contextCreateWithRegistry(registry, /*threading=*/false);
  // main_graph still carries unregistered hip.* ops; let them parse generically.
  api.contextSetAllowUnregisteredDialects(ctx, true);
  api.contextLoadAllAvailableDialects(ctx);

  MlirStringRef text{moduleText.data(), moduleText.size()};
  MlirModule module = api.moduleCreateParse(ctx, text);
  if (!module.ptr) {
    llvm::errs() << "error: librockCompiler.so failed to parse the "
                    "tosa module\n";
    api.contextDestroy(ctx);
    api.dialectRegistryDestroy(registry);
    return false;
  }
  MlirOperation moduleOp = api.moduleGetOperation(module);

  auto fail = [&](const char *msg) {
    llvm::errs() << "error: " << msg << "\n";
    api.moduleDestroy(module);
    api.contextDestroy(ctx);
    api.dialectRegistryDestroy(registry);
    return false;
  };

  // 1. High-level pipeline: tosa -> rock.gemm.
  {
    MlirPassManager pm = api.passManagerCreate(ctx);
    api.hipEpAddHighLevelPipeline(api.passManagerGetAsOpPassManager(pm));
    int r = api.passManagerRunOnOp(pm, moduleOp);
    api.passManagerDestroy(pm);
    if (r == 0)
      return fail("rocMLIR high-level pipeline failed");
  }

  // 2. Enumerate the perfConfig search space and affix the first entry to the
  //    gemm op. mlirRockTuningSpaceCreate reads the gemm problem out of the
  //    module; mlirRockTuningSetFromStr stamps `perf_config` onto the gemm op.
  char perfConfig[1024]; // ROCMLIR_TUNING_PARAM_STRING_BUFSZ
  {
    MlirRockTuningSpace space =
        api.rockTuningSpaceCreate(module, RocmlirTuningParamSetKindFull);
    unsigned num = api.rockTuningGetNumParams(space);
    if (num == 0) {
      api.rockTuningSpaceDestroy(space);
      return fail("perfConfig search space is empty");
    }

    MlirRockTuningParam param = api.rockTuningParamCreate();
    if (!api.rockTuningParamGet(space, /*pos=*/0, param)) {
      api.rockTuningParamDestroy(param);
      api.rockTuningSpaceDestroy(space);
      return fail("failed to read the first perfConfig entry");
    }

    size_t n = api.rockTuningParamToString(param, perfConfig, sizeof(perfConfig));
    if (n >= sizeof(perfConfig)) {
      api.rockTuningParamDestroy(param);
      api.rockTuningSpaceDestroy(space);
      return fail("perfConfig string too long");
    }
    perfConfig[n] = '\0';
    llvm::errs() << "[hip-rocmlir-compiler] perfConfig search space size: "
                 << num << "; affixing first entry: " << perfConfig << "\n";

    MlirStringRef perf{perfConfig, n};
    bool stamped = api.rockTuningSetFromStr(module, perf);
    api.rockTuningParamDestroy(param);
    api.rockTuningSpaceDestroy(space);
    if (!stamped)
      return fail("failed to affix perfConfig to the gemm op");
  }

  // 3. Backend pipeline: rock (with the affixed perf_config) -> LLVM / binary.
  //    The Triton/backend knobs are the tuning fields of the chosen perfConfig,
  //    so read them straight out of the string rather than re-defaulting them
  //    (the perfConfig field `numWaves` is the backend's `numWarps`). A missing
  //    field falls back to the rock default; kKnobDefault (-1) leaves a bool
  //    knob at its arch default.
  {
    llvm::StringRef perf(perfConfig);
    HipEpBackendOptions opts{};
    opts.arch = arch.c_str();
    opts.optLevel = 3;
    opts.numWarps = static_cast<int>(perfConfigField(perf, "numWaves", 4));
    opts.numCTAs = static_cast<int>(perfConfigField(perf, "numCTAs", 1));
    opts.numStages = static_cast<int>(perfConfigField(perf, "numStages", 1));
    opts.matrixInstrNonkdim =
        static_cast<int>(perfConfigField(perf, "matrixInstrNonkdim", 0));
    opts.kpack = static_cast<int>(perfConfigField(perf, "kpack", 1));
    opts.wavesPerEU = static_cast<int>(perfConfigField(perf, "wavesPerEU", 0));
    opts.useAsyncCopy = perfConfigField(perf, "useAsyncCopy", -1);
    opts.useBlockPingpong = perfConfigField(perf, "useBlockPingpong", -1);
    opts.useInThreadTranspose =
        perfConfigField(perf, "useInThreadTranspose", -1);
    opts.useBufferOps = perfConfigField(perf, "useBufferOps", -1);
    opts.useBufferAtomics = perfConfigField(perf, "useBufferAtomics", -1);
    opts.useReductionLayout = perfConfigField(perf, "useReductionLayout", -1);
    opts.useOptimizeEpilogue = perfConfigField(perf, "useOptimizeEpilogue", -1);

    MlirPassManager pm = api.passManagerCreate(ctx);
    if (!api.hipEpAddBackendPipeline(api.passManagerGetAsOpPassManager(pm),
                                     &opts)) {
      api.passManagerDestroy(pm);
      return fail("failed to build the backend pipeline");
    }
    int r = api.passManagerRunOnOp(pm, moduleOp);
    api.passManagerDestroy(pm);
    if (r == 0)
      return fail("rocMLIR backend pipeline failed");
  }

  // 4. Extract the compiled artifact: the gpu.binary blob plus launch geometry.
  //    mlirGetKernelAttrs returns uint32_t[3] = {block_size, grid_size,
  //    cluster_size}.
  uint32_t attrs[3] = {0, 0, 0};
  api.getKernelAttrs(module, attrs);
  out.blockSize = attrs[0];
  out.gridSize = attrs[1];

  size_t binSize = 0;
  if (!api.getBinary(module, &binSize, nullptr) || binSize == 0)
    return fail("failed to query compiled binary size");
  out.binary.resize(binSize);
  if (!api.getBinary(module, nullptr, out.binary.data()))
    return fail("failed to extract compiled binary");

  api.moduleDestroy(module);
  api.contextDestroy(ctx);
  api.dialectRegistryDestroy(registry);
  return true;
}

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-rocmlir-compiler");

  std::string inputFilename;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '-')
      inputFilename = argv[i];
  }
  if (inputFilename.empty()) {
    llvm::errs() << "Usage: " << argv[0] << " <input.onnx.mlir>\n"
                 << "  Runs ONNX->HIP head passes + hip->tosa conversion, then\n"
                 << "  the rocMLIR high-level pipeline (via librockCompiler.so)\n"
                 << "  and prints the result to stdout.\n"
                 << "  Set ROCK_COMPILER_SO to override the .so path "
                    "(default: build/librockCompiler.so).\n";
    return 1;
  }

  auto bufOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!bufOrErr) {
    llvm::errs() << "error: cannot open '" << inputFilename
                 << "': " << bufOrErr.getError().message() << "\n";
    return 1;
  }

  // Our own context, loaded with the same dialects the EP uses (adds tosa
  // lazily as a dependent dialect of convert-hip-to-tosa).
  mlir::MLIRContext context;
  hip::compiler::loadAllDialects(context);
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
  // buildOnnxToHipPipeline (simplify-onnx, hip-add-context-arg, loop/if outline,
  // infer-loop-body-shapes, convert-onnx-to-hip; plain path, no hipdnn handle)
  // followed by fuse-rocmlir + duplicate-function-elimination, which outline the
  // fused GEMM into a `rock.kernel` func and create the `hip.rocmlir` dispatch.
  // `module` is LEFT in this hip form: it is the artifact we mutate at the end.
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::hip::createSimplifyOnnxPass());
  pm.addPass(mlir::hip::createHipAddContextArgPass());
  pm.addPass(mlir::hip::createOnnxLoopOutlinePass());
  pm.addPass(mlir::hip::createOnnxIfOutlinePass());
  pm.addPass(mlir::hip::createInferLoopBodyShapesPass());
  pm.addPass(mlir::hip::createConvertOnnxToHipPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hip::createFuseROCMlirPass());
  pm.addPass(mlir::func::createDuplicateFunctionEliminationPass());

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "error: ONNX->HIP + fuse-rocmlir passes failed\n";
    return 1;
  }

  // Stage 2: on a CLONE, run the hip->tosa conversion (front of
  // buildRocMlirPipeline, minus its terminal hipEpAddHighLevelPipeline -- that
  // runs in the .so), then serialize only the `rock.kernel` funcs and compile
  // them through the rocMLIR pipeline in librockCompiler.so. The clone is
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

  // The rocMLIR high-level pipeline errors on any non-kernel func (its
  // tosa->rock passes assert a `rock.kernel` attribute and walk ops assuming
  // registered dialects). So hand it ONLY the `rock.kernel` functions: drop
  // everything else (e.g. `main_graph` and its hip.* ops). Generic-form text is
  // the portable interchange between this executable's MLIR and the .so's.
  for (auto func :
       llvm::make_early_inc_range(tosaModule->getOps<mlir::func::FuncOp>())) {
    if (!func->hasAttr("rock.kernel"))
      func.erase();
  }
  std::string moduleText;
  {
    llvm::raw_string_ostream os(moduleText);
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm();
    tosaModule->print(os, flags);
  }

  CompiledKernel compiled;
  if (!runRocmlirInSo(moduleText, resolveSoPath(), resolveArch(), compiled))
    return 1;

  // Stage 3: embed the compiled artifact back into `module`'s `hip.rocmlir`
  // dispatch op (kernel_binary + grid_size + block_size), then delete the
  // now-compiled `rock.kernel` funcs. This matches what hip-compiler consumes:
  // a self-contained hip module carrying the GPU binary inline.
  //
  // The .so tuning/backend path is single-GEMM, so exactly one kernel binary is
  // produced; stamp it onto every hip.rocmlir op whose callee is a compiled
  // rock.kernel func.
  mlir::Builder b(&context);
  auto binaryAttr = mlir::StringAttr::get(
      &context, llvm::StringRef(compiled.binary.data(), compiled.binary.size()));
  auto i64 = mlir::IntegerType::get(&context, 64);

  llvm::SmallVector<mlir::func::FuncOp> kernelFuncs;
  for (auto func : module->getOps<mlir::func::FuncOp>())
    if (func->hasAttr("rock.kernel"))
      kernelFuncs.push_back(func);

  unsigned stamped = 0;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "hip.rocmlir")
      return;
    op->setAttr("kernel_binary", binaryAttr);
    op->setAttr("grid_size", mlir::IntegerAttr::get(i64, compiled.gridSize));
    op->setAttr("block_size", mlir::IntegerAttr::get(i64, compiled.blockSize));
    ++stamped;
  });
  if (stamped == 0) {
    llvm::errs() << "error: no hip.rocmlir op found to embed the binary into\n";
    return 1;
  }

  // Delete the successfully compiled kernel funcs; their body now lives in the
  // embedded binary and the symbol is no longer needed.
  for (auto func : kernelFuncs)
    func.erase();

  llvm::errs() << "[hip-rocmlir-compiler] embedded " << compiled.binary.size()
               << "-byte binary into " << stamped << " hip.rocmlir op(s); "
               << "grid_size=" << compiled.gridSize
               << " block_size=" << compiled.blockSize << "; deleted "
               << kernelFuncs.size() << " kernel func(s)\n";

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
