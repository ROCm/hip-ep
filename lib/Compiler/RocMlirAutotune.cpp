/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Benchmarked perfConfig selection. See RocMlirAutotune.h for why this lives
// in LibHipCompiler rather than in hip-rocmlir-compiler's main().

#include "hip/Compiler/RocMlirAutotune.h"

#include "hip/Compiler/RocMlirKernelCompiler.h"
#include "hip/InitAllPasses.h"
// hip_get_env, not std::getenv: this runs inside the static-CRT EP DLL, whose
// CRT cannot see env vars the host process sets afterwards.
#include "hip/debug_log.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/InitRocMLIRDialects.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

// After the MLIR/LLVM headers: the HIP headers pull in platform definitions
// that collide otherwise. Same ordering hip-rocmlir-compiler used.
#if defined(HIPDNN_EP_LINK_HIP_HOST) && defined(HIPDNN_HIP_RUNTIME_LIB)
#define HIPDNN_EP_HAVE_HIP_RUNTIME 1
#include <hip/hip_runtime.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace mlir {
namespace hip {

const char *autotuneSpaceName(AutotuneSpace space) {
  switch (space) {
  case AutotuneSpace::Quick:
    return "quick";
  case AutotuneSpace::Full:
    return "full";
  case AutotuneSpace::Exhaustive:
    return "exhaustive";
  }
  return "unknown";
}

bool parseAutotuneSpace(StringRef name, AutotuneSpace &out) {
  if (name == "quick")
    out = AutotuneSpace::Quick;
  else if (name == "full")
    out = AutotuneSpace::Full;
  else if (name == "exhaustive")
    out = AutotuneSpace::Exhaustive;
  else
    return false;
  return true;
}

bool rocMlirAutotuneFromEnv(AutotuneSpace &out) {
  // hip_get_env, not std::getenv: this runs inside the static-CRT EP DLL, and
  // std::getenv there cannot see env vars set by the host process -- the
  // override would be silently dropped. Same reasoning as resolveRocMlirArch.
  std::string mode = hip_get_env("HIPDNN_EP_ROCMLIR_AUTOTUNE");
  if (mode.empty() || mode == "0" || mode == "off")
    return false;
  if (parseAutotuneSpace(mode, out))
    return true;
  llvm::errs() << "warning: ignoring unknown HIPDNN_EP_ROCMLIR_AUTOTUNE='"
               << mode << "' (expected quick, full, exhaustive, or off)\n";
  return false;
}

#ifdef HIPDNN_EP_HAVE_HIP_RUNTIME

namespace {

#ifdef _WIN32
// Declared rather than including <windows.h>, whose macros collide with the
// MLIR/LLVM headers above. Same pattern as include/hip/debug_log.h and the
// single-symbol lookup in RocMlirKernelCompiler.cpp.
extern "C" __declspec(dllimport) void *__stdcall GetModuleHandleA(const char *);
extern "C" __declspec(dllimport) void *__stdcall LoadLibraryExA(const char *,
                                                                void *,
                                                                unsigned long);
extern "C" __declspec(dllimport) void *__stdcall GetProcAddress(void *,
                                                                const char *);
// From libloaderapi.h. System32 plus the application/AddDllDirectory set;
// notably excludes the working directory and PATH, which users can write to.
#define HIPDNN_LOADER_SEARCH_SYSTEM32 0x00000800
#define HIPDNN_LOADER_SEARCH_DEFAULT_DIRS 0x00001000
#endif

// LibHipCompiler is headers-only against hip::host on purpose: hip-compiler
// links it but only compiles, and a load-time HIP import stops it starting
// where no driver is installed. Benchmarking needs to allocate, launch, and
// time real kernels, so it needs ~18 functions rather than the one
// resolveRocMlirArch looks up. Resolving them at call time keeps the
// dependency out of the import table. Types and launch macros come from the
// HIP headers, which this library does have; only the calls need indirection.
void *hipRuntimeHandle() {
  static void *handle = []() -> void * {
#ifdef _WIN32
    // Already-loaded first: the EP runs inside a process that has HIP mapped.
    if (void *mod = GetModuleHandleA(HIPDNN_HIP_RUNTIME_LIB))
      return mod;
    // Restricted search: the name is unqualified, and the default order would
    // also trust the working directory and PATH.
    return LoadLibraryExA(HIPDNN_HIP_RUNTIME_LIB, nullptr,
                          HIPDNN_LOADER_SEARCH_SYSTEM32 |
                              HIPDNN_LOADER_SEARCH_DEFAULT_DIRS);
#else
    if (void *mod = dlopen(HIPDNN_HIP_RUNTIME_LIB, RTLD_LAZY | RTLD_NOLOAD))
      return mod;
    return dlopen(HIPDNN_HIP_RUNTIME_LIB, RTLD_LAZY);
#endif
  }();
  return handle;
}

void *hipRuntimeSymbol(const char *name) {
  void *mod = hipRuntimeHandle();
  if (!mod)
    return nullptr;
#ifdef _WIN32
  return GetProcAddress(mod, name);
#else
  return dlsym(mod, name);
#endif
}

#define HIPDNN_LOADER_STRINGIFY_(x) #x
#define HIPDNN_LOADER_STRINGIFY(x) HIPDNN_LOADER_STRINGIFY_(x)

// Each entry resolves the symbol the *header* names. Several HIP functions are
// #defined to a versioned export (hipGetDeviceProperties ->
// hipGetDevicePropertiesR0600), and stringifying through the macro picks up
// whichever spelling this HIP version actually exports.
// Return type and parameter list stay separate so the same entry can spell
// both a member declarator -- `ret (*name) params` -- and a cast target --
// `ret (*) params`. A single function-pointer type cannot do both.
#define HIPDNN_HIP_RUNTIME_FUNCTIONS(X)                                        \
  X(hipGetDevice, hipError_t, (int *))                                         \
  X(hipGetDeviceProperties, hipError_t, (hipDeviceProp_t *, int))              \
  X(hipGetLastError, hipError_t, ())                                           \
  X(hipGetErrorString, const char *, (hipError_t))                             \
  X(hipStreamCreate, hipError_t, (hipStream_t *))                              \
  X(hipStreamDestroy, hipError_t, (hipStream_t))                               \
  X(hipStreamSynchronize, hipError_t, (hipStream_t))                           \
  X(hipMalloc, hipError_t, (void **, size_t))                                  \
  X(hipFree, hipError_t, (void *))                                             \
  X(hipMemsetAsync, hipError_t, (void *, int, size_t, hipStream_t))            \
  X(hipModuleLoadData, hipError_t, (hipModule_t *, const void *))              \
  X(hipModuleUnload, hipError_t, (hipModule_t))                                \
  X(hipModuleGetFunction, hipError_t,                                          \
    (hipFunction_t *, hipModule_t, const char *))                              \
  X(hipModuleLaunchKernel, hipError_t,                                         \
    (hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned,          \
     unsigned, unsigned, hipStream_t, void **, void **))                       \
  X(hipEventCreate, hipError_t, (hipEvent_t *))                                \
  X(hipEventDestroy, hipError_t, (hipEvent_t))                                 \
  X(hipEventRecord, hipError_t, (hipEvent_t, hipStream_t))                     \
  X(hipEventElapsedTime, hipError_t, (float *, hipEvent_t, hipEvent_t))

// Resolved together on first use, all or nothing: a partial table would fail
// deep inside a benchmark loop instead of at the point where the caller can
// still fall back to the default perfConfig.
struct HipRuntime {
#define HIPDNN_DECLARE_FN(name, ret, params) ret(*name) params = nullptr;
  HIPDNN_HIP_RUNTIME_FUNCTIONS(HIPDNN_DECLARE_FN)
#undef HIPDNN_DECLARE_FN

  bool complete = false;
};

const HipRuntime &hipRuntime() {
  static HipRuntime table = [] {
    HipRuntime t;
    bool ok = hipRuntimeHandle() != nullptr;
#define HIPDNN_RESOLVE_FN(name, ret, params)                                   \
  if (ok) {                                                                    \
    t.name = reinterpret_cast<ret(*) params>(                                  \
        hipRuntimeSymbol(HIPDNN_LOADER_STRINGIFY(name)));                      \
    ok = t.name != nullptr;                                                    \
  }
    HIPDNN_HIP_RUNTIME_FUNCTIONS(HIPDNN_RESOLVE_FN)
#undef HIPDNN_RESOLVE_FN
    t.complete = ok;
    return t;
  }();
  return table;
}

rock::TuningParamSetKind toRockKind(AutotuneSpace space) {
  switch (space) {
  case AutotuneSpace::Quick:
    return rock::TuningParamSetKind::Quick;
  case AutotuneSpace::Full:
    return rock::TuningParamSetKind::Full;
  case AutotuneSpace::Exhaustive:
    return rock::TuningParamSetKind::Exhaustive;
  }
  return rock::TuningParamSetKind::Quick;
}

double millisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

bool getBufferSize(Type type, size_t &bytes) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return false;

  Type elementType = shaped.getElementType();
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

// Dialects the backend pipeline needs, built once. DialectRegistry is
// move-only; each worker context reads this registry while loading its own
// copy of the dialects. A throwaway context loads them on the calling thread
// first so dialect static init is not raced by the pool.
DialectRegistry &autotuneRegistry() {
  static DialectRegistry registry = [] {
    DialectRegistry reg;
    ::hip::compiler::registerAllDialects(reg);
    registerRocMLIRDialects(reg);
    return reg;
  }();
  return registry;
}

void ensureDialectsLoaded() {
  static bool loaded = [] {
    MLIRContext warmup(autotuneRegistry(), MLIRContext::Threading::DISABLED);
    warmup.loadAllAvailableDialects();
    return true;
  }();
  (void)loaded;
}

// One perfConfig compile. A private context is required: PassManager::run is
// not safe on two modules that share an MLIRContext. The post-high-level rock
// module is parsed from text so workers never touch the caller's IR.
bool compilePerfConfig(StringRef rockModule, StringRef arch,
                       StringRef perfConfig, CompiledKernel &out) {
  MLIRContext context(autotuneRegistry(), MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> parsed =
      parseSourceString<ModuleOp>(rockModule, parserConfig);
  if (!parsed)
    return false;
  ModuleOp parsedModule = *parsed;
  if (!rock::tuningSetStr(parsedModule, perfConfig))
    return false;
  return compileRocMlirBackend(parsedModule, arch, perfConfig, out);
}

struct CompiledCandidate {
  std::string perfConfig;
  CompiledKernel kernel;
  bool compiled = false;
};

// A kernel whose problem key already has a winner. Only its own symbol needs
// compiling; benchmarking happened on the first kernel with that key.
struct DeferredCacheHit {
  std::string kernelName;
  std::string rockModule;
  std::string perfConfig;
  std::string cacheKey;
  CompiledKernel kernel;
  bool compiled = false;
};

// One entry per kernel whose problem key has not been seen yet. Candidates are
// filled with perfConfig strings here and compiled together later.
struct DeferredSearch {
  std::string kernelName;
  std::string rockModule;
  std::string cacheKey;
  std::vector<CompiledCandidate> candidates;
};

// A later kernel whose problem is already queued for search. The winning
// perfConfig is not known until that search is benchmarked, so the compile
// waits for the winner pool.
struct PendingWinnerCompile {
  std::string kernelName;
  std::string rockModule;
  std::string cacheKey;
};

// Device buffers for one kernel's arguments, zero-filled. Sized from the func
// signature, so every argument must be statically shaped.
class AutotuneBuffers {
public:
  AutotuneBuffers() = default;
  AutotuneBuffers(const AutotuneBuffers &) = delete;
  AutotuneBuffers &operator=(const AutotuneBuffers &) = delete;

  ~AutotuneBuffers() {
    const HipRuntime &fns = hipRuntime();
    for (void *buffer : deviceBuffers)
      if (buffer)
        (void)fns.hipFree(buffer);
    if (stream)
      (void)fns.hipStreamDestroy(stream);
  }

  bool initialize(ModuleOp module, const AutotuneOptions &options);

  hipStream_t getStream() const { return stream; }
  std::vector<void *> &getDeviceBuffers() { return deviceBuffers; }

private:
  hipStream_t stream = nullptr;
  std::vector<void *> deviceBuffers;
};

} // namespace

// Holds one compile's search state. Instance-scoped rather than process-global
// so concurrent sessions cannot consume each other's queues.
struct RocMlirAutotuner::Impl {
  std::string arch;
  AutotuneOptions options;
  rock::TuningParamSetKind kind;
  AutotunePhaseTimes times;
  bool usable = false;

  // Winners for this compile, same role as MIGraphX's problem cache. The first
  // kernel with a given problem key is searched; every other kernel with that
  // key compiles only the winning perfConfig. The problem string carries the
  // chip and the conv/gemm shape but not which space produced the winner, so
  // the kind is part of the key. The binary is not reused: each outlined kernel
  // has its own symbol and must be compiled.
  llvm::StringMap<std::string> winnerCache;
  std::vector<DeferredSearch> searches;
  std::vector<DeferredCacheHit> cacheHits;
  std::vector<PendingWinnerCompile> pendingWinners;
  llvm::StringSet<> searchesInFlight;

  // RocMlirEmbedOptions holds function_refs, which do not own their callable.
  // The callbacks live here so they outlive installInto's caller frame.
  std::function<bool(ModuleOp, StringRef, CompiledKernel &)> compileOneFn;
  std::function<bool(llvm::StringMap<CompiledKernel> &)> finishFn;

  Impl(StringRef archIn, const AutotuneOptions &opts)
      : arch(archIn.str()), options(opts), kind(toRockKind(opts.space)) {}

  llvm::raw_ostream *log() const { return options.log; }
  // Per-candidate detail: one line per config per kernel.
  llvm::raw_ostream *detail() const {
    return options.verbose ? options.log : nullptr;
  }

  // Diagnostics always go to errs(): a failed benchmark changes which config
  // ships, so it is not something a silent caller should miss.
  bool reportError(hipError_t status, StringRef operation) const {
    if (status == hipSuccess)
      return true;
    llvm::errs() << "error: " << operation
                 << " failed: " << hipRuntime().hipGetErrorString(status)
                 << "\n";
    return false;
  }

  std::string cacheKeyFor(StringRef problem) const {
    std::string key;
    key.reserve(problem.size() + 16);
    key.append(autotuneSpaceName(options.space));
    key.push_back('\n');
    key.append(problem.data(), problem.size());
    return key;
  }

  // Width of the perfConfig compile pool. Every config is queued; this only
  // bounds how many rocMLIR backend compiles run at once. Unset uses one job
  // per hardware thread. GPU benchmarking stays serial either way.
  unsigned compileJobs(unsigned numConfigs) const {
    unsigned jobs = std::thread::hardware_concurrency();
    if (jobs == 0)
      jobs = 1;
    std::string env = hip_get_env("HIP_ROCMLIR_COMPILE_JOBS");
    if (!env.empty()) {
      unsigned parsed = 0;
      if (!StringRef(env).getAsInteger(10, parsed) && parsed > 0)
        jobs = parsed;
      else
        llvm::errs() << "warning: ignoring invalid HIP_ROCMLIR_COMPILE_JOBS='"
                     << env << "'\n";
    }
    return std::max(1u, std::min(jobs, numConfigs));
  }

  bool checkDevice();
  bool launch(hipFunction_t function, const CompiledKernel &kernel,
              AutotuneBuffers &buffers) const;
  bool benchmark(const CompiledKernel &kernel, StringRef kernelName,
                 AutotuneBuffers &buffers, double &milliseconds) const;
  // Times every compiled candidate and keeps the fastest. `numbered` drives
  // the "N/M" progress lines.
  bool pickWinner(std::vector<CompiledCandidate> &candidates,
                  StringRef kernelName, ModuleOp benchModule,
                  std::string &bestConfig, CompiledKernel &winner);
  bool autotuneKernel(ModuleOp module, StringRef kernelName,
                      CompiledKernel &out, bool deferCacheHits);
  bool compileAndBenchmarkSearches(llvm::StringMap<CompiledKernel> &byKernel);
  bool compileDeferredCacheHits(llvm::StringMap<CompiledKernel> &byKernel);
  bool finish(llvm::StringMap<CompiledKernel> &byKernel) {
    return compileAndBenchmarkSearches(byKernel) &&
           compileDeferredCacheHits(byKernel);
  }
};

namespace {

bool AutotuneBuffers::initialize(ModuleOp module,
                                 const AutotuneOptions &options) {
  const HipRuntime &fns = hipRuntime();
  auto funcs = module.getOps<func::FuncOp>();
  if (funcs.empty()) {
    llvm::errs() << "error: autotune module has no function\n";
    return false;
  }
  func::FuncOp func = *funcs.begin();
  llvm::SmallVector<Type> kernelArgTypes(func.getArgumentTypes());
  llvm::append_range(kernelArgTypes, func.getResultTypes());

  if (kernelArgTypes.empty()) {
    llvm::errs() << "error: autotune kernel has no buffer arguments\n";
    return false;
  }
  if (fns.hipStreamCreate(&stream) != hipSuccess) {
    llvm::errs() << "error: hipStreamCreate failed\n";
    return false;
  }

  for (Type type : kernelArgTypes) {
    size_t bytes = 0;
    if (!getBufferSize(type, bytes)) {
      llvm::errs() << "error: autotune requires statically-shaped "
                      "tensor/memref kernel arguments; unsupported type: "
                   << type << "\n";
      return false;
    }
    void *buffer = nullptr;
    if (hipError_t status = fns.hipMalloc(&buffer, bytes);
        status != hipSuccess) {
      llvm::errs() << "error: hipMalloc failed for " << bytes
                   << " bytes: " << fns.hipGetErrorString(status) << "\n";
      return false;
    }
    deviceBuffers.push_back(buffer);
    if (hipError_t status = fns.hipMemsetAsync(buffer, 0, bytes, stream);
        status != hipSuccess) {
      llvm::errs() << "error: hipMemsetAsync failed: "
                   << fns.hipGetErrorString(status) << "\n";
      return false;
    }
  }
  (void)options;
  // Named, like the failures above: this is where an async fault from the
  // memsets above actually surfaces, and reporting it as a bare false left the
  // caller with nothing but a generic deferred-compilation failure.
  if (hipError_t status = fns.hipStreamSynchronize(stream);
      status != hipSuccess) {
    llvm::errs() << "error: hipStreamSynchronize failed: "
                 << fns.hipGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

} // namespace

bool RocMlirAutotuner::Impl::checkDevice() {
  const HipRuntime &fns = hipRuntime();
  if (!fns.complete) {
    llvm::errs() << "warning: rocMLIR autotuning requested but the HIP runtime "
                    "could not be loaded; keeping the default perfConfig\n";
    return false;
  }
  int device = 0;
  hipDeviceProp_t properties{};
  if (!reportError(fns.hipGetDevice(&device), "hipGetDevice") ||
      !reportError(fns.hipGetDeviceProperties(&properties, device),
                   "hipGetDeviceProperties"))
    return false;
  StringRef requestedArch = StringRef(arch).split(':').first;
  StringRef deviceArch = StringRef(properties.gcnArchName).split(':').first;
  if (requestedArch != deviceArch)
    llvm::errs() << "warning: autotuning for " << requestedArch << " on device "
                 << deviceArch << "; compiled candidates may not load\n";
  return true;
}

bool RocMlirAutotuner::Impl::launch(hipFunction_t function,
                                    const CompiledKernel &kernel,
                                    AutotuneBuffers &buffers) const {
  const HipRuntime &fns = hipRuntime();
  std::vector<void *> &deviceBuffers = buffers.getDeviceBuffers();
  size_t kernargSize = deviceBuffers.size() * sizeof(void *);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, deviceBuffers.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargSize,
                    HIP_LAUNCH_PARAM_END};
  (void)fns.hipGetLastError();
  hipError_t status = fns.hipModuleLaunchKernel(
      function, static_cast<unsigned>(kernel.gridSize), 1, 1,
      static_cast<unsigned>(kernel.blockSize), 1, 1, 0, buffers.getStream(),
      nullptr, config);
  // Both statuses, unconditionally: `&&` would skip hipGetLastError whenever
  // the launch call itself reported an error, leaving that error sticky for
  // whatever HIP call runs next -- which is a different candidate's benchmark,
  // so one bad perfConfig would be blamed on its successor.
  const bool launchOk = reportError(status, "hipModuleLaunchKernel");
  const bool pendingOk = reportError(fns.hipGetLastError(), "kernel launch");
  return launchOk && pendingOk;
}

bool RocMlirAutotuner::Impl::benchmark(const CompiledKernel &kernel,
                                       StringRef kernelName,
                                       AutotuneBuffers &buffers,
                                       double &milliseconds) const {
  const HipRuntime &fns = hipRuntime();
  hipModule_t hipModule = nullptr;
  if (!reportError(fns.hipModuleLoadData(&hipModule, kernel.binary.data()),
                   "hipModuleLoadData"))
    return false;

  hipFunction_t function = nullptr;
  std::string kernelNameStorage = kernelName.str();
  if (!reportError(fns.hipModuleGetFunction(&function, hipModule,
                                            kernelNameStorage.c_str()),
                   "hipModuleGetFunction")) {
    (void)fns.hipModuleUnload(hipModule);
    return false;
  }

  bool ok = true;
  for (unsigned i = 0; ok && i < options.warmupRuns; ++i)
    ok = launch(function, kernel, buffers);
  if (ok)
    ok = reportError(fns.hipStreamSynchronize(buffers.getStream()),
                     "hipStreamSynchronize(warmup)");

  std::vector<hipEvent_t> starts(options.measuredRuns, nullptr);
  std::vector<hipEvent_t> stops(options.measuredRuns, nullptr);
  for (unsigned i = 0; ok && i < options.measuredRuns; ++i)
    ok = reportError(fns.hipEventCreate(&starts[i]), "hipEventCreate(start)") &&
         reportError(fns.hipEventCreate(&stops[i]), "hipEventCreate(stop)");
  for (unsigned i = 0; ok && i < options.measuredRuns; ++i)
    ok = reportError(fns.hipEventRecord(starts[i], buffers.getStream()),
                     "hipEventRecord(start)") &&
         launch(function, kernel, buffers) &&
         reportError(fns.hipEventRecord(stops[i], buffers.getStream()),
                     "hipEventRecord(stop)");
  if (ok)
    ok = reportError(fns.hipStreamSynchronize(buffers.getStream()),
                     "hipStreamSynchronize(benchmark)");

  std::vector<float> samples;
  for (unsigned i = 0; ok && i < options.measuredRuns; ++i) {
    float elapsed = 0.0f;
    ok = reportError(fns.hipEventElapsedTime(&elapsed, starts[i], stops[i]),
                     "hipEventElapsedTime");
    if (ok)
      samples.push_back(elapsed);
  }
  for (hipEvent_t event : starts)
    if (event)
      (void)fns.hipEventDestroy(event);
  for (hipEvent_t event : stops)
    if (event)
      (void)fns.hipEventDestroy(event);
  (void)fns.hipModuleUnload(hipModule);

  if (!ok || samples.empty())
    return false;
  // Interquartile mean: drops the tail a shared GPU introduces without
  // letting one lucky sample decide the winner.
  std::sort(samples.begin(), samples.end());
  size_t trim = samples.size() / 4;
  auto first = samples.begin() + trim;
  auto last = samples.end() - trim;
  milliseconds = std::accumulate(first, last, 0.0) / std::distance(first, last);
  return true;
}

bool RocMlirAutotuner::Impl::pickWinner(
    std::vector<CompiledCandidate> &candidates, StringRef kernelName,
    ModuleOp benchModule, std::string &bestConfig, CompiledKernel &winner) {
  // Benchmarking is only meaningful when this kernel has the GPU to itself.
  // Search state is per-instance so concurrent sessions can each tune, but two
  // instances timing candidates at once would measure each other's work and
  // could pick a slower winner. Held across the whole candidate loop, not per
  // launch: interleaving at a finer grain reintroduces the interference. The
  // compile pools stay outside it and still overlap.
  static std::mutex benchmarkMutex;
  std::lock_guard<std::mutex> benchmarkGuard(benchmarkMutex);

  AutotuneBuffers buffers;
  if (!buffers.initialize(benchModule, options))
    return false;

  const unsigned numConfigs = static_cast<unsigned>(candidates.size());
  double bestMilliseconds = std::numeric_limits<double>::infinity();
  unsigned compiled = 0;
  unsigned benchmarked = 0;
  const auto benchmarkStart = std::chrono::steady_clock::now();
  for (unsigned index = 0; index < numConfigs; ++index) {
    CompiledCandidate &candidate = candidates[index];
    if (!candidate.compiled) {
      if (detail())
        *detail() << options.logPrefix << " autotune " << (index + 1) << "/"
                  << numConfigs << ": compile failed\n";
      continue;
    }
    ++compiled;

    double elapsed = 0.0;
    if (!benchmark(candidate.kernel, kernelName, buffers, elapsed)) {
      if (detail())
        *detail() << options.logPrefix << " autotune " << (index + 1) << "/"
                  << numConfigs << ": benchmark failed\n";
      continue;
    }
    ++benchmarked;
    if (detail())
      *detail() << options.logPrefix << " autotune " << (index + 1) << "/"
                << numConfigs << ": " << elapsed << " ms  "
                << candidate.perfConfig << "\n";
    if (elapsed < bestMilliseconds) {
      bestMilliseconds = elapsed;
      bestConfig = candidate.perfConfig;
      winner = std::move(candidate.kernel);
    }
  }
  times.benchmarkMs += millisecondsSince(benchmarkStart);

  if (bestConfig.empty()) {
    llvm::errs() << "error: autotune found no runnable perfConfig for '"
                 << kernelName << "' (compiled " << compiled << ", benchmarked "
                 << benchmarked << ")\n";
    return false;
  }
  if (detail())
    *detail() << options.logPrefix << " autotune winner for '" << kernelName
              << "': " << bestMilliseconds << " ms  " << bestConfig << "\n";
  return true;
}

bool RocMlirAutotuner::Impl::autotuneKernel(ModuleOp module,
                                            StringRef kernelName,
                                            CompiledKernel &out,
                                            bool deferCacheHits) {
  std::string rockModule;
  {
    llvm::raw_string_ostream os(rockModule);
    module.print(os);
  }
  ensureDialectsLoaded();

  // Look up before building the search space. A known winner is compiled in
  // the second pool. A problem already queued in this compile waits for that
  // same pool once the search has been benchmarked. A later compile failure
  // drops the entry and searches, so a config that won for a different
  // epilogue cannot fail this kernel.
  llvm::SmallString<2048> problem;
  std::string cacheKey;
  if (succeeded(rock::getTuningProblemStr(module, problem))) {
    cacheKey = cacheKeyFor(problem);
    auto cached = winnerCache.find(cacheKey);
    if (cached != winnerCache.end()) {
      if (deferCacheHits) {
        DeferredCacheHit hit;
        hit.kernelName = kernelName.str();
        hit.rockModule = std::move(rockModule);
        hit.perfConfig = cached->getValue();
        hit.cacheKey = std::move(cacheKey);
        cacheHits.push_back(std::move(hit));
        return true;
      }
      const auto compileStart = std::chrono::steady_clock::now();
      bool ok = compilePerfConfig(rockModule, arch, cached->getValue(), out);
      times.perfConfigCompileMs += millisecondsSince(compileStart);
      if (ok)
        return true;
      llvm::errs() << "warning: cached perfConfig failed to compile for '"
                   << kernelName << "'; searching the "
                   << autotuneSpaceName(options.space) << " space\n";
      winnerCache.erase(cached);
    }
    if (deferCacheHits && searchesInFlight.count(cacheKey)) {
      PendingWinnerCompile pending;
      pending.kernelName = kernelName.str();
      pending.rockModule = std::move(rockModule);
      pending.cacheKey = std::move(cacheKey);
      pendingWinners.push_back(std::move(pending));
      return true;
    }
  }

  std::unique_ptr<rock::TuningParamSet> space(
      rock::createTunableParamSpace(module, kind));
  if (!space || space->tuningRange.empty()) {
    llvm::errs() << "error: autotune perfConfig search space is empty\n";
    return false;
  }

  const unsigned numConfigs = space->tuningRange.size();
  std::vector<CompiledCandidate> candidates(numConfigs);
  for (auto [index, tuningAttr] : llvm::enumerate(space->tuningRange)) {
    llvm::SmallString<1024> perfConfig;
    tuningAttr.getPerfConfigStr(perfConfig);
    candidates[index].perfConfig = perfConfig.str().str();
  }

  // The fallback path (deferCacheHits == false) compiles this kernel on its
  // own. The normal path queues every candidate and returns; one pool compiles
  // them with every other kernel after high-level lowering has finished.
  if (deferCacheHits) {
    DeferredSearch search;
    search.kernelName = kernelName.str();
    search.rockModule = std::move(rockModule);
    search.cacheKey = std::move(cacheKey);
    search.candidates = std::move(candidates);
    if (!search.cacheKey.empty())
      searchesInFlight.insert(search.cacheKey);
    searches.push_back(std::move(search));
    return true;
  }

  const unsigned jobs = compileJobs(numConfigs);
  if (log())
    *log() << options.logPrefix << " compiling " << numConfigs << " "
           << autotuneSpaceName(options.space) << " perfConfigs for '"
           << kernelName << "' on " << jobs << " threads\n";
  {
    // Each task owns a context. Nested MLIR/LLVM parallelism stays off inside
    // compileRocMlirBackend (rocMLIR pins llvm::parallel::strategy to 1
    // thread).
    const auto compileStart = std::chrono::steady_clock::now();
    llvm::DefaultThreadPool pool(llvm::hardware_concurrency(jobs));
    for (unsigned index = 0; index < numConfigs; ++index) {
      pool.async([&, index] {
        candidates[index].compiled =
            compilePerfConfig(rockModule, arch, candidates[index].perfConfig,
                              candidates[index].kernel);
      });
    }
    pool.wait();
    times.perfConfigCompileMs += millisecondsSince(compileStart);
  }

  std::string bestConfig;
  if (!pickWinner(candidates, kernelName, module, bestConfig, out))
    return false;
  if (!cacheKey.empty())
    winnerCache[cacheKey] = bestConfig;
  return true;
}

bool RocMlirAutotuner::Impl::compileAndBenchmarkSearches(
    llvm::StringMap<CompiledKernel> &byKernel) {
  std::vector<DeferredSearch> queued = std::move(searches);
  searches.clear();
  searchesInFlight.clear();

  if (!queued.empty()) {
    unsigned total = 0;
    for (const DeferredSearch &search : queued)
      total += static_cast<unsigned>(search.candidates.size());
    const unsigned jobs = compileJobs(total);
    if (log())
      *log() << options.logPrefix << " compiling " << total << " "
             << autotuneSpaceName(options.space) << " perfConfigs for "
             << queued.size() << " kernels on " << jobs << " threads\n";
    {
      const auto compileStart = std::chrono::steady_clock::now();
      llvm::DefaultThreadPool pool(llvm::hardware_concurrency(jobs));
      for (unsigned searchIndex = 0; searchIndex < queued.size();
           ++searchIndex) {
        const unsigned numCandidates =
            static_cast<unsigned>(queued[searchIndex].candidates.size());
        for (unsigned candidateIndex = 0; candidateIndex < numCandidates;
             ++candidateIndex) {
          pool.async([&, searchIndex, candidateIndex] {
            DeferredSearch &search = queued[searchIndex];
            CompiledCandidate &candidate = search.candidates[candidateIndex];
            candidate.compiled =
                compilePerfConfig(search.rockModule, arch, candidate.perfConfig,
                                  candidate.kernel);
          });
        }
      }
      pool.wait();
      times.perfConfigCompileMs += millisecondsSince(compileStart);
    }

    for (DeferredSearch &search : queued) {
      MLIRContext context(autotuneRegistry(), MLIRContext::Threading::DISABLED);
      context.loadAllAvailableDialects();
      ParserConfig parserConfig(&context);
      OwningOpRef<ModuleOp> parsed =
          parseSourceString<ModuleOp>(search.rockModule, parserConfig);
      if (!parsed) {
        llvm::errs() << "error: failed to reparse rock module for '"
                     << search.kernelName << "'\n";
        return false;
      }
      CompiledKernel winner;
      std::string bestConfig;
      if (!pickWinner(search.candidates, search.kernelName, *parsed, bestConfig,
                      winner))
        return false;
      if (!search.cacheKey.empty())
        winnerCache[search.cacheKey] = bestConfig;
      byKernel[search.kernelName] = std::move(winner);
      search.candidates.clear();
    }
  }

  std::vector<PendingWinnerCompile> pending = std::move(pendingWinners);
  pendingWinners.clear();
  for (PendingWinnerCompile &item : pending) {
    auto cached = winnerCache.find(item.cacheKey);
    if (cached == winnerCache.end()) {
      llvm::errs() << "error: no autotune winner for repeated problem on '"
                   << item.kernelName << "'\n";
      return false;
    }
    DeferredCacheHit hit;
    hit.kernelName = std::move(item.kernelName);
    hit.rockModule = std::move(item.rockModule);
    hit.perfConfig = cached->getValue();
    hit.cacheKey = std::move(item.cacheKey);
    cacheHits.push_back(std::move(hit));
  }
  return true;
}

// One pool for every kernel that reused a winner. Each task compiles that
// kernel's own symbol; benchmarking already happened on the first kernel with
// the key. A compile failure erases the winner and searches that kernel.
bool RocMlirAutotuner::Impl::compileDeferredCacheHits(
    llvm::StringMap<CompiledKernel> &byKernel) {
  std::vector<DeferredCacheHit> hits = std::move(cacheHits);
  cacheHits.clear();
  if (hits.empty())
    return true;

  const unsigned jobs = compileJobs(static_cast<unsigned>(hits.size()));
  if (log())
    *log() << options.logPrefix << " compiling " << hits.size() << " cached "
           << autotuneSpaceName(options.space) << " perfConfigs on " << jobs
           << " threads\n";
  {
    const auto compileStart = std::chrono::steady_clock::now();
    llvm::DefaultThreadPool pool(llvm::hardware_concurrency(jobs));
    for (unsigned index = 0; index < hits.size(); ++index) {
      pool.async([&, index] {
        hits[index].compiled =
            compilePerfConfig(hits[index].rockModule, arch,
                              hits[index].perfConfig, hits[index].kernel);
      });
    }
    pool.wait();
    times.perfConfigCompileMs += millisecondsSince(compileStart);
  }

  for (DeferredCacheHit &hit : hits) {
    if (hit.compiled) {
      byKernel[hit.kernelName] = std::move(hit.kernel);
      continue;
    }
    llvm::errs() << "warning: cached perfConfig failed to compile for '"
                 << hit.kernelName << "'; searching the "
                 << autotuneSpaceName(options.space) << " space\n";
    winnerCache.erase(hit.cacheKey);
    MLIRContext context(autotuneRegistry(), MLIRContext::Threading::DISABLED);
    context.loadAllAvailableDialects();
    ParserConfig parserConfig(&context);
    OwningOpRef<ModuleOp> parsed =
        parseSourceString<ModuleOp>(hit.rockModule, parserConfig);
    CompiledKernel winner;
    if (!parsed || !autotuneKernel(*parsed, hit.kernelName, winner,
                                   /*deferCacheHits=*/false))
      return false;
    byKernel[hit.kernelName] = std::move(winner);
  }
  return true;
}

RocMlirAutotuner::RocMlirAutotuner(StringRef arch,
                                   const AutotuneOptions &options)
    : impl(std::make_unique<Impl>(arch, options)) {
  impl->usable = impl->checkDevice();
}

RocMlirAutotuner::~RocMlirAutotuner() = default;

bool RocMlirAutotuner::isUsable() const { return impl->usable; }

const AutotunePhaseTimes &RocMlirAutotuner::phaseTimes() const {
  return impl->times;
}

void RocMlirAutotuner::installInto(RocMlirEmbedOptions &opts) {
  if (!impl->usable)
    return;
  // `compileOne` only queues each kernel; `finishCompiles` compiles every
  // search config in one pool, benchmarks them, then compiles the winning
  // perfConfig for every repeated problem in a second pool. Both callbacks
  // capture `impl`, which this object owns, so it must outlive the embed call.
  Impl *state = impl.get();
  state->compileOneFn = [state](ModuleOp single, StringRef name,
                                CompiledKernel &out) {
    return state->autotuneKernel(single, name, out, /*deferCacheHits=*/true);
  };
  state->finishFn = [state](llvm::StringMap<CompiledKernel> &byKernel) {
    return state->finish(byKernel);
  };
  opts.compileOne = state->compileOneFn;
  opts.finishCompiles = state->finishFn;
}

#else // no HIP runtime to benchmark against

struct RocMlirAutotuner::Impl {
  AutotunePhaseTimes times;
};

RocMlirAutotuner::RocMlirAutotuner(StringRef, const AutotuneOptions &)
    : impl(std::make_unique<Impl>()) {}
RocMlirAutotuner::~RocMlirAutotuner() = default;
bool RocMlirAutotuner::isUsable() const { return false; }
const AutotunePhaseTimes &RocMlirAutotuner::phaseTimes() const {
  return impl->times;
}
void RocMlirAutotuner::installInto(RocMlirEmbedOptions &) {}

#endif

} // namespace hip
} // namespace mlir
