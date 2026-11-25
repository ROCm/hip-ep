/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Benchmarked perfConfig selection for the rocMLIR offload path.
//
// Without this, `compileAndEmbedRocMlirKernels` falls back to
// `compileRocMlirDefaultPerfConfig`, which takes the first config of the
// tuning space that compiles, moving past (with a warning) ones rocMLIR
// rejects as not applicable. That is not an arbitrary pick -- rocMLIR orders
// the space so the first *conservatively applicable* config leads, and
// prepends a safe default when no table entry qualifies -- but it is chosen
// for LDS-budget safety rather than speed, so a problem the per-arch table
// does not cover can land well off the best config.
//
// This compiles every config in the space, times each on the real GPU, and
// keeps the fastest. It used to live in hip-rocmlir-compiler's main(), which
// meant the EP could never reach it; it lives here for the same reason kernel
// compilation does, so the offline tool and the in-process CompilerDriver run
// the same code.
//
// HIP is resolved symbol-by-symbol at call time rather than linked: a
// load-time HIP import would stop hip-compiler starting on a machine with no
// driver, and LibHipCompiler is deliberately headers-only against hip::host.
// Construction therefore does not throw or abort when no runtime or device is
// present; it reports that through `isUsable`, and whether to fall back to the
// default perfConfig or refuse the compile is the caller's decision. Both
// callers today refuse, because the artifact records which mode was asked for
// and an untuned artifact must not be labelled tuned.
//
// Only built when ENABLE_ROCMLIRTRITON is on, which is also the macro
// consumers gate on.

#ifndef HIP_COMPILER_ROCMLIRAUTOTUNE_H
#define HIP_COMPILER_ROCMLIRAUTOTUNE_H

#include "hip/Compiler/RocMlirKernelCompiler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace mlir {
namespace hip {

// Breadth of the perfConfig search. Mirrors rock::TuningParamSetKind without
// pulling a rocMLIR header into this interface; `Full` and `Exhaustive` are
// supersets of `Quick` by rocMLIR's own contract. A tuning type skips the
// shipped problem-cache LUT. An external --enable-problem-cache-rw file still
// takes priority.
enum class AutotuneSpace { Quick, Full, Exhaustive };

// Parses "quick", "full", or "exhaustive". Returns false and leaves \p out
// alone on anything else.
bool parseAutotuneSpace(StringRef name, AutotuneSpace &out);

const char *autotuneSpaceName(AutotuneSpace space);

// How the candidates of one search are timed. `Adaptive` is MIGraphX's
// two-pass selection (see AdaptiveBenchmark.h): a short coarse pass over
// every candidate, then bundled launches for the few that can still win.
// `Simple` times every candidate with `warmupRuns` and `measuredRuns`; it
// costs time proportional to the slowest candidates and is kept for A/B
// comparison.
enum class AutotuneBenchmark { Adaptive, Simple };

// Parses "adaptive" or "simple". Returns false and leaves \p out alone on
// anything else.
bool parseAutotuneBenchmark(StringRef name, AutotuneBenchmark &out);

const char *autotuneBenchmarkName(AutotuneBenchmark benchmark);

struct AutotuneOptions {
  AutotuneSpace space = AutotuneSpace::Quick;
  AutotuneBenchmark benchmark = AutotuneBenchmark::Adaptive;
  // `Simple` only. Untimed launches before the timed ones, discarding
  // clock/cache warmup. These default high enough that run-to-run noise stays
  // below the effect: at 5 and 20 the same kernel measured 4.2e-02 to
  // 7.5e-02 ms, wider than most of the wins being searched for. `Adaptive`
  // gets the same stability from bundling launches instead.
  unsigned warmupRuns = 20;
  unsigned measuredRuns = 100;
  // Where the compile-pool progress lines go. Null keeps the search silent,
  // which is what the EP wants: its compiles happen inside ORT session
  // creation.
  llvm::raw_ostream *log = nullptr;
  // Owning: the autotuner keeps a copy of these options for the whole compile
  // and logs through it long after the constructor returns, so a StringRef
  // here would dangle for any caller that did not pass a literal.
  std::string logPrefix = "[rocmlir-autotune]";
  // True when the user chose the search space explicitly: --autotune=<space>
  // on hip-rocmlir-compiler, or a non-empty HIPDNN_EP_ROCMLIR_AUTOTUNE in the
  // EP. A provided tuning type skips the shipped LUT. With it false (the
  // default), the shipped LUT is consulted before the search.
  bool tuningTypeProvided = false;
  // Let a problem the table has not measured borrow the tile of its nearest
  // neighbour in the problem cache, shipped or external, instead of searching.
  // Off by default, so a table answers only for the exact problem strings it
  // holds and every other problem is searched. Set by
  // --enable-nearest-config on hip-rocmlir-compiler, or by
  // rocMlirNearestConfigFromEnv (HIPDNN_EP_ROCMLIR_NEAREST_CONFIG) in either
  // binary. It changes nothing when no problem cache is in use.
  bool nearestConfig = false;
  // Also print a line per candidate and the winning perfConfig. Separate from
  // `log` because the per-candidate lines are one per config per kernel --
  // useful when tuning by hand, far too much for a normal compile.
  bool verbose = false;
  // External rocMLIR LUT JSON, the document updateProblemCacheFile writes.
  // hip-rocmlir-compiler sets this from --enable-problem-cache-rw, and the
  // provider from HIPDNN_EP_ROCMLIR_PROBLEM_CACHE_RW. It is always JSON: an
  // RCAL FlatBuffer, the format of the shipped tables, is refused. When set,
  // this file takes priority over the search and the shipped table is not
  // consulted. When empty, the shipped LUT is consulted unless
  // tuningTypeProvided.
  std::string problemCachePath;
  // Write this compile's benchmarked winners back into `problemCachePath`
  // itself. Set by --enable-problem-cache-rw and the provider variable; with
  // it false the path is only read. The autotuner does not write the file
  // itself; the caller calls updateProblemCacheFile once the embed has
  // succeeded. A missing or empty file is a new table, not an error.
  bool problemCacheRewrite = false;
};

// Wall times accumulated by one autotuner. perfConfig compiles for the whole
// model share a pool, so that figure is the pool wait, not the sum of CPU
// times. Benchmarks are serial, so theirs is real elapsed GPU time.
struct AutotunePhaseTimes {
  double perfConfigCompileMs = 0;
  double benchmarkMs = 0;
};

// Holds one compile's search state: the per-problem winner cache, the queued
// searches, and the kernels waiting on a winner. Instance-scoped rather than
// process-global so concurrent sessions cannot consume each other's queues.
//
// Use it by handing `installInto` an embed options struct, then running
// `compileAndEmbedRocMlirKernels`. Must outlive that call.
class RocMlirAutotuner {
public:
  // \p arch is the target the candidates are built for; a mismatch against
  // the live device is reported and tuning continues, since the caller may be
  // cross-compiling deliberately.
  RocMlirAutotuner(StringRef arch, const AutotuneOptions &options);
  ~RocMlirAutotuner();

  RocMlirAutotuner(const RocMlirAutotuner &) = delete;
  RocMlirAutotuner &operator=(const RocMlirAutotuner &) = delete;

  // False when the HIP runtime or a device could not be reached, in which case
  // nothing was installed and the caller should keep the default path.
  bool isUsable() const;

  // Point \p opts at this autotuner. No-op when !isUsable().
  void installInto(RocMlirEmbedOptions &opts);

  const AutotunePhaseTimes &phaseTimes() const;

  // Call before the compile when options.problemCacheRewrite is set. Checks
  // that updateProblemCacheFile will be able to rewrite
  // AutotuneOptions::problemCachePath, and loads it as the lookup table. A
  // missing or blank file is a new, empty table. False, with \p error set and
  // the table left empty, for a file the rewrite would refuse: not a LUT, for
  // another gpu_arch, malformed, an RCAL FlatBuffer, or unreadable. Finding
  // that here, rather than after the search, saves the search. True, doing
  // nothing, without the rewrite option.
  bool checkProblemCacheFile(std::string &error);

  // Merges this compile's search winners into AutotuneOptions::problemCachePath
  // and replaces that file, so the next compile (this tool or another run)
  // reads them back. A no-op returning true unless options.problemCacheRewrite
  // is set. Only winners of a search are written: a perfConfig borrowed from
  // the cache, or from the shipped LUT, is not a measurement and never goes
  // into the file, and neither does any problem the search did not run for.
  // The file is read again at write time, so winners another compile added in
  // between are kept, and every point already in it is kept, including ones
  // this build cannot use. Replaced atomically via a temporary file. With no
  // winners an existing file is left as it is. False, with the file untouched,
  // when it cannot be read, is not a LUT, is for another gpu_arch, is an RCAL
  // FlatBuffer, or cannot be written.
  bool updateProblemCacheFile() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

// HIP_ROCMLIR_SKIP_BENCHMARKING, shared by hip-rocmlir-compiler and the ONNX
// Runtime provider. Unset or "0" means benchmark. A present empty string
// skips, and so does any other value. The search then keeps rocMLIR's
// default perfConfig.
bool rocMlirSkipBenchmarking();

// Reads HIPDNN_EP_ROCMLIR_AUTOTUNE for the search space only. Unset or empty
// selects quick. "quick", "full", and "exhaustive" are the only accepted
// values. Anything else warns and selects quick. A tuning type skips the
// shipped problem-cache LUT. An external --enable-problem-cache-rw file still
// takes priority. Skipping the search is rocMlirSkipBenchmarking(), not a
// value of this variable.
// Returns false, leaving \p out alone, only when that skip flag is set.
//
// Env-only on purpose for now. The custom-kernel autotuners also accept a
// provider option, but those are read in the Runtime, which already has the
// option map; CompilationOptions carries no rocMLIR autotune field, and adding
// one means touching the compilation schema. Worth doing when a caller needs
// per-session control -- an env var is process-wide.
bool rocMlirAutotuneFromEnv(AutotuneSpace &out);

// Reads HIPDNN_EP_ROCMLIR_AUTOTUNE_BENCH ("adaptive" or "simple"). Leaves
// \p out alone when unset or unparseable.
void rocMlirAutotuneBenchmarkFromEnv(AutotuneBenchmark &out);

// Reads HIPDNN_EP_ROCMLIR_NEAREST_CONFIG, the shell-wide switch for
// AutotuneOptions::nearestConfig. Unset, empty, "0", "false", "off" and "no"
// leave nearest-config lookup off, which is the default; any other value turns
// it on. hip-rocmlir-compiler uses it as the starting value and
// --enable-nearest-config turns it on for that run.
bool rocMlirNearestConfigFromEnv();

} // namespace hip
} // namespace mlir

#endif // HIP_COMPILER_ROCMLIRAUTOTUNE_H
