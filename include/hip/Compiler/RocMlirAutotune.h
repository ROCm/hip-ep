/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Benchmarked perfConfig selection for the rocMLIR offload path.
//
// Without this, `compileAndEmbedRocMlirKernels` falls back to
// `compileRocMlirDefaultPerfConfig`, which takes position 0 of the tuning
// space. That is not an arbitrary pick -- rocMLIR orders the space so the
// first *conservatively applicable* config leads, and prepends a safe default
// when no table entry qualifies -- but it is chosen for LDS-budget safety
// rather than speed, so a problem the per-arch table does not cover can land
// well off the best config.
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
// Construction fails softly when no runtime or device is present, and the
// caller then keeps the default perfConfig path.
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
// supersets of `Quick` by rocMLIR's own contract.
enum class AutotuneSpace { Quick, Full, Exhaustive };

// Parses "quick", "full", or "exhaustive". Returns false and leaves \p out
// alone on anything else.
bool parseAutotuneSpace(StringRef name, AutotuneSpace &out);

const char *autotuneSpaceName(AutotuneSpace space);

struct AutotuneOptions {
  AutotuneSpace space = AutotuneSpace::Quick;
  // Untimed launches before the timed ones, discarding clock/cache warmup.
  // The whole point of benchmarking is resolving differences of tens of
  // percent, so these default high enough that run-to-run noise stays below
  // the effect: at 5 and 20 the same kernel measured 4.2e-02 to 7.5e-02 ms,
  // wider than most of the wins being searched for.
  unsigned warmupRuns = 20;
  unsigned measuredRuns = 100;
  // Where the compile-pool progress lines go. Null keeps the search silent,
  // which is what the EP wants: its compiles happen inside ORT session
  // creation.
  llvm::raw_ostream *log = nullptr;
  StringRef logPrefix = "[rocmlir-autotune]";
  // Also print a line per candidate and the winning perfConfig. Separate from
  // `log` because the per-candidate lines are one per config per kernel --
  // useful when tuning by hand, far too much for a normal compile.
  bool verbose = false;
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

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

// Reads HIPDNN_EP_ROCMLIR_AUTOTUNE. Recognizes "quick", "full", "exhaustive",
// and "off"/"0"/"" for disabled; returns false when disabled or unparseable,
// leaving \p out alone.
//
// Env-only on purpose for now. The custom-kernel autotuners also accept a
// provider option, but those are read in the Runtime, which already has the
// option map; CompilationOptions carries no rocMLIR autotune field, and adding
// one means touching the compilation schema. Worth doing when a caller needs
// per-session control -- an env var is process-wide.
bool rocMlirAutotuneFromEnv(AutotuneSpace &out);

} // namespace hip
} // namespace mlir

#endif // HIP_COMPILER_ROCMLIRAUTOTUNE_H
