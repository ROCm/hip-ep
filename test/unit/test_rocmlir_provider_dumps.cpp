/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// HIPDNN_EP_ROCMLIR_DUMP_HIP, _DUMP_TOSA and _DUMP_PROBLEM_CACHE through
// CompilerDriver::compile, without a GPU: ROCK_ARCH names the target and
// HIP_ROCMLIR_SKIP_BENCHMARKING compiles every kernel with its default
// perfConfig. The driver is called several times in this one process because
// the dump counter is process-wide; a separate hip-compiler run per case would
// restart it at 0 and could not see two compiles tell their files apart.
//
//   - a compile without any dump variable writes nothing and leaves the
//     counter alone, so the first dump is number 0;
//   - one compile numbers all three dumps alike, and the skipped problem cache
//     still names its numbered file;
//   - repeated compiles write new files and leave the earlier ones in place;
//   - a dump that cannot be written is reported and does not fail the
//     compile, nor the other dumps;
//   - the problem cache is not written, and says why, when autotuning is
//     skipped.
//
// Writing the problem cache itself needs a device, so that file is covered by
// the numeric tests instead.

#include "compilation_options_generated.h"
#include "hip/Compiler/CompilerDriver.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <io.h>
#include <stdlib.h>
#define HIP_DUP _dup
#define HIP_DUP2 _dup2
#define HIP_CLOSE _close
#define HIP_FILENO _fileno
#else
#include <unistd.h>
#define HIP_DUP dup
#define HIP_DUP2 dup2
#define HIP_CLOSE close
#define HIP_FILENO fileno
#endif

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void check(bool cond, llvm::StringRef what) {
  if (cond) {
    llvm::outs() << "[ OK ] " << what << "\n";
    return;
  }
  llvm::errs() << "[FAIL] " << what << "\n";
  ++g_failures;
}

void setEnv(const char *name, const std::string &value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void unsetEnv(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

const char *const kDumpVars[] = {"HIPDNN_EP_ROCMLIR_DUMP_HIP",
                                 "HIPDNN_EP_ROCMLIR_DUMP_TOSA",
                                 "HIPDNN_EP_ROCMLIR_DUMP_PROBLEM_CACHE"};

void clearDumpVars() {
  for (const char *name : kDumpVars)
    unsetEnv(name);
}

// Everything written to stderr while alive. The driver reports through
// llvm::errs(), which is unbuffered and goes straight to file descriptor 2.
class StderrCapture {
public:
  explicit StderrCapture(const fs::path &file) : file_(file) {
    std::fflush(stderr);
    saved_ = HIP_DUP(HIP_FILENO(stderr));
    std::FILE *f = std::fopen(file_.string().c_str(), "w");
    if (f) {
      HIP_DUP2(HIP_FILENO(f), HIP_FILENO(stderr));
      std::fclose(f);
    }
  }
  std::string finish() {
    if (saved_ >= 0) {
      std::fflush(stderr);
      HIP_DUP2(saved_, HIP_FILENO(stderr));
      HIP_CLOSE(saved_);
      saved_ = -1;
    }
    std::ifstream in(file_);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
  }
  ~StderrCapture() { finish(); }

private:
  fs::path file_;
  int saved_ = -1;
};

// A 1x3x8x8 -> 1x16x8x8 convolution: one rocMLIR kernel.
constexpr const char *kConvModel = R"mlir(
module {
  func.func @main_graph(%arg0: tensor<1x3x8x8xf32> {onnx.name = "input_0"}) -> (tensor<1x16x8x8xf32> {onnx.name = "output_0"}) {
    %weights = "onnx.Constant"() {value = dense<1.000000e-02> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %0 = "onnx.Conv"(%arg0, %weights, %bias) {
      auto_pad = "NOTSET",
      dilations = [1, 1],
      group = 1 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1],
      onnx_node_name = "Conv_0"
    } : (tensor<1x3x8x8xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x8x8xf32>
    "onnx.Return"(%0) : (tensor<1x16x8x8xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
)mlir";

struct CompileResult {
  bool ok = false;
  std::string error;
  std::string log; // stderr of the compile
  bool outputWritten = false;
};

CompileResult compileOnce(const fs::path &scratch) {
  static int run = 0;
  const fs::path out = scratch / ("model." + std::to_string(run) + ".bc");
  StderrCapture capture(scratch / ("stderr." + std::to_string(run) + ".txt"));
  ++run;

  mlir::hip::CompilationOptionsT options;
  options.output_mode = mlir::hip::OutputMode::LLVM_IR;
  hip::compiler::CompilerDriver driver;
  CompileResult result;
  result.ok = driver.compile(kConvModel, out.string(), options, result.error);
  result.log = capture.finish();
  result.outputWritten = fs::exists(out);
  return result;
}

// The numbers N of the files <dir>/<stem>.N<ext>.
std::set<unsigned> numbersOf(const fs::path &dir, const std::string &stem,
                             const std::string &ext) {
  std::set<unsigned> numbers;
  std::error_code ec;
  if (!fs::is_directory(dir, ec))
    return numbers;
  const std::string prefix = stem + ".";
  for (const fs::directory_entry &entry : fs::directory_iterator(dir, ec)) {
    const std::string name = entry.path().filename().string();
    if (name.size() <= prefix.size() + ext.size() ||
        name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
      continue;
    const std::string mid =
        name.substr(prefix.size(), name.size() - prefix.size() - ext.size());
    if (mid.empty() || mid.find_first_not_of("0123456789") != std::string::npos)
      continue;
    numbers.insert(static_cast<unsigned>(std::stoul(mid)));
  }
  return numbers;
}

bool contains(const std::string &text, const std::string &needle) {
  return text.find(needle) != std::string::npos;
}

std::string slurp(const fs::path &file) {
  std::ifstream in(file);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

} // namespace

int main() {
  const fs::path root =
      fs::temp_directory_path() /
      ("hip-rocmlir-provider-dumps-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root, ec);
  const fs::path dumps = root / "dumps";
  const fs::path scratch = root / "scratch";
  fs::create_directories(scratch, ec);

  // GPU-free: a fixed target, and default perfConfigs instead of autotuning.
  setEnv("HIPDNN_EP_ROCMLIR", "1");
  setEnv("ROCK_ARCH", "gfx1151");
  setEnv("HIP_ROCMLIR_SKIP_BENCHMARKING", "1");
  clearDumpVars();

  // No dump variable: no files, and no number taken.
  CompileResult plain = compileOnce(scratch);
  if (!plain.ok) {
    llvm::errs() << "baseline compile failed: " << plain.error << "\n";
    return 1;
  }
  check(plain.outputWritten, "baseline: the compile writes its output");
  check(!fs::exists(dumps), "no dump variable: nothing is dumped");
  check(!contains(plain.log, "[CompilerDriver] wrote") &&
            !contains(plain.log, "needs rocMLIR autotune"),
        "no dump variable: the driver reports no dump");

  // One compile, all three dumps, one number. The problem cache is skipped
  // (no autotuner), but its message names the file it would have written.
  setEnv("HIPDNN_EP_ROCMLIR_DUMP_HIP", (dumps / "hip.mlir").string());
  setEnv("HIPDNN_EP_ROCMLIR_DUMP_TOSA", (dumps / "tosa.mlir").string());
  setEnv("HIPDNN_EP_ROCMLIR_DUMP_PROBLEM_CACHE",
         (dumps / "cache.json").string());
  CompileResult first = compileOnce(scratch);
  check(first.ok, "all three dumps: the compile succeeds");
  const std::set<unsigned> hip1 = numbersOf(dumps, "hip", ".mlir");
  const std::set<unsigned> tosa1 = numbersOf(dumps, "tosa", ".mlir");
  check(hip1.size() == 1 && tosa1.size() == 1,
        "all three dumps: one hip and one tosa file");
  check(hip1 == tosa1, "all three dumps: hip and tosa share a number");
  check(!hip1.empty() && *hip1.begin() == 0,
        "the first dump is number 0, so dump-free compiles took no number");
  check(!hip1.empty() &&
            contains(first.log,
                     "cache." + std::to_string(*hip1.begin()) + ".json"),
        "all three dumps: the problem cache message carries the same number");
  check(fs::exists(dumps / "hip.mlir") == false &&
            fs::exists(dumps / "tosa.mlir") == false,
        "the unnumbered paths are never written");
  if (!hip1.empty()) {
    const std::string hipText =
        slurp(dumps / ("hip." + std::to_string(*hip1.begin()) + ".mlir"));
    const std::string tosaText =
        slurp(dumps / ("tosa." + std::to_string(*hip1.begin()) + ".mlir"));
    check(contains(hipText, "hip.rocmlir") || contains(hipText, "hip.conv"),
          "the hip dump holds the post-fusion hip IR");
    check(contains(tosaText, "tosa."), "the tosa dump holds tosa ops");
  }

  // Skip case: HIP_ROCMLIR_SKIP_BENCHMARKING means no autotuner, so no cache.
  check(numbersOf(dumps, "cache", ".json").empty() &&
            !fs::exists(dumps / "cache.json"),
        "skipped autotune: no problem cache file is written");
  check(contains(first.log,
                 "HIPDNN_EP_ROCMLIR_DUMP_PROBLEM_CACHE needs rocMLIR "
                 "autotune") &&
            contains(first.log, "HIP_ROCMLIR_SKIP_BENCHMARKING"),
        "skipped autotune: the driver says why there is no cache");
  check(!contains(first.log, "wrote problem cache"),
        "skipped autotune: no 'wrote problem cache' claim");

  // Repeated compiles: new numbers, earlier files kept.
  CompileResult second = compileOnce(scratch);
  CompileResult third = compileOnce(scratch);
  check(second.ok && third.ok, "repeated compiles succeed");
  const std::set<unsigned> hipAll = numbersOf(dumps, "hip", ".mlir");
  const std::set<unsigned> tosaAll = numbersOf(dumps, "tosa", ".mlir");
  check(hipAll.size() == 3, "three compiles write three distinct hip files");
  check(hipAll == tosaAll, "each compile's hip and tosa files share a number");
  check(hipAll.count(0) == 1 && hipAll.count(1) == 1 && hipAll.count(2) == 1,
        "the numbers count up from 0");
  check(contains(second.log, "cache.1.json") &&
            contains(third.log, "cache.2.json"),
        "each compile's cache message names its own number");
  check(contains(second.log, "hip.1.mlir") &&
            !contains(second.log, "hip.0.mlir"),
        "a later compile does not report an earlier file");

  // A dump that cannot be written is reported and nothing else is lost. The
  // hip dump's parent is a regular file, so neither the directory nor the
  // file can be created; the tosa dump still goes through.
  const fs::path blocker = root / "blocker";
  std::ofstream(blocker) << "not a directory";
  setEnv("HIPDNN_EP_ROCMLIR_DUMP_HIP", (blocker / "sub" / "hip.mlir").string());
  const std::set<unsigned> tosaBefore = numbersOf(dumps, "tosa", ".mlir");
  CompileResult hipFails = compileOnce(scratch);
  check(
      hipFails.ok && hipFails.outputWritten,
      "unwritable hip dump: the compile still succeeds and writes its output");
  check(contains(hipFails.log, "cannot open") &&
            contains(hipFails.log, "hip MLIR"),
        "unwritable hip dump: the failure is reported");
  check(numbersOf(dumps, "tosa", ".mlir").size() == tosaBefore.size() + 1,
        "unwritable hip dump: the tosa dump is still written");

  // Both module dumps unwritable.
  setEnv("HIPDNN_EP_ROCMLIR_DUMP_TOSA",
         (blocker / "sub" / "tosa.mlir").string());
  CompileResult bothFail = compileOnce(scratch);
  check(bothFail.ok && bothFail.outputWritten,
        "unwritable hip and tosa dumps: the compile still succeeds");
  check(contains(bothFail.log, "hip MLIR") &&
            contains(bothFail.log, "tosa MLIR"),
        "unwritable hip and tosa dumps: both failures are reported");
  check(!contains(bothFail.log, "wrote hip MLIR") &&
            !contains(bothFail.log, "wrote tosa MLIR"),
        "unwritable hip and tosa dumps: nothing claims to have been written");

  // A failed dump does not release its number to a later one: the numbers
  // stay unique even across failures.
  setEnv("HIPDNN_EP_ROCMLIR_DUMP_HIP", (dumps / "hip.mlir").string());
  setEnv("HIPDNN_EP_ROCMLIR_DUMP_TOSA", (dumps / "tosa.mlir").string());
  const std::set<unsigned> hipBefore = numbersOf(dumps, "hip", ".mlir");
  CompileResult after = compileOnce(scratch);
  const std::set<unsigned> hipAfter = numbersOf(dumps, "hip", ".mlir");
  check(after.ok && hipAfter.size() == hipBefore.size() + 1 &&
            hipBefore.count(*hipAfter.rbegin()) == 0,
        "a dump after failed ones gets a number not used before");

  clearDumpVars();
  unsetEnv("HIP_ROCMLIR_SKIP_BENCHMARKING");
  unsetEnv("ROCK_ARCH");
  unsetEnv("HIPDNN_EP_ROCMLIR");
  fs::remove_all(root, ec);

  if (g_failures) {
    llvm::errs() << g_failures << " check(s) failed\n";
    return 1;
  }
  llvm::outs() << "rocmlir provider dumps passed\n";
  return 0;
}
