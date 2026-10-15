/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Package entry point for standalone custom-kernel unit tests. The leaf
// executables remain separate so CI identifies the failing op/dtype directly;
// this program selects the matching GPU-arch directory, runs every leaf
// serially, and writes their combined report to bin/out/results.csv.

#include <hip/hip_runtime.h>

#include <windows.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Leaf {
  const char* op;
  const char* name;
  const char* executable;
};

static const Leaf kLeaves[] = {
    {"matmul_nbits", "Xfp16_Wu2_Yfp16",
     "hipdnn-kernel-ut-matmul-nbits-xfp16-wu2-yfp16.exe"},
    {"matmul_nbits", "Xfp16_Wu3_Yfp16",
     "hipdnn-kernel-ut-matmul-nbits-xfp16-wu3-yfp16.exe"},
    {"matmul_nbits", "Xfp16_Wu4_Yfp16",
     "hipdnn-kernel-ut-matmul-nbits-xfp16-wu4-yfp16.exe"},
    {"matmul_nbits", "Xfp16_Wi8_Yfp16",
     "hipdnn-kernel-ut-matmul-nbits-xfp16-wi8-yfp16.exe"},
    {"matmul_nbits", "Xfp32_Wu2_Yfp32",
     "hipdnn-kernel-ut-matmul-nbits-xfp32-wu2-yfp32.exe"},
    {"matmul_nbits", "Xfp32_Wu3_Yfp32",
     "hipdnn-kernel-ut-matmul-nbits-xfp32-wu3-yfp32.exe"},
    {"matmul_nbits", "Xfp32_Wu4_Yfp32",
     "hipdnn-kernel-ut-matmul-nbits-xfp32-wu4-yfp32.exe"},
    {"matmul_nbits", "Xfp32_Wi8_Yfp32",
     "hipdnn-kernel-ut-matmul-nbits-xfp32-wi8-yfp32.exe"},
    {"gqa/decode", "Xfp16_Wfp16_Yfp16",
     "hipdnn-kernel-ut-gqa-decode-xfp16-wfp16-yfp16.exe"},
    {"gqa/decode", "Xfp16_Wi8_Yfp16",
     "hipdnn-kernel-ut-gqa-decode-xfp16-wi8-yfp16.exe"},
    {"gqa/prefill", "Xfp16_Wfp16_Yfp16",
     "hipdnn-kernel-ut-gqa-prefill-xfp16-wfp16-yfp16.exe"},
    {"gqa/prefill", "Xfp16_Wi8_Yfp16",
     "hipdnn-kernel-ut-gqa-prefill-xfp16-wi8-yfp16.exe"},
    {"qmoe_ragged_mm", "Xfp16_Wu4_Yfp16",
     "hipdnn-kernel-ut-qmoe-ragged-mm-xfp16-wu4-yfp16.exe"},
};

static std::string executableDirectory() {
  char path[MAX_PATH];
  const DWORD length = GetModuleFileNameA(nullptr, path, sizeof(path));
  if (length == 0 || length >= sizeof(path)) return {};
  return fs::path(path).parent_path().string();
}

static std::string detectedArch() {
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) return {};
  std::string arch(props.gcnArchName);
  const size_t suffix = arch.find(':');
  return arch.substr(0, suffix);
}

// kernel-tests/<arch> is named after the HIP_ARCHITECTURES entry, like
// custom_kernels_<arch>.dll, so the directory is picked by the rule the EP uses
// to pick the DLL (LlvmIrJit.cpp): the device ISA first, then its generic
// target. Mirrors genericTargetFor() in
// backend-mlir-compiler/custom-op-mlir/src/LlvmIrJit.cpp; keep them in sync.
struct GenericTarget {
  const char* generic;
  const char* members[8];
};

static const GenericTarget kGenericTargets[] = {
    {"gfx9-generic",
     {"gfx900", "gfx902", "gfx904", "gfx906", "gfx909", "gfx90c"}},
    {"gfx9-4-generic", {"gfx942", "gfx950"}},
    {"gfx10-1-generic", {"gfx1010", "gfx1011", "gfx1012", "gfx1013"}},
    {"gfx10-3-generic",
     {"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035",
      "gfx1036"}},
    {"gfx11-generic",
     {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
      "gfx1152", "gfx1153"}},
    {"gfx12-generic", {"gfx1200", "gfx1201"}},
};

static std::string genericTargetFor(const std::string& arch) {
  for (const GenericTarget& target : kGenericTargets) {
    for (const char* member : target.members) {
      if (member && arch == member) return target.generic;
    }
  }
  return {};
}

static fs::path leafDirectory(const fs::path& bin, const std::string& arch) {
  const fs::path exact = bin / "kernel-tests" / arch;
  if (fs::is_directory(exact)) return exact;
  const std::string generic = genericTargetFor(arch);
  if (generic.empty()) return exact;
  const fs::path generic_dir = bin / "kernel-tests" / generic;
  return fs::is_directory(generic_dir) ? generic_dir : exact;
}

static bool setEnvironment(const char* name, const std::string& value) {
  if (SetEnvironmentVariableA(name, value.c_str())) return true;
  std::fprintf(stderr, "failed to set %s: %lu\n", name, GetLastError());
  return false;
}

static HANDLE inheritableStdHandle(DWORD id) {
  const HANDLE handle = GetStdHandle(id);
  if (handle && handle != INVALID_HANDLE_VALUE)
    SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
  return handle;
}

// custom_kernels_<arch>.dll links its own static CRT, whose stderr stays on the
// handle the process started with, so a leaf cannot capture the kernels'
// diagnostics with freopen(). The leaf therefore starts with stderr on
// `stderr_log`, named in HIPDNN_KERNEL_UT_STDERR, and reads back what each case
// appends.
static bool runLeaf(const fs::path& executable, const fs::path& working_dir,
                    const fs::path& stderr_log, int coverage, DWORD& exit_code,
                    double& elapsed_ms) {
  std::string command = "\"" + executable.string() + "\" --coverage " +
                        std::to_string(coverage);
  std::vector<char> command_buffer(command.begin(), command.end());
  command_buffer.push_back('\0');

  exit_code = static_cast<DWORD>(-1);
  elapsed_ms = 0.0;
  SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
  const HANDLE log = CreateFileW(
      stderr_log.c_str(), FILE_APPEND_DATA,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &inherit,
      CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (log == INVALID_HANDLE_VALUE) {
    std::fprintf(stderr, "failed to create %s: %lu\n",
                 stderr_log.string().c_str(), GetLastError());
    return false;
  }
  if (!setEnvironment("HIPDNN_KERNEL_UT_STDERR", stderr_log.string())) {
    CloseHandle(log);
    return false;
  }

  STARTUPINFOA startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = inheritableStdHandle(STD_INPUT_HANDLE);
  startup.hStdOutput = inheritableStdHandle(STD_OUTPUT_HANDLE);
  startup.hStdError = log;
  PROCESS_INFORMATION process{};
  const auto start = std::chrono::steady_clock::now();
  const BOOL created = CreateProcessA(
      executable.string().c_str(), command_buffer.data(), nullptr, nullptr,
      TRUE, 0, nullptr, working_dir.string().c_str(), &startup, &process);
  CloseHandle(log);
  if (!created) {
    std::fprintf(stderr, "failed to launch %s: %lu\n", executable.string().c_str(),
                 GetLastError());
    return false;
  }

  WaitForSingleObject(process.hProcess, INFINITE);
  GetExitCodeProcess(process.hProcess, &exit_code);
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  elapsed_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();
  return true;
}

int main(int argc, char** argv) {
  std::string arch;
  std::string mode = "lookup";
  int coverage = 3;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--arch") == 0 && i + 1 < argc)
      arch = argv[++i];
    else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
      mode = argv[++i];
    else if (std::strcmp(argv[i], "--coverage") == 0 && i + 1 < argc)
      coverage = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("Usage: hipdnn-kernel-ut.exe [--arch gfxNNNN] "
                  "[--mode lookup|autotune] [--coverage 1|2|3]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
      return 2;
    }
  }

  if (mode != "lookup" && mode != "autotune") {
    std::fprintf(stderr, "--mode must be lookup or autotune, got %s\n",
                 mode.c_str());
    return 2;
  }
  if (coverage < 1 || coverage > 3) {
    std::fprintf(stderr, "--coverage must be 1, 2, or 3\n");
    return 2;
  }
  if (arch.empty()) arch = detectedArch();
  if (arch.empty()) {
    std::fprintf(stderr, "could not detect GPU architecture; pass --arch gfxNNNN\n");
    return 2;
  }

  const fs::path bin = executableDirectory();
  const fs::path leaf_dir = leafDirectory(bin, arch);
  if (!fs::is_directory(leaf_dir)) {
    std::fprintf(stderr, "no packaged kernel tests for %s at %s\n", arch.c_str(),
                 leaf_dir.string().c_str());
    return 2;
  }
  std::printf("Running %s kernel tests from %s\n", arch.c_str(),
              leaf_dir.string().c_str());
  std::fflush(stdout);

  const fs::path out_dir = bin / "out";
  fs::create_directories(out_dir);
  const fs::path csv_path = out_dir / "results.csv";
  {
    std::ofstream csv(csv_path, std::ios::trunc);
    csv << "op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n";
  }

  if (!setEnvironment("HIPDNN_RESULTS_CSV", "out\\results.csv") ||
      !setEnvironment("HIPDNN_RESULTS_ARCH", arch))
    return 1;

  // Windows searches System32 before a leaf's working directory, and the GPU
  // driver installs its own amdhip64_*.dll and amd_comgr*.dll there. Child
  // processes inherit this DLL directory (long-standing, though undocumented),
  // so the leaves load the HIP runtime from bin/ like this controller does.
  if (!SetDllDirectoryW(bin.wstring().c_str())) {
    const DWORD error = GetLastError();
    std::fprintf(stderr, "failed to add %s to the DLL search path: %lu\n",
                 bin.string().c_str(), error);
    return 1;
  }

  bool passed = true;
  for (const Leaf& leaf : kLeaves) {
    // Do not predict a table hit from the leaf name. Every leaf receives the
    // requested mode; the DLL resolver matches its actual request (dtype/bits,
    // shape, and arch) and falls through to autotune on a miss.
    const char* kernel_mode = mode == "lookup" ? "lookup" : "online";
    if (!setEnvironment("HIPDNN_RESULTS_OP", leaf.op) ||
        !setEnvironment("HIPDNN_RESULTS_LEAF", leaf.name) ||
        !setEnvironment("HIPDNN_RESULTS_MODE", mode) ||
        !setEnvironment("HIPDNN_KERNEL_UT_MODE", mode) ||
        !setEnvironment("HIPDNN_MATMUL_AUTOTUNE_MODE", kernel_mode) ||
        !setEnvironment("HIPDNN_GQA_AUTOTUNE_MODE", kernel_mode))
      return 1;

    DWORD exit_code = 0;
    double elapsed_ms = 0.0;
    const fs::path executable = leaf_dir / leaf.executable;
    const fs::path stderr_log =
        out_dir / (executable.stem().string() + ".stderr.log");
    const bool launched =
        fs::is_regular_file(executable) &&
        runLeaf(executable, bin, stderr_log, coverage, exit_code, elapsed_ms);
    const bool leaf_passed = launched && exit_code == 0;
    std::printf("[%s/%s] %s (%.1f ms)\n", leaf.op, leaf.name,
                leaf_passed ? "PASS" : "FAIL", elapsed_ms);
    std::error_code ignored;
    if (leaf_passed)
      fs::remove(stderr_log, ignored);
    else if (fs::exists(stderr_log, ignored))
      std::printf("  stderr: %s\n", stderr_log.string().c_str());
    std::fflush(stdout);

    // One summary row per leaf, alongside the per-case rows the leaf itself
    // appended. time_ms here is the leaf process's total wall time, the only
    // row in the file where that column is not a per-launch mean.
    std::ofstream csv(csv_path, std::ios::app);
    csv << leaf.op << ',' << leaf.name << ',' << arch << ',' << mode
        << ",suite,suite-total," << elapsed_ms << ",0,"
        << (leaf_passed ? "PASS" : "FAIL") << '\n';
    passed = passed && leaf_passed;
  }

  std::printf("Combined report: %s\n", csv_path.string().c_str());
  return passed ? 0 : 1;
}
