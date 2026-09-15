/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_EXAMPLE_COMMON_CSV_WRITER_H
#define HIPDNN_EP_EXAMPLE_COMMON_CSV_WRITER_H

/* Shared out/results.csv writer + autotune-candidate log capture, used by every
 * kernel-UT leaf's test_<op>.cpp (allowed exception to the "5 files per leaf"
 * rule -- see example/README.md). One file, no framework: a row struct, a
 * tiny appending writer keyed off HIPDNN_RESULTS_* env vars (set by each
 * leaf's Makefile), and a helper that redirects stderr around one kernel call
 * to recover the [custom_kernels] autotune/LUT log lines every op already
 * prints, without touching any kernel .hip numeric code.
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <regex>
#include <string>
#include <vector>

namespace hipdnn_ep_test {

struct CsvRow {
  std::string shape;
  std::string config;
  double time_ms = 0.0;
  double gflops = 0.0;
  double gbps = 0.0;
  int is_best = 0;
  std::string lut_source;  // "exact" | "nearest" | "fallback" | "" (autotune rows)
  double rel_l2 = 0.0;
  bool has_verdict = false;
  std::string verdict;  // "PASS" | "FAIL"
};

// Columns: op,leaf,arch,mode,shape,config,time_ms,gflops,gbps,is_best,lut_source,relL2,verdict
class CsvWriter {
 public:
  CsvWriter() {
    const char* path = std::getenv("HIPDNN_RESULTS_CSV");
    if (!path || !path[0]) return;  // CSV writing is opt-in via the Makefile
    path_ = path;
    op_ = env_or("HIPDNN_RESULTS_OP", "unknown");
    leaf_ = env_or("HIPDNN_RESULTS_LEAF", "unknown");
    arch_ = env_or("HIPDNN_RESULTS_ARCH", "unknown");
    mode_ = env_or("HIPDNN_RESULTS_MODE", "unknown");

    bool need_header = true;
    if (FILE* probe = std::fopen(path_.c_str(), "rb")) {
      std::fseek(probe, 0, SEEK_END);
      need_header = std::ftell(probe) == 0;
      std::fclose(probe);
    }
    f_ = std::fopen(path_.c_str(), "a");
    if (f_ && need_header) {
      std::fprintf(f_,
                    "op,leaf,arch,mode,shape,config,time_ms,gflops,gbps,"
                    "is_best,lut_source,relL2,verdict\n");
      std::fflush(f_);
    }
  }

  ~CsvWriter() {
    if (f_) std::fclose(f_);
  }

  bool enabled() const { return f_ != nullptr; }

  void write(const CsvRow& r) {
    if (!f_) return;
    std::fprintf(
        f_, "%s,%s,%s,%s,%s,%s,%.4f,%.3f,%.3f,%d,%s,%.6e,%s\n", op_.c_str(),
        leaf_.c_str(), arch_.c_str(), mode_.c_str(), r.shape.c_str(),
        r.config.c_str(), r.time_ms, r.gflops, r.gbps, r.is_best,
        r.lut_source.c_str(), r.rel_l2, r.has_verdict ? r.verdict.c_str() : "");
    std::fflush(f_);
  }

 private:
  static std::string env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? v : dflt;
  }

  std::string path_, op_, leaf_, arch_, mode_;
  FILE* f_ = nullptr;
};

// Sets `env_var=1` for the duration of `fn`, redirecting stderr to `tmp_path`
// so the [custom_kernels] debug lines the op already prints (gated on that
// same env var) land in a file instead of the console, then returns them
// split by line. Restores stderr to the console and the env var to unset
// before returning. `tmp_path` should sit under the leaf's BUILD_DIR (out/)
// so `make clean` removes it; it is also deleted here on success.
inline std::vector<std::string> captureLogLines(
    const char* env_var, const std::string& tmp_path,
    const std::function<void()>& fn) {
  std::vector<std::string> lines;
#ifdef _WIN32
  _putenv_s(env_var, "1");
#else
  setenv(env_var, "1", 1);
#endif
  std::fflush(stderr);
  FILE* redirected = std::freopen(tmp_path.c_str(), "w", stderr);
  if (!redirected) {
#ifdef _WIN32
    _putenv_s(env_var, "");
#else
    unsetenv(env_var);
#endif
    return lines;
  }
  fn();
  std::fflush(stderr);
#ifdef _WIN32
  std::freopen("CON", "w", stderr);
  _putenv_s(env_var, "");
#else
  std::freopen("/dev/tty", "w", stderr);
  unsetenv(env_var);
#endif

  std::ifstream in(tmp_path);
  std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  in.close();
  std::remove(tmp_path.c_str());
  return lines;
}

// Parses "[custom_kernels]   <descriptor> : <float> ms" candidate lines --
// the convention every autotune sweep in this repo already logs one line per
// timed config to (matmul_nbits' `[custom_kernels]   config[N] ... : X ms`,
// GQA decode's `[custom_kernels]   cand ... : X ms`, GQA prefill's
// `[prefill-v5-tune] ... : X ms/call`). Deliberately prefix-agnostic (not
// anchored to "[custom_kernels]") since prefill logs under its own tag; the
// capture window this runs over is narrow (one wrapped kernel call) so a
// stray unrelated "...: 1.23 ms" line is not a real risk. Marks the fastest
// candidate is_best=1. Lines without that "descriptor : float ms" shape
// (summaries, cache-store notices, LUT hit/miss lines) are not candidates and
// are skipped.
inline std::vector<CsvRow> parseCandidateLines(
    const std::vector<std::string>& lines) {
  static const std::regex re(R"(^(.*?)\s*:\s*([0-9]+\.[0-9]+)\s*ms)");
  std::vector<CsvRow> rows;
  for (const std::string& line : lines) {
    std::smatch m;
    if (!std::regex_search(line, m, re)) continue;
    CsvRow row;
    row.config = m[1].str();
    row.time_ms = std::stod(m[2].str());
    rows.push_back(row);
  }
  if (!rows.empty()) {
    size_t best = 0;
    for (size_t i = 1; i < rows.size(); ++i)
      if (rows[i].time_ms < rows[best].time_ms) best = i;
    rows[best].is_best = 1;
  }
  return rows;
}

}  // namespace hipdnn_ep_test

#endif  // HIPDNN_EP_EXAMPLE_COMMON_CSV_WRITER_H
