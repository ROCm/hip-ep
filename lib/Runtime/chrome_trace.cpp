/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "chrome_trace.h"

#include <cstdint>
#include <cstdio>

#ifdef _WIN32
#include <process.h> // _getpid
#else
#include <unistd.h> // getpid
#endif

namespace {

long current_pid() {
#ifdef _WIN32
  return (long)_getpid();
#else
  return (long)getpid();
#endif
}

// Escape a string for a JSON string literal. Kernel names/shapes are usually
// clean, but a stray quote/backslash would corrupt the whole trace.
std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out += (char)c;
      }
    }
  }
  return out;
}

// Insert `token` just before the file extension: "a/b.json" + ".x" ->
// "a/b.x.json". A '.' counts as the extension only in the final path component,
// so a dot in a parent directory doesn't mangle the path.
//
// Implemented with a hand-rolled reverse scan rather than
// std::string::find_last_of on purpose: modern MSVC lowers find_last_of to
// vectorized CRT intrinsics (__std_find_last_of_trivial_pos_1,
// __std_find_last_trivial_1) that the EP's LLVM_IR JIT cannot materialize, so
// their presence in runtime.bc breaks session init on the JIT artifact path
// (the NATIVE path links the CRT and is unaffected). A plain loop keeps the
// runtime bitcode free of those symbols.
std::string insert_before_ext(const std::string &path,
                              const std::string &token) {
  size_t sep = std::string::npos; // last '/' or '\\'
  size_t dot = std::string::npos; // last '.'
  for (size_t i = path.size(); i-- > 0;) {
    char c = path[i];
    if (dot == std::string::npos && c == '.')
      dot = i;
    if (sep == std::string::npos && (c == '/' || c == '\\'))
      sep = i;
    if (dot != std::string::npos && sep != std::string::npos)
      break;
  }
  bool has_ext =
      dot != std::string::npos && (sep == std::string::npos || dot > sep);
  return has_ext ? path.substr(0, dot) + token + path.substr(dot)
                 : path + token;
}

} // namespace

ChromeTrace::ChromeTrace() {
  char buf[64];
  snprintf(buf, sizeof(buf), "p%lu_%llx", (unsigned long)current_pid(),
           (unsigned long long)(uintptr_t)this);
  tag_ = buf;
}

void ChromeTrace::addOp(const std::string &name, const std::string &shape,
                        double gpuTsUs, double gpuDurUs, double cpuTsUs,
                        double cpuDurUs, int64_t bytes) {
  ops_.push_back({name, shape, gpuTsUs, gpuDurUs, cpuTsUs, cpuDurUs, bytes});
}

void ChromeTrace::addIoSpans(double epochAbsUs, double h2dMs, int64_t h2dBytes,
                             double computeMs, double d2hMs, int64_t d2hBytes) {
  double t = epochAbsUs;
  io_.push_back({"H2D", 2, t, h2dMs * 1000.0, h2dBytes});
  t += h2dMs * 1000.0;
  io_.push_back({"Compute", 4, t, computeMs * 1000.0, 0});
  t += computeMs * 1000.0;
  io_.push_back({"D2H", 3, t, d2hMs * 1000.0, d2hBytes});
}

void ChromeTrace::addComputeTotal(const std::string &name, double cpuTsUs,
                                  double cpuDurUs) {
  cpuTotals_.push_back({name, cpuTsUs, cpuDurUs});
}

void ChromeTrace::write(const std::string &basePath,
                        double rooflineGbps) const {
  if (basePath.empty())
    return;
  // Per-session file so concurrent EP sessions never clobber one shared file;
  // they share the absolute time axis, so merge_chrome_traces.py aligns them.
  std::string path = insert_before_ext(basePath, "." + tag_);
  FILE *f = fopen(path.c_str(), "w");
  if (!f)
    return;
  fputs("{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n", f);
  fprintf(f,
          "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,"
          "\"args\":{\"name\":\"hipdnn-ep session %s\"}},\n",
          tag_.c_str());
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":0,"
        "\"args\":{\"name\":\"CPU (wrapper)\"}},\n",
        f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":1,"
        "\"args\":{\"name\":\"GPU (stream)\"}},\n",
        f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":2,"
        "\"args\":{\"name\":\"H2D\"}},\n",
        f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":3,"
        "\"args\":{\"name\":\"D2H\"}},\n",
        f);
  fputs("{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":0,\"tid\":4,"
        "\"args\":{\"name\":\"Compute (phase)\"}}",
        f);
  for (const auto &s : io_) {
    fprintf(f,
            ",\n{\"name\":\"%s\",\"cat\":\"io\",\"ph\":\"X\",\"pid\":0,"
            "\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"bytes\":%lld}}",
            json_escape(s.name).c_str(), s.tid, s.tsUs, s.durUs,
            (long long)s.bytes);
  }
  for (const auto &e : ops_) {
    // Achieved bandwidth: bytes / (durUs * 1e-6) / 1e9 GB/s = bytes / durUs /
    // 1e3.
    double gbps =
        (e.bytes > 0 && e.durUs > 0) ? (double)e.bytes / 1e3 / e.durUs : 0.0;
    std::string name = json_escape(e.name), shape = json_escape(e.shape);
    // GPU span
    fprintf(f,
            ",\n{\"name\":\"%s\",\"cat\":\"gpu\",\"ph\":\"X\",\"pid\":0,"
            "\"tid\":1,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"shape\":\"%s\","
            "\"bytes\":%lld,\"GB/s\":%.1f,\"%%peak\":%.1f}}",
            name.c_str(), e.tsUs, e.durUs, shape.c_str(), (long long)e.bytes,
            gbps, rooflineGbps > 0 ? 100.0 * gbps / rooflineGbps : 0.0);
    // CPU span (host-side wrapper time: launch overhead + setup)
    fprintf(f,
            ",\n{\"name\":\"%s\",\"cat\":\"cpu\",\"ph\":\"X\",\"pid\":0,"
            "\"tid\":0,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"shape\":\"%s\"}}",
            name.c_str(), e.cpuTsUs, e.cpuDurUs, shape.c_str());
  }
  // Outer whole-Compute spans on the CPU (wrapper) track. Emitted last, but
  // Chrome/Perfetto orders by timestamp and nests by containment, so each of
  // these encloses that inference's per-op CPU spans; the uncovered width is
  // the bubble. cat "total" so it can be filtered/colored distinctly.
  for (const auto &c : cpuTotals_) {
    fprintf(f,
            ",\n{\"name\":\"%s\",\"cat\":\"total\",\"ph\":\"X\",\"pid\":0,"
            "\"tid\":0,\"ts\":%.3f,\"dur\":%.3f,\"args\":{}}",
            json_escape(c.name).c_str(), c.tsUs, c.durUs);
  }
  fputs("\n]}\n", f);
  fclose(f);
}
