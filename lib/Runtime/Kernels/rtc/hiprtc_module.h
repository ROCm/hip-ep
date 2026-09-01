/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPDNN_EP_KERNELS_RTC_HIPRTC_MODULE_H
#define HIPDNN_EP_KERNELS_RTC_HIPRTC_MODULE_H

#include <hip/hip_runtime.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_ep {
namespace rtc {

// True when HIPDNN_EP_RTC=1. Queried once per process.
bool enabled();

// One device translation unit compiled at runtime.
//
// Compilation is deferred to the first getFunction() call and happens at most
// once per process; callers may share a single instance across threads. A
// failed compile is sticky -- getFunction() then returns nullptr forever rather
// than retrying on every launch.
//
// Kernel names are the source-level expressions (for templates, the full
// "kernel<A,B>" spelling). They are resolved to mangled names via hipRTC, so
// callers never spell a mangled name themselves.
class Module {
 public:
  // source must outlive the Module; it is the embedded .h text, not a copy.
  Module(std::string name, const char *source, size_t source_size,
         std::vector<std::string> kernel_names);

  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;

  // nullptr if compilation failed or kernel_name was not registered.
  hipFunction_t getFunction(const std::string &kernel_name);

 private:
  void ensureLoaded();
  bool compileAndCache(std::vector<char> &code);
  bool loadFromCache(std::vector<char> &code);

  const std::string name_;
  const char *const source_;
  const size_t source_size_;
  const std::vector<std::string> kernel_names_;

  std::once_flag load_once_;
  bool load_ok_ = false;
  hipModule_t module_ = nullptr;
  // Source-level expression -> resolved device function.
  std::unordered_map<std::string, hipFunction_t> functions_;
  // Populated by the hipRTC path; empty when the code object came from cache,
  // in which case the mangled names are read back from the sidecar file.
  std::unordered_map<std::string, std::string> lowered_names_;
};

}  // namespace rtc
}  // namespace hipdnn_ep

#endif  // HIPDNN_EP_KERNELS_RTC_HIPRTC_MODULE_H
