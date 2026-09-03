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
  // source is borrowed, not copied, and must outlive the Module.
  Module(std::string name, const char *source, size_t source_size,
         std::vector<std::string> kernel_names);
  ~Module();

  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;

  // nullptr if compilation failed or kernel_name was not registered. Callers
  // must treat that as "RTC unavailable" and fall back to their AOT path.
  hipFunction_t getFunction(const std::string &kernel_name);

 private:
  void ensureLoaded();
  bool compile(std::vector<char> &code);

  const std::string name_;
  const char *const source_;
  const size_t source_size_;
  const std::vector<std::string> kernel_names_;

  std::once_flag load_once_;
  bool load_ok_ = false;
  hipModule_t module_ = nullptr;
  std::unordered_map<std::string, hipFunction_t> functions_;
  std::unordered_map<std::string, std::string> lowered_names_;
};

// Args are taken by value so the addresses handed to hipModuleLaunchKernel stay
// valid for the call; the driver requires an array of pointers to the actual
// arguments. Order and types have to track the __global__ signature by hand --
// nothing here is type-checked against it.
template <typename... Args>
hipError_t launchKernel(hipFunction_t fn, dim3 grid, dim3 block, size_t smem,
                        hipStream_t stream, Args... args) {
  void *argv[] = {static_cast<void *>(&args)...};
  return hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y,
                               block.z, static_cast<unsigned>(smem), stream,
                               argv, nullptr);
}

}  // namespace rtc
}  // namespace hipdnn_ep

// Dispatch one launch through hipRTC, falling back to the AOT kernel when the
// module is unavailable. The kernel expression goes last because it contains
// commas that would otherwise split the macro's argument list; the real launch
// arguments are parenthesised for the same reason.
//
//   HIPDNN_RTC_LAUNCH(fn_lookup, grid, block, 0, stream,
//                     (a, b, c), some_kernel<32, 1, true>);
#define HIPDNN_RTC_UNPAREN(...) __VA_ARGS__
#define HIPDNN_RTC_STR_(...) #__VA_ARGS__
#define HIPDNN_RTC_STR(...) HIPDNN_RTC_STR_(__VA_ARGS__)

// Cached per launch site: resolving a name expression costs a string
// construction and a hash probe, and these sites sit on the hot path.
#define HIPDNN_RTC_LAUNCH(LOOKUP, GRID, BLOCK, SMEM, STREAM, ARGS, ...)       \
  do {                                                                        \
    static hipFunction_t _rtc_fn = (LOOKUP)(HIPDNN_RTC_STR(__VA_ARGS__));     \
    if (_rtc_fn != nullptr) {                                                 \
      hipdnn_ep::rtc::launchKernel(_rtc_fn, GRID, BLOCK, SMEM, STREAM,        \
                                   HIPDNN_RTC_UNPAREN ARGS);                  \
    } else {                                                                  \
      hipLaunchKernelGGL(HIP_KERNEL_NAME(__VA_ARGS__), GRID, BLOCK, SMEM,     \
                         STREAM, HIPDNN_RTC_UNPAREN ARGS);                    \
    }                                                                         \
  } while (0)

#endif  // HIPDNN_EP_KERNELS_RTC_HIPRTC_MODULE_H
