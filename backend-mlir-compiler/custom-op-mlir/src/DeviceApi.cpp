/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define HIPDNN_EP_DEVICE_API_EXPORTS
#include "hip/device_api.h"

#ifdef HIPDNN_EP_LINK_HIP_HOST
#include "hip_custom_kernels.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

extern "C" void *hipdnn_ep_get_current_stream(void);

namespace {

using StateSlotCopyKernel = decltype(&hip_state_slot_copy);

static_assert(sizeof(hipdnn_ep_state_slot_desc) == 16);
static_assert(offsetof(hipdnn_ep_state_slot_desc, base) == 0);
static_assert(offsetof(hipdnn_ep_state_slot_desc, slot_bytes) == 8);

struct KernelLibrary {
#ifdef _WIN32
  HMODULE handle = nullptr;
#else
  void *handle = nullptr;
#endif
  StateSlotCopyKernel copy = nullptr;
};

std::mutex g_kernel_libraries_mutex;
std::unordered_map<std::string, KernelLibrary> g_kernel_libraries;

std::string moduleDirectory() {
#ifdef _WIN32
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&hipdnn_ep_device_api_version), &module))
    return {};
  char path[MAX_PATH] = {};
  const DWORD length = GetModuleFileNameA(module, path, sizeof(path));
  if (length == 0 || length >= sizeof(path))
    return {};
  std::string result(path, length);
  const size_t slash = result.find_last_of("\\/");
#else
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(&hipdnn_ep_device_api_version), &info) ||
      !info.dli_fname)
    return {};
  std::string result(info.dli_fname);
  const size_t slash = result.find_last_of('/');
#endif
  return slash == std::string::npos ? std::string{} : result.substr(0, slash);
}

std::string activeArchitecture() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    return {};
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, device) != hipSuccess)
    return {};
  std::string architecture(properties.gcnArchName);
  if (const size_t colon = architecture.find(':'); colon != std::string::npos)
    architecture.resize(colon);
  return architecture;
}

std::string genericArchitecture(const std::string &architecture) {
  static const std::vector<std::pair<std::vector<const char *>, const char *>>
      families = {
          {{"gfx900", "gfx902", "gfx904", "gfx906", "gfx909", "gfx90c"},
           "gfx9-generic"},
          {{"gfx942", "gfx950"}, "gfx9-4-generic"},
          {{"gfx1010", "gfx1011", "gfx1012", "gfx1013"}, "gfx10-1-generic"},
          {{"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035",
            "gfx1036"},
           "gfx10-3-generic"},
          {{"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
            "gfx1152", "gfx1153"},
           "gfx11-generic"},
          {{"gfx1200", "gfx1201"}, "gfx12-generic"},
      };
  for (const auto &[members, generic] : families) {
    if (std::find_if(members.begin(), members.end(), [&](const char *member) {
          return architecture == member;
        }) != members.end())
      return generic;
  }
  return {};
}

KernelLibrary loadKernelLibrary(const std::string &architecture) {
  const std::string directory = moduleDirectory();

#ifdef _WIN32
  const std::string basename = "custom_kernels_" + architecture + ".dll";
  const std::vector<std::string> paths =
      directory.empty()
          ? std::vector<std::string>{basename}
          : std::vector<std::string>{directory + "\\" + basename, basename};
  for (const std::string &path : paths) {
    HMODULE handle = LoadLibraryA(path.c_str());
    if (!handle)
      continue;
    auto copy = reinterpret_cast<StateSlotCopyKernel>(
        GetProcAddress(handle, "hip_state_slot_copy"));
    if (copy)
      return {handle, copy};
    FreeLibrary(handle);
  }
#else
  const std::string basename = "libcustom_kernels_" + architecture + ".so";
  const std::vector<std::string> paths =
      directory.empty()
          ? std::vector<std::string>{basename}
          : std::vector<std::string>{directory + "/" + basename, basename};
  for (const std::string &path : paths) {
    void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      continue;
    auto copy = reinterpret_cast<StateSlotCopyKernel>(
        dlsym(handle, "hip_state_slot_copy"));
    if (copy)
      return {handle, copy};
    dlclose(handle);
  }
#endif
  return {};
}

StateSlotCopyKernel resolveKernel() {
  const std::string architecture = activeArchitecture();
  if (architecture.empty())
    return nullptr;

  std::lock_guard<std::mutex> lock(g_kernel_libraries_mutex);
  if (const auto it = g_kernel_libraries.find(architecture);
      it != g_kernel_libraries.end())
    return it->second.copy;

  KernelLibrary library = loadKernelLibrary(architecture);
  if (!library.copy) {
    const std::string generic = genericArchitecture(architecture);
    if (!generic.empty())
      library = loadKernelLibrary(generic);
  }
  const StateSlotCopyKernel copy = library.copy;
  g_kernel_libraries.emplace(architecture, std::move(library));
  return copy;
}

bool validateDescriptors(const hipdnn_ep_state_slot_desc *descriptors,
                         int32_t count, int32_t source_slot,
                         int32_t destination_slot) {
  if (!descriptors || count <= 0 || count > 65535 || source_slot < 0 ||
      destination_slot < 0 || source_slot == destination_slot)
    return false;
  const uint64_t maximum_slot =
      static_cast<uint64_t>(std::max(source_slot, destination_slot));
  for (int32_t i = 0; i < count; ++i) {
    if (!descriptors[i].base || descriptors[i].slot_bytes == 0 ||
        maximum_slot + 1 >
            std::numeric_limits<uint64_t>::max() / descriptors[i].slot_bytes)
      return false;
  }
  return true;
}

} // namespace
#endif

extern "C" HIPDNN_EP_DEVICE_API uint32_t hipdnn_ep_device_api_version(void) {
  return HIPDNN_EP_DEVICE_API_VERSION;
}

extern "C" HIPDNN_EP_DEVICE_API int
hipdnn_ep_copy_state_slots(const hipdnn_ep_state_slot_desc *descriptors_device,
                           const hipdnn_ep_state_slot_desc *descriptors_host,
                           int32_t count, int32_t source_slot,
                           int32_t destination_slot) {
#ifdef HIPDNN_EP_LINK_HIP_HOST
  if (!descriptors_device ||
      !validateDescriptors(descriptors_host, count, source_slot,
                           destination_slot))
    return HIPDNN_EP_DEVICE_API_INVALID_ARGUMENT;

  void *stream = hipdnn_ep_get_current_stream();
  StateSlotCopyKernel kernel = resolveKernel();
  if (!stream || !kernel) {
    if (std::getenv("ORTGENAI_AMDGPU_TRACE_STATE_SLOT_COPY"))
      std::fprintf(
          stderr, "[HIP EP] batched state-slot copy unavailable: %s%s\n",
          stream ? "" : "no current provider stream",
          kernel ? ""
                 : (stream ? "kernel unavailable" : "; kernel unavailable"));
    return HIPDNN_EP_DEVICE_API_UNAVAILABLE;
  }

  const int launch_status =
      kernel(stream, descriptors_device, count, source_slot, destination_slot);
  if (std::getenv("ORTGENAI_AMDGPU_TRACE_STATE_SLOT_COPY"))
    std::fprintf(
        stderr, "[HIP EP] batched state-slot copy launch status=%d stream=%p\n",
        launch_status, stream);
  return launch_status == static_cast<int>(hipSuccess)
             ? HIPDNN_EP_DEVICE_API_SUCCESS
             : HIPDNN_EP_DEVICE_API_LAUNCH_ERROR;
#else
  (void)descriptors_device;
  (void)descriptors_host;
  (void)count;
  (void)source_slot;
  (void)destination_slot;
  return HIPDNN_EP_DEVICE_API_UNAVAILABLE;
#endif
}
