/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Diagnostic-only, process-local HIP allocation accounting. No on-disk
// patching. Install before constructing the model. Counts successful user
// allocations, including live allocations retained in EP pools; excludes
// driver-internal use.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <tlhelp32.h>
#include <unordered_map>
#include <vector>
using Alloc = int(__cdecl *)(void **, size_t);
using HostAlloc = int(__cdecl *)(void **, size_t, unsigned);
using Free = int(__cdecl *)(void *);
static Alloc real_alloc;
static HostAlloc real_host;
static Free real_free, real_host_free;
struct Entry {
  uint64_t bytes, host;
};
struct Snapshot {
  uint64_t device, host, peak, allocations, frees, unknown_frees, failures,
      live_count;
};
static std::mutex mutex;
static std::unordered_map<void *, Entry> entries;
static Snapshot stats{};
static void added(void *p, size_t bytes, bool host) {
  std::lock_guard<std::mutex> lock(mutex);
  entries[p] = {bytes, uint64_t(host)};
  (host ? stats.host : stats.device) += bytes;
  ++stats.allocations;
  if (stats.device + stats.host > stats.peak)
    stats.peak = stats.device + stats.host;
}
static void removed(void *p) {
  if (!p)
    return;
  std::lock_guard<std::mutex> lock(mutex);
  auto i = entries.find(p);
  if (i == entries.end()) {
    ++stats.unknown_frees;
    return;
  }
  (i->second.host ? stats.host : stats.device) -= i->second.bytes;
  entries.erase(i);
  ++stats.frees;
}
static void failed() {
  std::lock_guard<std::mutex> lock(mutex);
  ++stats.failures;
}
// Export HIP names for the ORC JIT's opt-in allocation-only search generator.
#pragma comment(linker, "/export:hipMalloc=tracked_alloc")
#pragma comment(linker, "/export:hipHostMalloc=tracked_host")
#pragma comment(linker, "/export:hipFree=tracked_free")
#pragma comment(linker, "/export:hipHostFree=tracked_host_free")
extern "C" __declspec(dllexport) int __cdecl tracked_alloc(void **p, size_t n) {
  int rc = real_alloc(p, n);
  if (!rc && *p)
    added(*p, n, false);
  else if (rc)
    failed();
  return rc;
}
extern "C" __declspec(dllexport) int __cdecl tracked_host(void **p, size_t n,
                                                          unsigned f) {
  int rc = real_host(p, n, f);
  if (!rc && *p)
    added(*p, n, true);
  else if (rc)
    failed();
  return rc;
}
extern "C" __declspec(dllexport) int __cdecl tracked_free(void *p) {
  int rc = real_free(p);
  if (!rc)
    removed(p);
  else
    failed();
  return rc;
}
extern "C" __declspec(dllexport) int __cdecl tracked_host_free(void *p) {
  int rc = real_host_free(p);
  if (!rc)
    removed(p);
  else
    failed();
  return rc;
}
extern "C" __declspec(dllexport) void snapshot(Snapshot *out, int reset) {
  std::lock_guard<std::mutex> lock(mutex);
  stats.live_count = entries.size();
  *out = stats;
  if (reset)
    stats.peak = stats.device + stats.host;
}
extern "C" __declspec(dllexport) size_t
    list_allocations(uint64_t *out, size_t count) {
  std::lock_guard<std::mutex> lock(mutex);
  size_t i = 0;
  for (auto &e : entries) {
    if (i < count) {
      out[3 * i] = uint64_t(e.first);
      out[3 * i + 1] = e.second.bytes;
      out[3 * i + 2] = e.second.host;
    }
    ++i;
  }
  return i;
}
extern "C" __declspec(dllexport) int install(HMODULE module, HMODULE hip) {
  real_alloc = (Alloc)GetProcAddress(hip, "hipMalloc");
  real_host = (HostAlloc)GetProcAddress(hip, "hipHostMalloc");
  real_free = (Free)GetProcAddress(hip, "hipFree");
  real_host_free = (Free)GetProcAddress(hip, "hipHostFree");
  if (!real_alloc || !real_host || !real_free || !real_host_free)
    return -1;
  auto base = (unsigned char *)module;
  auto dos = (IMAGE_DOS_HEADER *)base;
  if (dos->e_magic != IMAGE_DOS_SIGNATURE)
    return -2;
  auto nt = (IMAGE_NT_HEADERS64 *)(base + dos->e_lfanew);
  auto rva = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]
                 .VirtualAddress;
  const char *names[] = {"hipMalloc", "hipHostMalloc", "hipFree",
                         "hipHostFree"};
  void *wrappers[] = {(void *)tracked_alloc, (void *)tracked_host,
                      (void *)tracked_free, (void *)tracked_host_free};
  int patched = 0;
  for (auto d = rva ? (IMAGE_IMPORT_DESCRIPTOR *)(base + rva) : nullptr;
       d && d->Name; ++d) {
    if (!d->OriginalFirstThunk)
      continue;
    auto original = (IMAGE_THUNK_DATA64 *)(base + d->OriginalFirstThunk);
    auto iat = (IMAGE_THUNK_DATA64 *)(base + d->FirstThunk);
    for (size_t i = 0; original[i].u1.AddressOfData; ++i) {
      if (IMAGE_SNAP_BY_ORDINAL64(original[i].u1.Ordinal))
        continue;
      auto name = (IMAGE_IMPORT_BY_NAME *)(base + original[i].u1.AddressOfData);
      for (int k = 0; k < 4; ++k)
        if (!strcmp((char *)name->Name, names[k])) {
          DWORD old;
          if (!VirtualProtect(&iat[i].u1.Function, sizeof(uint64_t),
                              PAGE_READWRITE, &old))
            return -3;
          iat[i].u1.Function = (uint64_t)wrappers[k];
          DWORD ignored;
          VirtualProtect(&iat[i].u1.Function, sizeof(uint64_t), old, &ignored);
          ++patched;
        }
    }
  }
  // hipgpu.dll intentionally uses MSVC delay-loaded HIP imports.
  struct Delay {
    DWORD attributes, name, module, iat, names, bound, unload, timestamp;
  };
  auto delay_rva =
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT]
          .VirtualAddress;
  for (auto d = delay_rva ? (Delay *)(base + delay_rva) : nullptr; d && d->name;
       ++d) {
    if (!(d->attributes & 1))
      return -4;
    auto original = (IMAGE_THUNK_DATA64 *)(base + d->names);
    auto iat = (IMAGE_THUNK_DATA64 *)(base + d->iat);
    for (size_t i = 0; original[i].u1.AddressOfData; ++i) {
      if (IMAGE_SNAP_BY_ORDINAL64(original[i].u1.Ordinal))
        continue;
      auto name = (IMAGE_IMPORT_BY_NAME *)(base + original[i].u1.AddressOfData);
      for (int k = 0; k < 4; ++k)
        if (!strcmp((char *)name->Name, names[k])) {
          DWORD old;
          if (!VirtualProtect(&iat[i].u1.Function, sizeof(uint64_t),
                              PAGE_READWRITE, &old))
            return -3;
          iat[i].u1.Function = (uint64_t)wrappers[k];
          DWORD ignored;
          VirtualProtect(&iat[i].u1.Function, sizeof(uint64_t), old, &ignored);
          ++patched;
        }
    }
  }
  return patched;
}
