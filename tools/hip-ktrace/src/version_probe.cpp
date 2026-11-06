/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <hip/amd_detail/hip_prof_str.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
using PfnApiName = const char *(*)(uint32_t);
using PfnGetVersion = int (*)(int *);

bool IsKnownName(const char *name) {
  return name && name[0] && std::strcmp(name, "unknown") != 0;
}

int FindRuntimeLast(PfnApiName api_name) {
  int last = 0;
  for (uint32_t id = 1; id < 4096; ++id) {
    if (IsKnownName(api_name(id)))
      last = static_cast<int>(id);
  }
  return last;
}

void PrintVersion(HMODULE hip, const char *symbol, const char *label) {
  const auto get_version =
      reinterpret_cast<PfnGetVersion>(GetProcAddress(hip, symbol));
  if (!get_version)
    return;

  int version = 0;
  if (get_version(&version) == 0)
    std::printf("%s: %d\n", label, version);
}
} // namespace

int main() {
  HMODULE hip = LoadLibraryA("amdhip64_7.dll");
  if (!hip) {
    std::fprintf(stderr, "ERROR: cannot load amdhip64_7.dll (%lu)\n",
                 GetLastError());
    return 1;
  }

  char path[MAX_PATH] = {};
  if (GetModuleFileNameA(hip, path, MAX_PATH))
    std::printf("HIP runtime: %s\n", path);

  PrintVersion(hip, "hipRuntimeGetVersion", "Runtime version");
  PrintVersion(hip, "hipDriverGetVersion", "Driver version");

  const auto api_name =
      reinterpret_cast<PfnApiName>(GetProcAddress(hip, "hipApiName"));
  if (!api_name) {
    std::fprintf(
        stderr,
        "ERROR: hipApiName is not exported; API table cannot be verified\n");
    FreeLibrary(hip);
    return 2;
  }

  const int runtime_last = FindRuntimeLast(api_name);
  const int headers_last = HIP_API_ID_LAST;
  const int overlap = std::min(runtime_last, headers_last);
  int compared = 0;
  int mismatches = 0;

  for (int id = HIP_API_ID_FIRST; id <= overlap; ++id) {
    const char *runtime_name = api_name(static_cast<uint32_t>(id));
    const char *header_name = hip_api_name(static_cast<uint32_t>(id));
    if (!IsKnownName(runtime_name) || !IsKnownName(header_name))
      continue;

    ++compared;
    if (std::strcmp(runtime_name, header_name) != 0) {
      if (mismatches < 10)
        std::printf("Mismatch at id %d: runtime='%s', headers='%s'\n", id,
                    runtime_name, header_name);
      ++mismatches;
    }
  }

  std::printf("API table: runtime last=%d, headers last=%d, compared=%d\n",
              runtime_last, headers_last, compared);

  FreeLibrary(hip);
  if (compared == 0 || mismatches != 0) {
    std::fprintf(stderr, "INCOMPATIBLE: %d API name mismatch(es)\n",
                 mismatches);
    return 3;
  }

  std::printf("COMPATIBLE: all overlapping API names are identical\n");
  return 0;
}
