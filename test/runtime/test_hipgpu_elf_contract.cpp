/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ORT dlcloses the EP when GetProvider is missing. -z nodelete keeps the DSO
// mapped across that close. The comgr loader pins libamd_comgr.so on its own,
// so a dlopen/dlclose cycle cannot reproduce the spirv-expand-step abort.
// Require DF_1_NODELETE, and check that the plugin entry points resolve.

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr uint64_t kDtFlags1 = 0x6ffffffb;
constexpr uint64_t kDf1Nodelete = 0x8;

uint64_t read_u64(const uint8_t *p) {
  uint64_t value = 0;
  std::memcpy(&value, p, sizeof(value));
  return value;
}

uint16_t read_u16(const uint8_t *p) {
  uint16_t value = 0;
  std::memcpy(&value, p, sizeof(value));
  return value;
}

bool has_nodelete(const char *path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<uint8_t> file(std::istreambuf_iterator<char>(in), {});
  if (file.size() < 64 || file[0] != 0x7f || file[1] != 'E' || file[2] != 'L' ||
      file[3] != 'F' || file[4] != 2 || file[5] != 1) {
    std::cerr << path << " is not an ELF64 LE object\n";
    return false;
  }

  const uint64_t phoff = read_u64(file.data() + 32);
  const uint16_t phentsize = read_u16(file.data() + 54);
  const uint16_t phnum = read_u16(file.data() + 56);
  if (phentsize < 56 ||
      phoff + static_cast<uint64_t>(phnum) * phentsize > file.size()) {
    std::cerr << path << " has a truncated program header\n";
    return false;
  }

  const uint8_t *dynamic = nullptr;
  uint64_t dynamic_size = 0;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint8_t *phdr = file.data() + phoff + i * phentsize;
    // p_type is a 4-byte field; the next 4 bytes are p_flags.
    if (read_u16(phdr) != 2 || phdr[2] != 0 || phdr[3] != 0)
      continue;
    const uint64_t offset = read_u64(phdr + 8);
    dynamic_size = read_u64(phdr + 32);
    if (offset + dynamic_size > file.size()) {
      std::cerr << path << " has a truncated PT_DYNAMIC\n";
      return false;
    }
    dynamic = file.data() + offset;
    break;
  }
  if (!dynamic) {
    std::cerr << path << " has no PT_DYNAMIC\n";
    return false;
  }

  for (uint64_t n = 0; n + 16 <= dynamic_size; n += 16) {
    const uint64_t tag = read_u64(dynamic + n);
    const uint64_t val = read_u64(dynamic + n + 8);
    if (tag == 0)
      break;
    if (tag == kDtFlags1 && (val & kDf1Nodelete) != 0)
      return true;
  }
  std::cerr << path << " is missing DF_1_NODELETE (-Wl,-z,nodelete)\n";
  return false;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: test-hipgpu-elf-contract <libhipgpu.so>\n";
    return 2;
  }

  const char *hipgpu = argv[1];
  if (!has_nodelete(hipgpu))
    return 1;

  dlerror();
  void *handle = dlopen(hipgpu, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    std::cerr << "dlopen failed: " << dlerror() << "\n";
    return 1;
  }

  dlerror();
  if (!dlsym(handle, "CreateEpFactories") ||
      !dlsym(handle, "ReleaseEpFactory")) {
    std::cerr << "plugin entry points missing: " << dlerror() << "\n";
    return 1;
  }

  dlerror();
  if (dlclose(handle) != 0) {
    std::cerr << "dlclose failed: " << dlerror() << "\n";
    return 1;
  }

  std::cout << hipgpu << " has DF_1_NODELETE and the plugin entry points\n";
  return 0;
}
