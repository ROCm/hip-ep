/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlss_conv.h"

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#ifndef HIPDNN_EP_AMDMLSS_HEADERS

int try_mlss_conv_forward(RuntimeState *, const void *, int64_t, int64_t,
                          int64_t, int64_t, const void *, int64_t, const void *,
                          void *, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t) {
  return 1;
}

#else

#include <amdmlss/amdmlss_api.h>

#include <hip/hip_runtime.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {

enum class ConvBackendMode { MiopenOnly, TryMlss, MlssOnly };

#ifdef _WIN32
void *tryLoadAmdmlssLibrary();
#else
void *tryLoadAmdmlssLibrary();
#endif

ConvBackendMode convBackendMode() {
  const char *env = std::getenv("HIPDNN_EP_CONV_BACKEND");
  if (!env || !*env)
    return ConvBackendMode::MiopenOnly;
  if (std::strcmp(env, "mlss") == 0)
    return ConvBackendMode::MlssOnly;
  if (std::strcmp(env, "auto") == 0)
    return ConvBackendMode::TryMlss;
  return ConvBackendMode::MiopenOnly;
}

struct MlssApi {
  void *lib = nullptr;
  MLSSstatus (*createContext)(MLSScontext *, const MLSSstring,
                              const MLSSstring) = nullptr;
  MLSSstatus (*setParameterByEnum)(MLSScontext *, const MLSSstring, MLSSenum,
                                   const MLSSvoid *) = nullptr;
  MLSSstatus (*getCaps)(MLSScontext, MLSSstatus **, MLSSsize *) = nullptr;
  MLSSstatus (*getBinariesEx)(MLSScontext, MLSSbinary **, MLSSsize *,
                              MLSSbinaryKind) = nullptr;
  MLSSstatus (*vectorRetrieveData)(MLSSvector, MLSSvoid **, MLSSsize *,
                                   MLSSenum *) = nullptr;
};

std::mutex g_mlss_api_mu;
MlssApi g_mlss_api;
std::atomic<bool> g_mlss_api_loaded{false};

void loadMlssApiLocked() {
#ifdef _WIN32
  g_mlss_api.lib = tryLoadAmdmlssLibrary();
  if (!g_mlss_api.lib)
    return;
#define LOAD(name, sym)                                                        \
  g_mlss_api.name = reinterpret_cast<decltype(g_mlss_api.name)>(               \
      GetProcAddress(static_cast<HMODULE>(g_mlss_api.lib), sym))
#else
  g_mlss_api.lib = tryLoadAmdmlssLibrary();
  if (!g_mlss_api.lib)
    return;
#define LOAD(name, sym)                                                        \
  g_mlss_api.name =                                                            \
      reinterpret_cast<decltype(g_mlss_api.name)>(dlsym(g_mlss_api.lib, sym))
#endif
  LOAD(createContext, "mlssCreateContext");
  LOAD(setParameterByEnum, "mlssSetParameterByEnum");
  LOAD(getCaps, "mlssGetCaps");
  LOAD(getBinariesEx, "mlssGetBinariesEx");
  LOAD(vectorRetrieveData, "mlssVectorRetrieveData");
#undef LOAD
}

MlssApi &mlssApi() {
  // std::call_once is deliberately avoided: its MSVC __std_init_once_* support
  // symbols are unresolvable in the JIT-linked runtime bitcode.
  if (!g_mlss_api_loaded.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(g_mlss_api_mu);
    if (!g_mlss_api_loaded.load(std::memory_order_relaxed)) {
      loadMlssApiLocked();
      g_mlss_api_loaded.store(true, std::memory_order_release);
    }
  }
  return g_mlss_api;
}

bool mlssApiReady() {
  MlssApi &api = mlssApi();
  return api.lib && api.createContext && api.setParameterByEnum &&
         api.getCaps && api.getBinariesEx && api.vectorRetrieveData;
}

#ifdef _WIN32
void *tryLoadAmdmlssLibrary() {
  const char *names[] = {"amdmlss.dll", "libamdmlss.dll"};
  for (const char *name : names) {
    if (HMODULE mod = LoadLibraryA(name))
      return mod;
  }

  char module_path[MAX_PATH];
  auto try_in_dir = [&](const char *dir) -> void * {
    if (!dir || !*dir)
      return nullptr;
    size_t dir_len = std::strlen(dir);
    if (dir_len + 16 >= MAX_PATH)
      return nullptr;
    std::memcpy(module_path, dir, dir_len);
    if (dir_len > 0 && module_path[dir_len - 1] != '\\' &&
        module_path[dir_len - 1] != '/')
      module_path[dir_len++] = '\\';
    for (const char *name : names) {
      std::strcpy(module_path + dir_len, name);
      if (HMODULE mod = LoadLibraryA(module_path))
        return mod;
    }
    return nullptr;
  };

  if (DWORD n = GetModuleFileNameA(nullptr, module_path, MAX_PATH); n > 0) {
    while (n > 0 && module_path[n - 1] != '\\' && module_path[n - 1] != '/')
      --n;
    module_path[n] = '\0';
    if (void *mod = try_in_dir(module_path))
      return mod;
  }

  if (const char *therock = std::getenv("THEROCK_DIST")) {
    char therock_bin[MAX_PATH];
    std::snprintf(therock_bin, sizeof(therock_bin), "%s\\bin", therock);
    if (void *mod = try_in_dir(therock_bin))
      return mod;
  }
  return nullptr;
}
#else
void *tryLoadAmdmlssLibrary() {
  const char *names[] = {"libamdmlss.so", "amdmlss.so"};
  for (const char *name : names) {
    if (void *mod = dlopen(name, RTLD_NOW | RTLD_LOCAL))
      return mod;
  }

  char exe_path[4096];
  ssize_t exe_len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (exe_len > 0) {
    exe_path[exe_len] = '\0';
    char *slash = std::strrchr(exe_path, '/');
    if (slash)
      *slash = '\0';
    for (const char *name : names) {
      char candidate[4096];
      std::snprintf(candidate, sizeof(candidate), "%s/%s", exe_path, name);
      if (void *mod = dlopen(candidate, RTLD_NOW | RTLD_LOCAL))
        return mod;
    }
  }

  if (const char *therock = std::getenv("THEROCK_DIST")) {
    for (const char *name : names) {
      char candidate[4096];
      std::snprintf(candidate, sizeof(candidate), "%s/lib/%s", therock, name);
      if (void *mod = dlopen(candidate, RTLD_NOW | RTLD_LOCAL))
        return mod;
    }
  }
  return nullptr;
}
#endif

struct ArgValue {
  uint8_t bytes[8];
  size_t size = 0;

  explicit ArgValue(uint32_t v) : size(sizeof(v)) {
    std::memcpy(bytes, &v, sizeof(v));
  }
  explicit ArgValue(int32_t v) : size(sizeof(v)) {
    std::memcpy(bytes, &v, sizeof(v));
  }
  explicit ArgValue(uint64_t v) : size(sizeof(v)) {
    std::memcpy(bytes, &v, sizeof(v));
  }
  explicit ArgValue(float v) : size(sizeof(v)) {
    std::memcpy(bytes, &v, sizeof(v));
  }
  explicit ArgValue(uint8_t v) : size(sizeof(v)) { bytes[0] = v; }
  explicit ArgValue(const void *ptr) : size(sizeof(uint64_t)) {
    uint64_t p = reinterpret_cast<uint64_t>(ptr);
    std::memcpy(bytes, &p, sizeof(p));
  }

  void *data() { return bytes; }
};

struct NamedArg {
  const char *name;
  ArgValue value;
};

bool fitsU32(int64_t v) {
  return v >= 0 && v <= static_cast<int64_t>(UINT32_MAX);
}

uint64_t hashCStr(const char *s) {
  uint64_t h = 14695981039346656037ull;
  for (; s && *s; ++s)
    h = (h ^ static_cast<unsigned char>(*s)) * 1099511628211ull;
  return h;
}

void detectRuntimeGfx(char *out, size_t out_size) {
  out[0] = '\0';
  if (out_size < 1)
    return;
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
    std::strncpy(out, "gfx1100", out_size - 1);
    out[out_size - 1] = '\0';
    return;
  }
  const char *arch = props.gcnArchName;
  const char *colon = std::strchr(arch, ':');
  size_t len = colon ? static_cast<size_t>(colon - arch) : std::strlen(arch);
  if (len == 0 || len >= out_size) {
    std::strncpy(out, "gfx1100", out_size - 1);
    out[out_size - 1] = '\0';
    return;
  }
  std::memcpy(out, arch, len);
  out[len] = '\0';
}

// Winograd Base non-reloc archives ship as gfx1100 ELFs. Query MLSS with
// gfx1100 even on gfx1151 hardware, then patch ELF metadata before
// hipModuleLoadData.
constexpr const char *kMlssQueryAsic = MLSS_GFX1100;
constexpr const char *kArchiveGfx = "gfx1100";
constexpr const char *kMainKernelName = "main";
constexpr size_t kElf64EflagsOffset = 0x30U;
constexpr uint8_t kEfAmdgpuMachMask = 0xFFU;
constexpr uint8_t kMachGfx1100 = 0x041U;

struct GfxMachEntry {
  const char *gfx_name;
  uint8_t mach_code;
};

constexpr GfxMachEntry kGfxMachTable[] = {
    {"gfx1100", 0x041U}, {"gfx1101", 0x046U}, {"gfx1102", 0x047U},
    {"gfx1150", 0x043U}, {"gfx1151", 0x04AU}, {"gfx1152", 0x055U},
    {"gfx1153", 0x058U},
};

bool machCodeForGfx(const char *runtime_gfx, uint8_t &out) {
  for (const GfxMachEntry &entry : kGfxMachTable) {
    if (runtime_gfx && std::strcmp(runtime_gfx, entry.gfx_name) == 0) {
      out = entry.mach_code;
      return true;
    }
  }
  return false;
}

bool patchGfxMetadata(std::vector<uint8_t> &image, const char *from_gfx,
                      uint8_t mach_from, const char *to_gfx, uint8_t mach_to) {
  const size_t from_len = std::strlen(from_gfx);
  const size_t to_len = std::strlen(to_gfx);
  if (from_len != to_len || image.empty())
    return false;

  if (image.size() > kElf64EflagsOffset &&
      (image[kElf64EflagsOffset] & kEfAmdgpuMachMask) == mach_from) {
    image[kElf64EflagsOffset] = static_cast<uint8_t>(
        (image[kElf64EflagsOffset] & ~kEfAmdgpuMachMask) | mach_to);
  }

  for (size_t i = 0; i + from_len <= image.size(); ++i) {
    if (std::memcmp(image.data() + i, from_gfx, from_len) == 0)
      std::memcpy(image.data() + i, to_gfx, to_len);
  }
  return true;
}

bool prepareModuleImage(const MLSSbinary &bin, const char *runtime_gfx,
                        std::vector<uint8_t> &out) {
  const auto *raw = static_cast<const uint8_t *>(bin.m_binaries);
  out.assign(raw, raw + bin.m_binarySize);
  if (!runtime_gfx || std::strcmp(runtime_gfx, kArchiveGfx) == 0)
    return true;

  uint8_t mach_to = 0;
  if (!machCodeForGfx(runtime_gfx, mach_to))
    return false;
  return patchGfxMetadata(out, kArchiveGfx, kMachGfx1100, runtime_gfx, mach_to);
}

struct ConvConfigKey {
  uint64_t h[16]{};

  static ConvConfigKey make(int64_t n, int64_t c, int64_t h, int64_t w,
                            int64_t k, int64_t kh, int64_t kw, int64_t oh,
                            int64_t ow, int64_t sh, int64_t sw, int64_t pt,
                            int64_t pl, int64_t pb, int64_t pr, int64_t dh,
                            int64_t dw, int64_t group, int64_t has_bias,
                            int64_t dtype, const char *runtime_gfx) {
    ConvConfigKey key{};
    int64_t vals[] = {n,  c,  h,  w,  k,  kh, kw, oh,    ow,       sh,
                      sw, pt, pl, pb, pr, dh, dw, group, has_bias, dtype};
    for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); ++i)
      key.h[i] = static_cast<uint64_t>(vals[i]);
    key.h[15] = hashCStr(runtime_gfx);
    return key;
  }

  bool operator==(const ConvConfigKey &o) const {
    return std::memcmp(h, o.h, sizeof(h)) == 0;
  }
};

struct ConvConfigKeyHash {
  size_t operator()(const ConvConfigKey &k) const {
    size_t result = 0;
    for (uint64_t v : k.h)
      result ^= v + 0x9e3779b97f4a7c15ULL + (result << 6) + (result >> 2);
    return result;
  }
};

struct ModuleCacheEntry {
  hipModule_t module = nullptr;
  std::vector<uint8_t> image;
  char kernel_name[256]{};
  MLSSdim3 grid{};
};

std::mutex g_module_mu;
ConvConfigKey g_module_key{};
ModuleCacheEntry g_module_entry{};
bool g_module_valid = false;

const MLSSbinary *pickNonRelocBinary(const MLSSbinary *bins, MLSSsize count) {
  // MIGraphX consumer pattern: only the linked "main" non-reloc ELF is
  // loadable via hipModuleLoadData. Other non-reloc blobs (e.g.
  // "_amdgpu_cs_main") are not valid module images for this path.
  for (MLSSsize i = 0; i < count; ++i) {
    const MLSSbinary *b = bins + i;
    if (b->m_isRelocatable)
      continue;
    if (b->m_pKernelName && std::strcmp(b->m_pKernelName, kMainKernelName) == 0)
      return b;
  }
  return nullptr;
}

const ArgValue *findNamedArg(const NamedArg *args, size_t count,
                             const char *name) {
  for (size_t i = 0; i < count; ++i) {
    if (args[i].name && name && std::strcmp(args[i].name, name) == 0)
      return &args[i].value;
  }
  return nullptr;
}

bool buildOrderedArgs(const MLSSbinary &bin, MlssApi &api,
                      const NamedArg *named, size_t named_count,
                      std::vector<ArgValue> &storage) {
  MLSSvoid *raw = nullptr;
  MLSSsize count = 0;
  MLSSenum type = 0;
  if (api.vectorRetrieveData(bin.m_argList, &raw, &count, &type) !=
          MLSS_SUCCESS ||
      !raw || count == 0)
    return false;

  const MLSSarg *args = static_cast<const MLSSarg *>(raw);
  std::vector<size_t> order(static_cast<size_t>(count));
  for (size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  for (size_t i = 1; i < order.size(); ++i) {
    size_t j = i;
    while (j > 0 && args[order[j - 1]].m_place > args[order[j]].m_place) {
      size_t tmp = order[j - 1];
      order[j - 1] = order[j];
      order[j] = tmp;
      --j;
    }
  }

  storage.clear();
  storage.reserve(order.size());

  for (size_t idx : order) {
    const char *name = args[idx].m_name ? args[idx].m_name : "";
    const ArgValue *val = findNamedArg(named, named_count, name);
    if (!val)
      return false;
    storage.push_back(*val);
  }
  return true;
}

bool serializeKernelArgs(const std::vector<ArgValue> &args,
                         std::vector<uint8_t> &out) {
  if (args.empty()) {
    out.clear();
    return true;
  }

  size_t total_size = args[0].size;
  for (size_t idx = 1; idx < args.size(); ++idx) {
    const size_t alignment = args[idx].size;
    const size_t padding = (alignment - (total_size % alignment)) % alignment;
    total_size += padding + args[idx].size;
  }

  out.resize(total_size);
  size_t offset = args[0].size;
  std::memcpy(out.data(), args[0].bytes, args[0].size);

  for (size_t idx = 1; idx < args.size(); ++idx) {
    const size_t alignment = args[idx].size;
    const size_t padding = (alignment - (offset % alignment)) % alignment;
    const size_t write_offset = offset + padding;
    std::memcpy(out.data() + write_offset, args[idx].bytes, args[idx].size);
    offset = write_offset + args[idx].size;
  }
  return true;
}

bool ensureModule(const MLSSbinary &bin, const char *runtime_gfx,
                  ModuleCacheEntry &entry) {
  if (entry.module)
    return true;
  if (!bin.m_binaries || bin.m_binarySize == 0)
    return false;

  if (!prepareModuleImage(bin, runtime_gfx, entry.image)) {
    RUNTIME_DEBUG_LOG(
        "[REAL] try_mlss_conv_forward: prepareModuleImage(%s -> %s) failed\n",
        kArchiveGfx, runtime_gfx ? runtime_gfx : "?");
    return false;
  }
  if (entry.image.size() >= 4) {
    RUNTIME_DEBUG_LOG("[REAL] try_mlss_conv_forward: module image size=%zu "
                      "magic=%02x%02x%02x%02x "
                      "eflags@0x30=0x%02x kernel=%s runtime=%s\n",
                      entry.image.size(), entry.image[0], entry.image[1],
                      entry.image[2], entry.image[3],
                      entry.image.size() > kElf64EflagsOffset
                          ? entry.image[kElf64EflagsOffset]
                          : 0,
                      bin.m_pKernelName ? bin.m_pKernelName : "?",
                      runtime_gfx ? runtime_gfx : "?");
  }
  std::strncpy(entry.kernel_name, kMainKernelName,
               sizeof(entry.kernel_name) - 1);
  entry.kernel_name[sizeof(entry.kernel_name) - 1] = '\0';
  entry.grid = bin.m_grid;

  hipModule_t module = nullptr;
  hipError_t load_err = hipModuleLoadData(&module, entry.image.data());
  if (load_err != hipSuccess) {
    RUNTIME_DEBUG_LOG(
        "[REAL] try_mlss_conv_forward: hipModuleLoadData(%zu bytes): %s\n",
        entry.image.size(), hipGetErrorString(load_err));
    return false;
  }
  entry.module = module;
  return true;
}

MLSSstatus setU32(MlssApi &api, MLSScontext *ctx, MLSSenum attr, uint32_t v) {
  MLSSstring op_name = const_cast<MLSSstring>(MLSS_CONV);
  return api.setParameterByEnum(ctx, op_name, attr, &v);
}

MLSSstatus setBool(MlssApi &api, MLSScontext *ctx, MLSSenum attr, MLSSbool v) {
  MLSSstring op_name = const_cast<MLSSstring>(MLSS_CONV);
  return api.setParameterByEnum(ctx, op_name, attr, &v);
}

bool configureMlssContext(MlssApi &api, MLSScontext *ctx, uint32_t n,
                          uint32_t c, uint32_t h, uint32_t w, uint32_t k,
                          uint32_t r, uint32_t s, uint32_t out_h,
                          uint32_t out_w, uint32_t pad_top, uint32_t pad_left,
                          uint32_t pad_bottom, uint32_t pad_right,
                          uint32_t stride_y, uint32_t stride_x,
                          uint32_t dilation_y, uint32_t dilation_x,
                          uint32_t groups, bool has_bias) {
  const uint32_t c_per_g = c / groups;
  const uint32_t k_per_g = k / groups;
  const uint32_t d_n = groups * c_per_g * h * w;
  const uint32_t d_c = h * w;
  const uint32_t d_h = w;
  const uint32_t f_k = c_per_g * r * s;
  const uint32_t f_c = r * s;
  const uint32_t f_r = s;
  const uint32_t f_s = 1;
  const uint32_t o_n = groups * k_per_g * out_h * out_w;
  const uint32_t o_k = out_h * out_w;
  const uint32_t o_h = out_w;
  const MLSSenum dtype = MLSS_FLOAT16;
  const MLSSenum precision = MLSS_PRECISION_FLOAT16_ADD_FLOAT32;
  const MLSSenum activation = MLSS_ACTIVATION_IDENTITY;
  // Match MIGraphX / convmxn_hip_module_validate sample: archived Winograd Base
  // bins were built with crossCorrelation=false (convolution, not
  // cross-correlation).
  const MLSSbool cross_corr = static_cast<MLSSbool>(false);
  const MLSSbool backward = static_cast<MLSSbool>(false);

  if (api.createContext(ctx, const_cast<MLSSstring>(kMlssQueryAsic),
                        const_cast<MLSSstring>(MLSS_CONV)) != MLSS_SUCCESS)
    return false;

#define SETU32(attr, val)                                                      \
  do {                                                                         \
    if (setU32(api, ctx, attr, (val)) != MLSS_SUCCESS)                         \
      return false;                                                            \
  } while (0)

  SETU32(MLSS_ATTR_CONV_W, w);
  SETU32(MLSS_ATTR_CONV_H, h);
  SETU32(MLSS_ATTR_CONV_C, c);
  SETU32(MLSS_ATTR_CONV_N, n);
  SETU32(MLSS_ATTR_CONV_K, k);
  SETU32(MLSS_ATTR_CONV_S, s);
  SETU32(MLSS_ATTR_CONV_R, r);
  SETU32(MLSS_ATTR_CONV_OUTW, out_w);
  SETU32(MLSS_ATTR_CONV_OUTH, out_h);
  SETU32(MLSS_ATTR_CONV_DILATIONX, dilation_x);
  SETU32(MLSS_ATTR_CONV_DILATIONY, dilation_y);
  SETU32(MLSS_ATTR_CONV_STARTPADX, pad_left);
  SETU32(MLSS_ATTR_CONV_STARTPADY, pad_top);
  SETU32(MLSS_ATTR_CONV_ENDPADX, pad_right);
  SETU32(MLSS_ATTR_CONV_ENDPADY, pad_bottom);
  SETU32(MLSS_ATTR_CONV_OUTPADX, 0);
  SETU32(MLSS_ATTR_CONV_OUTPADY, 0);
  SETU32(MLSS_ATTR_CONV_CONVSTRIDEX, stride_x);
  SETU32(MLSS_ATTR_CONV_CONVSTRIDEY, stride_y);
  SETU32(MLSS_ATTR_CONV_INPUTSTRIDEX, 1);
  SETU32(MLSS_ATTR_CONV_INPUTSTRIDEY, 1);
  SETU32(MLSS_ATTR_CONV_FILTERSTRIDEX, 1);
  SETU32(MLSS_ATTR_CONV_FILTERSTRIDEY, 1);
  SETU32(MLSS_ATTR_CONV_GROUPS, groups);
  if (setBool(api, ctx, MLSS_ATTR_CONV_HASBIAS, has_bias ? 1 : 0) !=
      MLSS_SUCCESS)
    return false;
  if (setBool(api, ctx, MLSS_ATTR_CONV_CROSSCORRELATION, cross_corr) !=
      MLSS_SUCCESS)
    return false;
  if (setBool(api, ctx, MLSS_ATTR_CONV_BACKWARD, backward) != MLSS_SUCCESS)
    return false;
  SETU32(MLSS_ATTR_CONV_DNSTRIDE, d_n);
  SETU32(MLSS_ATTR_CONV_DHSTRIDE, d_h);
  SETU32(MLSS_ATTR_CONV_DCSTRIDE, d_c);
  SETU32(MLSS_ATTR_CONV_FKSTRIDE, f_k);
  SETU32(MLSS_ATTR_CONV_FCSTRIDE, f_c);
  SETU32(MLSS_ATTR_CONV_FRSTRIDE, f_r);
  SETU32(MLSS_ATTR_CONV_FSSTRIDE, f_s);
  SETU32(MLSS_ATTR_CONV_ONSTRIDE, o_n);
  SETU32(MLSS_ATTR_CONV_OHSTRIDE, o_h);
  SETU32(MLSS_ATTR_CONV_OKSTRIDE, o_k);
  SETU32(MLSS_ATTR_CONV_DOFFSET, 0);
  SETU32(MLSS_ATTR_CONV_OOFFSET, 0);
  SETU32(MLSS_ATTR_CONV_FOFFSET, 0);
  SETU32(MLSS_ATTR_CONV_BOFFSET, 0);
  SETU32(MLSS_ATTR_CONV_DATATYPE, dtype);
  SETU32(MLSS_ATTR_CONV_PRECISION, precision);
  SETU32(MLSS_ATTR_CONV_ACTIVATION, activation);
#undef SETU32
  return true;
}

bool mlssSupportedShape(int64_t n, int64_t c, int64_t h, int64_t w, int64_t k,
                        int64_t kh, int64_t kw, int64_t oh, int64_t ow,
                        int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                        int64_t group, int64_t data_type) {
  if (data_type != HIPDNN_EP_DATATYPE_HALF)
    return false;
  if (dh != 1 || dw != 1)
    return false;
  if (group <= 0)
    return false;
  if (c % group != 0 || k % group != 0)
    return false;
  const int64_t fields[] = {n, c, h, w, k, kh, kw, oh, ow, sh, sw, group};
  for (int64_t v : fields) {
    if (!fitsU32(v))
      return false;
  }
  (void)dh;
  (void)dw;
  return true;
}

} // namespace

int try_mlss_conv_forward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group, int64_t data_type) {
  const ConvBackendMode mode = convBackendMode();
  auto fail = [&](const char *why) -> int {
    RUNTIME_DEBUG_LOG("[REAL] try_mlss_conv_forward: %s\n", why);
    return mode == ConvBackendMode::MlssOnly ? -1 : 1;
  };
  if (mode == ConvBackendMode::MiopenOnly)
    return 1;
  if (!state || !input || !weights || !output)
    return fail("invalid arguments");
  if (!mlssApiReady())
    return fail("amdmlss API not available");
  if (!mlssSupportedShape(input_n, input_c, input_h, input_w, weights_k,
                          kernel_h, kernel_w, output_h, output_w, stride_h,
                          stride_w, dilation_h, dilation_w, group, data_type)) {
    return fail("unsupported shape or dtype for MLSS v1");
  }

  MlssApi &api = mlssApi();
  char runtime_gfx[32];
  detectRuntimeGfx(runtime_gfx, sizeof(runtime_gfx));
  const bool has_bias = bias != nullptr;
  const uint32_t n = static_cast<uint32_t>(input_n);
  const uint32_t c = static_cast<uint32_t>(input_c);
  const uint32_t h = static_cast<uint32_t>(input_h);
  const uint32_t w = static_cast<uint32_t>(input_w);
  const uint32_t k = static_cast<uint32_t>(weights_k);
  const uint32_t r = static_cast<uint32_t>(kernel_h);
  const uint32_t s = static_cast<uint32_t>(kernel_w);
  const uint32_t out_h = static_cast<uint32_t>(output_h);
  const uint32_t out_w = static_cast<uint32_t>(output_w);
  const uint32_t groups = static_cast<uint32_t>(group);

  MLSScontext ctx = 0;
  if (!configureMlssContext(
          api, &ctx, n, c, h, w, k, r, s, out_h, out_w,
          static_cast<uint32_t>(pad_top), static_cast<uint32_t>(pad_left),
          static_cast<uint32_t>(pad_bottom), static_cast<uint32_t>(pad_right),
          static_cast<uint32_t>(stride_h), static_cast<uint32_t>(stride_w),
          static_cast<uint32_t>(dilation_h), static_cast<uint32_t>(dilation_w),
          groups, has_bias)) {
    return fail("configureMlssContext failed");
  }

  MLSSstatus *statuses = nullptr;
  MLSSsize status_count = 0;
  if (api.getCaps(ctx, &statuses, &status_count) != MLSS_SUCCESS) {
    return fail("mlssGetCaps failed");
  }

  MLSSbinary *bins = nullptr;
  MLSSsize bin_count = 0;
  if (api.getBinariesEx(ctx, &bins, &bin_count,
                        MLSS_BINARY_KIND_NON_RELOCATABLE) != MLSS_SUCCESS ||
      !bins || bin_count == 0) {
    return fail("mlssGetBinariesEx returned no binaries");
  }

  const MLSSbinary *bin = pickNonRelocBinary(bins, bin_count);
  if (!bin) {
    for (MLSSsize i = 0; i < bin_count; ++i) {
      RUNTIME_DEBUG_LOG("[REAL] try_mlss_conv_forward: bin[%zu] kernel=%s "
                        "reloc=%d size=%zu\n",
                        static_cast<size_t>(i),
                        bins[i].m_pKernelName ? bins[i].m_pKernelName : "?",
                        bins[i].m_isRelocatable ? 1 : 0,
                        static_cast<size_t>(bins[i].m_binarySize));
    }
    return fail("no non-relocatable MLSS binary");
  }

  const ConvConfigKey key = ConvConfigKey::make(
      input_n, input_c, input_h, input_w, weights_k, kernel_h, kernel_w,
      output_h, output_w, stride_h, stride_w, pad_top, pad_left, pad_bottom,
      pad_right, dilation_h, dilation_w, group, has_bias ? 1 : 0, data_type,
      runtime_gfx);

  ModuleCacheEntry module_entry;
  {
    std::lock_guard<std::mutex> lock(g_module_mu);
    if (g_module_valid && g_module_key == key)
      module_entry = g_module_entry;
    else {
      module_entry = ModuleCacheEntry{};
      if (!ensureModule(*bin, runtime_gfx, module_entry))
        return fail("hipModuleLoadData failed");
      g_module_key = key;
      g_module_entry = module_entry;
      g_module_valid = true;
    }
  }

  hipFunction_t func = nullptr;
  if (hipModuleGetFunction(&func, module_entry.module,
                           module_entry.kernel_name) != hipSuccess) {
    return fail("hipModuleGetFunction failed");
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (module_entry.grid.m_x == 0 || n == 0 || groups == 0)
    return fail("invalid launch grid");
  const uint32_t n_groups = module_entry.grid.m_x / (n * groups);
  if (n_groups == 0)
    return fail("invalid nGroups derived from grid");
  const size_t sync_bytes = static_cast<size_t>(n_groups) * sizeof(uint32_t);
  const size_t acc_bytes =
      static_cast<size_t>(n) * k * out_h * out_w * sizeof(float);

  void *sync_dev = nullptr;
  void *acc_dev = nullptr;
  if (hipMalloc(&sync_dev, sync_bytes) != hipSuccess ||
      hipMalloc(&acc_dev, acc_bytes) != hipSuccess) {
    if (sync_dev)
      (void)hipFree(sync_dev);
    if (acc_dev)
      (void)hipFree(acc_dev);
    return fail("hipMalloc for sync/acc buffers failed");
  }
  if (hipMemset(sync_dev, 0, sync_bytes) != hipSuccess ||
      hipMemset(acc_dev, 0, acc_bytes) != hipSuccess) {
    (void)hipFree(sync_dev);
    (void)hipFree(acc_dev);
    return fail("hipMemset for sync/acc buffers failed");
  }

  const uint64_t flags64 = (has_bias ? (1ull << 7) : 0ull) | (1ull << 9) |
                           (1ull << 14) | (1ull << 15);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const uint8_t activation_mode = 0;
  const uint8_t sync_limit = 255;
  const uint8_t sync_period = 0;

  const NamedArg named_args[] = {
      {"n", ArgValue(n)},
      {"c", ArgValue(c)},
      {"h", ArgValue(h)},
      {"w", ArgValue(w)},
      {"k", ArgValue(k)},
      {"nGroups", ArgValue(n_groups)},
      {"flags64", ArgValue(flags64)},
      {"dataAddr", ArgValue(input)},
      {"filterAddr", ArgValue(weights)},
      {"outputAddr", ArgValue(output)},
      {"reserved3", ArgValue(static_cast<uint64_t>(0))},
      {"r", ArgValue(r)},
      {"s", ArgValue(s)},
      {"padH", ArgValue(static_cast<int32_t>(pad_top))},
      {"padW", ArgValue(static_cast<int32_t>(pad_left))},
      {"outH", ArgValue(out_h)},
      {"outW", ArgValue(out_w)},
      {"biasAddr", ArgValue(has_bias ? bias : nullptr)},
      {"alpha", ArgValue(alpha)},
      {"beta", ArgValue(beta)},
      {"dOffset", ArgValue(static_cast<uint64_t>(0))},
      {"fOffset", ArgValue(static_cast<uint64_t>(0))},
      {"oOffset", ArgValue(static_cast<uint64_t>(0))},
      {"bOffset", ArgValue(static_cast<uint64_t>(0))},
      {"dNStride", ArgValue(groups * (c / groups) * h * w)},
      {"dCStride", ArgValue(h * w)},
      {"dHStride", ArgValue(w)},
      {"reserved4", ArgValue(static_cast<uint32_t>(0))},
      {"fKStride", ArgValue((c / groups) * r * s)},
      {"fCStride", ArgValue(r * s)},
      {"fRStride", ArgValue(s)},
      {"reserved5", ArgValue(static_cast<uint32_t>(0))},
      {"oNStride", ArgValue(groups * (k / groups) * out_h * out_w)},
      {"oKStride", ArgValue(out_h * out_w)},
      {"oHStride", ArgValue(out_w)},
      {"reserved6", ArgValue(static_cast<uint32_t>(0))},
      {"G", ArgValue(groups)},
      {"dGStride", ArgValue(static_cast<uint32_t>(0))},
      {"fGStride", ArgValue(static_cast<uint32_t>(0))},
      {"oGStride", ArgValue(static_cast<uint32_t>(0))},
      {"activationMode", ArgValue(activation_mode)},
      {"syncLimit", ArgValue(sync_limit)},
      {"syncPeriod", ArgValue(sync_period)},
      {"reserved8", ArgValue(static_cast<uint8_t>(0))},
      {"reserved9", ArgValue(static_cast<uint32_t>(0))},
      {"syncAddr", ArgValue(sync_dev)},
      {"accAddr", ArgValue(acc_dev)},
      {"aOffset", ArgValue(static_cast<uint64_t>(0))},
  };

  std::vector<ArgValue> arg_storage;
  std::vector<uint8_t> kern_args;
  if (!buildOrderedArgs(*bin, api, named_args,
                        sizeof(named_args) / sizeof(named_args[0]),
                        arg_storage) ||
      !serializeKernelArgs(arg_storage, kern_args)) {
    (void)hipFree(sync_dev);
    (void)hipFree(acc_dev);
    return fail("buildOrderedArgs/serializeKernelArgs failed");
  }

  dim3 grid(module_entry.grid.m_x, module_entry.grid.m_y,
            module_entry.grid.m_z);
  dim3 block(256, 1, 1);

  size_t kern_arg_size = kern_args.size();
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kern_args.data(),
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kern_arg_size,
                   HIP_LAUNCH_PARAM_END};

  (void)hipGetLastError();
  hipError_t launch_err = hipModuleLaunchKernel(
      func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
      static_cast<unsigned int>(bin->m_sharedMemInBytes),
      static_cast<hipStream_t>(stream), nullptr, extra);

  if (launch_err != hipSuccess) {
    fprintf(stderr, "[REAL] try_mlss_conv_forward: hipModuleLaunchKernel: %s\n",
            hipGetErrorString(launch_err));
    (void)hipFree(sync_dev);
    (void)hipFree(acc_dev);
    return fail("hipModuleLaunchKernel failed");
  }
  launch_err = hipGetLastError();
  if (launch_err != hipSuccess) {
    fprintf(stderr, "[REAL] try_mlss_conv_forward: hipGetLastError: %s\n",
            hipGetErrorString(launch_err));
    (void)hipFree(sync_dev);
    (void)hipFree(acc_dev);
    return fail("hipGetLastError after launch failed");
  }

  if (hipStreamSynchronize(static_cast<hipStream_t>(stream)) != hipSuccess) {
    (void)hipFree(sync_dev);
    (void)hipFree(acc_dev);
    return fail("hipStreamSynchronize failed");
  }

  (void)hipFree(sync_dev);
  (void)hipFree(acc_dev);
  RUNTIME_DEBUG_LOG("[REAL] try_mlss_conv_forward: completed via MLSS "
                    "(query=%s runtime=%s)\n",
                    kMlssQueryAsic, runtime_gfx);
  return 0;
}

#endif // HIPDNN_EP_AMDMLSS_HEADERS
