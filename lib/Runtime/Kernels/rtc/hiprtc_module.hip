/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "rtc/hiprtc_module.h"

#include <hip/hiprtc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hipdnn_ep {
namespace rtc {
namespace {

// Bumping this invalidates every on-disk code object from an older binary.
constexpr const char kCacheFormatTag[] = "RTC1";

std::string getEnvVar(const char *key) {
#ifdef _WIN32
  char *v = nullptr;
  size_t len = 0;
  std::string out;
  if (_dupenv_s(&v, &len, key) == 0 && v != nullptr) {
    out = v;
    free(v);
  }
  return out;
#else
  const char *v = getenv(key);
  return v ? std::string(v) : std::string();
#endif
}

std::string tempDir() {
#ifdef _WIN32
  return getEnvVar("TEMP");
#else
  std::string v = getEnvVar("TEMP");
  if (v.empty()) v = getEnvVar("TMPDIR");
  if (v.empty()) v = "/tmp";
  return v;
#endif
}

// gcnArchName can carry feature suffixes (":xnack-"), which are not portable
// in a filename.
std::string sanitize(const std::string &s) {
  std::string out = s;
  for (char &c : out) {
    const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                    (c >= 'A' && c <= 'Z') || c == '_' || c == '-';
    if (!ok) c = '_';
  }
  return out;
}

std::string archName() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      prop.gcnArchName[0] == '\0') {
    return std::string();
  }
  return std::string(prop.gcnArchName);
}

// FNV-1a. Only needs to detect source edits, not resist collisions.
std::string sourceHash(const char *data, size_t size) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < size; ++i) {
    h ^= static_cast<unsigned char>(data[i]);
    h *= 1099511628211ULL;
  }
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return std::string(buf);
}

}  // namespace

bool enabled() {
  static const bool on = (getEnvVar("HIPDNN_EP_RTC") == "1");
  return on;
}

Module::Module(std::string name, const char *source, size_t source_size,
               std::vector<std::string> kernel_names)
    : name_(std::move(name)),
      source_(source),
      source_size_(source_size),
      kernel_names_(std::move(kernel_names)) {}

// <TEMP>/morphizen_rtc_<name>_<arch>_<hash>.co plus a ".map" sidecar holding
// the mangled names, which cannot be recovered from the code object alone.
static std::string cachePathBase(const std::string &name, const char *source,
                                 size_t source_size) {
  const std::string dir = tempDir();
  const std::string arch = archName();
  if (dir.empty() || arch.empty()) return std::string();
  return dir + "/morphizen_rtc_" + name + "_" + sanitize(arch) + "_" +
         sourceHash(source, source_size);
}

bool Module::loadFromCache(std::vector<char> &code) {
  const std::string base = cachePathBase(name_, source_, source_size_);
  if (base.empty()) return false;

  FILE *mf = fopen((base + ".map").c_str(), "rb");
  if (!mf) return false;
  char tag[8] = {0};
  if (!fgets(tag, sizeof(tag), mf)) {
    fclose(mf);
    return false;
  }
  if (strncmp(tag, kCacheFormatTag, strlen(kCacheFormatTag)) != 0) {
    fclose(mf);
    return false;
  }
  std::unordered_map<std::string, std::string> names;
  char line[1024];
  while (fgets(line, sizeof(line), mf)) {
    char *tab = strchr(line, '\t');
    if (!tab) continue;
    *tab = '\0';
    std::string expr(line);
    std::string lowered(tab + 1);
    while (!lowered.empty() &&
           (lowered.back() == '\n' || lowered.back() == '\r')) {
      lowered.pop_back();
    }
    names.emplace(std::move(expr), std::move(lowered));
  }
  fclose(mf);

  for (const std::string &k : kernel_names_) {
    if (names.find(k) == names.end()) return false;
  }

  FILE *cf = fopen((base + ".co").c_str(), "rb");
  if (!cf) return false;
  fseek(cf, 0, SEEK_END);
  const long size = ftell(cf);
  fseek(cf, 0, SEEK_SET);
  if (size <= 0) {
    fclose(cf);
    return false;
  }
  code.resize(static_cast<size_t>(size));
  const size_t got = fread(code.data(), 1, code.size(), cf);
  fclose(cf);
  if (got != code.size()) return false;

  lowered_names_ = std::move(names);
  return true;
}

// comgr keeps its own on-disk cache of hipRTC compilations, in a machine-wide
// 1 GiB LRU shared with every other comgr user (MIOpen, hipBLASLt). We cannot
// redirect it from here: comgr latches AMD_COMGR_CACHE_DIR when it initialises,
// which happens with the HIP runtime long before this DLL is loaded, so setting
// the variable at this point measurably has no effect. Doing so would have to
// happen in the host process before HIP loads. The code object cache below
// makes it largely moot anyway -- a warm start never reaches hipRTC.
bool Module::compileAndCache(std::vector<char> &code) {
  const std::string arch = archName();
  if (arch.empty()) {
    fprintf(stderr, "[custom_kernels] rtc(%s): no device architecture\n",
            name_.c_str());
    return false;
  }

  hiprtcProgram prog = nullptr;
  const std::string src(source_, source_size_);
  hiprtcResult rc = hiprtcCreateProgram(&prog, src.c_str(),
                                        (name_ + ".hip").c_str(), 0, nullptr,
                                        nullptr);
  if (rc != HIPRTC_SUCCESS) {
    fprintf(stderr, "[custom_kernels] rtc(%s): createProgram: %s\n",
            name_.c_str(), hiprtcGetErrorString(rc));
    return false;
  }

  // Must precede compilation; hipRTC resolves these while parsing.
  for (const std::string &k : kernel_names_) {
    hiprtcAddNameExpression(prog, k.c_str());
  }

  const std::string arch_opt = "--gpu-architecture=" + arch;
  const char *opts[] = {arch_opt.c_str()};
  rc = hiprtcCompileProgram(prog, 1, opts);
  if (rc != HIPRTC_SUCCESS) {
    size_t log_size = 0;
    hiprtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size ? log_size : 1, '\0');
    if (log_size) hiprtcGetProgramLog(prog, &log[0]);
    fprintf(stderr, "[custom_kernels] rtc(%s): compile failed: %s\n%s\n",
            name_.c_str(), hiprtcGetErrorString(rc), log.c_str());
    hiprtcDestroyProgram(&prog);
    return false;
  }

  for (const std::string &k : kernel_names_) {
    const char *lowered = nullptr;
    if (hiprtcGetLoweredName(prog, k.c_str(), &lowered) == HIPRTC_SUCCESS &&
        lowered != nullptr) {
      lowered_names_[k] = lowered;
    } else {
      fprintf(stderr, "[custom_kernels] rtc(%s): no lowered name for %s\n",
              name_.c_str(), k.c_str());
      hiprtcDestroyProgram(&prog);
      return false;
    }
  }

  size_t code_size = 0;
  hiprtcGetCodeSize(prog, &code_size);
  code.resize(code_size);
  const bool got_code =
      (hiprtcGetCode(prog, code.data()) == HIPRTC_SUCCESS) && code_size > 0;
  hiprtcDestroyProgram(&prog);
  if (!got_code) {
    fprintf(stderr, "[custom_kernels] rtc(%s): getCode failed\n",
            name_.c_str());
    return false;
  }

  // Cache write failures are not fatal: the module is already usable, the next
  // process just pays for the compile again.
  const std::string base = cachePathBase(name_, source_, source_size_);
  if (!base.empty()) {
    FILE *cf = fopen((base + ".co").c_str(), "wb");
    if (cf) {
      fwrite(code.data(), 1, code.size(), cf);
      fclose(cf);
      FILE *mf = fopen((base + ".map").c_str(), "wb");
      if (mf) {
        fprintf(mf, "%s\n", kCacheFormatTag);
        for (const auto &kv : lowered_names_) {
          fprintf(mf, "%s\t%s\n", kv.first.c_str(), kv.second.c_str());
        }
        fclose(mf);
      }
    }
  }
  return true;
}

void Module::ensureLoaded() {
  std::call_once(load_once_, [this]() {
    std::vector<char> code;
    if (!loadFromCache(code) && !compileAndCache(code)) {
      return;
    }

    hipError_t err = hipModuleLoadData(&module_, code.data());
    if (err != hipSuccess) {
      fprintf(stderr, "[custom_kernels] rtc(%s): hipModuleLoadData: %s\n",
              name_.c_str(), hipGetErrorString(err));
      return;
    }

    for (const std::string &k : kernel_names_) {
      hipFunction_t f = nullptr;
      err = hipModuleGetFunction(&f, module_, lowered_names_[k].c_str());
      if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] rtc(%s): getFunction(%s): %s\n",
                name_.c_str(), k.c_str(), hipGetErrorString(err));
        return;
      }
      functions_[k] = f;
    }
    load_ok_ = true;
  });
}

hipFunction_t Module::getFunction(const std::string &kernel_name) {
  ensureLoaded();
  if (!load_ok_) return nullptr;
  auto it = functions_.find(kernel_name);
  return it == functions_.end() ? nullptr : it->second;
}

}  // namespace rtc
}  // namespace hipdnn_ep
