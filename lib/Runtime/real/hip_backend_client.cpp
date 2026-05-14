/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip::Backend implementation.
//
// Bitcode-compiled into model.dll (see lib/Runtime/CMakeLists.txt). The
// portable DLL shim here lets the SAME source build for Windows
// (LoadLibraryA / GetProcAddress / FreeLibrary) and Linux (dlopen /
// dlsym / dlclose). We deliberately do NOT pull <windows.h> into the
// bitcode TU -- it bloats the include graph and triggers warnings under
// clang -emit-llvm host-only mode. The three Win32 symbols we need are
// declared inline.

#include "hip_backend_client.h"
#include "hipdnn_ep_backend.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef _WIN32
extern "C" {
__declspec(dllimport) void *__stdcall LoadLibraryA(const char *);
__declspec(dllimport) void *__stdcall GetProcAddress(void *, const char *);
__declspec(dllimport) int __stdcall FreeLibrary(void *);
}
#else
#include <dlfcn.h>
#endif

namespace hip {
namespace {

// -- portable DLL shim ------------------------------------------------------

void *load_dll(const char *name) {
#ifdef _WIN32
  return LoadLibraryA(name);
#else
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void *get_sym(void *handle, const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(GetProcAddress(handle, name));
#else
  return dlsym(handle, name);
#endif
}

void close_dll(void *handle) {
  if (!handle)
    return;
#ifdef _WIN32
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif
}

std::string backend_dll_filename(const std::string &arch) {
#ifdef _WIN32
  return "hip-backend-" + arch + ".dll";
#else
  return "libhip-backend-" + arch + ".so";
#endif
}

// -- arch detection ---------------------------------------------------------

// Returns the device's gfx arch (e.g. "gfx1151"), stripping any
// target-id suffix like ":xnack-" / ":sramecc+". Throws on HIP error.
std::string detect_gfx_arch() {
  hipDeviceProp_t prop{};
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                             hipGetErrorString(err));
  }
  std::string name(prop.gcnArchName);
  if (auto colon = name.find(':'); colon != std::string::npos)
    name.resize(colon);
  if (name.empty())
    throw std::runtime_error("device reports empty gcnArchName");
  return name;
}

// -- weak_ptr cache ---------------------------------------------------------

std::mutex &cache_mu() {
  static std::mutex m;
  return m;
}

std::weak_ptr<Backend> &cache_weak() {
  static std::weak_ptr<Backend> w;
  return w;
}

} // namespace

Backend::Backend() {
  std::string arch = detect_gfx_arch();
  std::string dll = backend_dll_filename(arch);

  dll_handle_ = load_dll(dll.c_str());
  if (!dll_handle_) {
    throw std::runtime_error("failed to load backend DLL: " + dll +
                             " (ensure install/dist/bin is on PATH; build "
                             "may not include this gfx)");
  }

  void *sym = get_sym(dll_handle_, HIP_BACKEND_API_SYMBOL);
  if (!sym) {
    void *h = dll_handle_;
    dll_handle_ = nullptr;
    close_dll(h);
    throw std::runtime_error(dll +
                             " is missing required export 'HIPBackendAPI'");
  }

  // The exported symbol is `const HIPBackendVTable *const HIPBackendAPI`,
  // i.e. dlsym returns the address of a pointer variable. Deref once to
  // get the vtable pointer value.
  vtable_ = *reinterpret_cast<const HIPBackendVTable *const *>(sym);
  if (!vtable_) {
    void *h = dll_handle_;
    dll_handle_ = nullptr;
    close_dll(h);
    throw std::runtime_error(dll + " HIPBackendAPI is null");
  }

  if (vtable_->abi_version != HIP_BACKEND_API_VERSION) {
    unsigned got = vtable_->abi_version;
    vtable_ = nullptr;
    void *h = dll_handle_;
    dll_handle_ = nullptr;
    close_dll(h);
    throw std::runtime_error(dll + " ABI version " + std::to_string(got) +
                             " does not match expected " +
                             std::to_string(HIP_BACKEND_API_VERSION));
  }

  if (vtable_->arch && std::string(vtable_->arch) != arch) {
    std::string backend_arch = vtable_->arch;
    vtable_ = nullptr;
    void *h = dll_handle_;
    dll_handle_ = nullptr;
    close_dll(h);
    throw std::runtime_error(dll + " reports arch " + backend_arch +
                             " but device is " + arch +
                             " (wrong DLL on PATH or filename collision)");
  }

  if (vtable_->init) {
    int rc = vtable_->init();
    if (rc != 0) {
      vtable_ = nullptr;
      void *h = dll_handle_;
      dll_handle_ = nullptr;
      close_dll(h);
      throw std::runtime_error(dll + " init() returned " + std::to_string(rc));
    }
  }

  fprintf(stderr, "[hip::Backend] loaded %s (arch %s, ABI v%u)\n", dll.c_str(),
          arch.c_str(), vtable_->abi_version);
}

Backend::~Backend() {
  if (vtable_ && vtable_->shutdown)
    vtable_->shutdown();
  vtable_ = nullptr;
  close_dll(dll_handle_);
  dll_handle_ = nullptr;
}

const char *Backend::Arch() const noexcept {
  return vtable_ && vtable_->arch ? vtable_->arch : "";
}

unsigned Backend::AbiVersion() const noexcept {
  return vtable_ ? vtable_->abi_version : 0u;
}

void Backend::SetScratchProvider(void *ctx,
                                 hip_backend_scratch_provider_fn provider) {
  if (!vtable_ || !vtable_->set_scratch_provider) {
    throw std::runtime_error(std::string("backend ") + Arch() +
                             " does not implement set_scratch_provider");
  }
  vtable_->set_scratch_provider(ctx, provider);
}

void Backend::Conv(void *stream, const void *input, int N, int C, int H, int W,
                   const void *weights, int K, int kernel_h, int kernel_w,
                   const void *bias, void *output, int Ho, int Wo, int stride_h,
                   int stride_w, int pad_top, int pad_left, int pad_bottom,
                   int pad_right, int dilation_h, int dilation_w, int group) {
  if (!vtable_ || !vtable_->conv_fwd_fp16_nchw) {
    throw std::runtime_error(std::string("backend ") + Arch() +
                             " does not implement conv_fwd_fp16_nchw");
  }
  int rc = vtable_->conv_fwd_fp16_nchw(
      stream, input, N, C, H, W, weights, K, kernel_h, kernel_w, bias, output,
      Ho, Wo, stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right,
      dilation_h, dilation_w, group);
  if (rc != 0) {
    throw std::runtime_error("backend conv_fwd_fp16_nchw returned " +
                             std::to_string(rc));
  }
}

std::shared_ptr<Backend> GetBackend() {
  std::lock_guard<std::mutex> lk(cache_mu());
  if (auto sp = cache_weak().lock())
    return sp;
  // ctor is private to Backend (friended); std::make_shared can't see it.
  std::shared_ptr<Backend> sp(new Backend());
  cache_weak() = sp;
  return sp;
}

} // namespace hip
