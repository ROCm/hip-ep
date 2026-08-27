/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "LlvmIrJit.h"

// HIP host API must precede the LLVM ORC headers: an ORC header pulls in
// <intrin.h> on MSVC, which otherwise breaks HIP's vector-type headers.
#ifdef HIPDNN_EP_LINK_HIP_HOST
#include <hip/hip_runtime_api.h>
#endif

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Triple.h>

// Plugin-registry accessors (pluginBitcodeBuffers / pluginLibraries /
// pluginLibraryPaths). Provided by LibHipCompiler, which every host linking
// this lib also links (see this dir's CMakeLists).
#include "hip/Compiler/PluginRegistry.h"
#include "hip/env.h" // shared cross-platform env reader (single Win32 call)

#include <glog/logging.h>

// Per-OS runtime bitcode embedded as bytes; emitted by
// backend-mlir-compiler/custom-op-mlir/CMakeLists.txt from runtime.bc.
extern "C" const unsigned char runtime_bc_data[];
extern "C" const std::size_t runtime_bc_data_size;

#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <typeinfo>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mlir_compilation::customop {

namespace {

void ensureLLVMNativeTargetInitialized() {
  static std::once_flag once;
  std::call_once(once, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
}

#ifdef _WIN32

// Windows-only: the bitcode references MSVC allocator helpers and the
// std::type_info vtable that no DLL exports (the latter via std::bad_alloc
// unwind metadata, even under -fno-rtti). Define them as absolute symbols
// pointing inside this DLL. Linux resolves these from libstdc++ via the
// process search generator, so the shim is a no-op there.
llvm::Error installPlatformSymbolShims(llvm::orc::LLJIT &jit) {
  llvm::orc::SymbolMap absolute_syms;
  auto add = [&](llvm::StringRef name, void *addr) {
    absolute_syms[jit.getExecutionSession().intern(name)] =
        llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr(reinterpret_cast<uintptr_t>(addr)),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute);
  };

  void *op_new_sz =
      reinterpret_cast<void *>(static_cast<void *(*)(size_t)>(::operator new));
  void *op_delete = reinterpret_cast<void *>(
      static_cast<void (*)(void *) noexcept>(::operator delete));
  void *op_delete_sz = reinterpret_cast<void *>(
      static_cast<void (*)(void *, size_t) noexcept>(::operator delete));
  add("??2@YAPEAX_K@Z", op_new_sz);     // operator new(size_t)
  add("??3@YAXPEAX@Z", op_delete);      // operator delete(void*)
  add("??3@YAXPEAX_K@Z", op_delete_sz); // operator delete(void*, size_t)

  // Read the `std::type_info` vtable pointer out of a live polymorphic
  // instance; `typeid(int)` shares the vtable with every other
  // `type_info` in the same image (MSVC ABI).
  const std::type_info &ti = typeid(int);
  void *type_info_vtbl =
      *reinterpret_cast<void *const *>(static_cast<const void *>(&ti));
  add("??_7type_info@@6B@", type_info_vtbl);

  return jit.getMainJITDylib().define(
      llvm::orc::absoluteSymbols(std::move(absolute_syms)));
}

#else // !_WIN32

inline llvm::Error installPlatformSymbolShims(llvm::orc::LLJIT & /*jit*/) {
  return llvm::Error::success();
}

#endif // _WIN32

// The runtime's current-session-stream accessors are defined in host-native
// code (lib/Runtime/tls_stream.cpp), not in runtime.bc, so the JIT'd runtime
// references them as externals. Register them as absolute symbols pointing at
// the host functions: this guarantees resolution (no reliance on export-table
// / process-generator quirks) and the reference here forces tls_stream.o into
// the host image (EP DLL / hip-test / hip-inspect) so the addresses are valid.
extern "C" void *hipdnn_ep_get_current_stream(void);
extern "C" void hipdnn_ep_set_current_stream(void *stream);

llvm::Error installRuntimeStreamShims(llvm::orc::LLJIT &jit) {
  llvm::orc::SymbolMap syms;
  auto add = [&](llvm::StringRef name, void *addr) {
    syms[jit.getExecutionSession().intern(name)] = llvm::orc::ExecutorSymbolDef(
        llvm::orc::ExecutorAddr(reinterpret_cast<uintptr_t>(addr)),
        llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute);
  };
  add("hipdnn_ep_get_current_stream",
      reinterpret_cast<void *>(&hipdnn_ep_get_current_stream));
  add("hipdnn_ep_set_current_stream",
      reinterpret_cast<void *>(&hipdnn_ep_set_current_stream));
  return jit.getMainJITDylib().define(
      llvm::orc::absoluteSymbols(std::move(syms)));
}

// Per-JIT atexit interposer. Static dtors registered from runtime.bc's
// @llvm.global_ctors must run before ~LlvmIrJit unmaps the JIT slab; routing
// __cxa_atexit/atexit here captures them per-JIT instead of letting them
// dangle in libc/ucrt (Linux faults in __run_exit_handlers otherwise).
struct AtexitEntry {
  void *fn;
  void *arg;
  bool has_arg; // true: __cxa_atexit shape; false: atexit shape
};

thread_local std::vector<AtexitEntry> *tls_current_atexit_list = nullptr;

extern "C" int bitcodeJitCxaAtexit(void (*fn)(void *), void *arg,
                                   void * /*dso*/) noexcept {
  if (tls_current_atexit_list && fn) {
    tls_current_atexit_list->push_back(
        {reinterpret_cast<void *>(fn), arg, /*has_arg=*/true});
  }
  return 0;
}

extern "C" int bitcodeJitAtexit(void (*fn)(void)) noexcept {
  if (tls_current_atexit_list && fn) {
    tls_current_atexit_list->push_back(
        {reinterpret_cast<void *>(fn), nullptr, /*has_arg=*/false});
  }
  return 0;
}

void defineShimOrSkip(llvm::orc::LLJIT &jit, llvm::StringRef name, void *addr) {
  llvm::orc::SymbolMap one;
  one[jit.getExecutionSession().intern(name)] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr(reinterpret_cast<uintptr_t>(addr)),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute);
  auto err =
      jit.getMainJITDylib().define(llvm::orc::absoluteSymbols(std::move(one)));
  if (!err)
    return;
  // LLJIT's platform pre-defines `atexit`; skip on duplicate and let
  // its `runDestructors()` run that list during `deinitialize()`. The
  // duplicate Error type isn't stable across LLVM versions, hence the
  // string match.
  std::string msg = llvm::toString(std::move(err));
  if (msg.find("duplicate definition") == std::string::npos) {
    LOG(WARNING) << "LlvmIrJit::installAtexitShims: define " << name.str()
                 << " failed: " << msg;
  }
}

void installAtexitShims(llvm::orc::LLJIT &jit) {
  defineShimOrSkip(jit, "atexit", reinterpret_cast<void *>(&bitcodeJitAtexit));
  defineShimOrSkip(jit, "__cxa_atexit",
                   reinterpret_cast<void *>(&bitcodeJitCxaAtexit));
}

// Absolute directory of the EP DLL/SO that owns this code. Per-arch kernel
// libs install side-by-side with it, so anchoring dlopen here avoids relying
// on the loader's search path. Empty string on failure.
std::string thisModuleDirectory() {
#ifdef _WIN32
  HMODULE mod = nullptr;
  if (!::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(&installAtexitShims),
                            &mod)) {
    return {};
  }
  char buf[MAX_PATH] = {};
  DWORD n = ::GetModuleFileNameA(mod, buf, sizeof(buf));
  if (n == 0 || n >= sizeof(buf))
    return {};
  std::string path(buf, n);
  auto slash = path.find_last_of("\\/");
  return (slash == std::string::npos) ? std::string{} : path.substr(0, slash);
#else
  ::Dl_info info{};
  if (!::dladdr(reinterpret_cast<void *>(&installAtexitShims), &info) ||
      !info.dli_fname) {
    return {};
  }
  std::string path = info.dli_fname;
  auto slash = path.find_last_of('/');
  return (slash == std::string::npos) ? std::string{} : path.substr(0, slash);
#endif
}

#ifdef HIPDNN_EP_LINK_HIP_HOST
// Active GPU ISA name, e.g. "gfx1100" (the ":<feature>" suffix stripped).
// Falls back to gfx1151 on any failure (incl. the empty gcnArchName seen on
// some Windows TheRock builds).
std::string detectCustomKernelArch() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    LOG(WARNING) << "LlvmIrJit::detectCustomKernelArch: "
                    "hipGetDeviceProperties failed; defaulting to gfx1151";
    return "gfx1151";
  }
  std::string arch = prop.gcnArchName;
  if (arch.empty()) {
    LOG(WARNING) << "LlvmIrJit::detectCustomKernelArch: gcnArchName empty; "
                    "defaulting to gfx1151";
    return "gfx1151";
  }
  if (auto colon = arch.find(':'); colon != std::string::npos)
    arch.resize(colon); // "gfx1100:sramecc+:xnack-" -> "gfx1100"
  return arch;
}
#endif // HIPDNN_EP_LINK_HIP_HOST

// Resolve a plugin's addLibrary() argument to a file on disk. Accepts a path
// that already exists as-is, or a bare name resolved against the plugin's
// addLibraryPath() dirs using the platform's library naming (mirroring what the
// native lld-link path resolves). Empty string when nothing matches.
std::string resolvePluginLibrary(llvm::StringRef nameOrPath,
                                 llvm::ArrayRef<std::string> searchPaths) {
  if (llvm::sys::fs::exists(nameOrPath))
    return nameOrPath.str();

  std::string name = nameOrPath.str();
  std::vector<std::string> candidates;
#ifdef _WIN32
  candidates = {name + ".dll", name + ".lib"};
#else
  candidates = {"lib" + name + ".so", "lib" + name + ".a"};
#endif
  for (const std::string &dir : searchPaths) {
    for (const std::string &cand : candidates) {
      llvm::SmallString<256> full(dir);
      llvm::sys::path::append(full, cand);
      if (llvm::sys::fs::exists(full))
        return std::string(full.data(), full.size());
    }
  }
  return {};
}

// A shared library is loaded into the JIT via DynamicLibrarySearchGenerator; a
// static archive via StaticLibraryDefinitionGenerator. Dispatch on extension.
bool isSharedLibraryFile(llvm::StringRef path) {
#ifdef _WIN32
  return path.ends_with_insensitive(".dll");
#else
  return path.ends_with(".so") || path.contains(".so.");
#endif
}

// Install symbol search generators on the JIT's main JITDylib, in priority
// order: per-OS ROCm libs, the per-arch kernel DLL/SO, plugin-contributed
// libraries, then the process image.
// The kernel DLL is added before the process generator so its hip_* launchers
// win over any look-alike symbol in another loaded module. Returns false only
// when a mandatory load fails.
bool installSearchGenerators(llvm::orc::LLJIT &jit) {
  auto &jd = jit.getMainJITDylib();
  const char global_prefix = jit.getDataLayout().getGlobalPrefix();

  std::string last_load_error;
  auto loadFirst = [&](llvm::ArrayRef<const char *> names) {
    for (const char *lib : names) {
      auto gen =
          llvm::orc::DynamicLibrarySearchGenerator::Load(lib, global_prefix);
      if (!gen) {
        last_load_error = llvm::toString(gen.takeError());
        continue;
      }
      jd.addGenerator(std::move(*gen));
      return true;
    }
    return false;
  };

  // ROCm libs the runtime calls into (the EP only imports amdhip64 directly).
  // A missing lib is tolerated: a model that never touches it still JITs.
  const std::vector<std::vector<const char *>> rocm_libs = {
#ifdef _WIN32
      {"hipblaslt.dll", "libhipblaslt.dll"},
#else
      {"libhipblaslt.so"},
#endif
  };
  for (const std::vector<const char *> &names : rocm_libs) {
    if (loadFirst(names))
      continue;
    std::string tried;
    for (const char *lib : names) {
      if (!tried.empty())
        tried += ", ";
      tried += lib;
    }
    if (hipdnn_ep::env_enabled("HIPDNN_EP_DEBUG")) {
      LOG(WARNING) << "LlvmIrJit: Load(" << tried
                   << ") failed: " << last_load_error
                   << "; models needing it will fail at lookup.";
    }
  }

#ifdef _WIN32
  const char *const kHipdnnBackend = "hipdnn_backend.dll";
#else
  const char *const kHipdnnBackend = "libhipdnn_backend.so";
#endif
  if (!loadFirst({kHipdnnBackend})) {
    if (hipdnn_ep::env_enabled("HIPDNN_EP_DEBUG")) {
      LOG(INFO) << "LlvmIrJit: Load(" << kHipdnnBackend
                << ") failed: " << last_load_error
                << "; hipDNN graph models will fail at lookup.";
    }
  }

#ifdef HIPDNN_EP_LOAD_KERNEL_DLLS
  // Per-arch GPU kernel DLL/SO, dlopen'd from the EP binary's own directory.
  {
    const std::string arch = detectCustomKernelArch();
#ifdef _WIN32
    const std::string basename = "custom_kernels_" + arch + ".dll";
    const char sep = '\\';
#else
    const std::string basename = "libcustom_kernels_" + arch + ".so";
    const char sep = '/';
#endif
    std::string kernel_lib = thisModuleDirectory();
    if (!kernel_lib.empty())
      kernel_lib.push_back(sep);
    kernel_lib += basename;

    auto gen = llvm::orc::DynamicLibrarySearchGenerator::Load(
        kernel_lib.c_str(), global_prefix);
    if (!gen) {
      LOG(ERROR) << "LlvmIrJit: Load(" << kernel_lib << ") for arch=" << arch
                 << " failed: " << llvm::toString(gen.takeError())
                 << ". Install " << basename << " next to the EP DLL/SO.";
      return false;
    }
    jd.addGenerator(std::move(*gen));
  }
#endif // HIPDNN_EP_LOAD_KERNEL_DLLS

  // Plugin-contributed libraries (addLibrary / addLibraryPath): load each so a
  // plugin's kernel/host symbols resolve, the same way the in-tree
  // custom_kernels dylib is loaded above -- a shared lib via
  // DynamicLibrarySearchGenerator, a static archive via
  // StaticLibraryDefinitionGenerator. Added before the process generator so a
  // plugin symbol is preferred over a coincidental process-image look-alike.
  // No-op when no plugin contributed a library.
  {
    // SmallVector<std::string> binds to the ArrayRef parameter directly.
    auto searchPaths = ::hip::compiler::pluginLibraryPaths();
    for (llvm::StringRef lib : ::hip::compiler::pluginLibraries()) {
      std::string file = resolvePluginLibrary(lib, searchPaths);
      if (file.empty()) {
        LOG(WARNING) << "LlvmIrJit: plugin library '" << lib.str()
                     << "' not found on its search paths; symbols it provides "
                        "will fail at lookup.";
        continue;
      }
      if (isSharedLibraryFile(file)) {
        auto gen = llvm::orc::DynamicLibrarySearchGenerator::Load(
            file.c_str(), global_prefix);
        if (!gen) {
          LOG(ERROR) << "LlvmIrJit: plugin library Load(" << file
                     << ") failed: " << llvm::toString(gen.takeError());
          return false;
        }
        jd.addGenerator(std::move(*gen));
      } else {
        auto gen = llvm::orc::StaticLibraryDefinitionGenerator::Load(
            jit.getObjLinkingLayer(), file.c_str());
        if (!gen) {
          LOG(ERROR) << "LlvmIrJit: plugin static library Load(" << file
                     << ") failed: " << llvm::toString(gen.takeError());
          return false;
        }
        jd.addGenerator(std::move(*gen));
      }
    }
  }

#ifdef _WIN32
  // Pin the C runtime to the shared UCRT before the process generator (ORC
  // tries generators in add-order). Binds the runtime's stdio family to the
  // same UCRT that owns the stderr/_iob FILE; otherwise the process search
  // picks legacy msvcrt.dll's fputs over a ucrtbase FILE -> heap corruption
  // under perf=1. Match this build's CRT: debug (/MTd, /MDd; _DEBUG) ships
  // ucrtbased.dll, release ships ucrtbase.dll. operator new/delete and
  // type_info stay pinned to the EP DLL via the absolute shims (which outrank
  // generators), so the C++ heap is untouched.
#ifdef _DEBUG
  constexpr const char *kUcrtDll = "ucrtbased.dll";
#else
  constexpr const char *kUcrtDll = "ucrtbase.dll";
#endif
  if (auto ucrt = llvm::orc::DynamicLibrarySearchGenerator::Load(
          kUcrtDll, global_prefix)) {
    jd.addGenerator(std::move(*ucrt));
  } else {
    LOG(INFO) << "LlvmIrJit: Load(" << kUcrtDll
              << ") failed: " << llvm::toString(ucrt.takeError())
              << "; stdio may bind to a foreign CRT under perf=1.";
  }
#endif // _WIN32

  // Process image: EP DLL, amdhip64, the libs above, the CRT, ORT, ...
  auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      global_prefix);
  if (!gen) {
    LOG(ERROR) << "LlvmIrJit: process search generator failed: "
               << llvm::toString(gen.takeError());
    return false;
  }
  jd.addGenerator(std::move(*gen));
  return true;
}

} // namespace

// PImpl keeps the LLVM ORC headers out of LlvmIrJit.h.
struct LlvmIrJit::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::vector<AtexitEntry> atexit_dtors;
  bool initialized = false;
};

LlvmIrJit::LlvmIrJit() : impl_(std::make_unique<Impl>()) {}

LlvmIrJit::~LlvmIrJit() {
  if (impl_ && impl_->jit && impl_->initialized) {
    // LIFO-drain JIT'd dtors before the slab unmaps in LLJIT teardown.
    for (auto it = impl_->atexit_dtors.rbegin();
         it != impl_->atexit_dtors.rend(); ++it) {
      if (!it->fn)
        continue;
      if (it->has_arg) {
        reinterpret_cast<void (*)(void *)>(it->fn)(it->arg);
      } else {
        reinterpret_cast<void (*)(void)>(it->fn)();
      }
    }
    impl_->atexit_dtors.clear();

    if (auto err = impl_->jit->deinitialize(impl_->jit->getMainJITDylib())) {
      LOG(WARNING) << "LlvmIrJit::~LlvmIrJit: deinitialize failed: "
                   << llvm::toString(std::move(err));
    }
  }
}

std::unique_ptr<LlvmIrJit>
LlvmIrJit::create(const std::vector<uint8_t> &bitcode,
                  const std::string &module_name) {
  ensureLLVMNativeTargetInitialized();

  // LLJIT auto-selects the object linking layer by triple
  // (LLJITBuilderState::prepareForConstruction): RuntimeDyld on Windows
  // (x86_64 COFF), JITLink on Linux (x86_64 ELF). Both are linked in (see this
  // dir's CMakeLists) for that reason.
  auto jit_or_err = llvm::orc::LLJITBuilder().create();
  if (!jit_or_err) {
    LOG(ERROR) << "LlvmIrJit::create: LLJITBuilder failed: "
               << llvm::toString(jit_or_err.takeError());
    return nullptr;
  }
  auto jit = std::move(*jit_or_err);

  if (auto err = installPlatformSymbolShims(*jit)) {
    LOG(ERROR) << "LlvmIrJit::create: platform symbol shims failed: "
               << llvm::toString(std::move(err));
    return nullptr;
  }
  installAtexitShims(*jit);

  if (auto err = installRuntimeStreamShims(*jit)) {
    LOG(ERROR) << "LlvmIrJit::create: runtime stream shims failed: "
               << llvm::toString(std::move(err));
    return nullptr;
  }

  if (!installSearchGenerators(*jit))
    return nullptr;

  // runtime.bc and the per-model bitcode share one LLVMContext so cross-module
  // type identity holds (LLVM interns types per-context).
  if (runtime_bc_data_size == 0) {
    LOG(ERROR) << "LlvmIrJit::create: embedded runtime.bc is empty -- EP DLL "
                  "built without Clang; per-model hipdnn_ep_runtime_* symbols "
                  "have nowhere to resolve. Rebuild with Clang.";
    return nullptr;
  }

  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::LLVMContext &ctx = *context;

  // Parse a .bc blob and stamp the JIT's host triple/datalayout onto it. The
  // per-model module is emitted with both stripped (see LLVMBackend::
  // emitLlvmIr); without the stamp IRCompileLayer rejects the module.
  auto parseAndStamp =
      [&](llvm::StringRef bytes,
          const std::string &name) -> std::unique_ptr<llvm::Module> {
    auto buf = llvm::MemoryBuffer::getMemBufferCopy(bytes, name);
    auto mod_or_err = llvm::parseBitcodeFile(buf->getMemBufferRef(), ctx);
    if (!mod_or_err) {
      LOG(ERROR) << "LlvmIrJit::create: parse " << name
                 << " failed: " << llvm::toString(mod_or_err.takeError());
      return nullptr;
    }
    auto mod = std::move(*mod_or_err);
    mod->setSourceFileName(name);
    mod->setDataLayout(jit->getDataLayout());
    mod->setTargetTriple(jit->getTargetTriple());
    return mod;
  };

  auto module = parseAndStamp(
      llvm::StringRef(reinterpret_cast<const char *>(bitcode.data()),
                      bitcode.size()),
      module_name);
  if (!module)
    return nullptr;

  auto runtime_module = parseAndStamp(
      llvm::StringRef(reinterpret_cast<const char *>(runtime_bc_data),
                      runtime_bc_data_size),
      "runtime.bc");
  if (!runtime_module)
    return nullptr;

  // Plugin-contributed runtime bitcode (addRuntimeBitcode): parse each into the
  // same shared context so it can be added as a sibling module alongside
  // runtime.bc below. This is what makes a plugin's wrap_* definitions resolve
  // for the JIT'd model on the default (LLVM_IR/JIT) path -- the native
  // linkRuntime merge, which links plugin bitcode too, does not run here.
  // Empty when no plugin contributed bitcode.
  std::vector<std::unique_ptr<llvm::Module>> plugin_modules;
  {
    unsigned idx = 0;
    for (const auto &buf : ::hip::compiler::pluginBitcodeBuffers()) {
      auto plugin_module = parseAndStamp(
          llvm::StringRef(static_cast<const char *>(buf.data), buf.sizeBytes),
          "plugin." + std::to_string(idx++) + ".bc");
      if (!plugin_module)
        return nullptr;
      plugin_modules.push_back(std::move(plugin_module));
    }
  }

  auto tsc = llvm::orc::ThreadSafeContext(std::move(context));

  // Add runtime first so the per-model module's external hipdnn_ep_* refs
  // resolve in the JITDylib.
  if (auto err = jit->addIRModule(
          llvm::orc::ThreadSafeModule(std::move(runtime_module), tsc))) {
    LOG(ERROR) << "LlvmIrJit::create: addIRModule(runtime) failed: "
               << llvm::toString(std::move(err));
    return nullptr;
  }
  // Then plugin bitcode, so plugin-defined wrap_* symbols are visible to the
  // per-model module too. Vendors must prefix their symbols: two modules
  // defining the same name in one JITDylib is a duplicate-definition error, so
  // the native path's OverrideFromSrc shadowing is not available under JIT.
  for (auto &plugin_module : plugin_modules) {
    if (auto err = jit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(plugin_module), tsc))) {
      LOG(ERROR) << "LlvmIrJit::create: addIRModule(plugin bitcode) failed: "
                 << llvm::toString(std::move(err));
      return nullptr;
    }
  }
  if (auto err = jit->addIRModule(
          llvm::orc::ThreadSafeModule(std::move(module), tsc))) {
    LOG(ERROR) << "LlvmIrJit::create: addIRModule failed: "
               << llvm::toString(std::move(err));
    return nullptr;
  }

  std::unique_ptr<LlvmIrJit> result(new LlvmIrJit());

  // Run @llvm.global_ctors (ORC doesn't do this automatically); skipping it
  // leaves static caches zero-filled. Scope the TLS pointer so the atexit
  // shims capture this JIT's dtor registrations into Impl::atexit_dtors.
  auto *prev_atexit_list = tls_current_atexit_list;
  tls_current_atexit_list = &result->impl_->atexit_dtors;
  auto init_err = jit->initialize(jit->getMainJITDylib());
  tls_current_atexit_list = prev_atexit_list;
  if (init_err) {
    LOG(ERROR) << "LlvmIrJit::create: initialize (global_ctors) failed: "
               << llvm::toString(std::move(init_err));
    return nullptr;
  }

  result->impl_->jit = std::move(jit);
  result->impl_->initialized = true;
  return result;
}

void *LlvmIrJit::lookup_raw(const char *name) const {
  if (!impl_ || !impl_->jit || !name)
    return nullptr;

  auto sym_or_err = impl_->jit->lookup(name);
  if (!sym_or_err) {
    llvm::consumeError(sym_or_err.takeError());
    return nullptr;
  }
  return reinterpret_cast<void *>(
      static_cast<uintptr_t>(sym_or_err->getValue()));
}

} // namespace mlir_compilation::customop
