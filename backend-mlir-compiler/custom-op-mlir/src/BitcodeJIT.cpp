/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "BitcodeJIT.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>

#include <glog/logging.h>

#include <cstring>
#include <mutex>
#include <new>
#include <typeinfo>

namespace mlir_compilation::customop {

namespace {

// Initialize the LLVM native target + asm printer + asm parser exactly
// once per process. ORC's `LLJITBuilder` calls
// `JITTargetMachineBuilder::detectHost` which requires the host target
// to be registered. Idempotent across concurrent BitcodeJIT::create
// invocations via std::call_once.
void ensureLLVMNativeTargetInitialized() {
  static std::once_flag once;
  std::call_once(once, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
}

#ifdef _WIN32

// JIT shim for C++ runtime symbols that the bitcode references but that
// are not exported by any DLL on the system.
//
// We link `runtime.bc` against the static MSVC CRT (/MT) into the EP DLL,
// and we feed clang `-fno-threadsafe-statics -fno-sized-deallocation
// -fno-rtti` (see `lib/Runtime/CMakeLists.txt`) so the bitcode does not
// reference magic-static guards, sized `operator delete`, or RTTI
// vtables. What's left is the small set of allocation helpers (operator
// new / delete) plus the LLVM-codegen-only `__emutls_get_address` /
// `__emutls_v.*` pair that ORC LLJIT emits on Windows when it lowers
// `thread_local` globals via emulated TLS.
//
// We define a tiny per-thread emutls helper below and register it
// alongside operator new/delete + the std::type_info vtable as absolute
// symbols in the JITDylib.
//
// On Linux this whole shim is unnecessary: LLVM's ORC native target uses
// real ELF TLS (not emulated), and `DynamicLibrarySearchGenerator::
// GetForCurrentProcess` resolves the Itanium-mangled allocation helpers
// (`_Znwm`, `_ZdlPv`, ...) directly from the running process via
// `dlsym(RTLD_DEFAULT, ...)`. See the no-op overload at the bottom of
// the `#else` branch.

// Maximum number of distinct emulated-TLS control blocks tracked per
// thread. The current merged runtime bitcode only ever asks for one
// (`_Init_thread_epoch`); the headroom is for forward compatibility.
constexpr size_t kEmutlsCapacity = 8;

// Emulated-TLS control block layout, matching libcompiler_rt:
//
//   struct __emutls_control {
//     size_t size;
//     size_t align;
//     union { uintptr_t index; void *address; } object;
//     void *value;          // initial value (or nullptr)
//   };
//
// LLVM's TLSEmulation pass replaces every `thread_local` global load
// with `__emutls_get_address(&__emutls_v.<name>)`. We provide a single
// implementation here -- thread-safe (per-thread storage), lock-free,
// suitable for the small number of TLS globals JITted bitcode can
// reach inside our process.
struct EmutlsControl {
  size_t size;
  size_t align;
  union {
    uintptr_t index;
    void *address;
  } object;
  void *value;
};

// Per-thread allocation table. Keyed by control-block pointer so we can
// support an arbitrary number of TLS globals; in practice the JIT only
// asks for one or two. Linear scan over kEmutlsCapacity entries is
// trivially fast.
thread_local void *g_emutls_storage[kEmutlsCapacity];
thread_local EmutlsControl *g_emutls_keys[kEmutlsCapacity];

void *windowsEmutlsGetAddress(EmutlsControl *ctrl) {
  for (size_t i = 0; i < kEmutlsCapacity; ++i) {
    if (g_emutls_keys[i] == ctrl)
      return g_emutls_storage[i];
  }
  for (size_t i = 0; i < kEmutlsCapacity; ++i) {
    if (g_emutls_keys[i] != nullptr)
      continue;
    void *p = ::operator new(ctrl->size);
    if (ctrl->value)
      std::memcpy(p, ctrl->value, ctrl->size);
    else
      std::memset(p, 0, ctrl->size);
    g_emutls_keys[i] = ctrl;
    g_emutls_storage[i] = p;
    return p;
  }
  // Table exhausted. With the current bitcode this is unreachable; if
  // we ever grow past kEmutlsCapacity TLS globals, bump the constant.
  LOG(FATAL) << "BitcodeJIT __emutls_get_address: per-thread table full";
  return nullptr;
}

// Register operator new/delete + emutls helper + std::type_info vtable
// as `absoluteSymbols` in the JIT's MainJITDylib so the bitcode's
// references resolve to addresses inside the EP DLL. See the comment
// block above for the full rationale.
llvm::Error installPlatformSymbolShims(llvm::orc::LLJIT &jit) {
  llvm::orc::SymbolMap absolute_syms;
  auto add = [&](llvm::StringRef name, void *addr) {
    absolute_syms[jit.getExecutionSession().intern(name)] =
        llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr(reinterpret_cast<uintptr_t>(addr)),
            llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute);
  };

  // Operator new / delete (sized + unsized). Mangled names follow the
  // MSVC x64 convention so they appear verbatim in the bitcode.
  void *op_new_sz =
      reinterpret_cast<void *>(static_cast<void *(*)(size_t)>(::operator new));
  void *op_delete = reinterpret_cast<void *>(
      static_cast<void (*)(void *) noexcept>(::operator delete));
  void *op_delete_sz = reinterpret_cast<void *>(
      static_cast<void (*)(void *, size_t) noexcept>(::operator delete));
  add("??2@YAPEAX_K@Z", op_new_sz);     // operator new(size_t)
  add("??3@YAXPEAX@Z", op_delete);      // operator delete(void*)
  add("??3@YAXPEAX_K@Z", op_delete_sz); // operator delete(void*, size_t)

  // Emulated TLS shim defined above. The control-block global
  // `__emutls_v.<name>` is emitted by LLVM codegen into the bitcode's
  // module itself, so we don't need to provide it -- we just need to
  // satisfy the `__emutls_get_address` call.
  add("__emutls_get_address",
      reinterpret_cast<void *>(&windowsEmutlsGetAddress));

  // `??_7type_info@@6B@` is the vtable of `std::type_info`. Clang on
  // Windows MSVC emits RTTI descriptors for any catchable exception
  // type referenced in the runtime (e.g. `std::bad_alloc`,
  // `std::exception`), and each descriptor's first field is a pointer
  // to this vtable. `-fno-rtti` does NOT suppress the exception RTTI
  // path, so we resolve the symbol by reading the vtable pointer from
  // a live `std::type_info` instance in the EP DLL. `typeid(int)` is
  // a stable polymorphic object whose vtable pointer is identical to
  // `??_7type_info@@6B@` in the same image (MSVC ABI).
  const std::type_info &ti = typeid(int);
  void *type_info_vtbl =
      *reinterpret_cast<void *const *>(static_cast<const void *>(&ti));
  add("??_7type_info@@6B@", type_info_vtbl);

  return jit.getMainJITDylib().define(
      llvm::orc::absoluteSymbols(std::move(absolute_syms)));
}

#else // !_WIN32

// On Linux the bitcode's operator new/delete come from libstdc++.so via
// `dlsym(RTLD_DEFAULT, ...)` through the process-wide search generator
// installed in BitcodeJIT::create. ORC's ELF target uses native TLS, so
// `__emutls_get_address` is never referenced. `-fno-rtti` keeps the
// MSVC `??_7type_info@@6B@` vtable out of the picture too. Nothing to
// inject.
inline llvm::Error installPlatformSymbolShims(llvm::orc::LLJIT & /*jit*/) {
  return llvm::Error::success();
}

#endif // _WIN32

} // namespace

// PImpl: keep all LLVM ORC types out of the public header so callers
// (InferenceState, MlirCustomOp) don't need an LLVM include path.
struct BitcodeJIT::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  // Whether `jit->initialize(MainJITDylib)` has run successfully. We
  // mirror it in the destructor so we only call `deinitialize` when
  // initialization actually completed, and so we can skip the call
  // entirely if the JIT was torn down half-built.
  bool initialized = false;
};

BitcodeJIT::BitcodeJIT() : impl_(std::make_unique<Impl>()) {}

BitcodeJIT::~BitcodeJIT() {
  if (impl_ && impl_->jit && impl_->initialized) {
    // Run `@llvm.global_dtors` for the JIT'd module. Errors here are
    // logged but not propagated -- we're in a destructor and any
    // failure would just mean static destructors are skipped, which
    // is no worse than what we did before this destructor existed.
    if (auto err = impl_->jit->deinitialize(impl_->jit->getMainJITDylib())) {
      LOG(WARNING) << "BitcodeJIT::~BitcodeJIT: deinitialize failed: "
                   << llvm::toString(std::move(err));
    }
  }
}

std::unique_ptr<BitcodeJIT>
BitcodeJIT::create(const std::vector<uint8_t> &bitcode,
                   const std::string &module_name) {
  ensureLLVMNativeTargetInitialized();

  // Stage 1: build a default LLJIT instance. The default builder picks a
  // host TargetMachine (X86 on the current builds) with O2-class codegen
  // and the legacy RuntimeDyld+ORC linking layer, which is the
  // best-tested configuration on Windows.
  auto jit_or_err = llvm::orc::LLJITBuilder().create();
  if (!jit_or_err) {
    LOG(ERROR) << "BitcodeJIT::create: LLJITBuilder failed: "
               << llvm::toString(jit_or_err.takeError());
    return nullptr;
  }
  auto jit = std::move(*jit_or_err);

  // Stage 2-pre: install platform-specific absolute-symbol shims. No-op
  // on Linux; see installPlatformSymbolShims for the Windows rationale.
  if (auto err = installPlatformSymbolShims(*jit)) {
    LOG(ERROR) << "BitcodeJIT::create: failed to install platform "
                  "symbol shims: "
               << llvm::toString(std::move(err));
    return nullptr;
  }

  // Stage 2a: explicitly LoadLibrary the ROCm shared libraries that the
  // per-model bitcode calls into through the merged runtime.bc (MIOpen,
  // hipBLASLt, hipDNN backend). The EP DLL itself only imports amdhip64
  // directly -- under the previous per-model-DLL build the higher-level
  // ROCm libs came in transitively because each generated model imported
  // them as DLL imports, but the bitcode path has no static-link step,
  // so we load them once at JIT-init time and ORC's
  // GetForCurrentProcess resolver picks up their symbols on subsequent
  // lookups.
  //
  // Each Load failure is logged but tolerated: a model that doesn't
  // touch MIOpen (e.g. a pure-kernel test bitcode) should still JIT,
  // and the eventual symbol lookup will fail loudly with an actionable
  // "undefined symbol" message if the lib is truly required.
  const char *const rocm_libs[] = {
#ifdef _WIN32
      "MIOpen.dll",
      "libhipblaslt.dll",
      "hipdnn_backend.dll",
#else
      "libMIOpen.so",
      "libhipblaslt.so",
      "libhipdnn_backend.so",
#endif
  };
  const char global_prefix = jit->getDataLayout().getGlobalPrefix();
  for (const char *lib : rocm_libs) {
    auto rocm_gen_or_err =
        llvm::orc::DynamicLibrarySearchGenerator::Load(lib, global_prefix);
    if (!rocm_gen_or_err) {
      LOG(WARNING) << "BitcodeJIT::create: failed to Load(" << lib
                   << "): " << llvm::toString(rocm_gen_or_err.takeError())
                   << ". Symbols from this DLL will be unavailable; "
                      "models that require it will fail at lookup.";
      continue;
    }
    jit->getMainJITDylib().addGenerator(std::move(*rocm_gen_or_err));
  }

  // Stage 2b: install a process-wide symbol search generator covering
  // every other already-loaded module -- the EP DLL itself (which now
  // exports the `hip_*` kernel launchers), `amdhip64_7.dll` (already
  // loaded as a transitive import of the EP DLL), the MSVC C runtime
  // (`memcpy`, `memset`, exception machinery), and anything else ORT
  // pulled in. `GetForCurrentProcess` walks loaded modules in load
  // order; the ROCm-specific generators added above are consulted
  // first, then the process-wide one, but none of these libraries
  // define overlapping symbols so order doesn't actually matter for
  // correctness.
  auto gen_or_err =
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          global_prefix);
  if (!gen_or_err) {
    LOG(ERROR) << "BitcodeJIT::create: process search generator failed: "
               << llvm::toString(gen_or_err.takeError());
    return nullptr;
  }
  jit->getMainJITDylib().addGenerator(std::move(*gen_or_err));

  // Stage 3: parse the bitcode into an LLVM Module. We copy the bytes
  // into a freshly allocated MemoryBuffer so the caller can free the
  // input vector immediately after returning from this function -- the
  // ThreadSafeModule below takes ownership of the parsed module and the
  // LLVMContext that owns its types.
  auto context = std::make_unique<llvm::LLVMContext>();
  auto buf = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(reinterpret_cast<const char *>(bitcode.data()),
                      bitcode.size()),
      module_name);
  auto module_or_err = llvm::parseBitcodeFile(buf->getMemBufferRef(), *context);
  if (!module_or_err) {
    LOG(ERROR) << "BitcodeJIT::create: parseBitcodeFile failed: "
               << llvm::toString(module_or_err.takeError());
    return nullptr;
  }
  auto module = std::move(*module_or_err);
  module->setSourceFileName(module_name);

  // Stage 4: hand the module to the JIT. Codegen runs lazily on the
  // first lookup, so this call is cheap; the heavy lifting moves to
  // `lookup_raw` (or, equivalently, to the first symbol resolution
  // triggered by InferenceState::create).
  llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
  if (auto err = jit->addIRModule(std::move(tsm))) {
    LOG(ERROR) << "BitcodeJIT::create: addIRModule failed: "
               << llvm::toString(std::move(err));
    return nullptr;
  }

  // Stage 5: run `@llvm.global_ctors`.
  //
  // The merged runtime bitcode pulls in ~8 static initializers (RNG
  // state in hipdnn_ep_runtime_tensor.cpp, op-profile counters, MIOpen
  // descriptor caches, gemm/matmul fast-path tables, ...). In the
  // previous native-DLL artifact path, MSVC's linker wired these into
  // the per-model DLL's `.CRT$XCU` section and the Windows loader ran
  // them on `LoadLibraryW`. In the bitcode path there is no DLL load,
  // and ORC LLJIT does NOT run constructors on its own -- the caller
  // is required to invoke `LLJIT::initialize` on the JITDylib that
  // owns the module.
  //
  // Skipping this call manifests as a hard-to-diagnose access
  // violation deep inside compute: `inference_init` completes (it
  // touches only zero-initialized state), then the first
  // `inference_compute` reads from a static cache that is still
  // zero-filled, dereferences a null vtable pointer, and crashes at
  // a fixed offset inside a JIT'd page. Symmetrically, we run
  // `deinitialize` in the destructor so static destructors get a
  // chance to release resources (cudaFree-equivalent on cached
  // descriptors, etc.) when the InferenceState is dropped.
  if (auto err = jit->initialize(jit->getMainJITDylib())) {
    LOG(ERROR) << "BitcodeJIT::create: initialize (global_ctors) failed: "
               << llvm::toString(std::move(err));
    return nullptr;
  }

  std::unique_ptr<BitcodeJIT> result(new BitcodeJIT());
  result->impl_->jit = std::move(jit);
  result->impl_->initialized = true;
  return result;
}

void *BitcodeJIT::lookup_raw(const char *name) const {
  if (!impl_ || !impl_->jit || !name)
    return nullptr;

  auto sym_or_err = impl_->jit->lookup(name);
  if (!sym_or_err) {
    // Absent symbols are normal for optional hooks (e.g.
    // `hipdnn_ep_runtime_begin_compute` on older modules during the
    // transition); consume the error so it doesn't become a
    // fatal-on-destruction unhandled `Error`.
    llvm::consumeError(sym_or_err.takeError());
    return nullptr;
  }
  return reinterpret_cast<void *>(
      static_cast<uintptr_t>(sym_or_err->getValue()));
}

} // namespace mlir_compilation::customop
