/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Target/LLVM/LLVMBackend.h"

#include "hip/Compiler/PluginRegistry.h"

#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include "hip/debug_log.h"

#include <system_error>

namespace hipdnn {

LLVMBackend::LLVMBackend() : target_initialized_(false) {
  // Target initialization is deferred to first use (native path only).
}

LLVMBackend::~LLVMBackend() = default;

void LLVMBackend::initializeTarget() {
  if (target_initialized_) {
    return;
  }

  // Initialize native target for object file emission
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  target_initialized_ = true;
}

std::unique_ptr<llvm::Module>
LLVMBackend::translateMLIRtoLLVMIR(mlir::ModuleOp mlirModule,
                                   llvm::LLVMContext &llvmContext) {
  // Register LLVM IR translation dialects
  mlir::registerBuiltinDialectTranslation(*mlirModule->getContext());
  mlir::registerLLVMDialectTranslation(*mlirModule->getContext());

  // Translate MLIR to LLVM IR using C++ library API
  auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate MLIR to LLVM IR\n";
    return nullptr;
  }

  // Verify the generated LLVM IR
  std::string error_msg;
  llvm::raw_string_ostream error_stream(error_msg);
  if (llvm::verifyModule(*llvmModule, &error_stream)) {
    llvm::errs() << "LLVM IR verification failed:\n" << error_msg << "\n";
    return nullptr;
  }

  return llvmModule;
}

void LLVMBackend::optimizeLLVMIR(llvm::Module *module, int optLevel) {
  if (!module || optLevel < 0 || optLevel > 3) {
    llvm::errs() << "Invalid arguments to optimizeLLVMIR\n";
    return;
  }

  if (optLevel == 0) {
    // No optimization
    return;
  }

  // Create pass builder with optimization level
  llvm::PassBuilder PB;

  // Create analysis managers
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // Register all the basic analyses with the managers
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create optimization pipeline based on level
  llvm::ModulePassManager MPM;
  if (optLevel == 1) {
    MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1);
  } else if (optLevel == 2) {
    MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
  } else if (optLevel == 3) {
    MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  }

  // Run optimization passes
  MPM.run(*module, MAM);
}

// ---- Bitcode backend -------------------------------------------------------

// LLJIT fills triple + data layout from detectHost at load time.
static void stripTargetMetadata(llvm::Module &module) {
  module.setTargetTriple(llvm::Triple());
  module.setDataLayout(llvm::DataLayout(""));
}

bool LLVMBackend::emitLlvmIr(llvm::Module *module,
                             const std::string &outputPath) {
  if (!module) {
    llvm::errs() << "Null module in emitLlvmIr\n";
    return false;
  }

  stripTargetMetadata(*module);

  std::error_code EC;
  llvm::raw_fd_ostream out(outputPath, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Failed to open bitcode output file: " << EC.message()
                 << "\n";
    return false;
  }

  llvm::WriteBitcodeToFile(*module, out);
  out.flush();
  if (out.has_error()) {
    llvm::errs() << "Failed to write bitcode to: " << outputPath << "\n";
    return false;
  }

  COMPILER_DEBUG_LOG("Emitted LLVM bitcode to: " << outputPath << "\n");
  return true;
}

// ---- Native backend --------------------------------------------------------

llvm::TargetMachine *LLVMBackend::createTargetMachine() {
  initializeTarget();

  // Get target triple for current platform
  llvm::Triple target_triple(llvm::sys::getDefaultTargetTriple());

  // Look up target
  std::string error_msg;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(target_triple.str(), error_msg);
  if (!target) {
    llvm::errs() << "Failed to lookup target: " << error_msg << "\n";
    return nullptr;
  }

  // Configure target machine
  std::string cpu = "generic";
  std::string features = "";
  llvm::TargetOptions options;
  // Route C++ static ctors into DT_INIT_ARRAY (LLVM's legacy default is the
  // `.ctors` section which glibc's loader silently drops on dlopen,
  // leaving every static unordered_map / string in the generated DLL at
  // zero-init BSS → SIGFPE on first emplace's `hash % bucket_count(0)`).
  //
  // Not an LLVM bug — `UseInitArray=false` is a backwards-compat default
  // for legacy ELF targets that lacked .init_array. clang's own driver
  // sets this to true for modern Linux (see clang/lib/Driver/ToolChains/
  // Gnu.cpp), but we bypass the driver and drive the TargetMachine API
  // directly, so we have to flip it ourselves.
  options.UseInitArray = true;
  llvm::Reloc::Model RM =
      llvm::Reloc::PIC_; // Position-independent code for DLL

  llvm::TargetMachine *TM =
      target->createTargetMachine(target_triple, cpu, features, options, RM);

  if (!TM) {
    llvm::errs() << "Failed to create target machine\n";
    return nullptr;
  }

  return TM;
}

bool LLVMBackend::compileToObjectFile(llvm::Module *module,
                                      const std::string &outputPath) {
  if (!module) {
    llvm::errs() << "Null module in compileToObjectFile\n";
    return false;
  }

  // Create target machine
  std::unique_ptr<llvm::TargetMachine> TM(createTargetMachine());
  if (!TM) {
    return false;
  }

  // Set module data layout and target triple
  module->setDataLayout(TM->createDataLayout());
  module->setTargetTriple(TM->getTargetTriple());

  // Open output file
  std::error_code EC;
  llvm::raw_fd_ostream out(outputPath, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Failed to open output file: " << EC.message() << "\n";
    return false;
  }

  // Create legacy pass manager for code generation
  llvm::legacy::PassManager pass;

  // Add pass to emit object file
  if (TM->addPassesToEmitFile(pass, out, nullptr,
                              llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "TargetMachine can't emit object file\n";
    return false;
  }

  // Run code generation passes
  pass.run(*module);
  out.flush();

  COMPILER_DEBUG_LOG("Compiled object file to: " << outputPath << "\n");
  return true;
}

bool LLVMBackend::compileToObjectInMemory(llvm::Module *module,
                                          std::vector<uint8_t> &outBytes) {
  if (!module) {
    llvm::errs() << "Null module in compileToObjectInMemory\n";
    return false;
  }

  // Create target machine
  std::unique_ptr<llvm::TargetMachine> TM(createTargetMachine());
  if (!TM) {
    return false;
  }

  // Set module data layout and target triple
  module->setDataLayout(TM->createDataLayout());
  module->setTargetTriple(TM->getTargetTriple());

  // Use SmallVector with raw_svector_ostream for in-memory compilation
  llvm::SmallVector<char, 0> objBuffer;
  llvm::raw_svector_ostream objStream(objBuffer);

  // Create legacy pass manager for code generation
  llvm::legacy::PassManager pass;

  // Add pass to emit object file to memory stream
  if (TM->addPassesToEmitFile(pass, objStream, nullptr,
                              llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "TargetMachine can't emit object file\n";
    return false;
  }

  // Run code generation passes
  pass.run(*module);
  // Note: raw_svector_ostream auto-flushes, flush() is deleted in newer LLVM

  // Copy result to output vector
  outBytes.assign(objBuffer.begin(), objBuffer.end());

  COMPILER_DEBUG_LOG("Compiled object to memory: " << outBytes.size()
                                                   << " bytes\n");
  return true;
}

// Extern declarations for embedded Runtime bitcode
// Generated by CMake: runtime.bc → xxd.py → runtime_ir_data.cpp
extern "C" const unsigned char runtime_bc_data[];
extern "C" const size_t runtime_bc_data_size;

// Helper: parse a bitcode buffer and link it into `linker` with the
// caller-supplied `flags`. Used by linkRuntimeModule for both the
// in-tree runtime (no flags) and plugin contributions
// (Linker::Flags::OverrideFromSrc, so a plugin's symbols shadow any
// in-tree definition with the same name).
static bool linkBitcodeBuffer(llvm::Linker &linker, llvm::LLVMContext &context,
                              const void *data, std::size_t sizeBytes,
                              llvm::StringRef bufName, unsigned flags) {
  auto memBuf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(data), sizeBytes), bufName,
      /*RequiresNullTerminator=*/false);

  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOrErr =
      llvm::parseBitcodeFile(memBuf->getMemBufferRef(), context);
  if (!moduleOrErr) {
    llvm::errs() << "Error: Failed to parse bitcode '" << bufName
                 << "': " << llvm::toString(moduleOrErr.takeError()) << "\n";
    return false;
  }

  if (linker.linkInModule(std::move(*moduleOrErr), flags)) {
    llvm::errs() << "Error: Failed to link bitcode '" << bufName
                 << "' into destination\n";
    return false;
  }
  return true;
}

bool LLVMBackend::linkRuntimeModule(llvm::Module *destModule) {
  if (!destModule) {
    llvm::errs() << "Error: Null destination module\n";
    return false;
  }

  llvm::Linker linker(*destModule);

  // Step 1: in-tree runtime bitcode.
  //
  // Use pre-calculated bitcode size from the xxd-generated header.
  size_t bcSize = runtime_bc_data_size;
  if (bcSize == 0) {
    // Empty bitcode - this means Clang wasn't available during build.
    // Runtime IR merging is disabled, but plugin bitcode may still be
    // contributed below.
    COMPILER_DEBUG_LOG(
        "Warning: Runtime bitcode is empty (Clang not available during "
        "build).\n"
        "         Runtime IR merging disabled - accessor functions will have "
        "call overhead.\n"
        "         To enable zero-cost abstraction, rebuild with Clang "
        "installed.\n");
  } else {
    if (!linkBitcodeBuffer(linker, destModule->getContext(), runtime_bc_data,
                           bcSize, "runtime.bc", /*flags=*/0)) {
      return false;
    }
  }

  // Step 2: plugin-contributed bitcode.
  //
  // Linked AFTER the in-tree runtime with `OverrideFromSrc` so a vendor
  // `wrap_*` definition replaces the in-tree one. Overlaying through the
  // linker is safer than mutating the destination module after the fact.
  //
  // Order: insertion order from `pluginBitcodeBuffers()`. If two plugins
  // contribute the same symbol, the later one wins (link search-order
  // semantics); this is documented in the design doc rather than promising a
  // deterministic-by-name order.
  unsigned i = 0;
  for (const auto &buf : ::hip::compiler::pluginBitcodeBuffers()) {
    std::string name = "plugin." + std::to_string(i++) + ".bc";
    if (!linkBitcodeBuffer(linker, destModule->getContext(), buf.data,
                           buf.sizeBytes, name,
                           llvm::Linker::Flags::OverrideFromSrc)) {
      return false;
    }
  }

  return true;
}

} // namespace hipdnn
