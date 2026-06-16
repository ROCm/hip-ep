/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef LLVM_BACKEND_H
#define LLVM_BACKEND_H

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/IR/BuiltinOps.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn {

// LLVM Backend: MLIR -> LLVM IR, then one of two artifact backends selected
// by the compile option (CompilationOptions.output_mode):
//   * Bitcode (default): emitLlvmIr writes OS-portable .bc with the target
//     triple + data layout stripped; LlvmIrJit in the EP fills both from
//     detectHost at JIT time.
//   * Native: linkRuntimeModule merges the embedded runtime.bc at producer
//     time, compileToObjectFile emits a host object, and DLLLinker links a
//     per-OS .dll/.so. Used for benchmarking/dev (signed-DLL policy keeps it
//     out of production deployment).
class LLVMBackend {
public:
  LLVMBackend();
  ~LLVMBackend();

  // MLIR -> LLVM IR translation (used by both backends). Returns nullptr on
  // failure.
  std::unique_ptr<llvm::Module>
  translateMLIRtoLLVMIR(mlir::ModuleOp mlirModule,
                        llvm::LLVMContext &llvmContext);

  // Target-independent PerModule optimization pipeline at level 0-3 (used by
  // both backends).
  void optimizeLLVMIR(llvm::Module *module, int optLevel);

  // ---- Bitcode backend (default) -------------------------------------------
  // Emits OS-portable LLVM bitcode (triple + data layout stripped). Consumed
  // by LlvmIrJit in the EP DLL.
  bool emitLlvmIr(llvm::Module *module, const std::string &outputPath);

  // ---- Native backend ------------------------------------------------------
  // Compile LLVM IR to a host object file (.obj on Windows, .o on Linux) using
  // a PIC TargetMachine for the default host triple.
  bool compileToObjectFile(llvm::Module *module, const std::string &outputPath);

  // In-memory variant of compileToObjectFile.
  bool compileToObjectInMemory(llvm::Module *module,
                               std::vector<uint8_t> &outBytes);

  // Merge the embedded runtime.bc into destModule at producer time. Native
  // path only: the bitcode path JIT-loads runtime.bc separately in the EP so
  // the artifact stays OS-portable. Returns true (degraded no-op) when the
  // embedded runtime bitcode is empty (Clang absent at build configure).
  bool linkRuntimeModule(llvm::Module *destModule);

private:
  // Helper: Initialize LLVM native target for object file emission.
  void initializeTarget();

  // Helper: Create a PIC TargetMachine for the default host triple.
  llvm::TargetMachine *createTargetMachine();

  bool target_initialized_;
};

} // namespace hipdnn

#endif // LLVM_BACKEND_H
