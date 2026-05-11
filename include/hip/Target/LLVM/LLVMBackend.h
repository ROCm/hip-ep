//===- LLVMBackend.h - LLVM IR backend declarations for the HIP compiler - *-
// C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
#ifndef HIP_TARGET_LLVM_LLVMBACKEND_H
#define HIP_TARGET_LLVM_LLVMBACKEND_H

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/IR/BuiltinOps.h>

#include <memory>
#include <string>

namespace hipdnn {

// LLVM Backend: Translates MLIR to LLVM IR and optionally compiles to native
// object file Supports two compilation modes:
//   1. IR Mode: MLIR → LLVM IR (.ll text file) - for debugging and
//   cross-platform
//   2. Native Mode: MLIR → LLVM IR → Object File (.obj/.o) - for production DLL
//   generation

class LLVMBackend {
public:
  LLVMBackend();
  ~LLVMBackend();

  // MLIR → LLVM IR translation (used by both modes)
  // Translates MLIR module in LLVM dialect to LLVM IR using C++ library API
  // Returns nullptr on failure
  std::unique_ptr<llvm::Module>
  translateMLIRtoLLVMIR(mlir::ModuleOp mlirModule,
                        llvm::LLVMContext &llvmContext);

  // LLVM IR optimization (used by both modes)
  // Runs standard optimization passes at specified level (0-3)
  // Level 0: No optimization
  // Level 1: Basic optimization
  // Level 2: Default optimization (recommended)
  // Level 3: Aggressive optimization
  void optimizeLLVMIR(llvm::Module *module, int optLevel);

  // IR Mode: Emit LLVM IR to text file (.ll)
  // Returns true on success, false on failure
  bool emitLLVMIR(llvm::Module *module, const std::string &outputPath);

  // Native Mode: Compile LLVM IR to object file (.obj on Windows, .o on Linux)
  // Uses LLVM TargetMachine to emit native code
  // Returns true on success, false on failure
  bool compileToObjectFile(llvm::Module *module, const std::string &outputPath);

  // In-Memory Mode: Emit LLVM IR to string (for EPContext storage)
  // Returns true on success, false on failure
  bool emitLLVMIRToString(llvm::Module *module, std::string &outIR);

  // In-Memory Mode: Compile LLVM IR to object in memory (for EPContext storage)
  // Returns true on success, false on failure
  bool compileToObjectInMemory(llvm::Module *module,
                               std::vector<uint8_t> &outBytes);

  // Link embedded Runtime IR into destination module
  // Merges Runtime bitcode with generated IR for zero-cost abstraction
  // Returns true on success, false on failure
  bool linkRuntimeModule(llvm::Module *destModule);

private:
  // Helper: Initialize LLVM target for current platform
  void initializeTarget();

  // Helper: Create TargetMachine for native compilation
  llvm::TargetMachine *createTargetMachine();

  bool target_initialized_;
};

} // namespace hipdnn

#endif // HIP_TARGET_LLVM_LLVMBACKEND_H
