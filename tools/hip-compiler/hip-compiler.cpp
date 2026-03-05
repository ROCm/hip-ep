/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Pipelines.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  std::string inputFilename;
  std::string outputDll;
  for (int argIdx = 1; argIdx < argc; ++argIdx) {
    if (std::string(argv[argIdx]) == "-o" && argIdx + 1 < argc) {
      outputDll = argv[++argIdx];
    } else if (argv[argIdx][0] != '-') {
      inputFilename = argv[argIdx];
    }
  }

  if (inputFilename.empty() || outputDll.empty()) {
    llvm::errs() << "Usage: " << argv[0]
                 << " <input.hip.mlir> -o <output.dll>\n";
    return 1;
  }

  // 1. Setup MLIR context and parse input
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  mlir::MLIRContext context(registry);

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR file\n";
    return 1;
  }

  // 2. Run MLIR PassManager (input must be pre-bufferized memref IR)
  mlir::PassManager pm(&context);
  mlir::hip::buildHipToLLVMPipeline(pm);

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "error: MLIR pass pipeline failed\n";
    return 1;
  }

  // 3. Translate MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
  if (!llvmModule) {
    llvm::errs() << "error: failed to translate to LLVM IR\n";
    return 1;
  }

  for (auto &func : *llvmModule) {
    if (!func.isDeclaration()) {
      func.setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    }
  }

  // 4. Code Generation to Object File
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::string err;
  auto targetTripleStr = llvm::sys::getDefaultTargetTriple();
  llvm::Triple targetTriple(targetTripleStr);
  auto target = llvm::TargetRegistry::lookupTarget(targetTripleStr, err);
  if (!target) {
    llvm::errs() << "error: looking up target: " << err << "\n";
    return 1;
  }

  llvm::TargetOptions opt;
  auto rm = std::optional<llvm::Reloc::Model>();
  auto targetMachine =
      target->createTargetMachine(targetTriple, "generic", "", opt, rm);

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  std::string objFilename = llvm::sys::path::stem(inputFilename).str() + ".obj";
  std::error_code EC;
  llvm::raw_fd_ostream dest(objFilename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "error: could not open file: " << EC.message() << "\n";
    return 1;
  }

  llvm::legacy::PassManager pass;
  auto fileType = llvm::CodeGenFileType::ObjectFile;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
    llvm::errs() << "error: target machine can't emit a file of this type\n";
    return 1;
  }
  pass.run(*llvmModule);
  dest.flush();
  dest.close();

  llvm::outs() << "Generated temporary object: " << objFilename << "\n";

  // 5. Link into DLL
  auto linkerPath = llvm::sys::findProgramByName("lld-link");
  if (!linkerPath) {
    linkerPath = llvm::sys::findProgramByName("link.exe");
    if (!linkerPath) {
      llvm::errs() << "error: could not find lld-link or link.exe in PATH\n";
      return 1;
    }
  }

  std::string therockDist;
  if (auto env = llvm::sys::Process::GetEnv("THEROCK_DIST"))
    therockDist = *env;

  std::string outArg = "/OUT:" + outputDll;
  std::string dllArg = "/DLL";
  std::string noLogoArg = "/NOLOGO";

  llvm::SmallVector<llvm::StringRef, 16> linkArgs;
  linkArgs.push_back(*linkerPath);
  linkArgs.push_back(noLogoArg);
  linkArgs.push_back(dllArg);
  linkArgs.push_back(outArg);
  linkArgs.push_back(objFilename);

  std::string exePath =
      llvm::sys::fs::getMainExecutable(argv[0], (void *)(intptr_t)main);
  llvm::StringRef exeDir = llvm::sys::path::parent_path(exePath);

  llvm::SmallString<128> runtimeLibPath(exeDir);
  llvm::sys::path::append(runtimeLibPath, "hip_runtime_static.lib");
  linkArgs.push_back(runtimeLibPath.str());

  std::string cwdLibPath = "/LIBPATH:.";
  linkArgs.push_back(cwdLibPath);

  std::string exeDirLibPath = std::string("/LIBPATH:") + exeDir.str();
  linkArgs.push_back(exeDirLibPath);

  std::string libPathArg;
  if (!therockDist.empty()) {
    libPathArg = "/LIBPATH:" + therockDist + "\\lib";
    linkArgs.push_back(libPathArg);
  }

  linkArgs.push_back("amdhip64.lib");
  linkArgs.push_back("hipblaslt.lib");
  linkArgs.push_back("MIOpen.lib");

  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(*linkerPath, linkArgs, std::nullopt,
                                         {}, 0, 0, &errMsg);
  if (result != 0) {
    llvm::errs() << "error: linker failed with code " << result << "\n";
    if (!errMsg.empty())
      llvm::errs() << "  " << errMsg << "\n";
    return 1;
  }

  llvm::outs() << "Successfully generated " << outputDll
               << " and its import library\n";

  return 0;
}
