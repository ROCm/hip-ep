/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
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

#include "HipBufferize.h"
#include "HipDialect.h"
#include "HipPasses.h"

#include <iostream>

int main(int argc, char** argv) {
  std::string inputFilename;
  std::string outputDll;
  bool skipBufferize = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-o" && i + 1 < argc) {
      outputDll = argv[++i];
    } else if (std::string(argv[i]) == "--no-bufferize") {
      skipBufferize = true;
    } else if (argv[i][0] != '-') {
      inputFilename = argv[i];
    }
  }

  if (inputFilename.empty() || outputDll.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " [--no-bufferize] <input.mlir> -o <output.dll>\n";
    return 1;
  }

  // 1. Setup MLIR context and parse input
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  mlir::MLIRContext context(registry);

  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    std::cerr << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    std::cerr << "Error parsing MLIR file\n";
    return 1;
  }

  // 2. Run MLIR PassManager
  mlir::PassManager pm(&context);

  if (!skipBufferize) {
    mlir::bufferization::OneShotBufferizePassOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  }

  pm.addPass(mlir::hip::createConvertHipToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(*module))) {
    std::cerr << "Error running passes\n";
    return 1;
  }

  // 3. DLL Export Injection is done after translation to LLVM IR (step 4)

  // 4. Translate MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
  if (!llvmModule) {
    std::cerr << "Error translating to LLVM IR\n";
    return 1;
  }

  // Mark all non-declaration functions as dllexport so they appear in the DLL
  for (auto& func : *llvmModule) {
    if (!func.isDeclaration()) {
      func.setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    }
  }

  // 5. Code Generation to Object File
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::string err;
  auto targetTripleStr = llvm::sys::getDefaultTargetTriple();
  llvm::Triple targetTriple(targetTripleStr);
  auto target = llvm::TargetRegistry::lookupTarget(targetTripleStr, err);
  if (!target) {
    std::cerr << "Error looking up target: " << err << "\n";
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
    std::cerr << "Could not open file: " << EC.message() << "\n";
    return 1;
  }

  llvm::legacy::PassManager pass;
  auto fileType = llvm::CodeGenFileType::ObjectFile;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
    std::cerr << "TargetMachine can't emit a file of this type\n";
    return 1;
  }
  pass.run(*llvmModule);
  dest.flush();
  dest.close();

  std::cout << "Generated temporary object: " << objFilename << "\n";

  // 6. Link into DLL
  auto linkerPath = llvm::sys::findProgramByName("lld-link");
  if (!linkerPath) {
    linkerPath = llvm::sys::findProgramByName("link.exe");
    if (!linkerPath) {
      std::cerr << "Could not find lld-link or link.exe in PATH\n";
      return 1;
    }
  }

  std::string therockDist = "";
  if (const char* env_p = std::getenv("THEROCK_DIST")) {
    therockDist = env_p;
  }

  std::string outArg = "/OUT:" + outputDll;
  std::string dllArg = "/DLL";
  std::string noLogoArg = "/NOLOGO";

  std::vector<llvm::StringRef> linkArgs;
  linkArgs.push_back(*linkerPath);
  linkArgs.push_back(noLogoArg);
  linkArgs.push_back(dllArg);
  linkArgs.push_back(outArg);
  linkArgs.push_back(objFilename);

  std::string exePath =
      llvm::sys::fs::getMainExecutable(argv[0], (void*)(intptr_t)main);
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
    std::cerr << "Linker failed with code " << result << "\n";
    if (!errMsg.empty()) {
      std::cerr << "Error: " << errMsg << "\n";
    }
    return 1;
  }

  std::cout << "Successfully generated " << outputDll
            << " and its import library\n";

  return 0;
}
