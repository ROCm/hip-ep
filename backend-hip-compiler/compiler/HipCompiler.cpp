/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipCompiler.h"
#include "InterfaceGenerator.h"

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

#include <fstream>
#include <iostream>

namespace hipdnn::compiler {

namespace {

mlir::MLIRContext createContext() {
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

  return mlir::MLIRContext(registry);
}

ModelMetadata extractMetadata(mlir::ModuleOp module) {
  ModelMetadata meta;

  // Extract pool_size and buffer_offsets from module or function attributes
  module.walk([&](mlir::func::FuncOp funcOp) {
    if (meta.entryFunction.empty())
      meta.entryFunction = funcOp.getName().str();

    if (auto poolAttr =
            funcOp->getAttrOfType<mlir::IntegerAttr>("hipdnn.pool_size"))
      meta.poolSize = poolAttr.getInt();

    if (auto offsetsAttr = funcOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "hipdnn.buffer_offsets")) {
      auto vals = offsetsAttr.asArrayRef();
      meta.bufferOffsets.assign(vals.begin(), vals.end());
    }

    // Count inputs vs outputs by looking at function signature and attributes
    auto funcType = funcOp.getFunctionType();
    int totalArgs = funcType.getNumInputs();

    // Args with "bufferize.result" attribute are outputs
    int outputCount = 0;
    for (int i = 0; i < totalArgs; ++i) {
      if (funcOp.getArgAttr(i, "bufferize.result"))
        ++outputCount;
    }
    meta.outputCount = outputCount > 0 ? outputCount : 0;
    meta.inputCount = totalArgs - meta.outputCount;

    // Extract ranks and shapes from memref types
    for (int i = 0; i < totalArgs; ++i) {
      auto type = funcType.getInput(i);
      int rank = 0;
      std::vector<int64_t> shape;
      if (auto memrefTy = type.dyn_cast<mlir::MemRefType>()) {
        rank = memrefTy.getRank();
        for (int64_t dim : memrefTy.getShape())
          shape.push_back(dim);
      }

      if (funcOp.getArgAttr(i, "bufferize.result")) {
        meta.outputRanks.push_back(rank);
        meta.outputShapes.push_back(shape);
      } else {
        meta.inputRanks.push_back(rank);
        meta.inputShapes.push_back(shape);
      }
    }
  });

  return meta;
}

bool runPasses(mlir::ModuleOp module, mlir::MLIRContext &context) {
  mlir::PassManager pm(&context);
  pm.addPass(mlir::hip::createConvertHipToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return mlir::succeeded(pm.run(module));
}

std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module,
                                                llvm::LLVMContext &llvmCtx,
                                                CompileMode mode) {
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmCtx);
  if (!llvmModule)
    return nullptr;

  for (auto &func : *llvmModule) {
    if (func.isDeclaration())
      continue;
    if (mode == CompileMode::Plugin) {
      // In plugin mode, only interface functions get dllexport;
      // rename internal compute function so it doesn't clash
      func.setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
    } else {
      func.setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    }
  }

  return llvmModule;
}

bool emitObjectFile(llvm::Module &llvmModule, const std::string &objPath) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::string err;
  auto tripleStr = llvm::sys::getDefaultTargetTriple();
  llvm::Triple triple(tripleStr);
  auto *target = llvm::TargetRegistry::lookupTarget(tripleStr, err);
  if (!target) {
    std::cerr << "Error looking up target: " << err << "\n";
    return false;
  }

  llvm::TargetOptions opt;
  auto rm = std::optional<llvm::Reloc::Model>();
  auto targetMachine =
      target->createTargetMachine(triple, "generic", "", opt, rm);

  llvmModule.setDataLayout(targetMachine->createDataLayout());
  llvmModule.setTargetTriple(triple);

  std::error_code EC;
  llvm::raw_fd_ostream dest(objPath, EC, llvm::sys::fs::OF_None);
  if (EC) {
    std::cerr << "Could not open file: " << objPath << ": " << EC.message()
              << "\n";
    return false;
  }

  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    std::cerr << "TargetMachine can't emit object file\n";
    return false;
  }
  pass.run(llvmModule);
  dest.flush();
  dest.close();
  return true;
}

bool linkDll(const std::vector<std::string> &objFiles,
             const std::string &outputDll, const CompileOptions &options) {
  auto linkerPath = llvm::sys::findProgramByName("lld-link");
  if (!linkerPath) {
    linkerPath = llvm::sys::findProgramByName("link.exe");
    if (!linkerPath) {
      std::cerr << "Could not find lld-link or link.exe in PATH\n";
      return false;
    }
  }

  std::string therockDist;
  if (const char *env = std::getenv("THEROCK_DIST"))
    therockDist = env;

  std::string outArg = "/OUT:" + outputDll;

  // Build link arg strings (keep alive for StringRef)
  std::vector<std::string> argStorage;
  argStorage.push_back(*linkerPath);
  argStorage.push_back("/NOLOGO");
  argStorage.push_back("/DLL");
  argStorage.push_back(outArg);
  for (auto &obj : objFiles)
    argStorage.push_back(obj);

  // Runtime lib
  if (!options.runtimeLibDir.empty()) {
    argStorage.push_back(options.runtimeLibDir + "/hip_runtime_static.lib");
    argStorage.push_back("/LIBPATH:" + options.runtimeLibDir);
  }

  argStorage.push_back("/LIBPATH:.");

  if (!therockDist.empty())
    argStorage.push_back("/LIBPATH:" + therockDist + "\\lib");

  argStorage.push_back("amdhip64.lib");
  argStorage.push_back("hipblaslt.lib");
  argStorage.push_back("MIOpen.lib");

  std::vector<llvm::StringRef> linkArgs;
  for (auto &s : argStorage)
    linkArgs.push_back(s);

  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(*linkerPath, linkArgs, std::nullopt,
                                         {}, 0, 0, &errMsg);
  if (result != 0) {
    std::cerr << "Linker failed with code " << result << "\n";
    if (!errMsg.empty())
      std::cerr << "Error: " << errMsg << "\n";
    return false;
  }
  return true;
}

std::optional<std::vector<uint8_t>> readFileBytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    return std::nullopt;

  auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
    return std::nullopt;
  return buffer;
}

std::optional<CompileResult>
compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
              mlir::MLIRContext &context, const std::string &outputPath,
              const CompileOptions &options) {
  // Extract metadata before lowering (func signatures still available)
  auto metadata = extractMetadata(module.get());
  std::cout << "Entry function: " << metadata.entryFunction << "\n";
  std::cout << "Inputs: " << metadata.inputCount
            << ", Outputs: " << metadata.outputCount << "\n";

  // Run MLIR passes (HIP -> LLVM)
  if (!runPasses(module.get(), context)) {
    std::cerr << "Error running MLIR passes\n";
    return std::nullopt;
  }

  // Translate to LLVM IR
  llvm::LLVMContext llvmCtx;
  auto llvmModule = translateToLLVMIR(module.get(), llvmCtx, options.mode);
  if (!llvmModule) {
    std::cerr << "Error translating to LLVM IR\n";
    return std::nullopt;
  }

  // In plugin mode, add interface wrapper functions to the LLVM module
  if (options.mode == CompileMode::Plugin) {
    InterfaceGenerator::addInterfaceFunctions(*llvmModule, metadata);
  }

  // Emit object file
  std::string objPath =
      llvm::sys::path::stem(outputPath).str() + "_model.obj";
  if (!emitObjectFile(*llvmModule, objPath)) {
    std::cerr << "Error emitting object file\n";
    return std::nullopt;
  }
  std::cout << "Generated object: " << objPath << "\n";

  // Link into DLL
  std::vector<std::string> objFiles = {objPath};

  if (!linkDll(objFiles, outputPath, options)) {
    return std::nullopt;
  }

  std::cout << "Successfully generated " << outputPath << "\n";

  // Read DLL bytes for the result
  auto dllBytes = readFileBytes(outputPath);
  if (!dllBytes) {
    std::cerr << "Failed to read generated DLL\n";
    return std::nullopt;
  }

  CompileResult result;
  result.dllBytes = std::move(*dllBytes);
  result.metadata = std::move(metadata);
  return result;
}

} // namespace

std::optional<CompileResult>
HipCompiler::compileFile(const std::string &inputPath,
                         const std::string &outputPath,
                         const CompileOptions &options) {
  auto context = createContext();

  std::string errorMessage;
  auto file = mlir::openInputFile(inputPath, &errorMessage);
  if (!file) {
    std::cerr << errorMessage << "\n";
    return std::nullopt;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    std::cerr << "Error parsing MLIR file\n";
    return std::nullopt;
  }

  return compileModule(module, context, outputPath, options);
}

std::optional<CompileResult>
HipCompiler::compile(const std::string &mlirText,
                     const std::string &outputPath,
                     const CompileOptions &options) {
  auto context = createContext();

  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(mlirText, "input.mlir");
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    std::cerr << "Error parsing MLIR text\n";
    return std::nullopt;
  }

  return compileModule(module, context, outputPath, options);
}

} // namespace hipdnn::compiler
