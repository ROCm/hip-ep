/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

// MLIR includes
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace morphizen;
using namespace morphizen_cxx;

DEF_ENV_PARAM(MLIR_PRINT_WITH_VERBOSE, "0")

namespace {

struct Level1MlirPass {
  Level1MlirPass(IPass &self) : self_{self} {}

  void process(IPass &self, Graph &graph) {
    LOG(INFO) << "Level1MlirPass::process() called";

    auto graph_ref = GraphConstRef(graph);
    auto graph_string = graph_ref.save_string();
    LOG(INFO) << "Graph serialized to bytecode, size: " << graph_string->size();

    // Parse MLIR bytecode to mlir::ModuleOp
    LOG(INFO) << "Parsing MLIR bytecode to ModuleOp...";
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.allowUnregisteredDialects();

    // Create a MemoryBuffer from the bytecode data
    auto memBuffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(graph_string->data(), graph_string->size()),
        "mlir-bytecode",
        /*RequiresNullTerminator=*/false);

    // Create SourceMgr and add the buffer
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

    // Parse bytecode using parseSourceFile with SourceMgr
    mlir::ParserConfig parserConfig(&context);
    auto moduleRef =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);

    if (!moduleRef) {
      LOG(INFO) << "Failed to parse MLIR bytecode to ModuleOp";
    } else {
      // Get the module operation
      mlir::ModuleOp module = *moduleRef;

      // Print module with detailed flags
      mlir::OpPrintingFlags flags;
      if (ENV_PARAM(MLIR_PRINT_WITH_VERBOSE)) {
        flags.printGenericOpForm();
        flags.enableDebugInfo();
        flags.printValueUsers();
      }
      std::cout << "ModuleOp content:" << std::endl;
      module.print(llvm::outs(), flags);
      std::cout << std::endl;

      // Future MLIR transformations would go here:
      // 1. Apply MLIR passes/optimizations
      // 2. Transform the IR
      // 3. Convert back to ONNX if needed
    }
  }

  IPass &self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1MlirPass, morphizen_pass_level1_mlir)
