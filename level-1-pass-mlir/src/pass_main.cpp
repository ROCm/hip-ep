/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <glog/logging.h>
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace morphizen;
using namespace morphizen_cxx;

namespace {

struct Level1MlirPass {
  Level1MlirPass(IPass& self) : self_{self} {}
  
  void process(IPass& self, Graph& graph) {
    LOG(INFO) << "Level1MlirPass::process() called";
   
    // Save graph to file for MLIR processing
    LOG(INFO) << "Saving graph to file...";
    auto graph_ref = GraphConstRef(graph);
    std::string graph_file = "graph_for_mlir.txt";
    graph_ref.save(graph_file);
    LOG(INFO) << "Graph saved to file: " << graph_file;
    
    // Parse MLIR file to mlir::ModuleOp
    LOG(INFO) << "Parsing MLIR file to ModuleOp...";
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.allowUnregisteredDialects();
    
    auto moduleRef = mlir::parseSourceFile<mlir::ModuleOp>(graph_file, &context);
    
    if (!moduleRef) {
      LOG(INFO) << "Failed to parse MLIR string to ModuleOp";
    } else {
      LOG(INFO) << "Successfully parsed MLIR string to ModuleOp";
      
      // Get the module operation
      mlir::ModuleOp module = *moduleRef;
      LOG(INFO) << "ModuleOp created, ready for MLIR transformations";
      
      
      // Print module with detailed flags
      mlir::OpPrintingFlags flags;
      flags.printGenericOpForm();
      flags.enableDebugInfo();
      flags.printValueUsers();
      std::cout << "ModuleOp content:" << std::endl;
      module.print(llvm::outs(), flags);
      std::cout << std::endl;
      
      // Future MLIR transformations would go here:
      // 1. Apply MLIR passes/optimizations
      // 2. Transform the IR
      // 3. Convert back to ONNX if needed
    }
  }

  IPass& self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1MlirPass, morphizen_pass_level1_mlir)
