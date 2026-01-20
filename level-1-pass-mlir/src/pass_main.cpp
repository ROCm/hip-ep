/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <glog/logging.h>
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR) >= n)

using namespace vaip_core;

namespace {

struct Level1MlirPass {
  Level1MlirPass(IPass& self) : self_{self} {}
  
  void process(IPass& self, Graph& graph) {
    MY_LOG(1) << "Level1MlirPass::process() called";
    std::cout << "================================================= " << std::endl;
    
    auto nodes = graph_nodes(graph);
    MY_LOG(1) << "Graph has " << nodes.size() << " nodes";
    
    // Save graph to file for MLIR processing
    MY_LOG(1) << "Saving graph to file...";
    auto graph_ref = vaip_cxx::GraphConstRef(graph);
    std::string graph_file = "graph_for_mlir.onnx";
    graph_ref.save(graph_file);
    MY_LOG(1) << "Graph saved to file: " << graph_file;
    
    // Parse MLIR file to mlir::ModuleOp
    MY_LOG(1) << "Parsing MLIR file to ModuleOp...";
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.allowUnregisteredDialects();
    
    auto moduleRef = mlir::parseSourceFile<mlir::ModuleOp>(graph_file, &context);
    
    if (!moduleRef) {
      MY_LOG(1) << "Failed to parse MLIR string to ModuleOp";
    } else {
      MY_LOG(1) << "Successfully parsed MLIR string to ModuleOp";
      
      // Get the module operation
      mlir::ModuleOp module = *moduleRef;
      MY_LOG(1) << "ModuleOp created, ready for MLIR transformations";
      
      // Walk operations and print module
      MY_LOG(1) << "Walking operations in module...";
      int op_count = 0;
      module.walk([&](mlir::Operation* op) {
        op_count++;
        MY_LOG(2) << "  Op #" << op_count << ": " << op->getName().getStringRef().str();
      });
      MY_LOG(1) << "Total operations in module: " << op_count;
      
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

DEFINE_VAIP_PASS(Level1MlirPass, vaip_pass_level1_mlir)
