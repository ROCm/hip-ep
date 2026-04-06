/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "../../HipDNNGraph/HipDNNGraph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace hip {

void GraphDeleter::operator()(void *ptr) const {
  delete static_cast<::hip::graph::HipDNNGraph *>(ptr);
}

namespace {

/// Inline the outline region back into the parent block, restoring the
/// original ONNX ops.  Used when hipDNN graph compilation fails.
static void unoutline(HipDNNGraphOutlineOp outlineOp) {
  Block &block = outlineOp.getBody().front();

  IRMapping mapping;
  for (auto [blockArg, input] :
       llvm::zip(block.getArguments(), outlineOp.getInputs()))
    mapping.map(blockArg, input);

  OpBuilder builder(outlineOp);
  for (Operation &op : block.without_terminator())
    builder.clone(op, mapping);

  auto *terminator = block.getTerminator();
  for (auto [yieldVal, result] :
       llvm::zip(terminator->getOperands(), outlineOp.getResults()))
    result.replaceAllUsesWith(mapping.lookup(yieldVal));

  outlineOp->erase();
}

//===----------------------------------------------------------------------===//
// CompileHipDNNGraphsPass
//===----------------------------------------------------------------------===//

struct CompileHipDNNGraphsPass
    : public PassWrapper<CompileHipDNNGraphsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompileHipDNNGraphsPass)

  CompileHipDNNGraphsPass(hipdnnHandle_t handle, CompiledGraphMap output_graphs)
      : handle_(handle), output_graphs_(std::move(output_graphs)) {}

  StringRef getArgument() const override { return "compile-hipdnn-graphs"; }

  StringRef getDescription() const override {
    return "Compile hip.hipdnn_graph_outline regions via hipDNN and replace "
           "with hip.hipdnn_graph (un-outlines on failure)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override;

private:
  hipdnnHandle_t handle_;
  CompiledGraphMap output_graphs_;
  int graph_count_{0};
};

void CompileHipDNNGraphsPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isDeclaration())
      continue;

    SmallVector<HipDNNGraphOutlineOp> outlineOps;
    func->walk([&](HipDNNGraphOutlineOp op) { outlineOps.push_back(op); });

    for (HipDNNGraphOutlineOp outlineOp : outlineOps) {
      auto graph = std::make_unique<::hip::graph::HipDNNGraph>(handle_);

      auto status = graph->BuildFromOnnxMLIR(outlineOp.getBody());
      if (status.failed()) {
        llvm::errs() << "[CompileHipDNNGraphs] BuildFromOnnxMLIR FAILED: "
                     << status.message() << "\n";
      }
      if (status.ok()) {
        status = graph->Compile();
        if (status.failed()) {
          llvm::errs() << "[CompileHipDNNGraphs] Compile FAILED: "
                       << status.message() << "\n";
        }
      }

      if (status.failed()) {
        llvm::errs() << "[CompileHipDNNGraphs] Falling back to standard "
                        "lowering for outlined op\n";
        unoutline(outlineOp);
        continue;
      }
      llvm::errs() << "[CompileHipDNNGraphs] SUCCESS: graph compiled\n";

      std::string graph_name = "hipdnn_graph_" + std::to_string(graph_count_);
      int32_t graph_id = graph_count_++;

      OpBuilder builder(outlineOp);
      auto loc = outlineOp->getLoc();

      SmallVector<Value> outs;
      for (auto result_type : outlineOp->getResultTypes()) {
        auto tensor_type = dyn_cast<RankedTensorType>(result_type);
        if (!tensor_type) {
          outlineOp->emitError("expected ranked tensor result type");
          signalPassFailure();
          return;
        }
        auto empty = tensor::EmptyOp::create(
            builder, loc, tensor_type.getShape(), tensor_type.getElementType());
        outs.push_back(empty);
      }

      auto input_uids = graph->getInputUids();
      auto output_uids = graph->getOutputUids();

      SmallVector<int64_t> input_uid_vec(input_uids.begin(), input_uids.end());
      SmallVector<int64_t> output_uid_vec(output_uids.begin(),
                                          output_uids.end());

      SmallVector<Value> inputs(outlineOp.getInputs());

      auto exec_op = HipDNNGraphOp::create(
          builder, loc,
          /*result_tensors=*/outlineOp->getResultTypes(),
          /*ctx=*/func.getArgument(0),
          /*inputs=*/inputs,
          /*outputs=*/outs,
          /*graph_id=*/builder.getI32IntegerAttr(graph_id),
          /*input_uids=*/builder.getI64ArrayAttr(input_uid_vec),
          /*output_uids=*/builder.getI64ArrayAttr(output_uid_vec));

      outlineOp->replaceAllUsesWith(exec_op->getResults());
      outlineOp->erase();

      // Defensive: output_graphs_ is always non-null when created via
      // CompilerDriver (see CompilerDriver.cpp). Guard retained since
      // createCompileHipDNNGraphsPass is a public API.
      if (output_graphs_)
        (*output_graphs_)[graph_name] = OwnedGraph(graph.release());
    }
  }
}

} // namespace

std::unique_ptr<Pass>
createCompileHipDNNGraphsPass(hipdnnHandle_t handle,
                              CompiledGraphMap output_graphs) {
  return std::make_unique<CompileHipDNNGraphsPass>(handle,
                                                   std::move(output_graphs));
}

} // namespace hip
} // namespace mlir
