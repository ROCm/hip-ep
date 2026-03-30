/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Ported from hipDNNEP's convert_onnx_to_hipdnn_pass.cc.
 *
 * Key behavioral difference: hipDNNEP fails the pass if any onnx.* op can't
 * be compiled (it expects ALL ops to go through hipDNN). This version SKIPS
 * unsupported ops, leaving them for ConvertOnnxToHip to handle (hybrid model).
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

namespace mlir::hip {

void GraphDeleter::operator()(void *ptr) const {
  delete static_cast<::hip::graph::HipDNNGraph *>(ptr);
}

namespace {

static bool isSupportedOp(Operation *op) {
  return op->getName().getStringRef() == "onnx.Conv";
}

/// Build a temporary function containing a single ONNX op, compile it via
/// HipDNNGraph::BuildFromOnnxMLIR, and return the compiled graph.
/// Ported from hipDNNEP convert_onnx_to_hipdnn_pass.cc lines 26-64.
static std::unique_ptr<::hip::graph::HipDNNGraph>
compileOnnxOp(Operation *onnx_op, ModuleOp module, hipdnnHandle_t handle) {
  auto loc = onnx_op->getLoc();

  auto func_type = FunctionType::get(module.getContext(),
                                     onnx_op->getOperandTypes(),
                                     onnx_op->getResultTypes());

  OpBuilder module_builder = OpBuilder::atBlockEnd(module.getBody());
  auto temp_func = func::FuncOp::create(module_builder, loc,
                                        "__hipdnn_compile_temp__", func_type);
  Block *entry = temp_func.addEntryBlock();

  OpBuilder body_builder(entry, entry->begin());
  IRMapping mapping;
  for (auto [operand, arg] :
       llvm::zip(onnx_op->getOperands(), entry->getArguments())) {
    mapping.map(operand, arg);
  }

  Operation *cloned = body_builder.clone(*onnx_op, mapping);
  func::ReturnOp::create(body_builder, loc, cloned->getResults());

  auto graph = std::make_unique<::hip::graph::HipDNNGraph>(handle);
  auto status = graph->BuildFromOnnxMLIR(temp_func.getBody());
  if (status.failed()) {
    temp_func->erase();
    return nullptr;
  }

  status = graph->Compile();
  temp_func->erase();
  if (status.failed())
    return nullptr;

  return graph;
}

//===----------------------------------------------------------------------===//
// ConvertOnnxToHipDNNPass
//===----------------------------------------------------------------------===//

struct ConvertOnnxToHipDNNPass
    : public PassWrapper<ConvertOnnxToHipDNNPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOnnxToHipDNNPass)

  ConvertOnnxToHipDNNPass(hipdnnHandle_t handle,
                          CompiledGraphMap output_graphs)
      : handle_(handle), output_graphs_(std::move(output_graphs)) {}

  StringRef getArgument() const override { return "convert-onnx-to-hipdnn"; }

  StringRef getDescription() const override {
    return "Convert supported ONNX ops to hip.hipdnn_execute via hipDNN graph "
           "compilation (unsupported ops pass through to ConvertOnnxToHip)";
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

void ConvertOnnxToHipDNNPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isDeclaration())
      continue;

    SmallVector<Operation *> onnx_ops;
    func->walk([&](Operation *op) {
      if (isSupportedOp(op))
        onnx_ops.push_back(op);
    });

    for (Operation *onnx_op : onnx_ops) {
      auto compiled = compileOnnxOp(onnx_op, module, handle_);
      if (!compiled) {
        onnx_op->emitError("supported op failed to compile via hipDNN");
        signalPassFailure();
        return;
      }

      std::string graph_name =
          "hipdnn_graph_" + std::to_string(graph_count_);
      int32_t graph_id = graph_count_++;

      OpBuilder builder(onnx_op);
      auto loc = onnx_op->getLoc();

      // Create tensor.empty for DPS output destinations
      SmallVector<Value> outs;
      for (auto result_type : onnx_op->getResultTypes()) {
        auto tensor_type = cast<RankedTensorType>(result_type);
        auto empty = tensor::EmptyOp::create(
            builder, loc, tensor_type.getShape(),
            tensor_type.getElementType());
        outs.push_back(empty);
      }

      // Bake UIDs from the compiled graph into IR attributes
      auto input_uids = compiled->getInputUids();
      auto output_uids = compiled->getOutputUids();

      SmallVector<int64_t> input_uid_vec(input_uids.begin(),
                                         input_uids.end());
      SmallVector<int64_t> output_uid_vec(output_uids.begin(),
                                          output_uids.end());

      SmallVector<Value> inputs(onnx_op->getOperands());

      auto exec_op = HipDNNExecuteOp::create(
          builder, loc,
          /*result_tensors=*/onnx_op->getResultTypes(),
          /*ctx=*/func.getArgument(0),
          /*inputs=*/inputs,
          /*outputs=*/outs,
          /*graph_id=*/builder.getI32IntegerAttr(graph_id),
          /*input_uids=*/builder.getI64ArrayAttr(input_uid_vec),
          /*output_uids=*/builder.getI64ArrayAttr(output_uid_vec));

      onnx_op->replaceAllUsesWith(exec_op->getResults());
      onnx_op->erase();

      if (output_graphs_)
        (*output_graphs_)[graph_name] = OwnedGraph(compiled.release());
    }
  }
}

} // namespace

std::unique_ptr<Pass>
createConvertOnnxToHipDNNPass(hipdnnHandle_t handle,
                              CompiledGraphMap output_graphs) {
  return std::make_unique<ConvertOnnxToHipDNNPass>(handle,
                                                   std::move(output_graphs));
}

} // namespace mlir::hip
