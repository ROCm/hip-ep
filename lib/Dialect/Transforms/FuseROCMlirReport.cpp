/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FuseROCMlirReport.cpp - JSON-lines report of rocMLIR kernels -------===//
//
// Describes the kernels a rocMLIR fusion pipeline made, so that two fusion
// pipelines can be compared on the same input IR. Each record is independent
// of the kernel symbol names and of the kernel order, because two pipelines
// may number and place equal kernels differently.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/xxhash.h"

#include <map>

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FUSEROCMLIRREPORTPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

std::string typeToString(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

llvm::json::Array typesToJson(TypeRange types) {
  llvm::json::Array array;
  for (Type type : types)
    array.push_back(typeToString(type));
  return array;
}

// std::map keeps the keys sorted, so equal multisets print equal JSON.
llvm::json::Object countsToJson(const std::map<std::string, int64_t> &counts) {
  llvm::json::Object object;
  for (const auto &[name, count] : counts)
    object[name] = count;
  return object;
}

std::string hashKernelBody(func::FuncOp kernel) {
  std::string text;
  llvm::raw_string_ostream os(text);
  kernel->print(os, OpPrintingFlags().useLocalScope());
  std::string symbol = ("@" + kernel.getSymName() + "(").str();
  if (size_t pos = text.find(symbol); pos != std::string::npos)
    text.replace(pos, symbol.size(), "@kernel(");
  return llvm::utohexstr(llvm::xxh3_64bits(text), /*LowerCase=*/true);
}

llvm::json::Object describeKernel(func::FuncOp kernel) {
  std::map<std::string, int64_t> counts;
  std::string anchor;
  kernel.walk([&](Operation *op) {
    if (op == kernel.getOperation() || isa<func::ReturnOp>(op))
      return;
    ++counts[op->getName().getStringRef().str()];
    if (anchor.empty() &&
        isa<ConvOp, MatmulOp, GemmOp, QuantConvOp, QuantMatmulOp>(op))
      anchor = op->getName().getStringRef().str();
  });
  FunctionType type = kernel.getFunctionType();
  return llvm::json::Object{
      {"kind", "kernel"},
      {"name", kernel.getSymName().str()},
      {"anchor", anchor},
      {"ops", countsToJson(counts)},
      {"inputs", typesToJson(type.getInputs())},
      {"outputs", typesToJson(type.getResults())},
      {"hash", hashKernelBody(kernel)},
  };
}

llvm::json::Object describeHost(func::FuncOp func) {
  std::map<std::string, int64_t> counts;
  func.walk([&](Operation *op) {
    if (op != func.getOperation())
      ++counts[op->getName().getStringRef().str()];
  });
  return llvm::json::Object{
      {"kind", "host"},
      {"name", func.getSymName().str()},
      {"ops", countsToJson(counts)},
  };
}

struct FuseROCMlirReportPass
    : public impl::FuseROCMlirReportPassBase<FuseROCMlirReportPass> {
  using impl::FuseROCMlirReportPassBase<
      FuseROCMlirReportPass>::FuseROCMlirReportPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::vector<llvm::json::Object> records;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr("rock.kernel") || func.isExternal())
        continue;
      records.push_back(describeHost(func));
      func.walk([&](RocMlirOp dispatch) {
        auto kernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            dispatch, dispatch.getKernelAttr());
        if (kernel)
          records.push_back(describeKernel(kernel));
      });
    }

    std::error_code error;
    std::unique_ptr<llvm::raw_fd_ostream> file;
    llvm::raw_ostream *os = &llvm::errs();
    if (output != "-") {
      file = std::make_unique<llvm::raw_fd_ostream>(output, error,
                                                    llvm::sys::fs::OF_Append);
      if (error) {
        module.emitError() << "cannot open fusion report '" << output
                           << "': " << error.message();
        return signalPassFailure();
      }
      os = file.get();
    }
    for (llvm::json::Object &record : records)
      *os << llvm::json::Value(std::move(record)) << "\n";
    os->flush();
    markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
