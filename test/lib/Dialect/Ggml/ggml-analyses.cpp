/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runs what the frontend works out from the graph - which states a split may
// update in place (StateDonation.h) - on each module of a file and prints it,
// so the tests can check it on its own:
//
//   --donation  ggml.state_pairs with each pair's input_donation_eligible,
//               and why a state may not be updated in place
//
// Each module is printed as its header line (`module attributes {...}`)
// first. --split-input-file and --verify-diagnostics are hip-mlir-opt's.

#include "hip/Dialect/Ggml/Transforms/StateDonation.h"

#include "hip/Dialect/Ggml/IR/GgmlDialect.h"
#include "hip/Dialect/Ggml/IR/GgmlStatePairs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace mlir;

namespace {

struct Options {
  bool donation = false;
  bool split = false;
  bool verify = false;
  std::string input;
};

void printModule(ModuleOp module, raw_ostream &os) {
  os << "module";
  if (!module->getAttrDictionary().empty()) {
    os << " attributes ";
    module->getAttrDictionary().print(os);
  }
  os << "\n";
}

LogicalResult analyze(ModuleOp module, const Options &options,
                      raw_ostream &os) {
  printModule(module, os);
  if (options.donation) {
    FailureOr<SmallVector<ggml::StateDonation>> donation =
        ggml::proveStateDonation(module);
    if (failed(donation))
      return failure();
    auto entry = module.lookupSymbol<func::FuncOp>("main_graph");
    std::string reason;
    FailureOr<SmallVector<ggml::StatePair>> pairs =
        entry ? ggml::readStatePairs(entry, reason)
              : FailureOr<SmallVector<ggml::StatePair>>(
                    SmallVector<ggml::StatePair>{});
    if (failed(pairs) || pairs->size() != donation->size())
      return failure();
    if (pairs->empty())
      return success();
    Builder builder(module.getContext());
    SmallVector<Attribute> decided;
    for (auto [pair, decision] : llvm::zip(*pairs, *donation)) {
      SmallVector<NamedAttribute> fields(pair.attributes.getValue());
      fields.emplace_back(builder.getStringAttr("input_donation_eligible"),
                          builder.getBoolAttr(decision.inPlace));
      decided.push_back(builder.getDictionaryAttr(fields));
    }
    os << "ggml.state_pairs = " << builder.getArrayAttr(decided) << "\n";
    for (auto [pair, decision] : llvm::zip(*pairs, *donation))
      if (!decision.inPlace)
        os << "state '" << pair.name.getValue()
           << "' is not donatable: " << decision.reason << "\n";
  }
  return success();
}

LogicalResult processBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer,
                            const Options &options, MLIRContext &context,
                            raw_ostream &os) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  const auto run = [&]() -> LogicalResult {
    OwningOpRef<ModuleOp> module =
        parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module)
      return failure();
    return analyze(*module, options, os);
  };
  if (!options.verify) {
    SourceMgrDiagnosticHandler handler(sourceMgr, &context);
    return run();
  }
  SourceMgrDiagnosticVerifierHandler handler(sourceMgr, &context);
  (void)run();
  return handler.verify();
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const llvm::StringRef argument = argv[index];
    if (argument == "--donation")
      options.donation = true;
    else if (argument == "--split-input-file")
      options.split = true;
    else if (argument == "--verify-diagnostics")
      options.verify = true;
    else if (!argument.starts_with("-") && options.input.empty())
      options.input = argument.str();
    else {
      llvm::errs() << "unknown argument " << argument << "\n";
      return 2;
    }
  }
  if (options.input.empty() || !options.donation) {
    llvm::errs() << "usage: ggml-analyses --donation [--split-input-file] "
                    "[--verify-diagnostics] FILE\n";
    return 2;
  }

  DialectRegistry registry;
  registry.insert<func::FuncDialect, ggml::GgmlDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  // As hip-mlir-opt: an expected diagnostic is the message alone.
  context.printOpOnDiagnostic(!options.verify);

  std::string error;
  std::unique_ptr<llvm::MemoryBuffer> input =
      openInputFile(options.input, &error);
  if (!input) {
    llvm::errs() << error << "\n";
    return 1;
  }
  const auto process = [&](std::unique_ptr<llvm::MemoryBuffer> chunk,
                           raw_ostream &os) {
    return processBuffer(std::move(chunk), options, context, os);
  };
  if (options.split)
    return failed(
               splitAndProcessBuffer(std::move(input), process, llvm::outs()))
               ? 1
               : 0;
  return failed(process(std::move(input), llvm::outs())) ? 1 : 0;
}
