/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/Pipeline.h"
#include "hip/InitAllPasses.h"
#include "hip/Support/DiskFileSystem.h"
#include "compilation_options_generated.h"

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/FileSystem.h"

namespace {

/// Pipeline options for morphizen-pipeline.
/// constants-file: embedded in the DLL metadata blob by generate-interface.
/// constants-dir:  directory where OnnxToHip writes the constants binary.
struct PipelineOptions : public mlir::PassPipelineOptions<PipelineOptions> {
  Option<std::string> constantsFile{
      *this, "constants-file",
      llvm::cl::desc("Filename for constants data embedded in module metadata "
                     "(default: constants.bin)"),
      llvm::cl::init("")};
  Option<std::string> constantsDir{
      *this, "constants-dir",
      llvm::cl::desc("Directory to write constants file into (default: .)"),
      llvm::cl::init("")};
  Option<int> optLevel{*this, "opt-level",
                       llvm::cl::desc("Optimization level 0-3 (default: 2)"),
                       llvm::cl::init(2)};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Enable verbose output"),
                       llvm::cl::init(false)};
};

} // namespace

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  udna::compiler::registerAllDialects(registry);
  udna::compiler::registerAllPasses();

  // compilationOpts and fs live in main() so they outlive MlirOptMain() and
  // therefore all pass executions triggered by it. Passes store
  // const CompilationOptionsT& and morphizen::FileSystem* — both must remain
  // valid for the full duration of pm.run().
  udna::compiler::CompilationOptionsT compilationOpts;
  std::unique_ptr<udna::DiskFileSystem> fs;

  // Full ONNX→HIP→LLVM→Interface pipeline.
  // Use when input is ONNX dialect MLIR (from onnx-mlir frontend).
  mlir::PassPipelineRegistration<PipelineOptions> morphizenPipeline(
      "morphizen-pipeline",
      "Complete Morphizen ONNX→HIP→LLVM→Interface pipeline",
      [&compilationOpts, &fs](mlir::OpPassManager &pm,
                              const PipelineOptions &opts) {
        compilationOpts.constants_file = opts.constantsFile;
        compilationOpts.opt_level = opts.optLevel;
        compilationOpts.verbose = opts.verbose;

        const std::string &dir = opts.constantsDir;
        if (!dir.empty())
          llvm::sys::fs::create_directories(dir);
        fs = std::make_unique<udna::DiskFileSystem>(
            dir.empty() ? "." : dir.c_str());

        udna::compiler::compiler::populateMorphizenPipeline(pm, compilationOpts,
                                                            fs.get());
      });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "hip-mlir-opt: HIP dialect compiler driver\n", registry));
}
