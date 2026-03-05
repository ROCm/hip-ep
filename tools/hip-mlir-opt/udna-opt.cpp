/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "hip-compiler/Compiler/Pipeline.h"
#include "hip-compiler/InitAllPasses.h"
#include "hip-compiler/Support/DiskFileSystem.h"
#include "compilation_options_generated.h"

#include "llvm/Support/FileSystem.h"

namespace {

/// CLI adapter for the Morphizen pipeline.
/// Only used by hip-opt; CompilerDriver calls populateMorphizenPipeline
/// directly with a CompilationOptionsT it already owns.
struct PipelineOptions : public mlir::PassPipelineOptions<PipelineOptions> {
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Enable verbose output"),
                       llvm::cl::init(false)};
  Option<int> optLevel{*this, "opt-level",
                       llvm::cl::desc("Optimization level (0-3)"),
                       llvm::cl::init(2)};
  Option<std::string> constantsDir{
      *this, "constants-dir",
      llvm::cl::desc("Directory to write constants file into (default: .)"),
      llvm::cl::init("")};
  Option<std::string> constantsFile{
      *this, "constants-file",
      llvm::cl::desc("Filename for constants data (default: constants.bin)"),
      llvm::cl::init("")};
};

} // namespace

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  hip::compiler::registerAllDialects(registry);

  hip::compiler::registerAllPasses();

  // compilationOpts and fs live in main() so they outlive MlirOptMain() and
  // therefore all pass executions triggered by it. Passes store
  // const CompilationOptionsT& and morphizen::FileSystem* — both must remain
  // valid for the full duration of pm.run().
  hip::compiler::CompilationOptionsT compilationOpts;
  std::unique_ptr<hip::DiskFileSystem> fs;

  // Register the Morphizen pipeline for use with --pass-pipeline on the CLI.
  // The lambda runs when the pipeline is instantiated during CLI processing,
  // before any pass executes. compilationOpts and fs are updated here and
  // then referenced by the passes added to pm.
  mlir::PassPipelineRegistration<PipelineOptions> morphizenPipeline(
      "morphizen-pipeline",
      "Complete Morphizen ONNX->HIP->LLVM->Interface pipeline",
      [&compilationOpts, &fs](mlir::OpPassManager& pm,
                              const PipelineOptions& opts) {
        compilationOpts.verbose = opts.verbose;
        compilationOpts.opt_level = opts.optLevel;
        compilationOpts.constants_file = opts.constantsFile;

        const std::string& dir = opts.constantsDir;
        if (!dir.empty())
          llvm::sys::fs::create_directories(dir);
        fs = std::make_unique<hip::DiskFileSystem>(
            dir.empty() ? "." : dir.c_str());

        hip::compiler::compiler::populateMorphizenPipeline(pm, compilationOpts,
                                                            fs.get());
      });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Morphizen MLIR Pass Runner\n", registry));
}
