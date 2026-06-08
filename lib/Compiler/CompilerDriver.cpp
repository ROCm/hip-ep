/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/CompilerDriver.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/InitAllPasses.h"

#include "hip/Target/LLVM/DLLLinker.h"
#include "hip/Target/LLVM/LLVMBackend.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "hip/debug_log.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <sstream>

namespace hip::compiler {

namespace {
bool fileExists(const std::string &path) {
  llvm::sys::fs::file_status status;
  std::error_code EC = llvm::sys::fs::status(path, status);
  return !EC && llvm::sys::fs::exists(status);
}
} // namespace

bool CompilerDriver::compile(llvm::StringRef input_mlir,
                             const std::string &output_path,
                             const mlir::hip::CompilationOptionsT &options,
                             std::string &error_message) {
  hip::compiler::registerAllPasses();

  mlir::MLIRContext context;
  hip::compiler::loadAllDialects(context);
  mlir::registerLLVMDialectTranslation(context);

  COMPILER_DEBUG_LOG("[CompilerDriver::compile] Input size: "
                     << input_mlir.size() << " bytes\n");

  // Wrap input in a non-owning buffer (no copy) for MLIR's parser.
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(input_mlir, "", false);

  // SourceMgr provides source-location tracking for parser diagnostics.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  auto t0 = timing_now();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (hipdnn_ep_timing_enabled()) {
    llvm::errs() << "[CompilerDriver] MLIR parsing: "
                 << llvm::format("%.3f", elapsed_since(t0)) << "s\n";
  }

  if (!module) {
    error_message = "Failed to parse MLIR input";
    return false;
  }

  return compileImpl(*module, output_path, options, error_message);
}

bool CompilerDriver::compileFromModule(
    mlir::ModuleOp module, const std::string &output_path,
    const mlir::hip::CompilationOptionsT &options, std::string &error_message) {
  hip::compiler::registerAllPasses();
  return compileImpl(module, output_path, options, error_message);
}

bool CompilerDriver::validate(llvm::StringRef input_mlir,
                              std::string &error_message) {
  mlir::MLIRContext context;
  hip::compiler::loadAllDialects(context);

  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(input_mlir, "", false);
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    error_message = "Failed to parse MLIR input";
    return false;
  }

  return true;
}

bool CompilerDriver::compileImpl(mlir::ModuleOp module,
                                 const std::string &output_path,
                                 const mlir::hip::CompilationOptionsT &options,
                                 std::string &error_message) {
  const bool timing = hipdnn_ep_timing_enabled();
  auto phaseStart = timing_now();
  auto totalStart = phaseStart;

  auto logPhase = [&](const char *name) {
    if (!timing)
      return;
    llvm::errs() << "[CompilerDriver] " << name << ": "
                 << llvm::format("%.3f", record_elapsed(phaseStart)) << "s\n";
  };

  if (timing)
    llvm::errs() << "[CompilerDriver] === Compilation phases ===\n";

  if (!runMLIRPasses(module, options, error_message))
    return false;
  logPhase("runMLIRPasses");

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateToLLVMIR(module, llvmContext, error_message);
  if (!llvmModule)
    return false;
  logPhase("translateToLLVMIR");

  if (!linkRuntime(llvmModule.get(), error_message))
    return false;
  logPhase("linkRuntime");

  optimizeLLVMIR(llvmModule.get(), options.opt_level);
  logPhase("optimizeLLVMIR");

  // Strip .dll extension to derive intermediate file paths (.ll, .obj).
  std::string base_path = output_path;
  llvm::StringRef dll_ext = ".dll";
  if (llvm::StringRef(base_path).ends_with(dll_ext))
    base_path.resize(base_path.size() - dll_ext.size());
  std::string ll_path = base_path + ".ll";
  std::string obj_path = base_path + ".obj";

  if (options.output_mode == mlir::hip::OutputMode::LLVM_IR) {
    if (!emitLLVMIR(llvmModule.get(), ll_path, error_message))
      return false;
    logPhase("emitLLVMIR");
    if (timing) {
      llvm::errs() << "[CompilerDriver] total: "
                   << llvm::format("%.3f", elapsed_since(totalStart)) << "s\n";
    }
    return true;
  }

  if (!compileToObject(llvmModule.get(), obj_path, error_message))
    return false;
  logPhase("compileToObject");

  // Symbols exported from the generated DLL:
  //   inference_init/compute/cleanup       — runtime entry points
  //   inference_get_metadata_json          — model metadata query
  //   test_hip_from_dll                    — diagnostic hook for test-model-dll
  //   hipdnn_ep_runtime_begin_compute      — per-Compute() cache invalidation
  //                                          hook (called from EP-side
  //                                          MlirCustomOp::Compute() entry)
  //   hipdnn_ep_set_output_allocator       — EP installs the output allocator
  //                                          before inference_compute (Phase 5)
  std::vector<std::string> export_symbols = {"inference_init",
                                             "inference_compute",
                                             "inference_cleanup",
                                             "inference_get_metadata_json",
                                             "test_hip_from_dll",
                                             "hipdnn_ep_runtime_begin_compute",
                                             "hipdnn_ep_set_output_allocator"};
  std::vector<std::string> libraries;
  std::vector<std::string> library_paths;
  discoverLibraries(libraries, library_paths);

  if (!linkToDLL(obj_path, output_path, libraries, library_paths,
                 export_symbols, error_message))
    return false;
  logPhase("linkToDLL");

  cleanupIntermediates(base_path);

  if (timing) {
    llvm::errs() << "[CompilerDriver] total: "
                 << llvm::format("%.3f", elapsed_since(totalStart)) << "s\n";
  }

  return true;
}

bool CompilerDriver::runMLIRPasses(
    mlir::ModuleOp module, const mlir::hip::CompilationOptionsT &options,
    std::string &error_message) {
  mlir::PassManager pm(module.getContext());

  if (hipdnn_ep_timing_enabled())
    pm.enableTiming();

  if (options.verbose) {
    COMPILER_DEBUG_LOG("Running ONNX->HIP->LLVM->Interface passes\n");
  }

  mlir::hip::OnnxToHipPipelineOptions onnxToHipOpts;
  onnxToHipOpts.externalizeMinNumElements =
      mlir::hip::kDefaultExternalizeMinNumElements;
  onnxToHipOpts.skipConstantData = options.skip_constant_data;

  if (hipdnnHandle_) {
    compiledGraphs_ =
        std::make_shared<llvm::StringMap<mlir::hip::OwnedGraph>>();
    mlir::hip::buildOnnxToHipPipeline(
        pm, onnxToHipOpts, fileSystem_,
        static_cast<hipdnnHandle_t>(hipdnnHandle_), compiledGraphs_);
  } else {
    mlir::hip::buildOnnxToHipPipeline(pm, onnxToHipOpts, fileSystem_);
  }

  mlir::hip::HipToLLVMPipelineOptions hipToLlvmOpts;
  hipToLlvmOpts.constantsFile = options.constants_file;
  mlir::hip::buildHipToLLVMPipeline(pm, hipToLlvmOpts);

  std::unique_ptr<llvm::raw_fd_ostream> irDumpStream;
  // hip_get_env, not std::getenv: this code may be linked into the static-CRT
  // EP DLL where std::getenv cannot see host-process env vars.
  std::string dumpPath = hip_get_env("HIPDNN_EP_IR_DUMP_PATH");
  if (!dumpPath.empty()) {
    // ORT can invoke the EP compiler multiple times per session (shape sub-
    // graphs, prefill specialization, decode specialization). A single sink
    // file gets overwritten on each call, masking divergence between
    // invocations. Insert a process-wide monotonic counter so each compile
    // dumps to its own file. The counter is inserted BEFORE the extension so
    // editors / file viewers still recognize the result as MLIR:
    //   /tmp/ir.mlir -> /tmp/ir.0.mlir, /tmp/ir.1.mlir, ...
    //   /tmp/ir      -> /tmp/ir.0,      /tmp/ir.1,      ...    (extensionless)
    // When HIPDNN_EP_IR_DUMP_SINGLE=1 the legacy single-file behaviour is
    // restored (no counter, output overwritten across compiles).
    static std::atomic<unsigned> sCompileSeq{0};
    std::string finalPath = dumpPath;
    if (hip_get_env("HIPDNN_EP_IR_DUMP_SINGLE").empty()) {
      // llvm::sys::path::extension/stem are path-separator aware, so
      // dots inside parent directory names (e.g. ".cache/ir") don't get
      // mistaken for an extension.
      llvm::StringRef pathRef(dumpPath);
      llvm::StringRef ext = llvm::sys::path::extension(pathRef);
      llvm::StringRef stem = pathRef.drop_back(static_cast<size_t>(ext.size()));
      std::string counter = "." + std::to_string(sCompileSeq.fetch_add(1));
      finalPath = (llvm::Twine(stem) + counter + ext).str();
    }
    // HIPDNN_EP_IR_DUMP_AFTER_ONLY=1 suppresses the per-pass "before" dump.
    // Combined with printAfterOnlyOnChange=true (always on), this leaves a
    // dump that contains only the IR after passes that actually changed
    // something -- typically halves the file size on a full pipeline run
    // while keeping every meaningful transformation visible.
    bool afterOnly = !hip_get_env("HIPDNN_EP_IR_DUMP_AFTER_ONLY").empty();
    auto shouldPrintBefore = [afterOnly](mlir::Pass *, mlir::Operation *) {
      return !afterOnly;
    };
    auto shouldPrintAfter = [](mlir::Pass *, mlir::Operation *) {
      return true;
    };
    // HIPDNN_EP_IR_DUMP_TREE=1 splits the dump into one .mlir file per pass
    // under finalPath as a DIRECTORY (vs. a single concatenated file). Per-
    // pass files are typically <1 MB even on a full LLM pipeline -- editable
    // in any editor, unlike the multi-MB monolithic dump. Files are named
    // `<idx>_<pass-name>.mlir` under `<finalPath>/<op>_<symbol>/`. See MLIR's
    // PassManager::enableIRPrintingToFileTree for the tree layout.
    bool treeMode = !hip_get_env("HIPDNN_EP_IR_DUMP_TREE").empty();
    if (treeMode) {
      module.getContext()->disableMultithreading();
      pm.enableIRPrintingToFileTree(shouldPrintBefore, shouldPrintAfter,
                                    /*printModuleScope=*/true,
                                    /*printAfterOnlyOnChange=*/true,
                                    /*printAfterOnlyOnFailure=*/false,
                                    /*printTreeDir=*/finalPath);
    } else {
      std::error_code ec;
      irDumpStream = std::make_unique<llvm::raw_fd_ostream>(finalPath, ec);
      if (!ec) {
        module.getContext()->disableMultithreading();
        pm.enableIRPrinting(shouldPrintBefore, shouldPrintAfter,
                            /*printModuleScope=*/true,
                            /*printAfterOnlyOnChange=*/true,
                            /*printAfterOnlyOnFailure=*/false, *irDumpStream);
      } else {
        llvm::errs() << "[CompilerDriver] Failed to open IR dump file: "
                     << dumpPath << ": " << ec.message() << "\n";
      }
    }
  }

  if (mlir::failed(pm.run(module))) {
    error_message = "MLIR pass pipeline failed";
    if (options.verbose) {
      llvm::errs() << "\n=== Failed Module IR ===\n";
      module.print(llvm::errs());
      llvm::errs() << "\n========================\n";
    }
    // Strict mode (opt-in): when HIPDNN_EP_STRICT=1 is set, abort so the
    // cpptrace SIGABRT handler prints a backtrace pinpointing the failing
    // pass. Use this when verifying that a model is fully offloaded — any
    // graph MorphiZenEP claims but cannot compile is a regression, and
    // catching it as a crash beats silent CPU fallback masking the bug
    // (e.g. accuracy tests passing cosine=1.0 because they compare CPU vs
    // CPU). Default behaviour returns false so ORT's CPU fallback handles
    // the graph normally — required for multi-session pipelines where
    // MorphiZenEP is registered only for HipDataTransferImpl visibility
    // (e.g. multi-stage VLM pipelines where embedding / vision sub-sessions
    // are intentionally not claimed by the EP).
    if (!hip_get_env("HIPDNN_EP_STRICT").empty()) {
      llvm::errs() << "[CompilerDriver] aborting on pass failure "
                      "(HIPDNN_EP_STRICT=1).\n";
      std::abort();
    }
    return false;
  }

  if (options.verbose)
    COMPILER_DEBUG_LOG("MLIR passes completed\n\n");

  return true;
}

std::unique_ptr<llvm::Module>
CompilerDriver::translateToLLVMIR(mlir::ModuleOp module,
                                  llvm::LLVMContext &llvmContext,
                                  std::string &error_message) {
  hipdnn::LLVMBackend backend;
  auto llvmModule = backend.translateMLIRtoLLVMIR(module, llvmContext);

  if (!llvmModule) {
    error_message = "Failed to translate MLIR to LLVM IR";
  }

  return llvmModule;
}

bool CompilerDriver::linkRuntime(llvm::Module *llvmModule,
                                 std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.linkRuntimeModule(llvmModule)) {
    error_message = "Failed to link runtime module";
    return false;
  }
  return true;
}

void CompilerDriver::optimizeLLVMIR(llvm::Module *llvmModule, int optLevel) {
  hipdnn::LLVMBackend backend;
  backend.optimizeLLVMIR(llvmModule, optLevel);
}

bool CompilerDriver::emitLLVMIR(llvm::Module *llvmModule,
                                const std::string &outputPath,
                                std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.emitLLVMIR(llvmModule, outputPath)) {
    error_message = "Failed to emit LLVM IR";
    return false;
  }
  return true;
}

bool CompilerDriver::compileToObject(llvm::Module *llvmModule,
                                     const std::string &outputPath,
                                     std::string &error_message) {
  hipdnn::LLVMBackend backend;
  if (!backend.compileToObjectFile(llvmModule, outputPath)) {
    error_message = "Failed to compile to object file";
    return false;
  }
  return true;
}

bool CompilerDriver::linkToDLL(const std::string &objPath,
                               const std::string &dllPath,
                               const std::vector<std::string> &libraries,
                               const std::vector<std::string> &library_paths,
                               const std::vector<std::string> &export_symbols,
                               std::string &error_message) {
  hipdnn::DLLLinker linker;

  if (!linker.linkDLL(objPath, dllPath, libraries, library_paths,
                      export_symbols)) {
    error_message = "Failed to link DLL";
    return false;
  }

  return true;
}

void CompilerDriver::discoverLibraries(
    std::vector<std::string> &libraries,
    std::vector<std::string> &library_paths) {
  // hip_get_env, not std::getenv: this code runs inside the static-CRT EP DLL
  // when invoked from the EP. std::getenv there has its own (empty) CRT env
  // table and silently returned NULL, leaving library_paths empty — lld then
  // failed with "could not open 'amdhip64.lib'" and the EP fell back to CPU.
  std::string dist = hip_get_env("THEROCK_DIST");
  if (dist.empty())
    return;

  std::string lib_dir = dist + "/lib";
  library_paths.push_back(lib_dir);
  COMPILER_DEBUG_LOG("THEROCK_DIST detected: " << dist << "\n");
  COMPILER_DEBUG_LOG("  Adding library path: " << lib_dir << "\n");

  libraries.push_back("amdhip64");
  libraries.push_back("MIOpen");

  // hipblaslt ships as .lib (Windows), .dll.a (cross-compiled), or
  // .so (native Linux). Bare name "hipblaslt" lets the linker resolve via
  // -L<lib_dir> to libhipblaslt.so on Linux.
  std::string hipblaslt_lib = lib_dir + "/hipblaslt.lib";
  std::string hipblaslt_dll_a = lib_dir + "/libhipblaslt.dll.a";
  std::string hipblaslt_so = lib_dir + "/libhipblaslt.so";
  if (llvm::sys::fs::exists(hipblaslt_lib))
    libraries.push_back("hipblaslt");
  else if (llvm::sys::fs::exists(hipblaslt_dll_a))
    libraries.push_back(hipblaslt_dll_a);
  else if (llvm::sys::fs::exists(hipblaslt_so))
    libraries.push_back("hipblaslt");
  else
    COMPILER_DEBUG_LOG("  WARNING: hipblaslt import library not found\n");

  // Custom kernels library discovery (priority high → low):
  //
  //   1. HIP_CUSTOM_KERNELS_DIR env var  — runtime override for end-users
  //      who deploy the library in a non-standard location.  The directory
  //      is added to library_paths so the linker can find it.
  //
  //   2. HIP_CUSTOM_KERNELS_LIB_PATH    — compile-time absolute path set
  //      by CMake (CMAKE_INSTALL_PREFIX/lib/hip_custom_kernels.lib).
  //      Used by developers whose kernels library is in the install tree.
  //
  //   3. Name-only fallback              — "hip_custom_kernels" is passed
  //      to the linker, which searches library_paths (/LIBPATH:) and the
  //      system LIB environment variable.
  {
    bool found = false;

    // hip_get_env, not std::getenv: see THEROCK_DIST comment above.
    std::string custom_dir = hip_get_env("HIP_CUSTOM_KERNELS_DIR");
    if (!custom_dir.empty()) {
      library_paths.push_back(custom_dir);
      libraries.push_back("hip_custom_kernels");
      found = true;
      COMPILER_DEBUG_LOG("  Custom kernels dir (env): " << custom_dir << "\n");
    }

#ifdef HIP_CUSTOM_KERNELS_LIB_PATH
    if (!found) {
      std::string custom_lib = HIP_CUSTOM_KERNELS_LIB_PATH;
      if (llvm::sys::fs::exists(custom_lib)) {
        libraries.push_back(custom_lib);
        found = true;
        COMPILER_DEBUG_LOG("  Custom kernels: " << custom_lib << "\n");
      } else {
        COMPILER_DEBUG_LOG("  WARNING: custom kernels lib not found at: "
                           << custom_lib << "\n");
      }
    }
#endif

    if (!found) {
      libraries.push_back("hip_custom_kernels");
      COMPILER_DEBUG_LOG(
          "  Custom kernels (name fallback): hip_custom_kernels\n");
    }
  }

  // hipDNN graph runtime: only needed when hipDNN graphs are compiled
  if (hipdnnHandle_) {
    std::string hipdnn_backend_lib = lib_dir + "/hipdnn_backend.lib";
    if (llvm::sys::fs::exists(hipdnn_backend_lib))
      libraries.push_back("hipdnn_backend");
    else
      COMPILER_DEBUG_LOG(
          "  WARNING: hipdnn_backend import library not found\n");

#ifdef HIPDNN_GRAPH_RUNTIME_LIB_PATH
    {
      std::string runtime_lib = HIPDNN_GRAPH_RUNTIME_LIB_PATH;
      if (llvm::sys::fs::exists(runtime_lib)) {
        libraries.push_back(runtime_lib);
        COMPILER_DEBUG_LOG("  hipDNN graph runtime: " << runtime_lib << "\n");
      } else {
        libraries.push_back("hipdnn_graph_runtime");
        COMPILER_DEBUG_LOG("  hipDNN graph runtime (name fallback): "
                           "hipdnn_graph_runtime.lib\n");
      }
    }
#else
    libraries.push_back("hipdnn_graph_runtime");
    COMPILER_DEBUG_LOG("  hipDNN graph runtime: hipdnn_graph_runtime\n");
#endif
  }

  for (const auto &lib : libraries) {
    COMPILER_DEBUG_LOG("  Linking library: " << lib << "\n");
  }
}

void CompilerDriver::cleanupIntermediates(const std::string &basePath) {
  std::string ll_path = basePath + ".ll";
  std::string obj_path = basePath + ".obj";

  if (fileExists(ll_path)) {
    llvm::sys::fs::remove(ll_path);
  }

  if (fileExists(obj_path)) {
    llvm::sys::fs::remove(obj_path);
  }
}

} // namespace hip::compiler
