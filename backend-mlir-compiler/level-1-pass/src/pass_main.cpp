/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Helper class headers
#include "CompilationArtifact.h"
#include "CompilationConfig.h"
#include "FusionManager.h"
#include "MetadataBuilder.h"
#include "MlirCompiler.h"

// Morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <chrono>
#include <glog/logging.h>

using namespace morphizen;
using namespace morphizen_cxx;
using namespace hipdnn::level1pass;

DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace {

struct Level1MlirPass {
  Level1MlirPass(IPass &self) : self_{self} {}

  void process(IPass &self, Graph &graph) {
    MY_LOG(1) << "Level1MlirPass::process() called";

    // 1. Load configuration from provider options
    auto config = CompilationConfig::fromProviderOptions(self.get_context());
    MY_LOG(1) << "Artifact format: "
              << (config.artifactFormat ==
                          CompilationConfig::ArtifactFormat::Native
                      ? "native"
                      : "llvm_ir");

    // 2. Get MLIR bytecode from graph
    //    Graph.save_string() returns MLIR bytecode format
    GraphRef graphRef(graph);
    auto graph_string = GraphConstRef(graphRef).save_string();

    if (graph_string->empty()) {
      LOG(WARNING) << "Empty graph bytecode, skipping compilation";
      return;
    }

    // Convert to std::string (bytecode is binary data)
    std::string mlir_bytecode(graph_string->data(), graph_string->size());

    MY_LOG(1) << "MLIR bytecode size: " << mlir_bytecode.size() << " bytes";

    // 3. Compile MLIR bytecode to artifact using C API (via
    // morphizen-mlir-compiler.dll)
    //    The new morphizen_mlir_compile_bytecode() function handles bytecode
    //    directly
    int64_t mlir_duration_ms = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    auto artifactOpt = MlirCompiler::compileFromBytecode(mlir_bytecode, config);

    auto end_time = std::chrono::high_resolution_clock::now();
    mlir_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_time - start_time)
                           .count();

    if (!artifactOpt) {
      LOG(WARNING) << "MLIR compilation failed, skipping";
      return;
    }
    CompilationArtifact artifact = *artifactOpt;

    // 4. Write artifact to EPContext
    MY_LOG(1) << "Writing artifact to EPContext: " << artifact.filename;
    auto stream = self.get_context()->open_file_for_write(artifact.filename);
    if (!stream) {
      LOG(WARNING) << "Failed to open EPContext file for writing: "
                   << artifact.filename;
      return;
    }

    size_t written =
        stream->fwrite(artifact.bytes.data(), artifact.bytes.size());
    if (written != artifact.bytes.size()) {
      LOG(WARNING) << "Failed to write complete artifact to EPContext";
      return;
    }
    stream.reset(); // Close file

    // 5. Extract output metadata from graph
    std::vector<OutputMetadata> output_metadata;
    for (const auto &output : graphRef.outputs()) {
      OutputMetadata meta;
      meta.name = output.name();

      // Get shape
      auto shape_ptr = output.shape();
      if (shape_ptr && !output.is_unknown_shape()) {
        meta.shape = *shape_ptr;
        meta.rank = static_cast<int32_t>(meta.shape.size());
      } else {
        meta.rank = -1; // Unknown rank
      }

      // Get element type and convert to dtype string
      int elem_type = output.element_type();
      switch (elem_type) {
      case 1:
        meta.dtype = "float32";
        break;
      case 2:
        meta.dtype = "uint8";
        break;
      case 3:
        meta.dtype = "int8";
        break;
      case 4:
        meta.dtype = "uint16";
        break;
      case 5:
        meta.dtype = "int16";
        break;
      case 6:
        meta.dtype = "int32";
        break;
      case 7:
        meta.dtype = "int64";
        break;
      case 10:
        meta.dtype = "float16";
        break;
      case 11:
        meta.dtype = "double";
        break;
      case 16:
        meta.dtype = "bfloat16";
        break;
      default:
        meta.dtype = "unknown";
        break;
      }

      output_metadata.push_back(meta);
      MY_LOG(2) << "Output " << meta.name << ": rank=" << meta.rank
                << ", dtype=" << meta.dtype;
    }

    // 6. Build metadata with output information
    auto metadataOpt = MetadataBuilder::build(
        artifact, config, mlir_duration_ms, output_metadata);
    if (!metadataOpt) {
      LOG(WARNING) << "Failed to build metadata, skipping fusion";
      return;
    }
    std::string metadata = *metadataOpt;

    // 7. Extract unique ID from artifact filename (remove extension)
    std::string unique_id =
        artifact.filename.substr(0, artifact.filename.find_last_of('.'));

    // 8. Fuse graph into single MLIR custom op
    if (!FusionManager::fuseGraph(self, graphRef, metadata, unique_id)) {
      LOG(WARNING) << "Graph fusion failed";
      return;
    }

    MY_LOG(1) << "MLIR compilation completed: " << artifact.filename << " ("
              << artifact.bytes.size() << " bytes, " << mlir_duration_ms
              << " ms)";
  }

  IPass &self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1MlirPass, morphizen_pass_level1_mlir)
