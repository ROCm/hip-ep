/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"
#include "MlirCompilerTypes.h"

// Morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <chrono>
#include <fstream>
#include <glog/logging.h>
#include <sstream>
#include <vector>

using namespace morphizen;
using namespace morphizen_cxx;
using namespace mlir_compiler_local;

DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace hipdnn {
namespace level1pass {

// Forward declarations for types used in pass_main.cpp
enum class ArtifactFormat { Native, LlvmIr };

struct CompilationConfig {
  ArtifactFormat artifactFormat;
  int optLevel;
};

struct CompilationArtifact {
  std::string filename;
  std::vector<uint8_t> bytes;
  ArtifactFormat format;
};

struct OutputMetadata {
  std::string name;
  int32_t rank;
  std::vector<int64_t> shape;
  int32_t elem_type;  // ONNX element type (e.g., 1=float, 7=int64)
};

} // namespace level1pass
} // namespace hipdnn

using namespace hipdnn::level1pass;

namespace {

// ============================================================================
// Static Helper Functions
// ============================================================================

// Step 1: Load configuration from provider options
static CompilationConfig load_config(PassContext *ctx) {
  CompilationConfig config;
  config.artifactFormat = ArtifactFormat::Native;
  config.optLevel = 2;

  try {
    // Parse artifact format
    std::string artifact_format_str =
        ctx->get_provider_option("artifact_format", "native");
    if (artifact_format_str == "llvm_ir") {
      config.artifactFormat = ArtifactFormat::LlvmIr;
    } else if (artifact_format_str != "native") {
      MY_LOG(1) << "Unknown artifact_format: " << artifact_format_str
                << ", using default: native";
    }

    // Parse optimization level
    std::string opt_level_str =
        ctx->get_provider_option("optimization_level", "2");
    config.optLevel = std::stoi(opt_level_str);

  } catch (const std::exception &ex) {
    MY_LOG(1) << "Failed to parse provider options: " << ex.what()
              << ", using defaults";
  }

  MY_LOG(1) << "Artifact format: "
            << (config.artifactFormat == ArtifactFormat::Native ? "native"
                                                                 : "llvm_ir");

  return config;
}

// Step 2: Get MLIR bytecode from graph
static std::string get_mlir_bytecode(Graph &graph) {
  auto bytecode = GraphConstRef(GraphRef(graph)).save_string();
  if (!bytecode || bytecode->empty()) {
    return "";
  }

  MY_LOG(1) << "MLIR bytecode size: " << bytecode->size() << " bytes";

  // Dump bytecode to file for troubleshooting if env var is set
  if (ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= 2) {
    std::ofstream dump_file("mlir_bytecode_dump.mlir", std::ios::binary);
    if (dump_file.is_open()) {
      dump_file.write(bytecode->data(), bytecode->size());
      MY_LOG(1) << "Dumped MLIR bytecode to mlir_bytecode_dump.mlir";
    } else {
      LOG(WARNING) << "Failed to dump MLIR bytecode";
    }
  }

  return std::string(bytecode->data(), bytecode->size());
}

// Step 3: Compile MLIR bytecode to artifact
static std::optional<CompilationArtifact>
compile_mlir(const std::string &mlir_bytecode,
             const CompilationConfig &config) {
  return MlirCompiler::compileFromBytecode(mlir_bytecode, config);
}

// Step 4: Write artifact to EPContext
static bool write_artifact_to_epcontext(PassContext *ctx,
                                        const CompilationArtifact &artifact) {
  MY_LOG(1) << "Writing artifact to EPContext: " << artifact.filename;

  auto stream = ctx->open_file_for_write(artifact.filename);
  if (!stream) {
    LOG(WARNING) << "Failed to open EPContext file for writing: "
                 << artifact.filename;
    return false;
  }

  size_t written = stream->fwrite(artifact.bytes.data(), artifact.bytes.size());
  if (written != artifact.bytes.size()) {
    LOG(WARNING) << "Failed to write complete artifact to EPContext";
    return false;
  }

  stream.reset(); // Close file
  MY_LOG(1) << "Wrote " << written << " bytes to EPContext";
  return true;
}

// Step 5: Extract output metadata from graph
static std::vector<OutputMetadata> extract_output_metadata(Graph &graph) {
  std::vector<OutputMetadata> output_metadata;
  GraphRef graphRef(graph);

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

    // Get element type (ONNX data type enum value)
    meta.elem_type = output.element_type();

    output_metadata.push_back(meta);
    MY_LOG(2) << "Output " << meta.name << ": rank=" << meta.rank
              << ", elem_type=" << meta.elem_type;
  }

  return output_metadata;
}

// Step 6: Build metadata JSON (replaces Protobuf)
static std::string
build_metadata_json(const CompilationArtifact &artifact,
                    const CompilationConfig &config,
                    const std::vector<OutputMetadata> &outputs) {
  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch())
                       .count();

  std::ostringstream json;
  json << "{";

  // Basic metadata
  json << "\"artifact_format\": \""
       << (config.artifactFormat == ArtifactFormat::Native ? "native"
                                                            : "llvm_ir")
       << "\",";
  json << "\"compilation_timestamp\": " << timestamp << ",";
  json << "\"compiler_version\": \"mlir-hip-compiler-1.0\",";
  json << "\"optimization_level\": " << config.optLevel << ",";
  json << "\"artifact_filename\": \"" << artifact.filename << "\",";
  json << "\"artifact_size\": " << artifact.bytes.size();

  // Output metadata
  if (!outputs.empty()) {
    json << ",\"outputs\": [";
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto &output = outputs[i];
      if (i > 0)
        json << ",";

      json << "{";
      json << "\"name\": \"" << output.name << "\",";
      json << "\"rank\": " << output.rank << ",";
      json << "\"elem_type\": " << output.elem_type;

      if (!output.shape.empty()) {
        json << ",\"shape\": [";
        for (size_t j = 0; j < output.shape.size(); ++j) {
          if (j > 0)
            json << ",";
          json << output.shape[j];
        }
        json << "]";
      }

      json << "}";
    }
    json << "]";
  }

  json << "}";

  std::string metadata = json.str();
  MY_LOG(1) << "Metadata JSON: " << metadata;
  return metadata;
}

// Step 7: Fuse graph into single MLIR custom op
static bool fuse_graph(IPass &self, Graph &graph,
                       const std::string &metadata,
                       const std::string &unique_id) {
  MY_LOG(1) << "Attempting to fuse entire graph with domain 'MLIR'";

  GraphRef graphRef(graph);

  // Extract graph input names
  std::vector<std::string> input_names;
  for (const auto &input : graphRef.inputs()) {
    input_names.push_back(input.name());
  }
  MY_LOG(1) << "Graph inputs: " << input_names.size();

  // Extract graph output names
  std::vector<std::string> output_names;
  for (const auto &output : graphRef.outputs()) {
    output_names.push_back(output.name());
  }
  MY_LOG(1) << "Graph outputs: " << output_names.size();

  // Try to fuse
  auto [meta_def, fuse_error] =
      self.try_fuse(graphRef, unique_id, input_names, output_names,
                    {},    // constant_initializers (empty for now)
                    "MLIR" // domain name
      );

  if (meta_def == nullptr) {
    MY_LOG(1) << "Fusion failed: " << fuse_error.comments;
    LOG(WARNING) << "Failed to create MLIR fusion: " << fuse_error.comments;
    return false;
  }

  MY_LOG(1) << "Fusion successful, attaching metadata";

  // Attach metadata to MetaDefProto
  self.attach_meta_def_param(*meta_def, metadata.c_str());

  // Fuse the graph (replace nodes with custom op)
  self.fuse(graphRef, std::move(*meta_def));

  MY_LOG(1) << "MLIR compilation and fusion completed";
  return true;
}

// ============================================================================
// Pass Implementation
// ============================================================================

struct Level1MlirPass {
  Level1MlirPass(IPass &self) : self_{self} {}

  void process(IPass &self, Graph &graph) {
    MY_LOG(1) << "Level1MlirPass::process() called";

    // Step 1: Load configuration from provider options
    auto config = load_config(self.get_context());

    // Step 2: Get MLIR bytecode from graph
    auto mlir_bytecode = get_mlir_bytecode(graph);
    if (mlir_bytecode.empty()) {
      LOG(WARNING) << "Empty graph bytecode, skipping compilation";
      return;
    }

    // Step 3: Compile bytecode to artifact
    auto artifactOpt = compile_mlir(mlir_bytecode, config);
    if (!artifactOpt) {
      LOG(WARNING) << "MLIR compilation failed, skipping";
      return;
    }
    CompilationArtifact artifact = *artifactOpt;

    // Step 4: Write artifact to EPContext
    if (!write_artifact_to_epcontext(self.get_context(), artifact)) {
      return;
    }

    // Step 5: Extract output metadata from graph
    auto output_metadata = extract_output_metadata(graph);

    // Step 6: Build metadata JSON
    auto metadata_json = build_metadata_json(artifact, config, output_metadata);

    // Step 7: Extract unique ID from artifact filename (remove extension)
    std::string unique_id =
        artifact.filename.substr(0, artifact.filename.find_last_of('.'));

    // Step 8: Fuse graph into single MLIR custom op
    if (!fuse_graph(self, graph, metadata_json, unique_id)) {
      LOG(WARNING) << "Graph fusion failed";
      return;
    }

    MY_LOG(1) << "MLIR compilation completed: " << artifact.filename << " ("
              << artifact.bytes.size() << " bytes)";
  }

  IPass &self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1MlirPass, morphizen_pass_level1_mlir)
