/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"

// Morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <sstream>
#include <vector>

// Protobuf
#include "google/protobuf/util/json_util.h"
#include "metadata.pb.h"

using namespace morphizen;
using namespace morphizen_cxx;

DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

// Types are now defined in MlirCompiler.h

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
static std::string get_mlir_bytecode(PassContext *ctx, Graph &graph) {
  auto bytecode = GraphConstRef(GraphRef(graph)).save_string();
  if (bytecode->empty()) {
    return "";
  }

  MY_LOG(1) << "MLIR bytecode size: " << bytecode->size() << " bytes";

  // Dump bytecode to file for troubleshooting if env var is set
  if (ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= 2) {
    auto dump_path = ctx->get_dump_directory() / "mlir_bytecode_dump.mlir";
    MY_LOG(1) << "Dumping MLIR bytecode to " << dump_path;
    CHECK(std::ofstream(dump_path, std::ios::binary)
              .write(bytecode->data(), bytecode->size())
              .good())
        << "Failed to dump MLIR bytecode to " << dump_path;
    MY_LOG(1) << "Dumped MLIR bytecode to " << dump_path;
  }

  return std::string(bytecode->data(), bytecode->size());
}

// Step 3: Compile MLIR bytecode to artifact
static std::optional<CompilationArtifact>
compile_mlir(const std::string &mlir_bytecode, const CompilationConfig &config,
             morphizen::FileSystem *fs) {
  return MlirCompiler::compileFromBytecode(mlir_bytecode, config, fs);
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

// Step 5: Build metadata JSON from graph outputs
static std::string build_metadata_json(const CompilationArtifact &artifact,
                                       Graph &graph) {
  mlir_metadata::Metadata metadata;
  metadata.set_artifact_filename(artifact.filename);

  GraphRef graphRef(graph);
  for (const auto &output : graphRef.outputs()) {
    auto *output_proto = metadata.add_outputs();
    output_proto->set_name(output.name());
    output_proto->set_elem_type(output.element_type());

    // Get shape and rank
    auto shape_ptr = output.shape();
    if (shape_ptr && !output.is_unknown_shape()) {
      output_proto->set_rank(static_cast<int32_t>(shape_ptr->size()));
      for (int64_t dim : *shape_ptr) {
        output_proto->add_shape(dim);
      }
    } else {
      output_proto->set_rank(-1); // Unknown rank
    }

    MY_LOG(2) << "Output " << output.name() << ": rank=" << output_proto->rank()
              << ", elem_type=" << output_proto->elem_type();
  }

  std::string json;
  auto status = google::protobuf::util::MessageToJsonString(metadata, &json);
  CHECK(status.ok()) << "Failed to serialize metadata to JSON: "
                     << status.ToString();

  MY_LOG(1) << "Metadata JSON: " << json;
  return json;
}

// Step 7: Fuse graph into single MLIR custom op
static bool fuse_graph(IPass &self, Graph &graph, const std::string &metadata,
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

    const bool timing = [] {
      const char *v = getenv("HIPDNN_EP_TIMING");
      return v && v[0] >= '1';
    }();
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    auto t_prev = t0;

    // Step 1: Load configuration from provider options
    auto config = load_config(self.get_context().get());

    if (timing) {
      auto t_now = Clock::now();
      fprintf(stderr, "[Session] load_config: %.3fs\n",
              std::chrono::duration<double>(t_now - t_prev).count());
      t_prev = t_now;
    }

    // Step 2: Get MLIR bytecode from graph
    auto mlir_bytecode = get_mlir_bytecode(self.get_context().get(), graph);
    if (mlir_bytecode.empty()) {
      LOG(WARNING) << "Empty graph bytecode, skipping compilation";
      return;
    }

    if (timing) {
      auto t_now = Clock::now();
      fprintf(stderr, "[Session] MLIR bytecode serialization: %.3fs (%zu bytes)\n",
              std::chrono::duration<double>(t_now - t_prev).count(),
              mlir_bytecode.size());
      t_prev = t_now;
    }

    // Step 3: Compile bytecode to artifact
    auto fs = self.get_context()->get_file_system();
    auto artifactOpt = compile_mlir(mlir_bytecode, config, fs.get());
    if (!artifactOpt) {
      LOG(WARNING) << "MLIR compilation failed, skipping";
      return;
    }
    CompilationArtifact artifact = *artifactOpt;

    if (timing) {
      auto t_now = Clock::now();
      fprintf(stderr, "[Session] MLIR compilation (CompilerDriver): %.3fs\n",
              std::chrono::duration<double>(t_now - t_prev).count());
      t_prev = t_now;
    }

    // Step 4: Write artifact to EPContext
    if (!write_artifact_to_epcontext(self.get_context().get(), artifact)) {
      return;
    }

    if (timing) {
      auto t_now = Clock::now();
      fprintf(stderr, "[Session] Write artifact to EPContext: %.3fs\n",
              std::chrono::duration<double>(t_now - t_prev).count());
      t_prev = t_now;
    }

    // Step 5: Build metadata JSON from graph outputs
    auto metadata_json = build_metadata_json(artifact, graph);

    if (timing) {
      auto t_now = Clock::now();
      fprintf(stderr, "[Session] Build metadata JSON: %.3fs\n",
              std::chrono::duration<double>(t_now - t_prev).count());
      t_prev = t_now;
    }

    // Step 6: Fuse graph into single MLIR custom op
    if (!fuse_graph(self, graph, metadata_json, artifact.filename)) {
      LOG(WARNING) << "Graph fusion failed";
      return;
    }

    if (timing) {
      auto t_now = Clock::now();
      fprintf(stderr, "[Session] Fuse graph: %.3fs\n",
              std::chrono::duration<double>(t_now - t_prev).count());
      fprintf(stderr, "[Session] Level1MlirPass::process total: %.3fs\n",
              std::chrono::duration<double>(t_now - t0).count());
    }

    MY_LOG(1) << "MLIR compilation completed: " << artifact.filename << " ("
              << artifact.bytes.size() << " bytes)";
  }

  IPass &self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1MlirPass, morphizen_pass_level1_mlir)
