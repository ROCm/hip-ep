/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"

// Morphizen headers
#include "hip/timing.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <limits>
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
  config.artifactFormat = ArtifactFormat::LLVM_IR;
  config.optLevel = 2;
  config.useDynamicDispatch = false;

  auto ep_ctx = ctx->get_session_config("ep.context_enable");
  bool epctxExport = ep_ctx.has_value() && ep_ctx.value() == "1";
  config.skipConstantData = !epctxExport;

  try {
    // Single compile option selecting the per-model artifact format.
    //   "LLVM_IR" (default) -> OS-portable LLVM IR (.bc), JIT-loaded by the EP.
    //   "NATIVE"            -> per-OS native .dll/.so loaded via LoadLibrary/
    //                          dlopen. Opt-in for benchmarking/dev; not the
    //                          production deployment format (signed-DLL
    //                          policy).
    // Unknown values are logged and coerced to LLVM_IR.
    std::string artifact_format_str =
        ctx->get_provider_option("artifact_format", "LLVM_IR");
    if (artifact_format_str == "NATIVE") {
      config.artifactFormat = ArtifactFormat::NATIVE;
    } else {
      config.artifactFormat = ArtifactFormat::LLVM_IR;
      if (artifact_format_str != "LLVM_IR") {
        MY_LOG(1) << "artifact_format=" << artifact_format_str
                  << " is not recognized; falling back to LLVM_IR.";
      }
    }

    std::string opt_level_str =
        ctx->get_provider_option("optimization_level", "2");
    config.optLevel = std::stoi(opt_level_str);

    // DynamicDispatch backend selection (NPU/IPU vs GPU)
    // Check provider option first, then fall back to env var
    std::string use_dd_str =
        ctx->get_provider_option("use_dynamic_dispatch", "");
    if (use_dd_str.empty()) {
      // Fallback to environment variable for tools that don't support provider options
      const char* env_dd = std::getenv("HIPEP_USE_DYNAMIC_DISPATCH");
      if (env_dd) {
        use_dd_str = env_dd;
      }
    }
    config.useDynamicDispatch = (use_dd_str == "true" || use_dd_str == "1");

    MY_LOG(1) << "use_dynamic_dispatch provider option: '"
              << ctx->get_provider_option("use_dynamic_dispatch", "<unset>") << "'";
    if (std::getenv("HIPEP_USE_DYNAMIC_DISPATCH")) {
      MY_LOG(1) << "HIPEP_USE_DYNAMIC_DISPATCH env var: '"
                << std::getenv("HIPEP_USE_DYNAMIC_DISPATCH") << "'";
    }

  } catch (const std::exception &ex) {
    MY_LOG(1) << "Failed to parse provider options: " << ex.what()
              << ", using defaults";
  }

  MY_LOG(1) << "Artifact format: "
            << (config.artifactFormat == ArtifactFormat::NATIVE ? "NATIVE"
                                                                : "LLVM_IR")
            << "; skipConstantData="
            << (config.skipConstantData ? "true" : "false")
            << (epctxExport ? " (EPContext export -> constants file)"
                            : " (streaming default)")
            << "; useDynamicDispatch="
            << (config.useDynamicDispatch ? "true" : "false");

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

// MLIR uses kDynamic (INT64_MIN) for dynamic dims; ONNX metadata uses -1.
// The MLIR backend may mutate graph shapes during bytecode construction,
// replacing -1 with kDynamic.  Normalize back to -1 for the metadata.
static constexpr int64_t kMLIRDynamic = std::numeric_limits<int64_t>::min();
static int64_t normalizeDim(int64_t dim) {
  return dim == kMLIRDynamic ? -1 : dim;
}

// Step 5: Build metadata JSON from graph inputs and outputs.
// Output shapes are emitted verbatim (static extent or -1 for dynamic dims);
// the DLL sizes dynamic outputs in-graph at runtime via the output-allocator
// callback.
static std::string build_metadata_json(const CompilationArtifact &artifact,
                                       Graph &graph) {
  mlir_metadata::Metadata metadata;
  metadata.set_artifact_filename(artifact.filename);
  // Record the artifact format so the EP picks the matching loader before it
  // opens the artifact (the artifact's own metadata blob is inside the
  // artifact and cannot drive the load decision).
  metadata.set_artifact_format(
      artifact.format == ArtifactFormat::NATIVE ? "NATIVE" : "LLVM_IR");

  GraphRef graphRef(graph);

  for (const auto &input : graphRef.inputs()) {
    auto *input_proto = metadata.add_inputs();
    input_proto->set_name(input.name());
    input_proto->set_elem_type(input.element_type());

    auto shape_ptr = input.shape();
    if (shape_ptr && !input.is_unknown_shape()) {
      input_proto->set_rank(static_cast<int32_t>(shape_ptr->size()));
      for (int64_t dim : *shape_ptr) {
        input_proto->add_shape(normalizeDim(dim));
      }
    } else {
      input_proto->set_rank(-1);
    }
  }

  for (const auto &output : graphRef.outputs()) {
    auto *output_proto = metadata.add_outputs();
    output_proto->set_name(output.name());
    output_proto->set_elem_type(output.element_type());

    auto shape_ptr = output.shape();
    if (shape_ptr && !output.is_unknown_shape()) {
      output_proto->set_rank(static_cast<int32_t>(shape_ptr->size()));

      // Emit the output shape verbatim (static extent, or -1 for a dynamic
      // dim). Dynamic output dims are sized in-graph at runtime by the DLL's
      // output-allocator callback.
      for (int d = 0; d < static_cast<int>(shape_ptr->size()); ++d) {
        output_proto->add_shape(normalizeDim((*shape_ptr)[d]));
      }
    } else {
      output_proto->set_rank(-1);
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

    auto t0 = timing_now();
    auto t_prev = t0;

    // Step 1: Load configuration from provider options
    auto config = load_config(self.get_context().get());

    TIMING_LOG("[Session] load_config: %.3fs\n", record_elapsed(t_prev));

    // Step 2: Get MLIR bytecode from graph
    auto mlir_bytecode = get_mlir_bytecode(self.get_context().get(), graph);
    if (mlir_bytecode.empty()) {
      LOG(WARNING) << "Empty graph bytecode, skipping compilation";
      return;
    }

    TIMING_LOG("[Session] MLIR bytecode serialization: %.3fs (%zu bytes)\n",
               record_elapsed(t_prev), mlir_bytecode.size());

    // Step 3: Compile bytecode to artifact
    auto fs = self.get_context()->get_file_system();
    auto artifactOpt = compile_mlir(mlir_bytecode, config, fs.get());
    if (!artifactOpt) {
      LOG(WARNING) << "MLIR compilation failed, skipping";
      return;
    }
    CompilationArtifact artifact = *artifactOpt;

    TIMING_LOG("[Session] MLIR compilation (CompilerDriver): %.3fs\n",
               record_elapsed(t_prev));

    // Step 4: Write artifact to EPContext
    if (!write_artifact_to_epcontext(self.get_context().get(), artifact)) {
      return;
    }

    TIMING_LOG("[Session] Write artifact to EPContext: %.3fs\n",
               record_elapsed(t_prev));

    // Step 5: Build metadata JSON from graph outputs
    auto metadata_json = build_metadata_json(artifact, graph);

    TIMING_LOG("[Session] Build metadata JSON: %.3fs\n",
               record_elapsed(t_prev));

    // Step 6: Fuse graph into single MLIR custom op
    if (!fuse_graph(self, graph, metadata_json, artifact.filename)) {
      LOG(WARNING) << "Graph fusion failed";
      return;
    }

    TIMING_LOG("[Session] Fuse graph: %.3fs\n", record_elapsed(t_prev));
    TIMING_LOG("[Session] Level1MlirPass::process total: %.3fs\n",
               elapsed_since(t0));

    MY_LOG(1) << "MLIR compilation completed: " << artifact.filename << " ("
              << artifact.bytes.size() << " bytes)";
  }

  IPass &self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1MlirPass, morphizen_pass_level1_mlir)
