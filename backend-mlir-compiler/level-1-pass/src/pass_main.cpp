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

  auto ep_ctx = ctx->get_session_config("ep.context_enable");
  bool epctxExport = ep_ctx.has_value() && ep_ctx.value() == "1";
  config.skipConstantData = !epctxExport;

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
                                                                : "llvm_ir")
            << "; skipConstantData="
            << (config.skipConstantData ? "true" : "false")
            << (epctxExport ? " (EPContext export -> sidecar)"
                            : " (streaming default)");

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

// Parse "name1:p0,p1;name2:p0,p1,p2;..." into {name → [dim_params]}.
//
// Resolution rules (locked into docs/design/morphizen-ep-integration.md
// "dim_params_map Contract"):
//   * Empty trailing/leading segments are skipped silently (the encoding
//     allows a trailing ';').
//   * Malformed segments (no ':' separator) are skipped with a
//     LOG(WARNING) — historically silent, which masked upstream
//     mis-encodings; the audit log surfaces them in the build dump.
//   * Empty tensor names (segment of the form ":p0,p1") are skipped with
//     a LOG(WARNING) — same rationale, plus an empty key in the result
//     map would silently shadow a real lookup.
//   * Duplicate tensor names: first occurrence wins, later ones are
//     dropped with a LOG(WARNING).  This matches the dim_param_map
//     first-occurrence convention used below in build_metadata_json and
//     is the safer choice (later overrides could silently flip a
//     resolved DimSource to the wrong input/dim).
static std::unordered_map<std::string, std::vector<std::string>>
parse_dim_params_map(const std::string &encoded) {
  std::unordered_map<std::string, std::vector<std::string>> result;
  if (encoded.empty())
    return result;
  std::istringstream outer(encoded);
  std::string entry;
  while (std::getline(outer, entry, ';')) {
    if (entry.empty())
      continue;
    auto colon = entry.find(':');
    if (colon == std::string::npos) {
      LOG(WARNING) << "dim_params_map: skipping malformed segment '" << entry
                   << "' (no ':' separator)";
      continue;
    }
    auto name = entry.substr(0, colon);
    if (name.empty()) {
      LOG(WARNING) << "dim_params_map: skipping segment with empty tensor "
                      "name (raw='"
                   << entry << "')";
      continue;
    }
    if (result.count(name) != 0) {
      LOG(WARNING) << "dim_params_map: tensor '" << name
                   << "' has duplicate entry; keeping first occurrence";
      continue;
    }
    auto params_str = entry.substr(colon + 1);
    std::vector<std::string> params;
    std::istringstream inner(params_str);
    std::string p;
    while (std::getline(inner, p, ','))
      params.push_back(p);
    result[name] = std::move(params);
  }
  return result;
}

// Step 5: Build metadata JSON from graph inputs and outputs.
// For dynamic shapes, records DimSource entries that map each dynamic output
// dimension to the input tensor + dimension index that provides its runtime
// value. Uses dim_params_map model metadata (populated by IR converter from
// ORT's GetSymbolicDimensions) to match dimensions across tensors.
static std::string build_metadata_json(const CompilationArtifact &artifact,
                                       Graph &graph) {
  mlir_metadata::Metadata metadata;
  metadata.set_artifact_filename(artifact.filename);

  GraphRef graphRef(graph);

  // Retrieve dim_params_map from model metadata (set by IR converter).
  auto dim_params_encoded = morphizen::model_get_meta_data(
      morphizen::graph_get_model(graph), "dim_params_map");
  auto all_dim_params = parse_dim_params_map(dim_params_encoded);

  // Build dim_param → (input_idx, dim_idx) map from graph inputs.
  // Only the first occurrence of each symbolic name is recorded — later inputs
  // sharing the same dim_param inherit from this one at runtime.
  std::unordered_map<std::string, std::pair<int, int>> dim_param_map;
  int input_idx = 0;
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

      auto it = all_dim_params.find(input.name());
      if (it != all_dim_params.end()) {
        const auto &dp = it->second;
        for (int d = 0; d < static_cast<int>(dp.size()); ++d) {
          if (!dp[d].empty() &&
              dim_param_map.find(dp[d]) == dim_param_map.end()) {
            dim_param_map[dp[d]] = {input_idx, d};
          }
        }
      }
    } else {
      input_proto->set_rank(-1);
    }
    ++input_idx;
  }

  for (const auto &output : graphRef.outputs()) {
    auto *output_proto = metadata.add_outputs();
    output_proto->set_name(output.name());
    output_proto->set_elem_type(output.element_type());

    auto shape_ptr = output.shape();
    if (shape_ptr && !output.is_unknown_shape()) {
      output_proto->set_rank(static_cast<int32_t>(shape_ptr->size()));

      auto it = all_dim_params.find(output.name());
      const std::vector<std::string> *dp =
          (it != all_dim_params.end()) ? &it->second : nullptr;

      for (int d = 0; d < static_cast<int>(shape_ptr->size()); ++d) {
        int64_t dim_val = normalizeDim((*shape_ptr)[d]);
        output_proto->add_shape(dim_val);

        // Classify each output dim into one of three runtime-resolution
        // strategies (see DimResolution in metadata.proto):
        //   - static (dim_val >= 0)            -> DIM_STATIC
        //   - dynamic, identity copy of an input dim (shares a dim_param with
        //     a graph input)                   -> DIM_FROM_INPUT  (DimSource)
        //   - dynamic, non-identity function of inputs (no shared dim_param;
        //     e.g. a vision patch-merger seq collapse) -> DIM_FROM_SHAPE_FN
        // The SHAPE_FN case is resolved at runtime by the model.dll's
        // inference_infer_shapes program (emitted by BuildShapeFunctionPass).
        // We intentionally do NOT hard-fail here for a missing dim_param: the
        // shape function can express dims that no single input dim does, so a
        // dim_param-less dynamic dim is delegated to it instead of aborting
        // the compile. If the shape function ALSO cannot resolve it (returns
        // the kDynamic sentinel at runtime), the EP raises a clear per-output
        // CHECK at inference time.
        auto *ds = output_proto->add_dim_sources();
        if (dim_val != -1) {
          // Static dim: emit explicit sentinels so the wire format
          // unambiguously distinguishes "not populated" from
          // "intentionally references (input 0, dim 0)" without relying
          // on the proto-default-zero convention.  Consumers that read
          // `resolved` first see the same behavior as before; consumers
          // that don't see -1 instead of a misleading 0.
          ds->set_input_idx(-1);
          ds->set_dim_idx(-1);
          ds->set_resolved(false);
          ds->set_resolution(mlir_metadata::DIM_STATIC);
        } else {
          const bool has_param =
              dp && d < static_cast<int>(dp->size()) && !(*dp)[d].empty();
          auto pit =
              has_param ? dim_param_map.find((*dp)[d]) : dim_param_map.end();
          if (has_param && pit != dim_param_map.end()) {
            // Identity dynamic dim: equals input[pit].shape[pit] at runtime.
            ds->set_input_idx(pit->second.first);
            ds->set_dim_idx(pit->second.second);
            ds->set_resolved(true);
            ds->set_resolution(mlir_metadata::DIM_FROM_INPUT);
          } else {
            // Non-identity dynamic dim: delegate to inference_infer_shapes.
            ds->set_input_idx(-1);
            ds->set_dim_idx(-1);
            ds->set_resolved(false);
            ds->set_resolution(mlir_metadata::DIM_FROM_SHAPE_FN);
            MY_LOG(1) << "Output '" << output.name() << "' dim " << d
                      << " has no input-shared dim_param; will be resolved by "
                         "inference_infer_shapes at runtime.";
          }
        }
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
