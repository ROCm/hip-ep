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

  int output_idx = 0;
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

      // MLIR-refined output shape from the compiler (post `InferOnnxShapes`).
      // Drives `DimSource.static_value` for dims the original ONNX export
      // declared as symbolic but the compiler tightened to a static int.
      const std::vector<int64_t> *refined =
          (output_idx < static_cast<int>(artifact.refined_output_shapes.size()))
              ? &artifact.refined_output_shapes[output_idx]
              : nullptr;

      // Per-output-dim SSA origin from the compiler (post
      // `InferOnnxShapes` backward-trace). Each entry is
      // `(graph_arg_index, dim_idx)` into the function arguments;
      // `(-1, -1)` means no traceable origin. Drives
      // `DimSource.input_idx + dim_idx` for dims that are genuinely
      // dynamic AND whose dim_param doesn't match any input dim_param —
      // i.e. the as-shipped Gemma-3 vision case where input dim 0 is
      // `num_images` but output dim 0 is `num_image_tokens` (same
      // value, different label).
      const std::vector<CompilationArtifact::DimOriginTriple> *origins =
          (output_idx <
           static_cast<int>(artifact.refined_output_dim_origins.size()))
              ? &artifact.refined_output_dim_origins[output_idx]
              : nullptr;

      for (int d = 0; d < static_cast<int>(shape_ptr->size()); ++d) {
        int64_t dim_val = normalizeDim((*shape_ptr)[d]);
        output_proto->add_shape(dim_val);

        // DimSource carries one of three states; see metadata.proto for
        // the consumer precedence (`static_value > 0` first, then
        // `resolved`).
        auto *ds = output_proto->add_dim_sources();

        // STATIC: either declared static in the original graph, or
        // tightened to a static positive value by InferOnnxShapes during
        // compile. The pass never widens, so `max(graph, refined)` is
        // well-defined — pick the larger of the two when both are
        // positive (the refined value is at least as precise; the
        // graph value is at most as precise).
        int64_t static_v = dim_val;
        if (refined && d < static_cast<int>(refined->size()) &&
            (*refined)[d] > 0)
          static_v = std::max(static_v, (*refined)[d]);
        if (static_v > 0) {
          ds->set_input_idx(-1);
          ds->set_dim_idx(-1);
          ds->set_resolved(false);
          ds->set_static_value(static_v);
          continue;
        }

        // RUNTIME-INPUT-LOOKUP via dim_param name match.
        if (dp && d < static_cast<int>(dp->size()) && !(*dp)[d].empty()) {
          auto pit = dim_param_map.find((*dp)[d]);
          if (pit != dim_param_map.end()) {
            ds->set_input_idx(pit->second.first);
            ds->set_dim_idx(pit->second.second);
            ds->set_resolved(true);
            continue;
          }
        }

        // RUNTIME-INPUT-LOOKUP via SSA origin trace from the compiler.
        // Kicks in when dim_param names don't match across the EP <-> ONNX
        // boundary (e.g. Gemma-3 vision's `num_image_tokens` output dim
        // vs `num_images` input dim — semantically equivalent, labelled
        // differently). The InferOnnxShapes backward-trace walked
        // through Cast/Transpose/MatMul/Conv/Reshape/etc. and recorded
        // that this output dim ultimately reads from input `arg_idx`'s
        // dim `arg_dim_idx` with a scalar multiplier `mult` accumulated
        // across any Reshape ops that resize dim 0 (e.g. Qwen vision's
        // patch merger contributes mult=0.25 for divide-by-4).
        if (origins && d < static_cast<int>(origins->size())) {
          int64_t arg_idx = (*origins)[d].arg_idx;
          int64_t arg_dim = (*origins)[d].dim_idx;
          double mult = (*origins)[d].mult;
          if (arg_idx >= 0 && arg_idx < input_idx && arg_dim >= 0) {
            ds->set_input_idx(static_cast<int>(arg_idx));
            ds->set_dim_idx(static_cast<int>(arg_dim));
            ds->set_resolved(true);
            ds->set_mult(mult);
            MY_LOG(2) << "Output '" << output.name() << "' dim " << d
                      << " resolved via SSA-origin trace to input[" << arg_idx
                      << "].dim[" << arg_dim << "] * " << mult;
            continue;
          }
        }

        // UNRESOLVED: genuinely dynamic, no name match, no static
        // refinement, no SSA-traceable origin. Fail loudly so the user
        // sees a clear error pointing at the offending output / dim.
        LOG(FATAL)
            << "Output '" << output.name() << "' dim " << d
            << " is dynamic AND has no matching input dim_param AND "
            << "InferOnnxShapes did not tighten it to a static value AND "
            << "the SSA-origin backward-trace could not find a function-"
            << "argument origin. Either fix the model (add a dim_param "
            << "matching some input) or extend InferOnnxShapes' trace "
            << "rules to cover the producing op.";
      }
    } else {
      output_proto->set_rank(-1);
    }

    MY_LOG(2) << "Output " << output.name() << ": rank=" << output_proto->rank()
              << ", elem_type=" << output_proto->elem_type();
    ++output_idx;
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
  // morphizen-core's `IPass_try_fuse` augments meta_def's inputs/outputs
  // with values discovered during a reverse DFS of the body nodes:
  // "arguments" (inputs the body needs that weren't explicitly listed,
  // typically internal values produced by Cast(Constant) chains the DFS
  // didn't trace into body_nodes) and "return_values" (intermediates that
  // happen to escape the fused region's notion of "output"). Both lists
  // can grow far beyond the actual graph inputs/outputs.
  //
  // ORT, on the other hand, computes the fused node's actual input/output
  // arity from the graph boundary, which matches the explicit lists we
  // passed. The downstream `update_argument_indice` in morphizen-ep.cpp
  // compares meta_def's lists against the ORT node's actual lists with
  // `CHECK_LE(meta_def_args.size(), node_value_infos.size())` and aborts
  // on mismatch.
  //
  // Fix: clamp meta_def's inputs/outputs to the EXACT lists we passed to
  // try_fuse before handing it off. We do this post-try_fuse rather than
  // patching morphizen-core because the morphizen submodule is shared
  // across consumers; our use case (whole-graph fusion with explicit
  // boundary lists) is unusual enough that the upstream auto-augment is
  // wrong for us specifically. constant_initializers / nodes are kept
  // intact — the EP runtime doesn't compare those against the ORT node.
  meta_def->mutable_inputs()->Clear();
  for (const auto &n : input_names)
    meta_def->add_inputs(n);
  meta_def->mutable_outputs()->Clear();
  for (const auto &n : output_names)
    meta_def->add_outputs(n);

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
