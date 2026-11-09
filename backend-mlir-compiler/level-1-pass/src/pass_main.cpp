/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"
#include "corelib_model.h"
#include "hip/Compiler/RocMlirArtifactTarget.h"
#include "hip/debug_log.h"

// Morphizen headers
#include "hip/native_artifacts.h"
#include "hip/timing.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Protobuf
#include "google/protobuf/struct.pb.h"
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

  auto ep_ctx = ctx->get_session_config("ep.context_enable");
  bool epctxExport = ep_ctx.has_value() && ep_ctx.value() == "1";
  config.skipConstantData = !epctxExport;

  // Single compile option selecting the per-model artifact format.
  //   "LLVM_IR" (default) -> OS-portable LLVM IR (.bc), JIT-loaded by the EP.
  //   "NATIVE"            -> per-OS native .dll/.so loaded via LoadLibrary/
  //                          dlopen. Requires HIPDNN_EP_ENABLE_NATIVE_ARTIFACTS
  //                          (default OFF); otherwise this throws.
  // Unknown values are logged and coerced to LLVM_IR.
  std::string artifact_format_str = "LLVM_IR";
  try {
    artifact_format_str =
        ctx->get_provider_option("artifact_format", "LLVM_IR");

    std::string opt_level_str =
        ctx->get_provider_option("optimization_level", "2");
    config.optLevel = std::stoi(opt_level_str);

  } catch (const std::exception &ex) {
    MY_LOG(1) << "Failed to parse provider options: " << ex.what()
              << ", using defaults";
    artifact_format_str = "LLVM_IR";
  }

  // Thrown outside the parse try/catch so a disabled native build cannot
  // swallow the failure and silently compile LLVM IR instead.
  if (artifact_format_str == "NATIVE") {
#if !HIPDNN_EP_ENABLE_NATIVE_ARTIFACTS
    throw std::runtime_error(hipdnn::kNativeArtifactsDisabledMessage);
#else
    config.artifactFormat = ArtifactFormat::NATIVE;
#endif
  } else if (artifact_format_str != "LLVM_IR") {
    MY_LOG(1) << "artifact_format=" << artifact_format_str
              << " is not recognized; falling back to LLVM_IR.";
  }

  MY_LOG(1) << "Artifact format: "
            << (config.artifactFormat == ArtifactFormat::NATIVE ? "NATIVE"
                                                                : "LLVM_IR")
            << "; skipConstantData="
            << (config.skipConstantData ? "true" : "false")
            << (epctxExport ? " (EPContext export -> constants file)"
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
  // Same process and env as the compile that just ran, so this is the arch
  // CompilerDriver embedded rocMLIR kernels for (empty if it did not).
  metadata.set_rocmlir_arch(hip::compiler::rocMlirArtifactTarget());

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
    // Skip producerless (constant) outputs, mirroring fuse_graph(): excluding
    // them here keeps the artifact metadata outputs consistent with the
    // meta_def outputs (see build_output_index_map).
    if (!output.find_producer().has_value()) {
      continue;
    }
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
                       const std::string &unique_id,
                       const char *domain = "MLIR") {
  MY_LOG(1) << "Attempting to fuse entire graph with domain '" << domain << "'";

  GraphRef graphRef(graph);

  // Extract graph input names
  std::vector<std::string> input_names;
  for (const auto &input : graphRef.inputs()) {
    input_names.push_back(input.name());
  }
  MY_LOG(1) << "Graph inputs: " << input_names.size();

  // Skip graph outputs with no producer node -- a constant initializer surfaced
  // directly as a graph output (e.g. an Identity(constant) that ORT constant
  // folding collapses to a bare initializer). It has no value inside the fused
  // region, so claiming it would build an invalid fused-function return; ORT
  // serves such initializer-backed outputs directly.
  std::vector<std::string> output_names;
  for (const auto &output : graphRef.outputs()) {
    if (!output.find_producer().has_value()) {
      MY_LOG(1) << "Skipping producerless (constant) graph output: "
                << output.name();
      continue;
    }
    output_names.push_back(output.name());
  }
  MY_LOG(1) << "Graph outputs: " << output_names.size();

  // Try to fuse
  auto [meta_def, fuse_error] =
      self.try_fuse(graphRef, unique_id, input_names, output_names,
                    {},    // constant_initializers (empty for now)
                    domain // domain name
      );

  if (meta_def == nullptr) {
    MY_LOG(1) << "Fusion failed: " << fuse_error.comments;
    LOG(WARNING) << "Failed to create " << domain
                 << " fusion: " << fuse_error.comments;
    return false;
  }

  // try_fuse follows explicit operands, so a graph input used only through
  // an If/Loop/Scan body is missing from MetaDef::inputs. Put those used
  // inputs back, in graph-argument order. Leave an argument out when nothing
  // in the graph reads it: it is not a fused-node input, and listing it makes
  // MetaDef longer than the ORT node.
  {
    std::vector<std::string> discovered{meta_def->inputs().begin(),
                                        meta_def->inputs().end()};
    std::unordered_set<std::string> discoveredSet(discovered.begin(),
                                                  discovered.end());
    std::vector<std::string> kept;
    kept.reserve(input_names.size());
    for (const auto &input : graphRef.inputs()) {
      const std::string &name = input.name();
      if (discoveredSet.count(name) != 0 || !input.find_consumers().empty())
        kept.push_back(name);
    }
    for (const auto &name : discovered) {
      if (std::find(kept.begin(), kept.end(), name) == kept.end())
        kept.push_back(name);
    }
    if (kept != discovered) {
      MY_LOG(1) << "Restoring used graph inputs (" << kept.size()
                << ") over try_fuse inputs (" << discovered.size() << ")";
      meta_def->clear_inputs();
      for (const auto &name : kept)
        meta_def->add_inputs(name);
    }
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
// Model-level CoreLib path
// ============================================================================

// Whole-model offload: provider option first, then the env var for tools that
// cannot set one.
static bool use_corelib_model(PassContext *ctx) {
  std::string v = ctx->get_provider_option("use_ryzenai_corelib_model", "");
  if (v.empty())
    v = hip_get_env("HIPEP_USE_RYZENAI_CORELIB_MODEL");
  return v == "1" || v == "true";
}

// An onnxruntime-genai decoder: attention_mask and a KV cache in, logits and
// the updated cache out. What it is and which of its other inputs a model
// takes, CoreLib decides; filled into `sig` when the graph qualifies.
static bool is_genai_decoder(Graph &graph,
                             hipdnn_ep::corelib::GraphSignature *sig) {
  GraphConstRef g(GraphRef{graph});
  bool mask = false, past = false, logits = false, present = false;
  sig->inputs.clear();
  sig->vocab_size = -1;
  for (const auto &in : g.inputs()) {
    const std::string &n = in.name();
    sig->inputs.push_back(n);
    mask |= n == "attention_mask";
    past |= n.rfind("past_key_values.", 0) == 0;
  }
  for (const auto &out : g.outputs()) {
    const std::string &n = out.name();
    present |= n.rfind("present.", 0) == 0;
    if (n != "logits")
      continue;
    logits = true;
    auto shape = out.shape();
    if (shape && !out.is_unknown_shape() && !shape->empty() &&
        shape->back() > 0)
      sig->vocab_size = shape->back();
  }
  return mask && past && logits && present;
}

// What the genai_config.json beside the model says: the context
// onnxruntime-genai runs it at (search.max_length, 0 for CoreLib's own) and
// its model type (model.type, e.g. "qwen3"), empty when absent.
struct GenaiConfig {
  int64_t max_length = 0;
  std::string model_type;
};

static GenaiConfig read_genai_config(const std::filesystem::path &model) {
  GenaiConfig out;
  std::ifstream f(model.parent_path() / "genai_config.json");
  if (!f)
    return out;
  std::stringstream ss;
  ss << f.rdbuf();
  google::protobuf::Struct config;
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  if (!google::protobuf::util::JsonStringToMessage(ss.str(), &config, options)
           .ok())
    return out;
  auto object = [](const google::protobuf::Struct &s,
                   const char *key) -> const google::protobuf::Struct * {
    auto it = s.fields().find(key);
    return it != s.fields().end() && it->second.has_struct_value()
               ? &it->second.struct_value()
               : nullptr;
  };
  if (const auto *search = object(config, "search")) {
    auto it = search->fields().find("max_length");
    if (it != search->fields().end() && it->second.has_number_value())
      out.max_length = int64_t(it->second.number_value());
  }
  if (const auto *m = object(config, "model")) {
    auto it = m->fields().find("type");
    if (it != m->fields().end() && it->second.has_string_value())
      out.model_type = it->second.string_value();
  }
  return out;
}

// Hands the whole graph to a CoreLib model when one takes it: fuses it under
// CORELIBMODEL with the parameter {arch, path, max_sequence_length} and
// leaves the session for the custom op. False, after one line saying why, when
// the graph is to be compiled as usual. No model is named here: CoreLib's own
// are tried, so a CoreLib with new models needs no change.
static bool try_corelib_model(IPass &self, Graph &graph) {
  PassContext *ctx = self.get_context().get();
  hipdnn_ep::corelib::GraphSignature sig;
  if (!use_corelib_model(ctx) || !is_genai_decoder(graph, &sig))
    return false;
  const std::filesystem::path model = ctx->get_model_path();
  if (model.empty()) {
    fprintf(stderr, "[corelib] model: no model file path (an in-memory "
                    "model); compiling it for the GPU\n");
    return false;
  }
  const GenaiConfig genai = read_genai_config(model);
  const int64_t max_len = genai.max_length;
  // A hint, not a filter: the model onnxruntime-genai names is tried first.
  sig.model_type = genai.model_type;
  std::string why;
  auto session = hipdnn_ep::corelib::create_model_session("", model.string(),
                                                          max_len, sig, &why);
  if (!session) {
    fprintf(stderr, "[corelib] model: %s; compiling it for the GPU\n",
            why.c_str());
    return false;
  }
  const std::string &arch = hipdnn_ep::corelib::architecture(*session);
  const std::string id = "corelib_model_" + arch;
  // The fused node's parameter must be a JSON object (a protobuf Struct).
  google::protobuf::Struct param;
  auto &fields = *param.mutable_fields();
  fields["arch"].set_string_value(arch);
  fields["path"].set_string_value(model.string());
  fields["max_sequence_length"].set_number_value(double(max_len));
  std::string param_json;
  CHECK(google::protobuf::util::MessageToJsonString(param, &param_json).ok());
  // A refused fusion falls through to the usual compile.
  if (!fuse_graph(self, graph, param_json, id, "CORELIBMODEL"))
    return false;
  hipdnn_ep::corelib::put_session(id, std::move(session));
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

    // A whole model CoreLib runs skips the MLIR compile.
    if (try_corelib_model(self, graph)) {
      TIMING_LOG("[Session] CoreLib model: %.3fs\n", record_elapsed(t_prev));
      return;
    }

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
