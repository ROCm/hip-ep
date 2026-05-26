/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "MlirCompiler.h"

// Morphizen headers
#include "hip/timing.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <sstream>
#include <stdexcept>
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

  // Dump-path resolution (Phase 1 — bisection repro):
  //
  //   1. HIPDNN_EP_BYTECODE_DUMP_PATH=<file>   -> dump bytecode here
  //      (most convenient; bypasses provider-options and cache-key
  //      hashing). A model that fails under the EP can be exported with
  //      one env var and then bisected with `hip-mlir-opt`. Triggers
  //      even if MORPHIZEN_DEBUG_MLIR_BACKEND is not set.
  //
  //      Distinct from the legacy `HIPDNN_EP_IR_DUMP_PATH` (consumed by
  //      `lib/Compiler/CompilerDriver.cpp::runMLIRPasses` to emit
  //      "IR after each pass" text via `pm.enableIRPrinting`). That
  //      path would overwrite a binary bytecode dump.
  //
  //   2. MORPHIZEN_DEBUG_MLIR_BACKEND>=2 -> dump to
  //      `<dump_dir>/mlir_bytecode_dump.mlir`. `dump_dir` is the
  //      `dump_dir` provider option if set, else
  //      `C:\temp\morphizen_dumps\<cache_key>` (Linux: `/tmp/...`).
  //
  // Both paths emit MLIR bytecode (the same format `Graph.save_string()`
  // produces). Convert to readable MLIR for diffing/inspection with
  // `hip-mlir-opt -o out.mlir <dump>`. See `tools/dump_imported_mlir.py`
  // and `docs/bisecting-ep-compile-failures.md` for the end-to-end
  // workflow.
  const char *env_dump_path = std::getenv("HIPDNN_EP_BYTECODE_DUMP_PATH");
  if (env_dump_path != nullptr && env_dump_path[0] != '\0') {
    LOG(INFO) << "Dumping MLIR bytecode to HIPDNN_EP_BYTECODE_DUMP_PATH = "
              << env_dump_path;
    if (!std::ofstream(env_dump_path, std::ios::binary)
             .write(bytecode->data(), bytecode->size())
             .good()) {
      LOG(WARNING) << "Failed to write MLIR bytecode dump to " << env_dump_path;
    }
  }

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
static CompilationResult compile_mlir(const std::string &mlir_bytecode,
                                      const CompilationConfig &config,
                                      morphizen::FileSystem *fs) {
  return MlirCompiler::compileFromBytecode(mlir_bytecode, config, fs);
}

// True when the user has opted in to silent CPU fallback on compile failure.
// Default is strict (returns false) so a compile failure throws an exception
// — matches the "Phase 0 — strict fallback" UX where a model that fails to
// lower to HIP must not look identical to a successful GPU run.
//
// Why throw and not abort: morphizen's outer std::exception catch will
// either skip the subgraph (XLNX_ENABLE_SKIP_FATAL=1, the morphizen default)
// or abort the process (XLNX_ENABLE_SKIP_FATAL=0). Either way the EP's
// supported-node list ends up empty, which ORT then catches IF the caller
// set `session.disable_cpu_ep_fallback=1` (e.g. our numeric test suite's
// `OrtEpBackend`). That converts our throw into a Python-visible
// `Fail`/`RuntimeError` from `InferenceSession.__init__` — caught cleanly
// by `pytest.raises` / `xfail(raises=Exception)`. Bare abort would crash
// pytest itself and break the K=0 xfail flow.
//
// Production callers who do NOT set `disable_cpu_ep_fallback=1` and want
// loud failures can additionally set `XLNX_ENABLE_SKIP_FATAL=0` to convert
// the morphizen catch into a process abort.
//
// Setting `HIPDNN_EP_ALLOW_CPU_FALLBACK=1` restores the legacy
// silent-skip behavior for users who knowingly want partial-EP execution.
static bool cpu_fallback_allowed() {
  const char *v = std::getenv("HIPDNN_EP_ALLOW_CPU_FALLBACK");
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Surface a fatal MLIR compile failure. By default this throws — see
// `cpu_fallback_allowed()` for the rationale (morphizen + ORT cooperate to
// turn the throw into a Python-visible session-init failure). Opt out with
// `HIPDNN_EP_ALLOW_CPU_FALLBACK=1`.
[[noreturn]] static void throw_on_compile_failure(const std::string &reason) {
  std::fprintf(stderr,
               "\n[MorphiZen EP] FATAL: MLIR compilation failed and "
               "HIPDNN_EP_ALLOW_CPU_FALLBACK is not set.\n"
               "  Reason: %s\n"
               "  Set HIPDNN_EP_ALLOW_CPU_FALLBACK=1 to let ORT route the "
               "failing subgraph to another EP (CPU). This hides real EP "
               "bugs - only use when you explicitly want partial-EP "
               "execution.\n\n",
               reason.c_str());
  std::fflush(stderr);
  LOG(ERROR) << "MorphiZen EP throwing on compile failure: " << reason;
  throw std::runtime_error("MorphiZen EP MLIR compilation failed: " + reason);
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
    auto compileOutcome = compile_mlir(mlir_bytecode, config, fs.get());
    if (!compileOutcome.artifact) {
      // Strict by default: a compile failure means the EP would silently
      // hand the subgraph back to ORT, which then routes it to CPU. That
      // would mask real bugs (e.g. dominance violations from dyn input
      // shapes) as "EP succeeded" since pure-CPU and EP-on-CPU look
      // identical from the Python side. Throw so that — when combined
      // with `session.disable_cpu_ep_fallback=1` on the caller side —
      // session creation fails with a Python-visible exception.
      const std::string reason =
          compileOutcome.error_message.empty()
              ? "MLIR compilation failed (no diagnostic captured)"
              : compileOutcome.error_message;
      if (!cpu_fallback_allowed()) {
        throw_on_compile_failure(reason);
      }
      LOG(WARNING) << "MLIR compilation failed, falling back to CPU "
                      "(HIPDNN_EP_ALLOW_CPU_FALLBACK=1): "
                   << reason;
      return;
    }
    CompilationArtifact artifact = std::move(*compileOutcome.artifact);

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
