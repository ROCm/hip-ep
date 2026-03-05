/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Level-1 pass for the HIP compiler backend.
 * Orchestrates: MLIR text -> HipCompiler -> DLL -> EPContext -> fuse graph.
 */
#include "HipCompiler.h"

#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <fstream>
#include <glog/logging.h>
#include <sstream>

using namespace morphizen;
using namespace morphizen_cxx;
using namespace hipdnn::compiler;

DEF_ENV_PARAM(HIP_COMPILER_DEBUG, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(HIP_COMPILER_DEBUG) >= n)

namespace {

// Step 1: Get MLIR text from graph
static std::string get_mlir_text(PassContext *ctx, Graph &graph) {
  auto bytecode = GraphConstRef(GraphRef(graph)).save_string();
  if (bytecode->empty())
    return "";

  MY_LOG(1) << "MLIR text size: " << bytecode->size() << " bytes";

  if (ENV_PARAM(HIP_COMPILER_DEBUG) >= 2) {
    auto dump_path = ctx->get_dump_directory() / "hip_mlir_dump.mlir";
    MY_LOG(1) << "Dumping MLIR to " << dump_path;
    CHECK(std::ofstream(dump_path, std::ios::binary)
              .write(bytecode->data(), bytecode->size())
              .good())
        << "Failed to dump MLIR";
  }

  return std::string(bytecode->data(), bytecode->size());
}

// Step 2: Compile MLIR to DLL
static std::optional<CompileResult>
compile_hip(const std::string &mlir_text, PassContext *ctx) {
  // Write MLIR to temp file (HipCompiler works with files)
  auto temp_mlir =
      ctx->get_dump_directory() / "hip_compile_input.mlir";
  {
    std::ofstream out(temp_mlir, std::ios::binary);
    out.write(mlir_text.data(), mlir_text.size());
  }

  auto temp_dll =
      ctx->get_dump_directory() / "hip_model_compiled.dll";

  CompileOptions opts;
  opts.optLevel = 2;

  return HipCompiler::compileFile(temp_mlir.string(), temp_dll.string(), opts);
}

// Step 3: Write DLL artifact to EPContext
static bool write_artifact(PassContext *ctx, const CompileResult &result,
                           const std::string &filename) {
  MY_LOG(1) << "Writing artifact to EPContext: " << filename;

  auto stream = ctx->open_file_for_write(filename);
  if (!stream) {
    LOG(WARNING) << "Failed to open EPContext file: " << filename;
    return false;
  }

  size_t written = stream->fwrite(result.dllBytes.data(), result.dllBytes.size());
  stream.reset();

  if (written != result.dllBytes.size()) {
    LOG(WARNING) << "Incomplete artifact write";
    return false;
  }
  MY_LOG(1) << "Wrote " << written << " bytes";
  return true;
}

// Step 4: Build metadata string for MetaDefProto
// Format: "artifact_filename=<name>;output:<name>,<rank>,<elem_type>,<dim0>x<dim1>x..."
static std::string build_metadata(const CompileResult &result,
                                  Graph &graph) {
  std::ostringstream ss;
  ss << "artifact_filename=hip_model_compiled";

  GraphRef gref(graph);
  for (const auto &output : gref.outputs()) {
    ss << ";output:" << output.name();
    ss << "," << 0; // rank placeholder

    auto shape_ptr = output.shape();
    if (shape_ptr && !output.is_unknown_shape()) {
      ss << "," << output.element_type();
      ss << ",";
      bool first = true;
      for (int64_t dim : *shape_ptr) {
        if (!first)
          ss << "x";
        ss << dim;
        first = false;
      }
    } else {
      ss << "," << output.element_type() << ",1";
    }
  }

  MY_LOG(1) << "Metadata: " << ss.str();
  return ss.str();
}

// Step 5: Fuse graph into a single HIP custom op node
static bool fuse_graph(IPass &self, Graph &graph,
                       const std::string &metadata,
                       const std::string &unique_id) {
  MY_LOG(1) << "Fusing graph with domain 'HIP'";

  GraphRef gref(graph);

  std::vector<std::string> input_names;
  for (const auto &input : gref.inputs())
    input_names.push_back(input.name());

  std::vector<std::string> output_names;
  for (const auto &output : gref.outputs())
    output_names.push_back(output.name());

  auto [meta_def, fuse_error] = self.try_fuse(
      gref, unique_id, input_names, output_names, {}, "HIP");

  if (meta_def == nullptr) {
    LOG(WARNING) << "Fusion failed: " << fuse_error.comments;
    return false;
  }

  self.attach_meta_def_param(*meta_def, metadata.c_str());
  self.fuse(gref, std::move(*meta_def));

  MY_LOG(1) << "HIP compilation and fusion completed";
  return true;
}

// ============================================================================
// Pass implementation
// ============================================================================
struct Level1HipPass {
  Level1HipPass(IPass &self) : self_{self} {}

  void process(IPass &self, Graph &graph) {
    MY_LOG(1) << "Level1HipPass::process()";

    auto mlir_text = get_mlir_text(self.get_context().get(), graph);
    if (mlir_text.empty()) {
      LOG(WARNING) << "Empty MLIR, skipping";
      return;
    }

    auto result = compile_hip(mlir_text, self.get_context().get());
    if (!result) {
      LOG(WARNING) << "HIP compilation failed, skipping";
      return;
    }

    const std::string artifact_name = "hip_model_compiled";

    if (!write_artifact(self.get_context().get(), *result, artifact_name))
      return;

    auto metadata = build_metadata(*result, graph);

    if (!fuse_graph(self, graph, metadata, artifact_name)) {
      LOG(WARNING) << "Graph fusion failed";
      return;
    }

    MY_LOG(1) << "HIP pass completed: " << result->dllBytes.size() << " bytes";
  }

  IPass &self_;
};

} // namespace

DEFINE_MORPHIZEN_PASS(Level1HipPass, morphizen_pass_level1_hip)
