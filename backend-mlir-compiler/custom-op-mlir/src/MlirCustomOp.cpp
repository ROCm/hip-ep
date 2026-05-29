/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "MlirCustomOp.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>

// Protobuf headers
#include "google/protobuf/util/json_util.h"
#include "metadata.pb.h"

// Component headers
#include "InferenceState.h"

// HIPDNN_EP_PERF instrumentation dependencies
#ifndef BUILD_MOCK_RUNTIME
#include <hip/hip_runtime_api.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {

// Tensor marshaling state - holds tensors, shapes, and span
struct TensorData {
  std::vector<tensor_t> tensors;
  std::vector<std::vector<int64_t>> shapes; // Storage for shape arrays
  span_t span;
};

static size_t ort_element_size(ONNXTensorElementDataType dtype) {
  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return 1;
  default:
    return 4;
  }
}

static size_t onnx_elem_type_size(int elem_type) {
  switch (elem_type) {
  case 1:
    return 4; // FLOAT
  case 2:
    return 1; // UINT8
  case 3:
    return 1; // INT8
  case 5:
    return 2; // INT16
  case 6:
    return 4; // INT32
  case 7:
    return 8; // INT64
  case 9:
    return 1; // BOOL
  case 10:
    return 2; // FLOAT16
  case 11:
    return 8; // DOUBLE
  case 12:
    return 4; // UINT32
  case 13:
    return 8; // UINT64
  case 16:
    return 2; // BFLOAT16
  default:
    return 4;
  }
}

// Build a mapping from compiler/meta_def input index to ORT kernel context
// input index. ORT's fused node may reorder inputs relative to the meta_def
// order; the mapping is recorded in meta_def.input_argument_indice by
// MorphiZen's Compile phase.
static std::vector<int>
build_input_index_map(const morphizen::MetaDefProto &meta_def) {
  int n = meta_def.inputs_size();
  std::vector<int> map(n);
  for (int i = 0; i < n; ++i) {
    map[i] = (!meta_def.input_argument_indice().empty())
                 ? meta_def.input_argument_indice(i)
                 : i;
    MY_LOG(3) << "Input map: compiler[" << i << "] '" << meta_def.inputs(i)
              << "' -> ort[" << map[i] << "]";
  }
  return map;
}

// Marshal input tensors from ORT context.
// input_index_map maps from compiler input index (= DLL input index) to
// the ORT kernel context input index.
TensorData marshal_input_tensors(OrtKernelContext *context,
                                 const std::vector<int> &input_index_map) {
  Ort::KernelContext ctx(context);
  size_t num_inputs = input_index_map.size();

  MY_LOG(2) << "Marshaling " << num_inputs << " input tensors";

  TensorData data;
  data.tensors.resize(num_inputs);
  data.shapes.resize(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    int ort_idx = input_index_map[i];
    auto input_tensor = ctx.GetInput(ort_idx);
    auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    data.shapes[i] = tensor_info.GetShape();

    data.tensors[i].data = const_cast<void *>(input_tensor.GetTensorRawData());
    data.tensors[i].shape = data.shapes[i].data();
    data.tensors[i].rank = data.shapes[i].size();
    data.tensors[i].element_size =
        ort_element_size(tensor_info.GetElementType());
    // Carry the OrtValue's OrtMemoryInfoDeviceType straight into
    // tensor_t.memory_type (the enum values are 1:1, see custom_op_mlir.hpp).
    // prepare_input fast-paths TENSOR_MEMORY_GPU into an alias (no H2D copy);
    // CPU / FPGA / NPU fall through to the legacy host H2D path.
    data.tensors[i].memory_type =
        static_cast<int>(input_tensor.GetTensorMemoryInfo().GetDeviceType());

    MY_LOG(3) << "Input[" << i << "] (ort_idx=" << ort_idx
              << "): rank=" << data.tensors[i].rank
              << " element_size=" << data.tensors[i].element_size
              << " memory_type=" << data.tensors[i].memory_type;
  }

  data.span.data = data.tensors.data();
  data.span.count = data.tensors.size();

  return data;
}

// Build a mapping from metadata output index to ORT kernel context output
// index. The metadata output order (which matches the compiled DLL's output
// order) may differ from the meta_def output order (which matches the fused
// node / ORT kernel context order). MorphiZen's try_fuse() computes outputs
// via calculate_return_values() in DFS-topological order rather than
// preserving the caller-supplied output order. We resolve this by matching
// output names between the two orderings.
static std::vector<int> build_output_index_map(
    const google::protobuf::RepeatedPtrField<mlir_metadata::Output> &outputs,
    const morphizen::MetaDefProto &meta_def) {
  std::vector<int> map(outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &name = outputs[i].name();
    int meta_def_idx = -1;
    for (int j = 0; j < meta_def.outputs_size(); ++j) {
      if (meta_def.outputs(j) == name) {
        meta_def_idx = j;
        break;
      }
    }
    CHECK(meta_def_idx >= 0)
        << "metadata output '" << name << "' not found in meta_def outputs";
    int ort_idx = (!meta_def.output_argument_indice().empty())
                      ? meta_def.output_argument_indice(meta_def_idx)
                      : meta_def_idx;
    map[i] = ort_idx;
    MY_LOG(3) << "Output map: metadata[" << i << "] '" << name
              << "' -> meta_def[" << meta_def_idx << "] -> ort[" << ort_idx
              << "]";
  }
  return map;
}

// Precompute, for each metadata output, the compiler-input index of the
// matching past_key_values input (or -1 if the output is not a `present.*`
// tensor / no matching past input was found). Built once at MlirCustomOp
// construction so the per-inference shape-override path is O(1) per output
// instead of an O(N×M) name-string scan on the decode hot path.
//
// TODO (follow-up PR): replace this name-based heuristic with explicit
// past↔present pairs serialized into the model_metadata ProtoBuf.  The
// emission point is the compiler-side MLIR pipeline (which has hip.gqa
// ops with past_key/present_key SSA values) -- NOT the Level-1 pass,
// which only sees the OrtGraph and would need a parallel GQA-node walker.
// The proposed schema adds `repeated PresentToPast pairs` to Metadata,
// emitted by an MLIR pass walking hip.gqa operands, and consumed here in
// place of (or as an override for) the name-based string match.  The
// name match works for OGA-exported models (the only flavor we ship
// today) but assumes the `past_key_values.N.K` ↔ `present.N.K` naming
// convention, which is brittle to future exporters.
static std::vector<int> build_present_to_past_input_idx(
    const google::protobuf::RepeatedPtrField<mlir_metadata::Output> &outputs,
    const google::protobuf::RepeatedPtrField<mlir_metadata::Input> &inputs) {
  std::unordered_map<std::string, int> input_name_to_idx;
  input_name_to_idx.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i)
    input_name_to_idx.emplace(inputs[i].name(), i);

  std::vector<int> result(outputs.size(), -1);
  for (int i = 0; i < outputs.size(); ++i) {
    const std::string &name = outputs[i].name();
    if (name.size() < 9 || name.substr(0, 8) != "present.")
      continue;
    std::string past_name = "past_key_values." + name.substr(8);
    auto it = input_name_to_idx.find(past_name);
    if (it != input_name_to_idx.end())
      result[i] = it->second;
  }
  return result;
}

// Marshal output tensors from ORT context using metadata outputs.
// output_index_map maps from metadata output index (= DLL output index) to
// the ORT kernel context output index.
// For dynamic shapes (dim == -1 in metadata), resolves the actual dimension
// value from the corresponding input tensor using DimSource references.
// present_to_past_input_idx is precomputed at MlirCustomOp construction;
// entry is -1 for outputs that are not `present.*` tensors.
TensorData marshal_output_tensors(
    OrtKernelContext *context,
    const google::protobuf::RepeatedPtrField<mlir_metadata::Output> &outputs,
    const std::vector<int> &output_index_map,
    const std::vector<int> &input_index_map,
    const std::vector<int> &present_to_past_input_idx) {
  if (outputs.size() == 0) {
    LOG(FATAL) << "No output shapes in metadata";
  }

  Ort::KernelContext ctx(context);
  MY_LOG(2) << "Marshaling " << outputs.size() << " output tensors";

  TensorData data;
  data.tensors.resize(outputs.size());
  data.shapes.resize(outputs.size());

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_meta = outputs[i];
    data.shapes[i].assign(output_meta.shape().begin(),
                          output_meta.shape().end());

    // Resolve dynamic dims (-1) using DimSource entries from metadata.
    // Each DimSource with resolved=true says "this output dim equals
    // input[X].shape[Y]". Static dims and unresolved dynamic dims have
    // resolved=false and are left alone here (the post-loop CHECK below
    // will catch any unresolved dynamic dim that survives).
    for (int d = 0; d < static_cast<int>(data.shapes[i].size()); ++d) {
      if (data.shapes[i][d] != -1)
        continue;
      if (d >= output_meta.dim_sources_size())
        continue;
      const auto &ds = output_meta.dim_sources(d);
      if (!ds.resolved())
        continue;
      int src_input = ds.input_idx();
      int src_dim = ds.dim_idx();
      CHECK(src_input >= 0 &&
            src_input < static_cast<int>(input_index_map.size()))
          << "Output '" << output_meta.name() << "' dim " << d
          << ": DimSource references input " << src_input << " but only "
          << input_index_map.size() << " inputs are mapped";
      int ort_input_idx = input_index_map[src_input];
      auto src_tensor = ctx.GetInput(ort_input_idx);
      auto src_shape = src_tensor.GetTensorTypeAndShapeInfo().GetShape();
      CHECK(src_dim >= 0 && src_dim < static_cast<int>(src_shape.size()))
          << "Output '" << output_meta.name() << "' dim " << d
          << ": DimSource references input[" << src_input << "] dim " << src_dim
          << " but that input has rank " << src_shape.size();
      data.shapes[i][d] = src_shape[src_dim];
    }

    // OGA's past_present_share_buffer binds the same OrtValue to both
    // past_key (input) and present_key (output). DimSource may resolve
    // present_key's seq dim from attention_mask (tight shape, e.g. 7) instead
    // of the pre-allocated buffer size (e.g. 128). Override from the matching
    // past input's actual shape BEFORE GetOutput so ORT returns the pre-
    // allocated buffer (preserving pointer identity for in-place GQA append).
    //
    // Heuristic gating: "past dim is strictly larger than DimSource result"
    // is the proxy for shared-buffer mode here.  Reliable in practice for
    // OGA (past = max_length pre-allocated, DimSource resolves to tight
    // current length), but a future model with non-shared buffers and a
    // past_total > current_total (e.g. historical context that legitimately
    // shrinks) would misfire this heuristic.  TODO: replace the heuristic
    // with an explicit `past_present_share_buffer` attribute plumbed from
    // the ONNX GQA node into the model_metadata; gate the override on the
    // attribute rather than on the per-dim shape comparison.
    int past_idx = (i < static_cast<int>(present_to_past_input_idx.size()))
                       ? present_to_past_input_idx[i]
                       : -1;
    if (past_idx >= 0 && past_idx < static_cast<int>(input_index_map.size())) {
      int ort_past_idx = input_index_map[past_idx];
      auto past_tensor = ctx.GetInput(ort_past_idx);
      auto past_shape = past_tensor.GetTensorTypeAndShapeInfo().GetShape();
      if (past_shape.size() != data.shapes[i].size()) {
        MY_LOG(2) << "Output[" << i << "] '" << output_meta.name()
                  << "': share-buffer override skipped (rank mismatch: past="
                  << past_shape.size() << " vs out=" << data.shapes[i].size()
                  << ")";
      } else {
        // Only override dimensions that were dynamic (-1) in the compiled
        // metadata.  Static dims are architecture constants (batch=1,
        // num_heads, head_dim) and must never change — restricting the
        // override to dynamic dims prevents accidental corruption.
        bool overridden = false;
        bool any_dynamic = false;
        for (int d = 0; d < static_cast<int>(past_shape.size()); ++d) {
          if (output_meta.shape(d) != -1)
            continue;
          any_dynamic = true;
          if (past_shape[d] > data.shapes[i][d]) {
            data.shapes[i][d] = past_shape[d];
            overridden = true;
          }
        }
        if (overridden) {
          MY_LOG(2) << "Output[" << i << "] '" << output_meta.name()
                    << "': overrode dynamic dims from past input shape";
        } else if (any_dynamic) {
          MY_LOG(2) << "Output[" << i << "] '" << output_meta.name()
                    << "': share-buffer override skipped (past not larger "
                       "than DimSource — non-shared-buffer mode or pre-grow)";
        }
      }
    }

    for (int d = 0; d < static_cast<int>(data.shapes[i].size()); ++d) {
      CHECK(data.shapes[i][d] >= 0)
          << "Output '" << output_meta.name() << "' dim " << d
          << " is still dynamic (-1) after DimSource resolution. "
          << "This means the compiler emitted no resolvable DimSource and "
          << "no past-input override fired. Check that the dynamic dim has "
          << "a dim_param shared with at least one input.";
    }

    int ort_idx = output_index_map[i];
    auto output_tensor = ctx.GetOutput(ort_idx, data.shapes[i]);

    data.tensors[i].data = output_tensor.GetTensorMutableRawData();
    data.tensors[i].shape = data.shapes[i].data();
    data.tensors[i].rank = data.shapes[i].size();
    data.tensors[i].element_size = onnx_elem_type_size(output_meta.elem_type());
    // Same memory-type carry-over as inputs (lets finalize_output skip the
    // per-inference D2H when ORT pre-allocated the output OrtValue in our
    // morphizen GPU-mapped memory; matters for present_key/value sharing
    // buffers with past_key/value under past_present_share_buffer=true).
    data.tensors[i].memory_type =
        static_cast<int>(output_tensor.GetTensorMemoryInfo().GetDeviceType());

    MY_LOG(3) << "Output[" << i << "] (ort_idx=" << ort_idx
              << "): rank=" << data.tensors[i].rank
              << " element_size=" << data.tensors[i].element_size
              << " memory_type=" << data.tensors[i].memory_type;
  }

  data.span.data = data.tensors.data();
  data.span.count = data.tensors.size();

  return data;
}

namespace {

// Parse metadata from JSON string in MetaDefProto
// Logs FATAL and terminates on failure
mlir_metadata::Metadata parse_metadata_from_metadef(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def) {

  auto metadata_json = context->get_meta_def_param(*meta_def);
  mlir_metadata::Metadata metadata;
  auto status =
      google::protobuf::util::JsonStringToMessage(metadata_json, &metadata);

  if (!status.ok()) {
    LOG(FATAL) << "Failed to parse MLIR metadata: " << status.ToString();
  }

  MY_LOG(1) << "Parsed metadata - Artifact filename: "
            << metadata.artifact_filename();
  return metadata;
}

// Load artifact bytes from EPContext file stream
// Logs FATAL and terminates on failure
std::vector<uint8_t> load_artifact_from_epcontext(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::string &artifact_filename) {

  auto artifact_stream = context->open_file_for_read(artifact_filename);
  if (!artifact_stream) {
    LOG(FATAL) << "Failed to open artifact from EPContext: "
               << artifact_filename;
  }

  std::vector<uint8_t> artifact_bytes;
  artifact_bytes.reserve(1024 * 1024); // Start with 1MB

  const size_t chunk_size = 64 * 1024; // 64KB chunks
  uint8_t buffer[chunk_size];
  size_t total_read = 0;

  while (true) {
    size_t bytes_read = artifact_stream->fread(buffer, chunk_size);
    if (bytes_read == 0) {
      break;
    }
    artifact_bytes.insert(artifact_bytes.end(), buffer, buffer + bytes_read);
    total_read += bytes_read;

    if (bytes_read < chunk_size) {
      break; // EOF reached
    }
  }

  if (artifact_bytes.empty()) {
    LOG(FATAL) << "Failed to read artifact bytes from EPContext";
  }

  MY_LOG(1) << "Loaded artifact: " << total_read << " bytes";
  return artifact_bytes;
}
} // anonymous namespace

// ============================================================================
// HIPDNN_EP_PERF: per-inference CPU + GPU timing for MlirCustomOp::Compute().
//
// When HIPDNN_EP_PERF is set to "1" (or higher) at process start, every
// Compute() call emits one [PERF] line to stderr and a [PERF SUMMARY] block
// on DLL unload.  The breakdown intentionally splits:
//
//   wall            = total CPU time spent inside Compute (incl. post-compute
//                     fence); closest analog to OGA's reported decode latency
//   marshal_in      = CPU time building input tensor_t span from ORT context
//   marshal_out     = CPU time building output tensor_t span from ORT context
//   compute_cpu     = CPU wall time across inference_state_->compute() (host
//                     dispatch + synchronous kernel launches)
//   gpu             = hipEvent-measured GPU time across the same compute()
//                     call on the EP stream
//   fence_residual  = CPU time blocked in hipDeviceSynchronize() issued AFTER
//                     compute() returns; non-zero means compute() returned
//                     with async GPU work still outstanding
//
// Interpretation grid for the decode gap vs llama.cpp Vulkan:
//   compute_cpu >> gpu       -> host dispatch / launch overhead dominates
//                               (per-kernel hipLaunchKernel cost, bitcode
//                               trampolines, or glue code in model.dll)
//   fence_residual >> 0      -> compute() is queueing async GPU work; the
//                               host returns early and we under-count GPU
//                               per call
//   gpu close to llama.cpp   -> kernels themselves are not the problem;
//                               the gap is in launch/dispatch / glue
//   gpu >> llama.cpp         -> kernels are the problem (GEMM, MatMulNBits,
//                               attention, etc.)
//
// Overhead when disabled: ~0 (single branch on a cached bool).
// Overhead when enabled:  small per-call cost dominated by the
//   hipDeviceSynchronize fence at the end of the timing block and the
//   per-Compute fprintf line. hipEventCreate/Destroy is now amortized
//   (events are session-scoped, lazy-allocated on first perf-enabled
//   Compute), and the events themselves are created with
//   hipEventDisableSystemFence so record() does not issue a system-scope
//   acquire/release fence -- both wasted work for events we only read
//   after the device-wide sync.
// ============================================================================
namespace {

bool perf_enabled() {
#ifdef BUILD_MOCK_RUNTIME
  return false;
#else
  static const bool enabled = [] {
    const char *v = std::getenv("HIPDNN_EP_PERF");
    return v && v[0] >= '1' && v[0] <= '9';
  }();
  return enabled;
#endif
}

struct PerfSample {
  double wall_ms;
  double marshal_in_ms;
  double marshal_out_ms;
  double compute_cpu_ms;
  double gpu_ms;
  double fence_residual_ms;
};

class PerfCollector {
public:
  void record(const PerfSample &s) {
    std::lock_guard<std::mutex> g(mu_);
    samples_.push_back(s);
    const size_t idx = samples_.size();
    std::fprintf(stderr,
                 "[PERF] #%zu wall=%.3f marshal_in=%.3f marshal_out=%.3f "
                 "compute_cpu=%.3f gpu=%.3f fence_residual=%.3f (ms)\n",
                 idx, s.wall_ms, s.marshal_in_ms, s.marshal_out_ms,
                 s.compute_cpu_ms, s.gpu_ms, s.fence_residual_ms);
    std::fflush(stderr);
  }

  void dump_summary() {
    std::lock_guard<std::mutex> g(mu_);
    if (samples_.empty()) {
      return;
    }
    std::fprintf(stderr, "[PERF SUMMARY] total_inferences=%zu\n",
                 samples_.size());
    print_stats("wall_ms", &PerfSample::wall_ms);
    print_stats("marshal_in_ms", &PerfSample::marshal_in_ms);
    print_stats("marshal_out_ms", &PerfSample::marshal_out_ms);
    print_stats("compute_cpu_ms", &PerfSample::compute_cpu_ms);
    print_stats("gpu_ms", &PerfSample::gpu_ms);
    print_stats("fence_residual_ms", &PerfSample::fence_residual_ms);
    std::fflush(stderr);
  }

private:
  // Called with mu_ held.
  void print_stats(const char *name, double PerfSample::*field) {
    std::vector<double> v;
    v.reserve(samples_.size());
    double sum = 0.0;
    for (const auto &s : samples_) {
      v.push_back(s.*field);
      sum += s.*field;
    }
    std::sort(v.begin(), v.end());
    const double min = v.front();
    const double max = v.back();
    const double median = v[v.size() / 2];
    // p99 index is clamped to back() for small sample counts.
    const size_t p99_idx = (v.size() * 99) / 100;
    const double p99 = v[p99_idx < v.size() ? p99_idx : v.size() - 1];
    const double mean = sum / static_cast<double>(v.size());
    std::fprintf(stderr,
                 "[PERF SUMMARY] %-18s  min=%.3f  mean=%.3f  median=%.3f  "
                 "p99=%.3f  max=%.3f\n",
                 name, min, mean, median, p99, max);
  }

  std::mutex mu_;
  std::vector<PerfSample> samples_;
};

// Declared at namespace scope *before* g_perf_printer so that construction
// order (declaration order within a TU) guarantees g_perf_printer is
// destroyed first at DLL unload, while g_perf_collector is still alive for
// its dump_summary() call.
PerfCollector g_perf_collector;

PerfCollector &perf_collector() { return g_perf_collector; }

// Dumps the summary at DLL unload.  See ordering note above.
struct PerfSummaryPrinter {
  ~PerfSummaryPrinter() {
    if (perf_enabled()) {
      g_perf_collector.dump_summary();
    }
  }
};
PerfSummaryPrinter g_perf_printer;

} // anonymous namespace

MlirCustomOp::MlirCustomOp(
    std::shared_ptr<const morphizen::PassContext> context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
    onnxruntime::Model *model)
    : morphizen::CustomOpImp(context, meta_def, model) {

  MY_LOG(1) << "MlirCustomOp constructor";

  // Parse metadata from JSON
  metadata_ = parse_metadata_from_metadef(context, meta_def);
  // Precompute index mappings (compiler order -> ORT kernel context order)
  input_index_map_ = build_input_index_map(*meta_def);
  output_index_map_ = build_output_index_map(metadata_.outputs(), *meta_def);
  // Precompute present.* -> past_key_values.* input index lookup so the
  // shape-override loop in marshal_output_tensors is O(1) per output instead
  // of an O(N×M) name-string scan on the per-token decode hot path.
  present_to_past_input_idx_ =
      build_present_to_past_input_idx(metadata_.outputs(), metadata_.inputs());
  // Get FileSystem from PassContext for constants file resolution.
  // const_cast follows the established morphizen pattern (custom_op_imp.hpp).
  auto fs =
      const_cast<morphizen::PassContext *>(context.get())->get_file_system();
  // Create inference state from DLL bytes (uses morphizen::Plugin)
  inference_state_ = customop::InferenceState::create(
      load_artifact_from_epcontext(context, metadata_.artifact_filename()),
      fs.get());
}

MlirCustomOp::~MlirCustomOp() {
#ifndef BUILD_MOCK_RUNTIME
  // Release the lazy-allocated HIPDNN_EP_PERF event pair. Created on first
  // perf-enabled Compute() and reused for the session lifetime; nullptr
  // here means perf was never enabled, so nothing to release.
  if (ep_perf_ev_start_) {
    (void)hipEventDestroy(static_cast<hipEvent_t>(ep_perf_ev_start_));
    ep_perf_ev_start_ = nullptr;
  }
  if (ep_perf_ev_stop_) {
    (void)hipEventDestroy(static_cast<hipEvent_t>(ep_perf_ev_stop_));
    ep_perf_ev_stop_ = nullptr;
  }
#endif
}

void MlirCustomOp::Compute(const OrtApi *api, OrtKernelContext *context) const {
  MY_LOG(2) << "MlirCustomOp::Compute() called";

  // Tell the runtime that a new forward pass is starting so per-Compute()
  // caches (currently the GQA seqlens_k cache) are invalidated. No-op if
  // the model.dll predates the begin_compute export -- but in that case
  // the cache must be disabled via HIPDNN_EP_GQA_CACHE_SEQLENS=0
  // (default-on), otherwise stale values would survive across forward
  // passes. The mismatch is detected at session creation and produces a
  // LOG(WARNING). The call is a single cached indirect dispatch, so
  // leaving it on the fast path costs ~1 ns.
  inference_state_->begin_compute();

  if (!perf_enabled()) {
    // --- Fast path: original behaviour, no timing overhead. ---
    auto inputs = marshal_input_tensors(context, input_index_map_);
    auto outputs =
        marshal_output_tensors(context, metadata_.outputs(), output_index_map_,
                               input_index_map_, present_to_past_input_idx_);

    int ret = inference_state_->compute(&inputs.span, &outputs.span);
    if (ret != 0) {
      LOG(ERROR) << "inference_compute() failed with code: " << ret;
      // TODO: Throw ORT exception
    }

    MY_LOG(2) << "Compute completed successfully";
    return;
  }

#ifndef BUILD_MOCK_RUNTIME
  // --- HIPDNN_EP_PERF path: wall + per-phase + GPU timing. ---
  using clock = std::chrono::steady_clock;
  auto elapsed_ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  const auto t_enter = clock::now();
  auto inputs = marshal_input_tensors(context, input_index_map_);
  const auto t_after_in = clock::now();
  auto outputs =
      marshal_output_tensors(context, metadata_.outputs(), output_index_map_,
                             input_index_map_, present_to_past_input_idx_);
  const auto t_after_out = clock::now();

  // Stream used by inference_compute (first field of RuntimeState).  May be
  // the default stream (hipStream_t(0)) — hipEvent APIs accept that fine.
  auto stream = static_cast<hipStream_t>(inference_state_->get_stream_raw());

  // HIP error codes are intentionally discarded below: any failure here only
  // corrupts diagnostic numbers, never the inference result.  Casting to void
  // also suppresses C4834 ([[nodiscard]] discarded) under MSVC /W4.
  // Session-lifetime event pair, lazy-allocated on first perf-enabled
  // Compute() and reused thereafter. Mirrors the event-pool design on
  // the runtime side (lib/Runtime/op_profile.cpp): avoids the
  // hipEventCreate / hipEventDestroy round-trip on every Compute() at
  // the cost of holding two events for the session lifetime, released
  // in ~MlirCustomOp().
  //
  // hipEventDisableSystemFence: same rationale + same "NEVER set on
  // events read without a follow-up sync" guardrail as the op_profile
  // event pool. We only read elapsed time AFTER the
  // hipDeviceSynchronize() below, which is a stronger ordering
  // guarantee than the per-record system fence would have provided.
  if (!ep_perf_ev_start_) {
    hipEvent_t ev = nullptr;
    (void)hipEventCreateWithFlags(&ev, hipEventDisableSystemFence);
    ep_perf_ev_start_ = static_cast<void *>(ev);
    ev = nullptr;
    (void)hipEventCreateWithFlags(&ev, hipEventDisableSystemFence);
    ep_perf_ev_stop_ = static_cast<void *>(ev);
  }
  auto ev_start = static_cast<hipEvent_t>(ep_perf_ev_start_);
  auto ev_stop = static_cast<hipEvent_t>(ep_perf_ev_stop_);
  (void)hipEventRecord(ev_start, stream);

  const int ret = inference_state_->compute(&inputs.span, &outputs.span);
  const auto t_after_compute = clock::now();

  (void)hipEventRecord(ev_stop, stream);
  // hipDeviceSynchronize() is the fence: measures residual async work left
  // on any stream after compute() returned.  For correctly-synchronous
  // dispatch paths this should be ~0.
  (void)hipDeviceSynchronize();
  const auto t_after_fence = clock::now();

  float gpu_ms_f = 0.0f;
  (void)hipEventSynchronize(ev_stop);
  (void)hipEventElapsedTime(&gpu_ms_f, ev_start, ev_stop);
  // Events are NOT destroyed here -- they are owned by MlirCustomOp and
  // released in the destructor (see MlirCustomOp::~MlirCustomOp).

  if (ret != 0) {
    LOG(ERROR) << "inference_compute() failed with code: " << ret;
    // TODO: Throw ORT exception
  }

  PerfSample s;
  s.wall_ms = elapsed_ms(t_enter, t_after_fence);
  s.marshal_in_ms = elapsed_ms(t_enter, t_after_in);
  s.marshal_out_ms = elapsed_ms(t_after_in, t_after_out);
  s.compute_cpu_ms = elapsed_ms(t_after_out, t_after_compute);
  s.gpu_ms = static_cast<double>(gpu_ms_f);
  s.fence_residual_ms = elapsed_ms(t_after_compute, t_after_fence);
  perf_collector().record(s);

  // Flush per-op profile AFTER the timing window has closed. The resolve
  // step (one hipEventElapsedTime per recorded op + std::map aggregation
  // + per-row fprintf) scales with the number of profiled events per
  // inference -- on a typical decoder graph (several hundred profiled
  // ops) it costs roughly 1 ms / Compute, large enough that keeping it
  // inside the timing window measurably inflated HIPDNN_EP_PERF=1 wall_ms.
  // The cost still happens, but it's now charged to "between-Compute"
  // time and never enters wall_ms / gpu_ms / fence_residual_ms.
  inference_state_->flush_op_profile();

  MY_LOG(2) << "Compute completed successfully";
#endif // BUILD_MOCK_RUNTIME
}

} // namespace mlir_compilation
