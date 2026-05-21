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
#include "DimSpecResolver.h"
#include "InferenceState.h"
#include "model_metadata_generated.h"

// HIPDNN_EP_PERF instrumentation dependencies
#ifndef BUILD_MOCK_RUNTIME
#include <hip/hip_runtime.h>
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

// Marshal output tensors from ORT context using metadata outputs.
// output_index_map maps from metadata output index (= DLL output index) to
// the ORT kernel context output index.
TensorData marshal_output_tensors(
    OrtKernelContext *context,
    const google::protobuf::RepeatedPtrField<mlir_metadata::Output> &outputs,
    const std::vector<int> &output_index_map) {
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

// Tries to resolve the full pre-compute shape for output `i` using its
// DimSpec tree. Returns true iff every dim resolved (Category A / B / D
// only); false signals "this output has at least one RuntimeSlot dim, so
// it must be deferred until post-compute".
bool tryResolveOutputShapePreCompute(
    const mlir::hip::HipModelMetaInfoT &fb_meta, size_t out_idx,
    const std::vector<std::vector<int64_t>> &input_shapes,
    const std::vector<const void *> &input_data,
    std::vector<int64_t> &out_shape) {
  if (out_idx >= fb_meta.outputs.size())
    return false;
  const auto &ti = fb_meta.outputs[out_idx];
  if (!ti)
    return false;
  if (ti->dim_specs.empty()) {
    // Legacy / all-static output. Fall back to the static shape array
    // verbatim.
    out_shape.assign(ti->shape.begin(), ti->shape.end());
    return true;
  }
  out_shape.assign(ti->dim_specs.size(), 0);
  for (size_t d = 0; d < ti->dim_specs.size(); ++d) {
    const auto &spec = ti->dim_specs[d];
    if (!spec || spec->nodes.empty()) {
      // No spec for this dim -- trust the legacy static value.
      if (d >= ti->shape.size())
        return false;
      out_shape[d] = ti->shape[d];
      continue;
    }
    int64_t v = 0;
    // Pre-compute pass: pass null state so RuntimeSlot leaves resolve to
    // "false" (deferred) without crashing.
    if (!customop::resolve(*spec, input_shapes, input_data,
                           /*state=*/nullptr, v)) {
      // Hit a RuntimeSlot leaf -- the whole output is deferred.
      return false;
    }
    out_shape[d] = v;
  }
  return true;
}

// Resolve a single Category-C output dim post-compute using the published
// slot value. LOG(FATAL) if the slot wasn't published. dim_index is the
// output dim that was the RuntimeSlot leaf; all other dims are assumed to
// have been resolved already pre-compute and live in `static_shape`.
std::vector<int64_t> resolveOutputShapePostCompute(
    const mlir::hip::HipModelMetaInfoT &fb_meta, size_t out_idx,
    const std::vector<std::vector<int64_t>> &input_shapes,
    const std::vector<const void *> &input_data,
    const customop::InferenceState &state) {
  CHECK_LT(out_idx, fb_meta.outputs.size());
  const auto &ti = fb_meta.outputs[out_idx];
  CHECK(ti);
  CHECK(!ti->dim_specs.empty()) << "Post-compute resolver called on output "
                                << out_idx << " but it has no dim_specs";
  std::vector<int64_t> resolved(ti->dim_specs.size(), 0);
  for (size_t d = 0; d < ti->dim_specs.size(); ++d) {
    const auto &spec = ti->dim_specs[d];
    CHECK(spec && !spec->nodes.empty());
    int64_t v = 0;
    if (!customop::resolve(*spec, input_shapes, input_data, &state, v)) {
      LOG(FATAL) << "Output " << out_idx << " dim " << d
                 << " is a RuntimeSlot that the wrap_* failed to publish "
                    "before inference_compute returned -- runtime invariant "
                    "violation. Check that the corresponding wrap_* "
                    "implementation calls hipdnn_ep_state_publish_dim().";
    }
    resolved[d] = v;
  }
  return resolved;
}

} // namespace

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
// Overhead when enabled:  ~25us/call (hipEventCreate+Record+Sync+Destroy +
//   one hipDeviceSynchronize + one fprintf) - negligible vs decode cost.
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
  // Get FileSystem from PassContext for constants file resolution.
  // const_cast follows the established morphizen pattern (custom_op_imp.hpp).
  auto fs =
      const_cast<morphizen::PassContext *>(context.get())->get_file_system();
  // Create inference state from DLL bytes (uses morphizen::Plugin)
  inference_state_ = customop::InferenceState::create(
      load_artifact_from_epcontext(context, metadata_.artifact_filename()),
      fs.get());

  // Build per-output dynamic-shape metadata cache.
  // Populated only when the DLL carries FB-JSON metadata (new DLLs) AND
  // declares dyn_dim_slots_count > 0. Legacy / all-static models leave the
  // vector empty -- Compute() then takes the fast path.
  output_dyn_info_.resize(metadata_.outputs_size());
  if (auto *fb_meta = inference_state_->metadata()) {
    const size_t n_fb = fb_meta->outputs.size();
    if (n_fb != static_cast<size_t>(metadata_.outputs_size())) {
      LOG(FATAL) << "FB metadata output count " << n_fb
                 << " does not match proto metadata count "
                 << metadata_.outputs_size();
    }
    for (size_t i = 0; i < n_fb; ++i) {
      const auto &ti = fb_meta->outputs[i];
      if (!ti)
        continue;
      for (size_t d = 0; d < ti->dim_specs.size(); ++d) {
        const auto &spec = ti->dim_specs[d];
        if (!spec)
          continue;
        if (!customop::containsRuntimeSlot(*spec))
          continue;
        // Find the (unique) RuntimeSlot leaf. Multi-slot trees are not
        // supported today -- bail loudly so a future op exercising the
        // path doesn't silently mis-resolve the dim.
        int32_t found_slot = -1;
        for (const auto &node : spec->nodes) {
          if (node && node->kind == mlir::hip::DimSpecKind::RuntimeSlot) {
            if (found_slot >= 0) {
              LOG(FATAL) << "Output " << i << " dim " << d
                         << " contains multiple RuntimeSlot leaves; the EP "
                            "host-side resolver only supports one slot per "
                            "dim today";
            }
            found_slot = node->slot_id;
          }
        }
        CHECK_GE(found_slot, 0)
            << "containsRuntimeSlot returned true but no RuntimeSlot leaf "
               "was found in the tree";
        if (output_dyn_info_[i].has_runtime_slot) {
          LOG(FATAL)
              << "Output " << i
              << " has more than one dim driven by a RuntimeSlot; the EP "
                 "host-side resolver only supports a single slot-driven dim "
                 "per output today";
        }
        output_dyn_info_[i].has_runtime_slot = true;
        output_dyn_info_[i].slot_id = found_slot;
        output_dyn_info_[i].slot_dim_index = static_cast<int32_t>(d);
        MY_LOG(2) << "Output " << i << ": Category-C, dim " << d
                  << " <- slot " << found_slot;
      }
    }
  }
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

  // Reset per-Compute() dyn-slot publishers so wrap_* can publish into a
  // clean slate. No-op on legacy DLLs (the symbol is absent and the cached
  // function pointer is null). Cheap (~1 ns indirect call).
  inference_state_->reset_dyn_slots();

  // Detect whether any output of this graph is Category-C
  // (data-dependent shape published via runtime slot). When no output is
  // dynamic the legacy fast path runs unchanged -- this preserves the
  // existing perf characteristics for every model that doesn't include
  // NonZero / Range / ConstantOfShape on intermediate values.
  bool any_dynamic_output = false;
  for (const auto &info : output_dyn_info_) {
    if (info.has_runtime_slot) {
      any_dynamic_output = true;
      break;
    }
  }

  if (!perf_enabled() && !any_dynamic_output) {
    // --- Fast path: original behaviour, no timing overhead. ---
    auto inputs = marshal_input_tensors(context, input_index_map_);
    auto outputs =
        marshal_output_tensors(context, metadata_.outputs(), output_index_map_);

    int ret = inference_state_->compute(&inputs.span, &outputs.span);
    if (ret != 0) {
      LOG(ERROR) << "inference_compute() failed with code: " << ret;
      // TODO: Throw ORT exception
    }

    MY_LOG(2) << "Compute completed successfully";
    return;
  }

  if (any_dynamic_output) {
    computeDynamic(context);
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
      marshal_output_tensors(context, metadata_.outputs(), output_index_map_);
  const auto t_after_out = clock::now();

  // Stream used by inference_compute (first field of RuntimeState).  May be
  // the default stream (hipStream_t(0)) — hipEvent APIs accept that fine.
  auto stream = static_cast<hipStream_t>(inference_state_->get_stream_raw());

  // HIP error codes are intentionally discarded below: any failure here only
  // corrupts diagnostic numbers, never the inference result.  Casting to void
  // also suppresses C4834 ([[nodiscard]] discarded) under MSVC /W4.
  hipEvent_t ev_start = nullptr, ev_stop = nullptr;
  (void)hipEventCreate(&ev_start);
  (void)hipEventCreate(&ev_stop);
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
  (void)hipEventDestroy(ev_start);
  (void)hipEventDestroy(ev_stop);

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

  MY_LOG(2) << "Compute completed successfully";
#endif // BUILD_MOCK_RUNTIME
}

void MlirCustomOp::computeDynamic(OrtKernelContext *context) const {
#ifdef BUILD_MOCK_RUNTIME
  // The mock build has no HIP runtime; the dynamic-output path requires
  // hipMemcpyAsync / hipStreamSynchronize to finalize Category-C outputs.
  // Fail loudly so models with NonZero etc. don't silently produce empty
  // outputs against the mock runtime.
  (void)context;
  LOG(FATAL) << "MlirCustomOp::computeDynamic() requires the real GPU "
                "runtime; rebuild without BUILD_MOCK_RUNTIME";
#else
  MY_LOG(2) << "MlirCustomOp::computeDynamic() called";

  const auto *fb_meta = inference_state_->metadata();
  CHECK(fb_meta) << "Dynamic-output path requires DLL FB metadata "
                    "(inference_get_metadata_json must be exported)";

  Ort::KernelContext ctx(context);

  // ===== Phase 1: marshal inputs and capture data ptrs for B-leaf reads =====
  auto inputs = marshal_input_tensors(context, input_index_map_);

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<const void *> input_data;
  input_shapes.reserve(inputs.tensors.size());
  input_data.reserve(inputs.tensors.size());
  for (size_t i = 0; i < inputs.tensors.size(); ++i) {
    input_shapes.push_back(inputs.shapes[i]);
    // InputValueI64 leaves expect a host-readable i64 buffer. CPU-memory
    // inputs are fine (the EP marshal already exposes the host ptr); GPU
    // (memory_type == TENSOR_MEMORY_GPU) inputs would need an explicit
    // D2H stage. Today we only generate B-leaves for func-arg inputs whose
    // OnnxToHip provenance proves they live in host memory (Range starts
    // / ConstantOfShape shape), so a GPU input feeding a B-leaf is a
    // compiler bug. Stash the pointer either way; the resolver
    // LOG(FATAL)s on null when it actually needs to read.
    input_data.push_back(inputs.tensors[i].data);
  }

  // ===== Phase 2: marshal outputs (resolved + Category-C sentinels) =====
  TensorData outputs;
  outputs.tensors.resize(metadata_.outputs_size());
  outputs.shapes.resize(metadata_.outputs_size());
  // Per-output cached resolved shape (filled either pre- or post-compute).
  // Indexed identically to metadata_.outputs().
  std::vector<std::vector<int64_t>> resolved_shapes(metadata_.outputs_size());
  // Sentinel shape array for Category-C outputs. We give the runtime a
  // valid shape ptr (so buildMemrefDescriptor can GEP it) but the values
  // are placeholder -1s; main_graph's hip.nonzero etc. ignore the
  // descriptor since the lowered wrap_* drops the output buffer pointer.
  // The per-output entry lives inside resolved_shapes for the resolved
  // case; for the deferred case we allocate a sibling sentinel shape.
  std::vector<std::vector<int64_t>> sentinel_shapes(metadata_.outputs_size());
  for (int i = 0; i < metadata_.outputs_size(); ++i) {
    const auto &output_meta = metadata_.outputs(i);
    const auto &dyn_info = output_dyn_info_[i];

    if (!dyn_info.has_runtime_slot) {
      // Try resolving pre-compute -- works for both legacy static outputs
      // and Category A/B/D outputs whose tree is RuntimeSlot-free.
      bool ok = tryResolveOutputShapePreCompute(*fb_meta, i, input_shapes,
                                                input_data, resolved_shapes[i]);
      if (!ok) {
        // Should not happen -- the static fallback path always populates
        // resolved_shapes[i] from the legacy `shape:` array when no spec
        // is available, and dyn_info.has_runtime_slot would have been
        // true otherwise.
        LOG(FATAL) << "Output " << i
                   << " failed pre-compute resolution despite having no "
                      "RuntimeSlot in its dim_specs";
      }
      outputs.shapes[i] = resolved_shapes[i];
      int ort_idx = output_index_map_[i];
      auto t = ctx.GetOutput(ort_idx, outputs.shapes[i]);
      outputs.tensors[i].data = t.GetTensorMutableRawData();
      outputs.tensors[i].shape = outputs.shapes[i].data();
      outputs.tensors[i].rank = outputs.shapes[i].size();
      outputs.tensors[i].element_size =
          onnx_elem_type_size(output_meta.elem_type());
      outputs.tensors[i].memory_type =
          static_cast<int>(t.GetTensorMemoryInfo().GetDeviceType());
      MY_LOG(3) << "Output[" << i << "] resolved pre-compute, rank="
                << outputs.tensors[i].rank;
    } else {
      // Category-C deferred sentinel. Build a shape array of length rank
      // filled with -1 (so any code path that mistakenly inspects it gets
      // an obvious "uninitialized" pattern). The runtime's prepare_output
      // sees data==null and skips GPU allocation + D2H; the lowered
      // wrap_* publishes the actual buffer into a slot, which we read in
      // phase 4 below.
      const size_t rank =
          fb_meta->outputs[i] ? fb_meta->outputs[i]->shape.size() : 0;
      CHECK_GT(rank, 0u) << "Category-C output " << i
                         << " has rank 0 -- not supported";
      sentinel_shapes[i].assign(rank, -1);
      outputs.tensors[i].data = nullptr;
      outputs.tensors[i].shape = sentinel_shapes[i].data();
      outputs.tensors[i].rank = rank;
      outputs.tensors[i].element_size =
          onnx_elem_type_size(output_meta.elem_type());
      outputs.tensors[i].memory_type = TENSOR_MEMORY_CPU;
      MY_LOG(3) << "Output[" << i << "] Category-C sentinel, rank=" << rank
                << " slot_id=" << dyn_info.slot_id;
    }
  }
  outputs.span.data = outputs.tensors.data();
  outputs.span.count = outputs.tensors.size();

  // ===== Phase 3: run inference_compute =====
  int ret = inference_state_->compute(&inputs.span, &outputs.span);
  if (ret != 0) {
    LOG(ERROR) << "inference_compute() failed with code: " << ret;
    // Still attempt to read slots so we surface as much info as possible.
  }

  // The runtime's inference_compute already stream-syncs (stream_sync
  // step) before returning, so any wrap_* publish_dim+publish_buffer is
  // visible at this point. No extra sync needed.

  // ===== Phase 4: resolve Category-C outputs post-compute and D2H =====
  auto stream_v = inference_state_->get_stream_raw();
  hipStream_t stream = static_cast<hipStream_t>(stream_v);
  bool any_async_d2h = false;
  for (int i = 0; i < metadata_.outputs_size(); ++i) {
    const auto &dyn_info = output_dyn_info_[i];
    if (!dyn_info.has_runtime_slot)
      continue;

    // Re-resolve the shape with the inference_state available -- now
    // RuntimeSlot leaves return their published value.
    auto resolved = resolveOutputShapePostCompute(*fb_meta, i, input_shapes,
                                                  input_data, *inference_state_);
    // Allocate the actual-sized ORT OrtValue. ctx.GetOutput is idempotent
    // per ort_idx, but for Category-C outputs we deliberately did NOT
    // call it pre-compute, so this is the first (and only) call.
    int ort_idx = output_index_map_[i];
    auto out_t = ctx.GetOutput(ort_idx, resolved);
    void *host_dst = out_t.GetTensorMutableRawData();
    void *gpu_src = inference_state_->read_buffer(dyn_info.slot_id);
    int64_t numel = 1;
    for (int64_t d : resolved)
      numel *= d;
    const size_t elem_size = onnx_elem_type_size(
        metadata_.outputs(i).elem_type());
    const size_t bytes = static_cast<size_t>(numel) * elem_size;

    MY_LOG(3) << "Output[" << i << "] Category-C resolved: numel=" << numel
              << " bytes=" << bytes << " slot=" << dyn_info.slot_id
              << " gpu_src=" << gpu_src << " host_dst=" << host_dst;

    if (bytes == 0) {
      // Empty result (e.g. NonZero returns no non-zero elements). The
      // OrtValue is already allocated with the right shape; nothing to
      // copy.
      continue;
    }
    CHECK(gpu_src) << "Output " << i
                   << " is Category-C with non-empty result but the wrap_* "
                      "did not publish a GPU buffer for slot "
                   << dyn_info.slot_id;
    CHECK(host_dst);
    hipError_t herr = hipMemcpyAsync(host_dst, gpu_src, bytes,
                                     hipMemcpyDeviceToHost, stream);
    if (herr != hipSuccess) {
      LOG(ERROR) << "Category-C output " << i << " D2H failed: "
                 << hipGetErrorString(herr);
      continue;
    }
    any_async_d2h = true;
  }
  if (any_async_d2h) {
    hipError_t herr = hipStreamSynchronize(stream);
    if (herr != hipSuccess) {
      LOG(ERROR) << "Category-C post-compute stream sync failed: "
                 << hipGetErrorString(herr);
    }
  }

  MY_LOG(2) << "Compute (dynamic-output path) completed successfully";
#endif // BUILD_MOCK_RUNTIME
}

} // namespace mlir_compilation
