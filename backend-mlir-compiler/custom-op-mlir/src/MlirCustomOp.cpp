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
#ifdef HIPDNN_EP_LINK_HIP_HOST
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
//
// Classic ABI only (use_output_allocator=0): every output shape in the
// metadata is fully static. Dynamic output dims (and the runtime DimSource
// resolution / OGA shared-KV-buffer override that classic mode once used to
// size them) were removed when the output-allocator ABI became the default --
// allocator mode sizes dynamic outputs in-graph via output_allocate_cb and
// never calls this function. The compiler rejects a dynamic output dim in
// classic mode at build time (see build_metadata_json), so a -1 surviving to
// here indicates a metadata/compiler bug and aborts.
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

    for (int d = 0; d < static_cast<int>(data.shapes[i].size()); ++d) {
      CHECK(data.shapes[i][d] >= 0)
          << "Output '" << output_meta.name() << "' dim " << d
          << " is dynamic (-1) in classic ABI metadata. Classic mode supports "
          << "only static output shapes; this model must be compiled with the "
          << "output allocator (the default ABI).";
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

// Decide which loader the EP should use for this artifact. The format is
// recorded in the EPContext metadata by the compiler (artifact_format), which
// always sets it (see pass_main.cpp); an empty/unknown value means a malformed
// or pre-PR EPContext and is fatal here rather than mis-loaded.
customop::ArtifactKind determine_artifact_kind(const std::string &format_str) {
  customop::ArtifactKind kind;
  if (!customop::artifactKindFromFormat(format_str, kind)) {
    LOG(FATAL) << "EPContext metadata has empty/unknown artifact_format='"
               << format_str << "'; expected '"
               << customop::kArtifactFormatLlvmIr << "' or '"
               << customop::kArtifactFormatNative << "'.";
  }

  MY_LOG(1) << "Artifact loader: "
            << (kind == customop::ArtifactKind::NATIVE ? "native (Plugin)"
                                                       : "LLVM IR (JIT)");
  return kind;
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
//                               trampolines, or glue code in the JITted
//                               per-model module)
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
#ifndef HIPDNN_EP_LINK_HIP_HOST
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

// ===========================================================================
// Output-allocator mode (2-arg inference_compute) plumbing.
//
// In allocator mode the DLL receives no pre-filled outputs span. Instead
// main_graph calls hipdnn_ep_alloc_output(out_idx, shape, rank, elem_size) for
// each graph output once its runtime shape is known; the runtime forwards to
// the callback the EP installed. The callback maps out_idx -> ORT output, asks
// ORT for the buffer at the DLL's in-graph shape, and returns a pointer the DLL
// writes into:
//   * GPU (aliased) output -> ORT's GPU-accessible pointer (zero-copy).
//   * host (CPU) output    -> an EP-owned GPU scratch pointer; Compute()
//                             D2H-copies it into ORT's host buffer afterwards.
// ===========================================================================

// One deferred device->host copy for a host (CPU) output.
struct PendingD2H {
  void *gpu_src;  // EP GPU scratch the DLL wrote into
  void *host_dst; // ORT host output buffer
  size_t nbytes;
};

// Per-Compute() context passed to the C callback via output_allocator_t.self.
// Lives on compute_with_output_allocator's stack for the duration of the 2-arg
// inference_compute; the allocator is cleared on the RuntimeState before it
// goes out of scope so `self` can never dangle.
struct OutputAllocatorCtx {
  Ort::KernelContext *ctx = nullptr;
  const google::protobuf::RepeatedPtrField<mlir_metadata::Output> *outputs =
      nullptr;
  const std::vector<int> *output_index_map = nullptr;
  // Borrowed from the MlirCustomOp instance (grow-on-demand, reused across
  // Compute()): one GPU scratch buffer per output index for host outputs.
  std::vector<HostOutputScratch> *host_out_scratch = nullptr;
  std::vector<PendingD2H> pending_d2h; // filled during compute, consumed after
  // Output-completeness guard: which metadata outputs received an alloc call.
  std::vector<bool> allocated;
};

#ifdef HIPDNN_EP_LINK_HIP_HOST
// Ensure scratch[idx] holds at least nbytes of device memory. Grows by
// hipFree + hipMalloc (never shrinks). Safe within a Compute(): each output
// index is allocated exactly once, so a grow here cannot invalidate a pointer
// the DLL is still about to write this pass (distinct indices = distinct
// buffers). LOG(FATAL) on failure -- never returns null.
static void *ensure_host_out_slot(std::vector<HostOutputScratch> &scratch,
                                  size_t idx, size_t nbytes) {
  if (idx >= scratch.size())
    scratch.resize(idx + 1);
  HostOutputScratch &slot = scratch[idx];
  if (slot.capacity < nbytes) {
    if (slot.ptr)
      (void)hipFree(slot.ptr);
    void *p = nullptr;
    hipError_t e = hipMalloc(&p, nbytes ? nbytes : 1);
    if (e != hipSuccess || !p)
      LOG(FATAL) << "hipMalloc(" << nbytes
                 << ") for output-allocator host scratch failed: "
                 << hipGetErrorString(e);
    slot.ptr = p;
    slot.capacity = nbytes;
  }
  return slot.ptr;
}
#endif // HIPDNN_EP_LINK_HIP_HOST

// C callback installed on the RuntimeState (output_allocator_t.allocate).
// noexcept: invoked from C (the model.dll runtime); a C++ exception crossing
// that boundary is UB. Any failure aborts via LOG(FATAL) with a clear message
// rather than returning null (a null would be written by the DLL -> segfault
// with no diagnostic).
void *output_allocate_cb(void *self, int64_t out_idx, const int64_t *shape,
                         int64_t rank, int64_t elem_size) noexcept {
  auto *octx = static_cast<OutputAllocatorCtx *>(self);
  try {
    if (out_idx < 0 || out_idx >= static_cast<int64_t>(octx->outputs->size())) {
      LOG(FATAL) << "output allocator: out_idx " << out_idx << " out of range ("
                 << octx->outputs->size() << " outputs)";
    }
    // Use the DLL's in-graph shape verbatim -- the EP never reshapes an output
    // here. The shape is computed in-graph by hip.alloc_output from its
    // producer's operands; when a dynamic output dim is sized from a graph
    // input's `memref.dim`, the requested shape already equals that input
    // buffer's allocated extent. ORT's GetOutput then returns the pre-bound
    // (IO-bound) buffer rather than allocating a fresh one, so when an input
    // and output are bound to the same buffer that identity is preserved with
    // no EP-side override.
    std::vector<int64_t> out_shape(shape, shape + rank);

    int ort_idx = (*octx->output_index_map)[static_cast<int>(out_idx)];
    auto out_tensor = octx->ctx->GetOutput(ort_idx, out_shape);
    int mem_type =
        static_cast<int>(out_tensor.GetTensorMemoryInfo().GetDeviceType());
    void *ort_ptr = out_tensor.GetTensorMutableRawData();

    octx->allocated[static_cast<size_t>(out_idx)] = true;

    if (mem_type == TENSOR_MEMORY_GPU) {
      // Zero-copy: ORT buffer is already GPU-accessible (e.g. the EP's
      // host-mapped allocator, or caller-provided device memory). The DLL
      // writes into it directly.
      return ort_ptr;
    }

    // Host (CPU) output: the DLL writes into EP GPU scratch; Compute()
    // D2H-copies into ort_ptr after inference_compute returns (the 2-arg
    // interface already stream-synced).
#ifdef HIPDNN_EP_LINK_HIP_HOST
    size_t nelem = 1;
    for (int64_t d : out_shape)
      nelem *= static_cast<size_t>(d);
    size_t nbytes = nelem * static_cast<size_t>(elem_size);
    void *gpu = ensure_host_out_slot(*octx->host_out_scratch,
                                     static_cast<size_t>(out_idx), nbytes);
    octx->pending_d2h.push_back({gpu, ort_ptr, nbytes});
    return gpu;
#else
    // Mock runtime writes host memory directly; no GPU staging needed.
    (void)elem_size;
    return ort_ptr;
#endif
  } catch (const std::exception &e) {
    LOG(FATAL) << "output allocator callback failed for out_idx " << out_idx
               << ": " << e.what();
  } catch (...) {
    LOG(FATAL) << "output allocator callback failed for out_idx " << out_idx
               << " (unknown exception)";
  }
  return nullptr; // unreachable: LOG(FATAL) aborts. Silences -Wreturn-type.
}

} // anonymous namespace

MlirCustomOp::MlirCustomOp(
    std::shared_ptr<const morphizen::PassContext> context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
    onnxruntime::Model *model)
    : morphizen::CustomOpImp(context, meta_def, model) {

  MY_LOG(1) << "MlirCustomOp constructor";

  // Parse metadata from JSON
  metadata_ = parse_metadata_from_metadef(context, meta_def);
  // Dispatch mode travels with the artifact's metadata (not a live provider
  // option), so it always matches the loaded DLL's ABI.
  use_output_allocator_ = metadata_.use_output_allocator();
  // Precompute index mappings (compiler order -> ORT kernel context order)
  input_index_map_ = build_input_index_map(*meta_def);
  output_index_map_ = build_output_index_map(metadata_.outputs(), *meta_def);
  // Get FileSystem from PassContext for constants file resolution.
  // const_cast follows the established morphizen pattern (custom_op_imp.hpp).
  auto fs =
      const_cast<morphizen::PassContext *>(context.get())->get_file_system();
  auto artifact_bytes =
      load_artifact_from_epcontext(context, metadata_.artifact_filename());
  auto kind = determine_artifact_kind(metadata_.artifact_format());
  inference_state_ =
      customop::InferenceState::create(artifact_bytes, fs.get(), kind);
}

MlirCustomOp::~MlirCustomOp() {
#ifdef HIPDNN_EP_LINK_HIP_HOST
  // Free the allocator-mode host-output GPU scratch (one buffer per output
  // index). No-op in classic mode (the vector stays empty).
  for (const HostOutputScratch &slot : host_out_scratch_)
    if (slot.ptr)
      (void)hipFree(slot.ptr);
#endif
}

// Output-allocator dispatch: 2-arg inference_compute with in-graph
// hip.alloc_output. The EP installs a per-Compute callback that hands each
// graph output an ORT (or GPU-scratch) buffer; host outputs are D2H-copied
// afterwards. See the OutputAllocatorCtx / output_allocate_cb block above.
void MlirCustomOp::compute_with_output_allocator(
    OrtKernelContext *context) const {
  MY_LOG(2) << "MlirCustomOp::compute_with_output_allocator()";

  auto inputs = marshal_input_tensors(context, input_index_map_);

  Ort::KernelContext ort_ctx(context);
  OutputAllocatorCtx octx;
  octx.ctx = &ort_ctx;
  octx.outputs = &metadata_.outputs();
  octx.output_index_map = &output_index_map_;
  octx.host_out_scratch = &host_out_scratch_;
  octx.allocated.assign(metadata_.outputs().size(), false);

  output_allocator_t alloc;
  alloc.self = &octx;
  alloc.allocate = &output_allocate_cb;
  inference_state_->set_output_allocator(&alloc);

  int ret = inference_state_->compute_with_output_allocator(&inputs.span);

  // Clear before octx leaves scope so a stale self pointer can never be used by
  // a later call (e.g. the next Compute() before it reinstalls its own ctx).
  inference_state_->set_output_allocator(nullptr);

  if (ret != 0) {
    LOG(ERROR) << "inference_compute (allocator) failed with code: " << ret;
  }

  // Output-completeness guard. Allocator mode requires every declared output to
  // be produced in-graph (each yields a hip.alloc_output -> callback). A
  // missing one means a passthrough / aliased output the DLL never allocates --
  // unsupported today. Fail loudly rather than hand ORT an uninitialized buffer
  // (which can silently pass a CPU-vs-CPU comparison).
  for (size_t i = 0; i < octx.allocated.size(); ++i) {
    if (!octx.allocated[i]) {
      LOG(FATAL) << "output allocator: output '"
                 << metadata_.outputs()[static_cast<int>(i)].name() << "' (idx "
                 << i
                 << ") was never allocated by the model.dll. Passthrough / "
                    "aliased outputs are not supported in output-allocator "
                    "mode yet.";
    }
  }

#ifdef HIPDNN_EP_LINK_HIP_HOST
  // Deferred host-output copies. The 2-arg interface already stream-synced, so
  // the GPU scratch holds final data; a blocking hipMemcpy is safe.
  for (const auto &p : octx.pending_d2h) {
    hipError_t e =
        hipMemcpy(p.host_dst, p.gpu_src, p.nbytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
      LOG(FATAL) << "output allocator: D2H copy of " << p.nbytes
                 << " bytes failed: " << hipGetErrorString(e);
    }
  }
#endif

  MY_LOG(2) << "compute_with_output_allocator completed";
}

void MlirCustomOp::Compute(const OrtApi *api, OrtKernelContext *context) const {
  MY_LOG(2) << "MlirCustomOp::Compute() called";

  // Tell the runtime that a new forward pass is starting so per-Compute()
  // caches (currently the GQA seqlens_k cache) are invalidated. No-op if
  // the per-model bitcode predates the begin_compute export -- but in that
  // case the cache must be disabled via HIPDNN_EP_GQA_CACHE_SEQLENS=0
  // (default-on), otherwise stale values would survive across forward
  // passes. The mismatch is detected at session creation and produces a
  // LOG(WARNING). The call is a single cached indirect dispatch, so
  // leaving it on the fast path costs ~1 ns.
  inference_state_->begin_compute();

  if (use_output_allocator_) {
    // Allocator mode owns its output staging (no pre-filled outputs span) and
    // its own post-compute host D2H; it does not share the classic perf path.
    compute_with_output_allocator(context);
    MY_LOG(2) << "Compute completed successfully (allocator mode)";
    return;
  }

  if (!perf_enabled()) {
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

#ifdef HIPDNN_EP_LINK_HIP_HOST
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
#endif // HIPDNN_EP_LINK_HIP_HOST
}

} // namespace mlir_compilation
