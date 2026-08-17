/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "MlirCustomOp.h"

// HIP must precede LLVM headers on MSVC. LLVM's __has_attribute fallback
// otherwise makes HIP select its GNU vector implementation.
#ifdef HIPDNN_EP_LINK_HIP_HOST
#include <hip/hip_runtime_api.h>
#endif

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "custom-op-compute-status.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>

#include "llvm/ADT/ScopeExit.h"

#include "metadata.pb.h"

// Component headers
#include "InferenceState.h"
#include "artifact_metadata.h"
#include "hip/env.h" // shared cross-platform env reader (single Win32 call)

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
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

namespace {

// Parse metadata from JSON string in MetaDefProto. Failures throw so the
// CreateState C-ABI callback can translate them into an OrtStatus.
mlir_metadata::Metadata parse_metadata_from_metadef(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def) {

  auto metadata_json = context->get_meta_def_param(*meta_def);
  mlir_metadata::Metadata metadata;
  std::string error;
  if (!customop::parseAndValidateArtifactMetadata(metadata_json, metadata,
                                                  error))
    throw std::runtime_error(error);

  MY_LOG(1) << "Parsed metadata - Artifact filename: "
            << metadata.artifact_filename();
  return metadata;
}

// Load artifact bytes from EPContext file stream. Failures throw so session
// creation returns a recoverable ORT-visible status.
std::vector<uint8_t> load_artifact_from_epcontext(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::string &artifact_filename) {

  auto artifact_stream = context->open_file_for_read(artifact_filename);
  if (!artifact_stream)
    throw std::runtime_error("failed to open artifact from EPContext: " +
                             artifact_filename);

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

  if (artifact_bytes.empty())
    throw std::runtime_error("failed to read artifact bytes from EPContext: " +
                             artifact_filename);

  MY_LOG(1) << "Loaded artifact: " << total_read << " bytes";
  return artifact_bytes;
}

// Decide which loader the EP should use for this artifact. The format is
// recorded in the EPContext metadata by the compiler (artifact_format), which
// always sets it (see pass_main.cpp); an empty/unknown value means a malformed
// or pre-PR EPContext and is rejected rather than mis-loaded.
customop::ArtifactKind determine_artifact_kind(const std::string &format_str) {
  customop::ArtifactKind kind;
  if (!customop::artifactKindFromFormat(format_str, kind)) {
    throw std::runtime_error(
        "EPContext metadata has empty/unknown artifact_format='" + format_str +
        "'; expected '" + customop::kArtifactFormatLlvmIr + "' or '" +
        customop::kArtifactFormatNative + "'");
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
#ifndef HIPDNN_EP_LINK_HIP_HOST
  return false;
#else
  // Consistent with the runtime's hipdnn_ep_perf_enabled()
  // (lib/Runtime/debug_log.h) -- both now read env through the shared
  // hip/env.h helper. PERF is on for HIPDNN_EP_PERF OR a set
  // HIPDNN_EP_TRACE_FILE. The trace case is load-bearing: without it a
  // trace-only run leaves perf=false here, so the per-op flush below never
  // runs and the chrome trace gets no per-op spans even though the runtime's
  // OP_PROFILE scopes are collecting them.
  static const bool enabled =
      hipdnn_ep::env_enabled("HIPDNN_EP_PERF") ||
      !hipdnn_ep::env_string("HIPDNN_EP_TRACE_FILE").empty();
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

#ifdef HIPDNN_EP_LINK_HIP_HOST
// Per-Compute() timer for HIPDNN_EP_PERF, keeping all the event handling and
// PerfSample assembly in one place. It times the GPU work with a
// session-lifetime hipEvent pair (owned by MlirCustomOp, created lazily here
// and released in its destructor) and the host phases with steady_clock.
//
// Usage: construct it after Compute() begins, call record_start() just before
// the inference dispatch and record_stop() just after it returns, fill in the
// host-phase fields on a PerfSample, then call finish() -- which fences the
// device, reads the GPU elapsed time, fills in wall/gpu/fence_residual, and
// records the sample. Only construct it when perf is enabled; the disabled
// path must do no allocation and no timing.
class EpPerfTimer {
public:
  using clock = std::chrono::steady_clock;

  static double ms(clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }

  // start_slot / stop_slot are MlirCustomOp's mutable void* event handles.
  // They are created once (on the first perf-enabled Compute()) and reused;
  // t_enter is captured after creation so the one-time cost is excluded.
  EpPerfTimer(void *&start_slot, void *&stop_slot, hipStream_t stream)
      : stream_(stream) {
    if (!start_slot) {
      // hipEventDisableSystemFence: see op_profile.cpp
      // (op_profile_make_marker) for the canonical rationale and
      // guardrail. Safe here -- elapsed time is read only after the
      // hipDeviceSynchronize() in finish().
      hipEvent_t a = nullptr, b = nullptr;
      (void)hipEventCreateWithFlags(&a, hipEventDisableSystemFence);
      (void)hipEventCreateWithFlags(&b, hipEventDisableSystemFence);
      start_slot = static_cast<void *>(a);
      stop_slot = static_cast<void *>(b);
    }
    start_ = static_cast<hipEvent_t>(start_slot);
    stop_ = static_cast<hipEvent_t>(stop_slot);
    t_enter_ = clock::now();
  }

  clock::time_point t_enter() const { return t_enter_; }
  clock::time_point t_after_compute() const { return t_after_compute_; }

  // HIP error codes are intentionally discarded: a failure here only corrupts
  // diagnostic numbers, never the inference result (the (void) cast also
  // silences MSVC C4834 on the [[nodiscard]] return).
  void record_start() { (void)hipEventRecord(start_, stream_); }
  void record_stop() {
    t_after_compute_ = clock::now();
    (void)hipEventRecord(stop_, stream_);
  }

  // Closes the timing window: fences the device (the residual measures async
  // work still queued after the dispatch returned), reads GPU elapsed time,
  // fills the device-side fields on the caller-populated sample, and records
  // it. The caller must have set the marshal_*/compute_cpu fields first.
  void finish(PerfSample &s) {
    (void)hipDeviceSynchronize();
    const auto t_after_fence = clock::now();
    float gpu_ms = 0.0f;
    (void)hipEventSynchronize(stop_);
    (void)hipEventElapsedTime(&gpu_ms, start_, stop_);
    s.wall_ms = ms(t_enter_, t_after_fence);
    s.gpu_ms = static_cast<double>(gpu_ms);
    s.fence_residual_ms = ms(t_after_compute_, t_after_fence);
    perf_collector().record(s);
  }

private:
  hipStream_t stream_ = nullptr;
  hipEvent_t start_ = nullptr;
  hipEvent_t stop_ = nullptr;
  clock::time_point t_enter_;
  clock::time_point t_after_compute_;
};
#endif // HIPDNN_EP_LINK_HIP_HOST

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
// Output-allocator (2-arg inference_compute) plumbing.
//
// The DLL receives no pre-filled outputs span. Instead, main_graph calls
// hipdnn_ep_alloc_output(out_idx, shape, rank, elem_size) for each graph
// output once its runtime shape is known; the runtime forwards to
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
  OutputAllocatorCtx(
      Ort::KernelContext &ctx,
      const google::protobuf::RepeatedPtrField<mlir_metadata::Output> &outputs,
      const std::vector<int> &outputIndexMap,
      std::vector<HostOutputScratch> &hostOutputScratch)
      : ctx(ctx), outputs(outputs), output_index_map(outputIndexMap),
        host_out_scratch(hostOutputScratch), allocated(outputs.size(), false) {
    if (output_index_map.size() != static_cast<size_t>(outputs.size()))
      throw std::runtime_error(
          "output allocator: metadata/output index map size mismatch");
  }

  Ort::KernelContext &ctx;
  const google::protobuf::RepeatedPtrField<mlir_metadata::Output> &outputs;
  const std::vector<int> &output_index_map;
  // Session-owned grow-on-demand GPU scratch, one slot per host output.
  std::vector<HostOutputScratch> &host_out_scratch;
  std::vector<PendingD2H> pending_d2h; // filled during compute, consumed after
  // Output-completeness guard: which metadata outputs received an alloc call.
  std::vector<bool> allocated;
  morphizen::detail::DeferredComputeException callback_error;
};

static size_t checked_output_byte_size(const std::vector<int64_t> &shape,
                                       int64_t elem_size, int64_t out_idx) {
  if (elem_size <= 0)
    throw std::runtime_error(
        "output allocator: non-positive element size for output " +
        std::to_string(out_idx));

  size_t nbytes = static_cast<size_t>(elem_size);
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    int64_t dim = shape[axis];
    if (dim < 0)
      throw std::runtime_error("output allocator: negative extent at axis " +
                               std::to_string(axis) + " for output " +
                               std::to_string(out_idx));
    size_t extent = static_cast<size_t>(dim);
    if (extent != 0 && nbytes > std::numeric_limits<size_t>::max() / extent)
      throw std::overflow_error("output allocator: byte size overflow for "
                                "output " +
                                std::to_string(out_idx));
    nbytes *= extent;
  }
  return nbytes;
}

#ifdef HIPDNN_EP_LINK_HIP_HOST
// Grow one persistent scratch slot without discarding the usable old allocation
// until its replacement exists.
static void *ensure_host_out_slot(std::vector<HostOutputScratch> &scratch,
                                  size_t idx, size_t nbytes) {
  if (idx >= scratch.size())
    scratch.resize(idx + 1);
  HostOutputScratch &slot = scratch[idx];
  const size_t requiredBytes = std::max(nbytes, size_t{1});
  if (slot.ptr && slot.capacity >= requiredBytes)
    return slot.ptr;

  void *replacement = nullptr;
  hipError_t error = hipMalloc(&replacement, requiredBytes);
  llvm::scope_exit releaseReplacement([&] {
    if (replacement)
      (void)hipFree(replacement);
  });
  if (error != hipSuccess || !replacement)
    throw std::runtime_error(
        "output allocator: hipMalloc(" + std::to_string(requiredBytes) +
        ") for host scratch failed: " + hipGetErrorString(error));

  // Publish the known-valid replacement before retiring the old pointer. If
  // hipFree reports an error, its disposition is uncertain; retaining it for
  // reuse would be unsafe, so leave the replacement live and abandon the old
  // allocation rather than risk a stale-pointer reuse.
  void *retired = slot.ptr;
  slot.ptr = replacement;
  slot.capacity = requiredBytes;
  replacement = nullptr;
  if (retired) {
    error = hipFree(retired);
    if (error != hipSuccess)
      throw std::runtime_error(
          "output allocator: hipFree of replaced host scratch failed: " +
          std::string(hipGetErrorString(error)));
  }
  return slot.ptr;
}
#endif // HIPDNN_EP_LINK_HIP_HOST

// C callback installed on the RuntimeState (output_allocator_t.allocate).
// It cannot unwind through the model's C ABI. Preserve the first exception and
// rethrow it after inference_compute returns and runs generated cleanup.
void *output_allocate_cb(void *self, int64_t out_idx, const int64_t *shape,
                         int64_t rank, int64_t elem_size) noexcept {
  auto *octx = static_cast<OutputAllocatorCtx *>(self);
  if (!octx || octx->callback_error.hasValue())
    return nullptr;

  try {
    if (out_idx < 0 || out_idx >= static_cast<int64_t>(octx->outputs.size()))
      throw std::runtime_error("output allocator: out_idx " +
                               std::to_string(out_idx) + " out of range (" +
                               std::to_string(octx->outputs.size()) +
                               " outputs)");
    if (rank < 0 || (rank > 0 && !shape))
      throw std::runtime_error("output allocator: invalid shape for output " +
                               std::to_string(out_idx));
    // Use the DLL's in-graph shape verbatim -- the EP never reshapes an output
    // here. The shape is computed in-graph by hip.alloc_output from its
    // producer's operands; when a dynamic output dim is sized from a graph
    // input's `memref.dim`, the requested shape already equals that input
    // buffer's allocated extent. ORT's GetOutput then returns the pre-bound
    // (IO-bound) buffer rather than allocating a fresh one, so when an input
    // and output are bound to the same buffer that identity is preserved with
    // no EP-side override.
    std::vector<int64_t> out_shape;
    if (rank > 0)
      out_shape.assign(shape, shape + rank);
    size_t nbytes = checked_output_byte_size(out_shape, elem_size, out_idx);

    const size_t outputIndex = static_cast<size_t>(out_idx);
    int ort_idx = octx->output_index_map[outputIndex];
    if (ort_idx < 0)
      throw std::runtime_error("output allocator: negative ORT output index");
    auto out_tensor = octx->ctx.GetOutput(ort_idx, out_shape);
    int mem_type =
        static_cast<int>(out_tensor.GetTensorMemoryInfo().GetDeviceType());
    void *ort_ptr = out_tensor.GetTensorMutableRawData();

    if (!ort_ptr) {
      if (nbytes != 0)
        throw std::runtime_error(
            "output allocator: ORT returned null for non-empty output " +
            std::to_string(out_idx));
#ifdef HIPDNN_EP_LINK_HIP_HOST
      // Some allocators represent an empty OrtValue with null. Generated code
      // still requires a non-null descriptor even though it performs no writes.
      void *sentinel =
          ensure_host_out_slot(octx->host_out_scratch, outputIndex, 0);
      octx->allocated[outputIndex] = true;
      return sentinel;
#else
      static std::byte emptyOutputSentinel;
      octx->allocated[outputIndex] = true;
      return &emptyOutputSentinel;
#endif
    }

    if (mem_type == TENSOR_MEMORY_GPU) {
      // Zero-copy: ORT buffer is already GPU-accessible (e.g. the EP's
      // host-mapped allocator, or caller-provided device memory). The DLL
      // writes into it directly.
      octx->allocated[outputIndex] = true;
      return ort_ptr;
    }

    // Host (CPU) output: the DLL writes into EP GPU scratch; Compute()
    // D2H-copies into ort_ptr after inference_compute returns (the 2-arg
    // interface already stream-synced).
#ifdef HIPDNN_EP_LINK_HIP_HOST
    void *gpu =
        ensure_host_out_slot(octx->host_out_scratch, outputIndex, nbytes);
    octx->pending_d2h.push_back({gpu, ort_ptr, nbytes});
    octx->allocated[outputIndex] = true;
    return gpu;
#else
    // Mock runtime writes host memory directly; no GPU staging needed.
    (void)elem_size;
    octx->allocated[outputIndex] = true;
    return ort_ptr;
#endif
  } catch (...) {
    octx->callback_error.captureCurrent();
    return nullptr;
  }
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
  // Free the host-output GPU scratch (one buffer per host output index).
  // No-op when every output was GPU-aliased (the vector stays empty).
  for (const HostOutputScratch &slot : host_out_scratch_)
    if (slot.ptr)
      (void)hipFree(slot.ptr);

  // Release the lazy-allocated HIPDNN_EP_PERF event pair. Created on first
  // perf-enabled Compute() and reused for the session lifetime; nullptr here
  // means perf was never enabled, so nothing to release. Guarded by
  // HIPDNN_EP_LINK_HIP_HOST (not BUILD_MOCK_RUNTIME): hipEvent_t /
  // hipEventDestroy are only declared when the HIP headers are linked, and
  // ep_perf_ev_* are only ever assigned by EpPerfTimer, which is itself behind
  // this same macro -- so in builds without it these handles are always null
  // and need no cleanup.
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

// Output-allocator dispatch: 2-arg inference_compute with in-graph
// hip.alloc_output. The EP installs a per-Compute callback that hands each
// graph output an ORT (or GPU-scratch) buffer; host outputs are D2H-copied
// afterwards. See the OutputAllocatorCtx / output_allocate_cb block above.
void MlirCustomOp::compute_with_output_allocator(
    OrtKernelContext *context) const {
  MY_LOG(2) << "MlirCustomOp::compute_with_output_allocator()";

#ifdef HIPDNN_EP_LINK_HIP_HOST
  // This dispatch emits its own PerfSample (§4 per-call distribution) and
  // per-op flush (§3 GPU table) here, so under HIPDNN_EP_PERF=1 a session
  // produces EP-side [PERF]/[PERF SUMMARY] lines. There is no marshal-out
  // pre-pass (outputs are allocated in-graph via output_allocate_cb), so
  // marshal_out_ms is 0.
  const bool perf = perf_enabled();
  std::optional<EpPerfTimer> timer;
  EpPerfTimer::clock::time_point t_after_in{};
  if (perf)
    timer.emplace(ep_perf_ev_start_, ep_perf_ev_stop_,
                  static_cast<hipStream_t>(inference_state_->get_stream_raw()));
#endif

  auto inputs = marshal_input_tensors(context, input_index_map_);
#ifdef HIPDNN_EP_LINK_HIP_HOST
  if (perf)
    t_after_in = EpPerfTimer::clock::now();
#endif

  Ort::KernelContext ort_ctx(context);
  OutputAllocatorCtx octx(ort_ctx, metadata_.outputs(), output_index_map_,
                          host_out_scratch_);

  output_allocator_t alloc;
  alloc.self = &octx;
  alloc.allocate = &output_allocate_cb;
  int ret = 0;
  {
    inference_state_->set_output_allocator(&alloc);
    llvm::scope_exit clearAllocator(
        [&] { inference_state_->set_output_allocator(nullptr); });

#ifdef HIPDNN_EP_LINK_HIP_HOST
    if (perf)
      timer->record_start();
#endif

    ret = inference_state_->compute(&inputs.span);

#ifdef HIPDNN_EP_LINK_HIP_HOST
    if (perf)
      timer->record_stop();
#endif
  }

  // The callback cannot throw through C. Surface its precise failure only
  // after inference_compute has completed its generated cleanup.
  octx.callback_error.rethrow();

  if (ret != 0) {
    // Do not validate, copy, profile, or report successful outputs after a
    // generated/runtime failure. The stable void CustomOp ABI carries this to
    // the ORT callback as an exception, where it becomes a non-null OrtStatus.
    morphizen::detail::throwComputeFailure("inference_compute", ret);
  }

  // Output-completeness guard. Every declared output must be produced in-graph
  // (each yields a hip.alloc_output -> callback). A missing one means a
  // passthrough / aliased output the DLL never allocates -- unsupported today.
  // Fail loudly rather than hand ORT an uninitialized buffer (which can
  // silently pass a CPU-vs-CPU comparison).
  for (size_t i = 0; i < octx.allocated.size(); ++i) {
    if (!octx.allocated[i])
      throw std::runtime_error(
          "output allocator: output '" +
          metadata_.outputs()[static_cast<int>(i)].name() + "' (idx " +
          std::to_string(i) +
          ") was never allocated by the model artifact; passthrough and "
          "aliased outputs are not supported yet");
  }

#ifdef HIPDNN_EP_LINK_HIP_HOST
  // Deferred host-output copies. The 2-arg interface already stream-synced, so
  // the GPU scratch holds final data; a blocking hipMemcpy is safe.
  for (const auto &p : octx.pending_d2h) {
    if (p.nbytes == 0)
      continue;
    hipError_t e =
        hipMemcpy(p.host_dst, p.gpu_src, p.nbytes, hipMemcpyDeviceToHost);
    if (e != hipSuccess)
      throw std::runtime_error("output allocator: D2H copy of " +
                               std::to_string(p.nbytes) +
                               " bytes failed: " + hipGetErrorString(e));
  }

  if (perf) {
    // finish() fences after the post-compute D2H copies, so wall_ms /
    // fence_residual_ms capture host-visible completion.
    PerfSample s;
    s.marshal_in_ms = EpPerfTimer::ms(timer->t_enter(), t_after_in);
    s.marshal_out_ms = 0.0; // outputs are allocated in-graph, not marshaled
    s.compute_cpu_ms = EpPerfTimer::ms(t_after_in, timer->t_after_compute());
    timer->finish(s);
    const double outer_start_us = std::chrono::duration<double, std::micro>(
                                      timer->t_enter().time_since_epoch())
                                      .count();
    inference_state_->add_cpu_profile("[outer] Compute", outer_start_us,
                                      s.wall_ms);
    // Resolve the per-op table AFTER the timing window closes (see the
    // flush_op_profile contract -- keeps the resolve cost out of wall_ms).
    inference_state_->flush_op_profile();
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

  // Does everything: input marshaling, in-graph output allocation via the
  // callback, post-compute host D2H, and (when HIPDNN_EP_PERF is set) its own
  // timing + per-op profile flush.
  compute_with_output_allocator(context);
  MY_LOG(2) << "Compute completed successfully";
}

} // namespace mlir_compilation
