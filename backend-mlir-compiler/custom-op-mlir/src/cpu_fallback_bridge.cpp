/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Debug-only CPU fallback: model.dll calls back with host-staged tensors; we
// run a tiny ORT CPU `Session` over a **pre-serialized ONNX** Gather graph — see
// docs/design/debug-cpu-fallback-plan.md and custom-op-mlir/onnx/README.md.
//
// MorphiZen **MLIR** backend: `model_proto_serialize_as_string` is **not**
// ONNX protobuf (it serializes the internal MLIR model). Feeding those bytes
// to `Ort::Session` fails with "protobuf parsing failed". The EP therefore
// embeds a minimal valid ONNX built offline (`onnx/gather_cpu_fb_*.onnx`).
//
// Historical: `morphizen::OpInvoker` + stock `Ort::OpAttr` also broke
// (`Ort::OpAttr*` reinterpreted as `AttributeProto*`). Do not revive that path.
//
// This path must never abort the process: use soft errors + exceptions.
//===----------------------------------------------------------------------===//

#include "hipdnn_ep_runtime.h"
#include "morphizen/onnxruntime_api.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ONNX bytes embedded at build time (see CMake + scripts/embed_binary_c_array.py).
// Included as .inc in this TU so we never rely on a second object file reaching
// morphizen-custom-op-mlir.lib (MSVC incremental / generator quirks caused LNK2019).
#include "gather_cpu_fb_onnx_data.inc"

namespace mlir_compilation::customop {

struct GatherCacheKey {
  int64_t axis = 0;
  int64_t data_hip_dtype = -1;
  int64_t indices_hip_dtype = -1;

  bool operator==(const GatherCacheKey &o) const {
    return axis == o.axis && data_hip_dtype == o.data_hip_dtype &&
           indices_hip_dtype == o.indices_hip_dtype;
  }
};

struct GatherCacheKeyHash {
  size_t operator()(const GatherCacheKey &k) const noexcept {
    return static_cast<size_t>(k.axis) ^
           (static_cast<size_t>(k.data_hip_dtype) * 1315423911u) ^
           (static_cast<size_t>(k.indices_hip_dtype) * 2654435761u);
  }
};

/// One ORT CPU session per (axis, data dtype, indices dtype). Session bytes
/// come from embedded ONNX (ORT-parseable), not MorphiZen MLIR serialization.
struct GatherOnnxCpuSession {
  Ort::Session session_{nullptr};

  static std::unique_ptr<GatherOnnxCpuSession> try_create(const GatherCacheKey &k);

  void invoke(Ort::Value *input_values, size_t input_count, Ort::Value *output_values,
              size_t output_count);
};

static std::mutex g_gather_fb_mutex;
static std::unordered_map<GatherCacheKey, std::unique_ptr<GatherOnnxCpuSession>,
                          GatherCacheKeyHash>
    g_gather_cpu_sessions;

static int64_t product_dims(const int64_t *shape, int64_t rank) {
  int64_t p = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] < 0)
      return -1;
    if (shape[i] > 0 &&
        p > (std::numeric_limits<int64_t>::max() / shape[i])) {
      return -1;
    }
    p *= shape[i];
  }
  return p;
}

/// ONNX Gather output rank = `indices_rank + (data_rank - 1)`; element count
/// matches the MLIR memref product even when MLIR uses a lower-rank `tensor`
/// with the same linear layout (e.g. `[1,4096]` vs `[1,1,4096]`). ORT needs the
/// canonical rank for `CreateTensor`; see invoke_gather_cpu.
static bool gather_onnx_output_shape(const HipdnnCpuFbGatherDesc *d,
                                     std::vector<int64_t> &out_shape) {
  out_shape.clear();
  if (!d->data_shape || d->data_rank <= 0 || d->indices_rank < 0 ||
      d->axis < 0 || d->axis >= d->data_rank)
    return false;
  if (d->indices_rank > 0 && !d->indices_shape)
    return false;
  for (int64_t i = 0; i < d->indices_rank; ++i)
    out_shape.push_back(d->indices_shape[i]);
  for (int64_t i = d->axis + 1; i < d->data_rank; ++i)
    out_shape.push_back(d->data_shape[i]);
  const int64_t po =
      product_dims(out_shape.data(), static_cast<int64_t>(out_shape.size()));
  return po >= 0 && po == d->output_num_elements;
}

static bool gather_shapes_match_elements(const HipdnnCpuFbGatherDesc *d) {
  if (!d->data_shape && d->data_rank > 0)
    return false;
  if (!d->indices_shape && d->indices_rank > 0)
    return false;
  if (!d->output_shape && d->output_rank > 0)
    return false;
  const int64_t pd = product_dims(d->data_shape, d->data_rank);
  const int64_t pi = product_dims(d->indices_shape, d->indices_rank);
  const int64_t po = product_dims(d->output_shape, d->output_rank);
  if (pd < 0 || pi < 0 || po < 0)
    return false;
  return pd == d->data_num_elements && pi == d->indices_num_elements &&
         po == d->output_num_elements;
}

static ONNXTensorElementDataType hip_dtype_to_onnx(int64_t hip_ty) {
  switch (hip_ty) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case HIPDNN_EP_DATATYPE_HALF:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case HIPDNN_EP_DATATYPE_INT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  default:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

static std::optional<Ort::Value>
make_cpu_tensor(const Ort::MemoryInfo &mi, void *base, const int64_t *shape,
                int64_t rank, ONNXTensorElementDataType et,
                int64_t expected_num_elements) {
  const int64_t nelem = product_dims(shape, rank);
  if (nelem < 0) {
    LOG(ERROR) << "cpu_fallback Gather: invalid shape (negative dim or "
                  "overflow in product)";
    return std::nullopt;
  }
  if (nelem != expected_num_elements) {
    LOG(ERROR) << "cpu_fallback Gather: shape product " << nelem
               << " != expected_num_elements " << expected_num_elements;
    return std::nullopt;
  }
  const size_t n = static_cast<size_t>(nelem);
  const int64_t *sh = shape;
  const size_t r = static_cast<size_t>(rank);
  switch (et) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return Ort::Value::CreateTensor<float>(
        mi, reinterpret_cast<float *>(base), n, sh, r);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return Ort::Value::CreateTensor<Ort::Float16_t>(
        mi, reinterpret_cast<Ort::Float16_t *>(base), n, sh, r);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return Ort::Value::CreateTensor<Ort::BFloat16_t>(
        mi, reinterpret_cast<Ort::BFloat16_t *>(base), n, sh, r);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return Ort::Value::CreateTensor<int32_t>(
        mi, reinterpret_cast<int32_t *>(base), n, sh, r);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return Ort::Value::CreateTensor<int64_t>(
        mi, reinterpret_cast<int64_t *>(base), n, sh, r);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return Ort::Value::CreateTensor<int8_t>(
        mi, reinterpret_cast<int8_t *>(base), n, sh, r);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return Ort::Value::CreateTensor<uint8_t>(
        mi, reinterpret_cast<uint8_t *>(base), n, sh, r);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return Ort::Value::CreateTensor<double>(
        mi, reinterpret_cast<double *>(base), n, sh, r);
  default:
    LOG(ERROR) << "cpu_fallback Gather: unsupported ONNX element type "
               << static_cast<int>(et);
    return std::nullopt;
  }
}

std::unique_ptr<GatherOnnxCpuSession>
GatherOnnxCpuSession::try_create(const GatherCacheKey &k) {
  // Bundled asset: opset-13 Gather, axis=0, fp16 data + int64 indices in ONNX.
  // Runtime may carry int32 indices (`indices_element_size_bytes==4`); we widen
  // to int64 before `CreateTensor` (same ONNX). Extend onnx/ + embed for more keys.
  if (k.axis != 0) {
    LOG(ERROR) << "cpu_fallback Gather: embedded ONNX is axis=0 only; got axis="
               << k.axis;
    return nullptr;
  }
  if (k.data_hip_dtype != HIPDNN_EP_DATATYPE_HALF) {
    LOG(ERROR) << "cpu_fallback Gather: embedded ONNX is fp16 data only; got "
                    "data_hip_dtype="
               << k.data_hip_dtype;
    return nullptr;
  }
  if (k.indices_hip_dtype != HIPDNN_EP_DATATYPE_INT64 &&
      k.indices_hip_dtype != HIPDNN_EP_DATATYPE_INT32) {
    LOG(ERROR) << "cpu_fallback Gather: embedded ONNX expects int32/int64 "
                    "indices; got indices_hip_dtype="
               << k.indices_hip_dtype;
    return nullptr;
  }

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hipdnn_cpu_fb_gather");
    Ort::SessionOptions options{};
    // Nested `Session::Run` from inside MorphiZen EP + tiny graphs: keep the
    // inner session predictable (external `Ort::Value` views + aggressive
    // optimizers have produced bogus Gather diagnostics in the field).
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    Ort::Session session(
        env, reinterpret_cast<const char *>(kGatherCpuFbOnnx),
        kGatherCpuFbOnnx_len, options);
    auto out = std::unique_ptr<GatherOnnxCpuSession>(new GatherOnnxCpuSession);
    out->session_ = std::move(session);
    return out;
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "cpu_fallback Gather: ORT Session from embedded ONNX failed: "
                 << e.what();
    return nullptr;
  }
}

void GatherOnnxCpuSession::invoke(Ort::Value *input_values, size_t input_count,
                                  Ort::Value *output_values,
                                  size_t output_count) {
  if (input_count != 2 || output_count != 1) {
    LOG(ERROR) << "cpu_fallback Gather: expected 2 inputs and 1 output, got "
                 << input_count << " in / " << output_count << " out";
    throw std::runtime_error("GatherOnnxCpuSession: bad IO arity");
  }

  Ort::RunOptions run_options{nullptr};
  const char *input_names[] = {"Input_0", "Input_1"};
  const char *output_names[] = {"Output_0"};
  // Prefer `Session::Run` with caller-owned `Ort::Value` outputs over `IoBinding`:
  // some ORT builds still mis-handle external-memory tensors in IoBinding for
  // small graphs (symptoms match binding the wrong logical input even when
  // names are spelled correctly — bogus Gather index diagnostics).
  session_.Run(run_options, input_names, input_values, input_count, output_names,
               output_values, output_count);
}

static GatherOnnxCpuSession *acquire_gather_cpu_session(const GatherCacheKey &k) {
  std::lock_guard<std::mutex> lock(g_gather_fb_mutex);
  auto it = g_gather_cpu_sessions.find(k);
  if (it != g_gather_cpu_sessions.end())
    return it->second.get();

  std::unique_ptr<GatherOnnxCpuSession> sess = GatherOnnxCpuSession::try_create(k);
  if (!sess)
    return nullptr;
  GatherOnnxCpuSession *raw = sess.get();
  g_gather_cpu_sessions.emplace(k, std::move(sess));
  return raw;
}

static int invoke_gather_cpu(const HipdnnCpuFbGatherDesc *d) {
  try {
    if (!gather_shapes_match_elements(d)) {
      LOG(ERROR) << "cpu_fallback Gather: shape metadata does not match element "
                    "counts (stale model.dll vs EP, or bad IR — rebuild cached "
                    "DLL or check lowering)";
      return -1;
    }

    ONNXTensorElementDataType data_et = hip_dtype_to_onnx(d->data_hip_dtype);
    ONNXTensorElementDataType indices_et =
        hip_dtype_to_onnx(d->indices_hip_dtype);
    if (data_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED ||
        indices_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      LOG(ERROR) << "cpu_fallback Gather: unsupported hip dtype pair (data="
                 << d->data_hip_dtype << ", indices=" << d->indices_hip_dtype
                 << ")";
      return -1;
    }

    const int64_t idx_bytes = d->indices_element_size_bytes;
    if (idx_bytes != 4 && idx_bytes != 8) {
      LOG(ERROR) << "cpu_fallback Gather: bad indices_element_size_bytes="
                 << idx_bytes << " (expected 4 or 8)";
      return -1;
    }
    if (idx_bytes == 4 &&
        d->indices_hip_dtype != HIPDNN_EP_DATATYPE_INT32) {
      LOG(ERROR) << "cpu_fallback Gather: indices_element_size_bytes=4 but "
                    "indices_hip_dtype="
                 << d->indices_hip_dtype;
      return -1;
    }
    if (idx_bytes == 8 &&
        d->indices_hip_dtype != HIPDNN_EP_DATATYPE_INT64) {
      LOG(ERROR) << "cpu_fallback Gather: indices_element_size_bytes=8 but "
                    "indices_hip_dtype="
                 << d->indices_hip_dtype;
      return -1;
    }

    GatherCacheKey key{d->axis, d->data_hip_dtype, d->indices_hip_dtype};
    GatherOnnxCpuSession *sess = acquire_gather_cpu_session(key);
    if (!sess)
      return -1;

    // Copy shapes off the model.dll stack — `CreateTensor` stores the shape
    // pointer internally; keep storage owned by this frame until Run returns.
    std::vector<int64_t> data_shape_vec;
    data_shape_vec.reserve(static_cast<size_t>(std::max<int64_t>(0, d->data_rank)));
    for (int64_t i = 0; i < d->data_rank; ++i)
      data_shape_vec.push_back(d->data_shape[i]);

    std::vector<int64_t> indices_shape_vec;
    indices_shape_vec.reserve(
        static_cast<size_t>(std::max<int64_t>(0, d->indices_rank)));
    for (int64_t i = 0; i < d->indices_rank; ++i)
      indices_shape_vec.push_back(d->indices_shape[i]);

    // Indices are tiny — allocate with ORT's default CPU allocator so the inner
    // `Session::Run` never holds a view into our D2H staging buffer (nested EP
    // runs have reproduced bogus Gather index reads with user-supplied tensor
    // memory for int64 indices even after IoBinding fixes).
    Ort::Value in_indices = Ort::Value::CreateTensor(
        Ort::AllocatorWithDefaultOptions(), indices_shape_vec.data(),
        indices_shape_vec.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    int64_t *idx_mut = in_indices.GetTensorMutableData<int64_t>();
    if (idx_bytes == 8) {
      memcpy(idx_mut, d->indices_host,
             static_cast<size_t>(d->indices_num_elements) * sizeof(int64_t));
    } else {
      const auto *src = static_cast<const int32_t *>(d->indices_host);
      for (int64_t i = 0; i < d->indices_num_elements; ++i)
        idx_mut[i] = static_cast<int64_t>(src[i]);
    }

    Ort::MemoryInfo mi =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto in_data = make_cpu_tensor(mi, const_cast<void *>(d->data_host),
                                   data_shape_vec.data(),
                                   static_cast<int64_t>(data_shape_vec.size()),
                                   data_et, d->data_num_elements);

    std::vector<int64_t> onnx_out_shape;
    const int64_t *out_shape = d->output_shape;
    int64_t out_rank = d->output_rank;
    if (gather_onnx_output_shape(d, onnx_out_shape)) {
      out_shape = onnx_out_shape.data();
      out_rank = static_cast<int64_t>(onnx_out_shape.size());
    }

    std::vector<int64_t> out_shape_vec;
    out_shape_vec.reserve(static_cast<size_t>(std::max<int64_t>(0, out_rank)));
    for (int64_t i = 0; i < out_rank; ++i)
      out_shape_vec.push_back(out_shape[i]);

    auto out_tensor =
        make_cpu_tensor(mi, d->output_host, out_shape_vec.data(), out_rank,
                        data_et, d->output_num_elements);
    if (!in_data || !out_tensor) {
      return -1;
    }

    Ort::Value inputs[2] = {std::move(*in_data), std::move(in_indices)};
    Ort::Value outputs[1] = {std::move(*out_tensor)};
    sess->invoke(inputs, 2, outputs, 1);
    return 0;
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "cpu_fallback Gather: ORT exception: " << e.what();
    return -1;
  } catch (const std::exception &e) {
    LOG(ERROR) << "cpu_fallback Gather: " << e.what();
    return -1;
  } catch (...) {
    LOG(ERROR) << "cpu_fallback Gather: unknown exception";
    return -1;
  }
}

static int morphizen_cpu_fallback_invoke(void * /*user*/, RuntimeState * /*state*/,
                                         int32_t op_kind, const void *detail,
                                         size_t detail_size) {
  if (op_kind != HIPDNN_CPU_FB_OP_GATHER) {
    LOG(ERROR) << "cpu_fallback: unknown op_kind " << op_kind;
    return -1;
  }
  if (!detail || detail_size != sizeof(HipdnnCpuFbGatherDesc)) {
    LOG(ERROR) << "cpu_fallback Gather: bad descriptor (size=" << detail_size
               << ", expected=" << sizeof(HipdnnCpuFbGatherDesc)
               << ") — rebuild model.dll after EP/runtime struct change";
    return -1;
  }
  const auto *d = static_cast<const HipdnnCpuFbGatherDesc *>(detail);
  return invoke_gather_cpu(d);
}

static hipdnn_cpu_fallback_iface_t g_iface{nullptr, morphizen_cpu_fallback_invoke};

const hipdnn_cpu_fallback_iface_t *morphizen_ep_cpu_fallback_iface() {
  return &g_iface;
}

} // namespace mlir_compilation::customop
