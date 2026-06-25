/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Debug-only CPU fallback: model.dll calls back with host-staged tensors; the EP
// runs a host reference Gather (large / fp16 tensors) or CPUGate + Ort::Op for
// small exotic cases. See docs/design/debug-cpu-fallback-plan.md.
//
// Do not use MorphiZen `model_proto_serialize_as_string` or `OpInvoker` here —
// MLIR serialization is not ONNX protobuf; OpInvoker mis-handles `Ort::OpAttr`.
//
// This path must never abort the process: use soft errors + exceptions.
//===----------------------------------------------------------------------===//

#include "cpu_fallback_cpugate.h"
#include "hipdnn_ep_runtime.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <vector>

namespace mlir_compilation::customop {

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

static int64_t hip_dtype_element_size_bytes(int64_t hip_ty) {
  switch (hip_ty) {
  case HIPDNN_EP_DATATYPE_HALF:
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return 2;
  case HIPDNN_EP_DATATYPE_FLOAT:
  case HIPDNN_EP_DATATYPE_INT32:
    return 4;
  case HIPDNN_EP_DATATYPE_INT64:
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return 8;
  case HIPDNN_EP_DATATYPE_INT8:
  case HIPDNN_EP_DATATYPE_UINT8:
    return 1;
  default:
    return -1;
  }
}

static int64_t read_gather_index(const void *indices_host, int64_t i,
                                 int64_t idx_bytes) {
  if (idx_bytes == 8) {
    return static_cast<const int64_t *>(indices_host)[i];
  }
  return static_cast<int64_t>(
      static_cast<const int32_t *>(indices_host)[i]);
}

// Host reference for ONNX Gather on dense row-major buffers. Matches the GPU
// gather_kernel layout ([outer, axis_size, inner]) and index rules (negative
// index += axis_size; OOB -> zero).
static int reference_gather_on_host(const HipdnnCpuFbGatherDesc *d) {
  if (!d->data_host || !d->indices_host || !d->output_host || !d->data_shape ||
      d->data_rank <= 0) {
    return -1;
  }
  if (d->axis < 0 || d->axis >= d->data_rank) {
    return -1;
  }

  const int64_t elem_size = hip_dtype_element_size_bytes(d->data_hip_dtype);
  if (elem_size <= 0) {
    return -1;
  }
  if (d->indices_element_size_bytes != 4 &&
      d->indices_element_size_bytes != 8) {
    return -1;
  }

  int64_t inner = 1;
  for (int64_t i = d->axis + 1; i < d->data_rank; ++i) {
    inner *= d->data_shape[i];
  }
  int64_t outer = 1;
  for (int64_t i = 0; i < d->axis; ++i) {
    outer *= d->data_shape[i];
  }
  const int64_t axis_dim = d->data_shape[d->axis];
  const int64_t indices_num = d->indices_num_elements;
  if (inner <= 0 || axis_dim <= 0 || indices_num < 0) {
    return -1;
  }
  if (d->output_num_elements != outer * indices_num * inner) {
    LOG(ERROR) << "cpu_fallback Gather: reference layout mismatch "
               << "output_num=" << d->output_num_elements << " expected "
               << (outer * indices_num * inner);
    return -1;
  }

  const auto *data = static_cast<const char *>(d->data_host);
  auto *out = static_cast<char *>(d->output_host);
  const size_t slice_bytes = static_cast<size_t>(inner * elem_size);

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < indices_num; ++i) {
      int64_t ix =
          read_gather_index(d->indices_host, i, d->indices_element_size_bytes);
      if (ix < 0) {
        ix += axis_dim;
      }
      char *out_slice =
          out + static_cast<size_t>((o * indices_num + i) * inner * elem_size);
      if (ix < 0 || ix >= axis_dim) {
        std::memset(out_slice, 0, slice_bytes);
        continue;
      }
      const char *data_slice = data + static_cast<size_t>(
                                           (o * axis_dim + ix) * inner *
                                           elem_size);
      std::memcpy(out_slice, data_slice, slice_bytes);
    }
  }
  return 0;
}

// CPUGate path only for small tensors (avoids fp16->fp32 full-table promotion).
static constexpr int64_t kCpugateGatherMaxDataBytes = 64 * 1024 * 1024;

static bool gather_needs_fp32_promotion(ONNXTensorElementDataType et) {
  return et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         et == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

static ONNXTensorElementDataType gather_ort_kernel_dtype(
    ONNXTensorElementDataType storage_et) {
  if (gather_needs_fp32_promotion(storage_et)) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  return storage_et;
}

static int invoke_gather_via_cpugate(const HipdnnCpuFbGatherDesc *d,
                                    ONNXTensorElementDataType data_et,
                                    ONNXTensorElementDataType ort_kernel_et,
                                    bool promote_fp32) {
  const int64_t idx_bytes = d->indices_element_size_bytes;

  cpugate::Manager &gate = cpugate::Manager::instance();
  Ort::Op *gather_op = gate.get_or_create_gather_op(
      d->axis, ort_kernel_et, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  if (!gather_op) {
    return -1;
  }

  std::vector<float> data_fp32;
  std::vector<float> out_fp32;
  void *data_ptr = const_cast<void *>(d->data_host);
  void *out_ptr = d->output_host;
  if (promote_fp32) {
    try {
      data_fp32.resize(static_cast<size_t>(d->data_num_elements));
      out_fp32.resize(static_cast<size_t>(d->output_num_elements));
    } catch (const std::bad_alloc &) {
      LOG(ERROR) << "cpu_fallback Gather: fp32 promotion staging OOM";
      return -1;
    }
    if (data_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      const auto *src = static_cast<const Ort::Float16_t *>(d->data_host);
      for (size_t i = 0; i < data_fp32.size(); ++i) {
        data_fp32[i] = src[i].ToFloat();
      }
    } else {
      const auto *src = static_cast<const Ort::BFloat16_t *>(d->data_host);
      for (size_t i = 0; i < data_fp32.size(); ++i) {
        data_fp32[i] = src[i].ToFloat();
      }
    }
    data_ptr = data_fp32.data();
    out_ptr = out_fp32.data();
  }

  std::vector<int64_t> data_shape_vec;
  data_shape_vec.reserve(
      static_cast<size_t>(std::max<int64_t>(0, d->data_rank)));
  for (int64_t i = 0; i < d->data_rank; ++i) {
    data_shape_vec.push_back(d->data_shape[i]);
  }

  std::vector<int64_t> indices_shape_vec;
  indices_shape_vec.reserve(
      static_cast<size_t>(std::max<int64_t>(0, d->indices_rank)));
  for (int64_t i = 0; i < d->indices_rank; ++i) {
    indices_shape_vec.push_back(d->indices_shape[i]);
  }

  Ort::Value in_indices = Ort::Value::CreateTensor(
      Ort::AllocatorWithDefaultOptions(), indices_shape_vec.data(),
      indices_shape_vec.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  int64_t *idx_mut = in_indices.GetTensorMutableData<int64_t>();
  if (idx_bytes == 8) {
    memcpy(idx_mut, d->indices_host,
           static_cast<size_t>(d->indices_num_elements) * sizeof(int64_t));
  } else {
    const auto *src = static_cast<const int32_t *>(d->indices_host);
    for (int64_t i = 0; i < d->indices_num_elements; ++i) {
      idx_mut[i] = static_cast<int64_t>(src[i]);
    }
  }

  Ort::MemoryInfo mi =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto in_data = make_cpu_tensor(mi, data_ptr, data_shape_vec.data(),
                                 static_cast<int64_t>(data_shape_vec.size()),
                                 ort_kernel_et, d->data_num_elements);

  std::vector<int64_t> onnx_out_shape;
  const int64_t *out_shape = d->output_shape;
  int64_t out_rank = d->output_rank;
  if (gather_onnx_output_shape(d, onnx_out_shape)) {
    out_shape = onnx_out_shape.data();
    out_rank = static_cast<int64_t>(onnx_out_shape.size());
  }

  std::vector<int64_t> out_shape_vec;
  out_shape_vec.reserve(static_cast<size_t>(std::max<int64_t>(0, out_rank)));
  for (int64_t i = 0; i < out_rank; ++i) {
    out_shape_vec.push_back(out_shape[i]);
  }

  auto out_tensor = make_cpu_tensor(mi, out_ptr, out_shape_vec.data(), out_rank,
                                    ort_kernel_et, d->output_num_elements);
  if (!in_data || !out_tensor) {
    return -1;
  }

  Ort::Value in_data_val = std::move(*in_data);
  Ort::Value out_val = std::move(*out_tensor);
  const OrtValue *inputs[] = {in_data_val, in_indices};
  OrtValue *outputs[] = {out_val};
  gate.invoke(*gather_op, inputs, 2, outputs, 1);

  if (promote_fp32) {
    if (data_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      auto *dst = static_cast<Ort::Float16_t *>(d->output_host);
      for (size_t i = 0; i < out_fp32.size(); ++i) {
        dst[i] = Ort::Float16_t(out_fp32[i]);
      }
    } else {
      auto *dst = static_cast<Ort::BFloat16_t *>(d->output_host);
      for (size_t i = 0; i < out_fp32.size(); ++i) {
        dst[i] = Ort::BFloat16_t(out_fp32[i]);
      }
    }
  }
  return 0;
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
    if (data_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      LOG(ERROR) << "cpu_fallback Gather: unsupported data_hip_dtype="
                 << d->data_hip_dtype;
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

    if (reference_gather_on_host(d) == 0) {
      return 0;
    }

    const int64_t elem_size = hip_dtype_element_size_bytes(d->data_hip_dtype);
    const int64_t data_bytes =
        elem_size > 0 ? d->data_num_elements * elem_size : -1;
    const ONNXTensorElementDataType ort_kernel_et =
        gather_ort_kernel_dtype(data_et);
    const bool promote_fp32 = ort_kernel_et != data_et;
    if (promote_fp32 && data_bytes > kCpugateGatherMaxDataBytes) {
      LOG(ERROR) << "cpu_fallback Gather: reference failed and CPUGate fp32 "
                    "promotion skipped for large tensor (data_bytes="
                 << data_bytes << ")";
      return -1;
    }

    return invoke_gather_via_cpugate(d, data_et, ort_kernel_et, promote_fp32);
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
  const auto *desc = static_cast<const HipdnnCpuFbGatherDesc *>(detail);
  return invoke_gather_cpu(desc);
}

static hipdnn_cpu_fallback_iface_t g_iface{nullptr, morphizen_cpu_fallback_invoke};

const hipdnn_cpu_fallback_iface_t *morphizen_ep_cpu_fallback_iface() {
  return &g_iface;
}

} // namespace mlir_compilation::customop
