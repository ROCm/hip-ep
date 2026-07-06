/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Debug-only CPU fallback: model.dll calls back with host-staged tensors; the EP
// runs ORT CPU kernels via CPUGate + Ort::Op::Invoke on the gate thread.
// See docs/design/debug-cpu-fallback-plan.md.
//
// Do not use MorphiZen `model_proto_serialize_as_string` or `OpInvoker` here —
// MLIR serialization is not ONNX protobuf; OpInvoker mis-handles `Ort::OpAttr`.
//
// This path must never abort the process: use soft errors + exceptions.
//===----------------------------------------------------------------------===//

#include "cpu_fallback_cpugate.h"
#include "cpu_fallback_invoke.h"
#include "hipdnn_ep_runtime.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
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

static int64_t gather_axis_attr(const HipdnnCpuFbGenericHostDesc *d) {
  for (int32_t i = 0; i < d->num_attrs; ++i) {
    if (d->attrs[i].name && std::strcmp(d->attrs[i].name, "axis") == 0 &&
        d->attrs[i].kind == HIPDNN_CPU_FB_ATTR_INT) {
      return d->attrs[i].i;
    }
  }
  return -1;
}

// ONNX Gather output rank = indices_rank + (data_rank - 1); may differ from the
// MLIR memref rank when layouts are linearly equivalent.
static bool gather_onnx_output_shape(const HipdnnCpuFbGenericHostDesc *d,
                                     int64_t axis,
                                     std::vector<int64_t> &out_shape) {
  out_shape.clear();
  if (d->num_inputs < 2 || d->num_outputs < 1)
    return false;
  const auto &data = d->inputs[0];
  const auto &indices = d->inputs[1];
  const auto &output = d->outputs[0];
  if (!data.shape || data.rank <= 0 || indices.rank < 0 || axis < 0 ||
      axis >= data.rank)
    return false;
  if (indices.rank > 0 && !indices.shape)
    return false;
  for (int64_t i = 0; i < indices.rank; ++i)
    out_shape.push_back(indices.shape[i]);
  for (int64_t i = axis + 1; i < data.rank; ++i)
    out_shape.push_back(data.shape[i]);
  const int64_t po =
      product_dims(out_shape.data(), static_cast<int64_t>(out_shape.size()));
  return po >= 0 && po == output.num_elements;
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
make_cpu_tensor(const char *op_name, const Ort::MemoryInfo &mi, void *base,
                const int64_t *shape, int64_t rank,
                ONNXTensorElementDataType et, int64_t expected_num_elements) {
  const int64_t nelem = product_dims(shape, rank);
  if (nelem < 0) {
    LOG(ERROR) << "cpu_fallback " << (op_name ? op_name : "?")
               << ": invalid shape (negative dim or overflow in product)";
    return std::nullopt;
  }
  if (nelem != expected_num_elements) {
    LOG(ERROR) << "cpu_fallback " << (op_name ? op_name : "?")
               << ": shape product " << nelem
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
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return Ort::Value::CreateTensor<bool>(
        mi, reinterpret_cast<bool *>(base), n, sh, r);
  default:
    LOG(ERROR) << "cpu_fallback " << (op_name ? op_name : "?")
               << ": unsupported ONNX element type " << static_cast<int>(et);
    return std::nullopt;
  }
}

static ONNXTensorElementDataType onnx_type_for_io(const char *op_name,
                                                  int32_t io_index,
                                                  int32_t num_inputs,
                                                  int64_t hip_dtype) {
  if (op_name && std::strcmp(op_name, "Gather") == 0 && io_index == 1) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  if (op_name && io_index >= num_inputs) {
    if (std::strcmp(op_name, "Equal") == 0 || std::strcmp(op_name, "Less") == 0) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
  }
  if (op_name && std::strcmp(op_name, "Where") == 0 && io_index == 0) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  }
  return hip_dtype_to_onnx(hip_dtype);
}

static int invoke_generic_cpu(const HipdnnCpuFbGenericHostDesc *d) {
  if (!d || !d->op_name) {
    return -1;
  }
  cpugate::Manager &gate = cpugate::Manager::instance();

  const Ort::MemoryInfo mi =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const bool is_gather = std::strcmp(d->op_name, "Gather") == 0;
  const int64_t gather_axis = is_gather ? gather_axis_attr(d) : -1;

  std::vector<Ort::Value> input_values;
  std::vector<Ort::Value> output_values;
  std::vector<ONNXTensorElementDataType> input_ort_types;
  std::vector<int64_t> indices_shape_storage;
  input_values.reserve(static_cast<size_t>(d->num_inputs));
  input_ort_types.reserve(static_cast<size_t>(d->num_inputs));

  for (int32_t i = 0; i < d->num_inputs; ++i) {
    const auto &t = d->inputs[i];
    const ONNXTensorElementDataType et =
        onnx_type_for_io(d->op_name, i, d->num_inputs, t.hip_dtype);
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      LOG(ERROR) << "cpu_fallback " << d->op_name << ": unsupported input dtype";
      return -1;
    }
    input_ort_types.push_back(et);

    if (is_gather && i == 1 &&
        t.hip_dtype == HIPDNN_EP_DATATYPE_INT32) {
      indices_shape_storage.assign(t.shape, t.shape + t.rank);
      Ort::Value in_indices = Ort::Value::CreateTensor(
          Ort::AllocatorWithDefaultOptions(), indices_shape_storage.data(),
          indices_shape_storage.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      int64_t *idx_mut = in_indices.GetTensorMutableData<int64_t>();
      const auto *src = static_cast<const int32_t *>(t.host);
      for (int64_t j = 0; j < t.num_elements; ++j) {
        idx_mut[j] = static_cast<int64_t>(src[j]);
      }
      input_values.push_back(std::move(in_indices));
      continue;
    }

    auto v = make_cpu_tensor(d->op_name, mi, const_cast<void *>(t.host), t.shape,
                             t.rank, et, t.num_elements);
    if (!v) {
      return -1;
    }
    input_values.push_back(std::move(*v));
  }

  std::vector<int64_t> gather_out_shape;
  for (int32_t i = 0; i < d->num_outputs; ++i) {
    const auto &t = d->outputs[i];
    const ONNXTensorElementDataType et = onnx_type_for_io(
        d->op_name, d->num_inputs + i, d->num_inputs, t.hip_dtype);
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      LOG(ERROR) << "cpu_fallback " << d->op_name
                 << ": unsupported output dtype";
      return -1;
    }
    const int64_t *out_shape = t.shape;
    int64_t out_rank = t.rank;
    if (is_gather && gather_axis >= 0 &&
        gather_onnx_output_shape(d, gather_axis, gather_out_shape)) {
      out_shape = gather_out_shape.data();
      out_rank = static_cast<int64_t>(gather_out_shape.size());
    }
    auto v = make_cpu_tensor(d->op_name, mi, t.host_mut, out_shape, out_rank, et,
                             t.num_elements);
    if (!v) {
      return -1;
    }
    output_values.push_back(std::move(*v));
  }

  std::vector<Ort::OpAttr> owned_attrs;
  owned_attrs.reserve(static_cast<size_t>(d->num_attrs));
  for (int32_t i = 0; i < d->num_attrs; ++i) {
    const auto &a = d->attrs[i];
    if (!a.name) {
      continue;
    }
    switch (a.kind) {
    case HIPDNN_CPU_FB_ATTR_INT:
      owned_attrs.emplace_back(a.name, &a.i, 1, OrtOpAttrType::ORT_OP_ATTR_INT);
      break;
    case HIPDNN_CPU_FB_ATTR_INTS:
      owned_attrs.emplace_back(a.name, a.ints,
                               static_cast<size_t>(a.ints_len),
                               OrtOpAttrType::ORT_OP_ATTR_INTS);
      break;
    case HIPDNN_CPU_FB_ATTR_FLOAT:
      owned_attrs.emplace_back(a.name, &a.f, 1, OrtOpAttrType::ORT_OP_ATTR_FLOAT);
      break;
    default:
      LOG(ERROR) << "cpu_fallback " << d->op_name << ": bad attr kind";
      return -1;
    }
  }

  const Ort::OpAttr *attrs_ptr =
      owned_attrs.empty() ? nullptr : owned_attrs.data();

  Ort::Op *op = gate.get_or_create_onnx_op(
      d->op_name, d->domain, d->opset, input_ort_types.data(),
      input_ort_types.size(), static_cast<size_t>(d->num_inputs),
      static_cast<size_t>(d->num_outputs), attrs_ptr, owned_attrs.size());
  if (!op) {
    return -1;
  }

  std::vector<const OrtValue *> in_ptrs;
  std::vector<OrtValue *> out_ptrs;
  in_ptrs.reserve(input_values.size());
  out_ptrs.reserve(output_values.size());
  for (auto &v : input_values) {
    in_ptrs.push_back(v);
  }
  for (auto &v : output_values) {
    out_ptrs.push_back(v);
  }

  try {
    gate.invoke(*op, in_ptrs.data(), in_ptrs.size(), out_ptrs.data(),
                out_ptrs.size());
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "cpu_fallback " << d->op_name << ": ORT exception: " << e.what();
    return -1;
  } catch (const std::exception &e) {
    LOG(ERROR) << "cpu_fallback " << d->op_name << ": " << e.what();
    return -1;
  }
  return 0;
}

static int morphizen_cpu_fallback_invoke(void * /*user*/, RuntimeState * /*state*/,
                                         int32_t op_kind, const void *detail,
                                         size_t detail_size) {
  if (op_kind != HIPDNN_CPU_FB_OP_GENERIC) {
    LOG(ERROR) << "cpu_fallback: unknown op_kind " << op_kind;
    return -1;
  }
  if (!detail || detail_size != sizeof(HipdnnCpuFbGenericHostDesc)) {
    LOG(ERROR) << "cpu_fallback: bad descriptor (size=" << detail_size
               << ", expected=" << sizeof(HipdnnCpuFbGenericHostDesc) << ")";
    return -1;
  }
  return invoke_generic_cpu(static_cast<const HipdnnCpuFbGenericHostDesc *>(detail));
}

static hipdnn_cpu_fallback_iface_t g_iface{nullptr, morphizen_cpu_fallback_invoke};

const hipdnn_cpu_fallback_iface_t *morphizen_ep_cpu_fallback_iface() {
  return &g_iface;
}

} // namespace mlir_compilation::customop
