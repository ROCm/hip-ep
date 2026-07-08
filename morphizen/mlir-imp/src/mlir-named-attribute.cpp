/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir-named-attribute.hpp"
#include "mlir-constants.hpp"
#include "mlir-context-manager.hpp"
#include "mlir-node-arg.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <glog/logging.h>
#include <vector>

// `create_tensor` copies ONNX little-endian `raw_data` verbatim into a
// `DenseElementsAttr` via `getFromRawBuffer`, i.e. in host byte order. A
// big-endian host would need an explicit byte-swap; fail the build there
// rather than silently emit corrupted attributes.
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "mlir-named-attribute.cpp assumes a little-endian host for ONNX raw_data"
#endif

namespace morphizen {
namespace mlir_impl {

// Double-underscore prefix avoids collision with ONNX attribute names;
// add_node filters this key out so it never appears in dumped IR.
static constexpr llvm::StringLiteral kSubgraphRefKey =
    "__morphizen_subgraph_ref";

// Factory method to create integer array attribute
std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_int_array(const std::string &name,
                                     const std::vector<int64_t> &data) {
  // Get MLIR context from ContextManager
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);

  // Convert vector to MLIR integer array attribute
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(data.size());
  for (int64_t value : data) {
    attrs.push_back(builder.getI64IntegerAttr(value));
  }
  mlir::ArrayAttr array_attr = builder.getArrayAttr(attrs);

  // Create and return unique_ptr to MLIRNamedAttribute with the integer array
  return std::make_unique<mlir::NamedAttribute>(name, array_attr);
}

// Factory method to create float array attribute
std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_float_array(const std::string &name,
                                       const std::vector<float> &data) {
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(data.size());
  for (float value : data) {
    attrs.push_back(builder.getF32FloatAttr(value));
  }
  mlir::ArrayAttr array_attr = builder.getArrayAttr(attrs);

  return std::make_unique<mlir::NamedAttribute>(name, array_attr);
}

// Factory method to create string array attribute
std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_string_array(const std::string &name,
                                        const std::vector<std::string> &data) {
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);

  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(data.size());
  for (const std::string &value : data) {
    attrs.push_back(builder.getStringAttr(value));
  }
  mlir::ArrayAttr array_attr = builder.getArrayAttr(attrs);

  return std::make_unique<mlir::NamedAttribute>(name, array_attr);
}

// Factory method to create single integer attribute
std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_int(const std::string &name, int64_t value) {
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);
  // In onnx-mlir,  all attribute type if is signed Integer use `si64`
  mlir::Type si64Type = builder.getIntegerType(64, /*isSigned=*/true);
  mlir::IntegerAttr int_attr = mlir::IntegerAttr::get(si64Type, value);
  return std::make_unique<mlir::NamedAttribute>(name, int_attr);
}

// Factory method to create single float attribute
std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_float(const std::string &name, float value) {
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);

  mlir::FloatAttr float_attr = builder.getF32FloatAttr(value);
  return std::make_unique<mlir::NamedAttribute>(name, float_attr);
}

// Factory method to create single string attribute
std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_string(const std::string &name,
                                  const std::string &value) {
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);

  mlir::StringAttr string_attr = builder.getStringAttr(value);
  return std::make_unique<mlir::NamedAttribute>(name, string_attr);
}

// Factory method to create tensor attribute.
//
// ONNX `TENSOR` attributes (e.g. `onnx.Constant.value`,
// `onnx.ConstantOfShape.value`) must carry the actual tensor *data*, not a host
// pointer, so MLIR consumer passes -- including DenseElementsAttr-based folders
// such as `ConstantOfShapeAsScalar` -- can read the bytes directly with no
// out-of-band dereference.
//
// When the source `MLIRNodeArg` has inline data (the standard path for
// `OpAttr_GetTensorAttributeAsOrtValue`, which materialises a freshly-allocated
// CPU tensor that the ort-bridge then memcpy-copies into the proto), we build a
// real `mlir::DenseElementsAttr` from the raw bytes. This keeps the attribute
// self-contained and survives any IR serialisation / round-trip.
//
// If `onnxElementTypeToMlirElementType` does not yet map this ONNX element type
// (it silently defaults BFLOAT16/FP8/INT4/STRING to F32), the dense byte-size
// guard below fails and we LOG(FATAL) with a fix hint, rather than silently
// degrading to a pointer-encoded attribute that DenseElementsAttr consumers
// cannot read.
//
// The resulting `DenseElementsAttr` is the single source of truth: consumers
// read the tensor value directly from it. The mlir-imp backend deliberately
// does NOT expose TENSOR attributes through the legacy `attr_proto_get_tensor`
// -> `tensor_proto_*` proto API (see the fatal in `morphizen-ort-api.cpp`).
std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_tensor(const std::string &name,
                                  const MLIRNodeArg &tensor) {
  auto &context = MLIRContextManager::getInstance().getContext();
  mlir::OpBuilder builder(&context);

  // The attribute path only ever receives data-carrying, inline tensors: every
  // `tensor_proto_new_*` factory produces a data-carrying proto, and
  // external-data protos reach `add_initialized_tensor` (initializer), never
  // here (ORT also materialises external attrs before this point). After this
  // CHECK, getData()/getDataSize() are non-null / non-zero (the inline ctor
  // only stores data when data_size > 0).
  CHECK(tensor.hasData()) << "create_tensor expects an inline (data-carrying) "
                             "tensor on the attribute path";
  CHECK(!tensor.isExternalData())
      << "create_tensor expects a non-external tensor on the attribute path";

  // Tensor attributes always carry concrete weight data; unranked storage is
  // only possible at the ORT NodeArg boundary, not for initializers.
  auto shape = tensor.getShape();
  CHECK(shape.has_value()) << "tensor attribute has no rank: "
                           << tensor.getName();

  const void *data = tensor.getData();
  size_t data_size = tensor.getDataSize();
  llvm::SmallVector<int64_t> mlir_shape(shape->begin(), shape->end());
  mlir::Type elem_type =
      onnxElementTypeToMlirElementType(tensor.getElementType(), builder);
  auto tensor_type = mlir::RankedTensorType::get(mlir_shape, elem_type);

  // `DenseElementsAttr::getFromRawBuffer` requires data_size to equal the
  // tensor's natural byte size (numElements * elemByteWidth). A mismatch means
  // `onnxElementTypeToMlirElementType` did not map this ONNX element type and
  // defaulted it to F32 (BFLOAT16 / FP8 / INT4 / STRING).
  const size_t elem_bytes = (elem_type.getIntOrFloatBitWidth() + 7) / 8;
  const size_t expected = tensor_type.getNumElements() * elem_bytes;
  if (expected != data_size) {
    LOG(FATAL)
        << "create_tensor: ONNX element type " << tensor.getElementType()
        << " is not representable as a DenseElementsAttr "
           "(onnxElementTypeToMlirElementType defaulted it to F32: "
        << elem_bytes << "B * " << tensor_type.getNumElements()
        << " elems != raw_data " << data_size
        << "B). Fix: add a case for this element type in "
           "onnxElementTypeToMlirElementType (mlir-constants.cpp) so the "
           "MLIR element width matches the ONNX raw_data byte size.";
  }

  // Note: ONNX `raw_data` is little-endian; `getFromRawBuffer` reads the bytes
  // in host byte order -- correct on little-endian hosts (x86, ROCm GPU hosts);
  // a big-endian host would need an explicit byte-swap.
  llvm::ArrayRef<char> raw(static_cast<const char *>(data), data_size);
  mlir::DenseElementsAttr dense =
      mlir::DenseElementsAttr::getFromRawBuffer(tensor_type, raw);
  return std::make_unique<mlir::NamedAttribute>(name, dense);
}

std::unique_ptr<mlir::NamedAttribute>
MLIRNamedAttribute::create_subgraph_ref(const std::string &name,
                                        MLIRGraph &sub) {
  static_assert(sizeof(void *) <= sizeof(int64_t),
                "raw MLIRGraph* must fit in int64 to embed in IntegerAttr");
  auto &context = MLIRContextManager::getInstance().getContext();
  auto i64_type = mlir::IntegerType::get(&context, 64);
  auto ptr_attr = mlir::IntegerAttr::get(
      i64_type, static_cast<int64_t>(reinterpret_cast<intptr_t>(&sub)));
  mlir::NamedAttribute marker(mlir::StringAttr::get(&context, kSubgraphRefKey),
                              ptr_attr);
  auto ref_dict = mlir::DictionaryAttr::get(&context, {marker});
  return std::make_unique<mlir::NamedAttribute>(
      mlir::StringAttr::get(&context, name), ref_dict);
}

MLIRGraph *MLIRNamedAttribute::get_subgraph_ref() const {
  auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(getValue());
  if (!dict) {
    return nullptr;
  }
  auto entry = dict.getNamed(kSubgraphRefKey);
  if (!entry) {
    return nullptr;
  }
  auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(entry->getValue());
  if (!int_attr) {
    return nullptr;
  }
  return reinterpret_cast<MLIRGraph *>(
      static_cast<intptr_t>(int_attr.getInt()));
}

int64_t MLIRNamedAttribute::get_int() const {
  if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(getValue())) {
    return int_attr.getSInt();
  }
  return 0;
}

double MLIRNamedAttribute::get_float() const {
  if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(getValue())) {
    return float_attr.getValueAsDouble();
  }
  return 0.0f;
}

const std::string &MLIRNamedAttribute::get_string() const {
  static thread_local std::string tmp;
  if (auto attr = mlir::dyn_cast<mlir::StringAttr>(getValue())) {
    tmp = attr.getValue().str();
  }
  return tmp;
}

std::vector<int64_t> MLIRNamedAttribute::get_ints() const {
  std::vector<int64_t> result;
  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(getValue())) {
    result.reserve(array_attr.size());
    for (auto element_attr : array_attr) {
      if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(element_attr)) {
        result.push_back(int_attr.getInt());
      }
    }
  }
  return result;
}

const std::vector<float> &MLIRNamedAttribute::get_floats() const {
  static thread_local std::vector<float> result;
  result.clear();
  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(getValue())) {
    result.reserve(array_attr.size());
    for (auto element_attr : array_attr) {
      if (auto int_attr = mlir::dyn_cast<mlir::FloatAttr>(element_attr)) {
        result.push_back(static_cast<float>(int_attr.getValueAsDouble()));
      }
    }
  }
  return result;
}

std::vector<std::string> MLIRNamedAttribute::get_strings() const {
  std::vector<std::string> result;
  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(getValue())) {
    result.reserve(array_attr.size());
    for (auto element_attr : array_attr) {
      if (auto int_attr = mlir::dyn_cast<mlir::StringAttr>(element_attr)) {
        result.push_back(int_attr.getValue().str());
      }
    }
  }
  return result;
}

// Get the ONNX attribute type for this MLIR attribute
int MLIRNamedAttribute::get_onnx_type() const {
  if (!getValue()) {
    return 0; // UNDEFINED type
  }

  // Map MLIR attribute types to ONNX AttributeProto type constants
  // Based on morphizen_onnx::AttributeProto enum values
  if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(getValue())) {
    return 2; // INT
  }

  if (auto float_attr = mlir::dyn_cast<mlir::FloatAttr>(getValue())) {
    return 1; // FLOAT
  }

  if (auto string_attr = mlir::dyn_cast<mlir::StringAttr>(getValue())) {
    return 3; // STRING
  }

  if (auto array_attr = mlir::dyn_cast<mlir::ArrayAttr>(getValue())) {
    if (!array_attr.empty()) {
      // Check the type of the first element to determine array type
      mlir::Attribute first_elem = array_attr[0];

      if (mlir::isa<mlir::IntegerAttr>(first_elem)) {
        return 7; // INTS
      }

      if (mlir::isa<mlir::FloatAttr>(first_elem)) {
        return 6; // FLOATS
      }

      if (mlir::isa<mlir::StringAttr>(first_elem)) {
        return 8; // STRINGS
      }
    }
    // Empty array or unknown element type
    return 7; // Default to INTS for empty arrays
  }

  if (auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(getValue())) {
    return 4; // TENSOR
  }

  if (get_subgraph_ref() != nullptr) {
    return 5; // GRAPH
  }

  // Unknown or unsupported attribute type
  return 0; // UNDEFINED
}

void MLIRNamedAttribute::set_name(const std::string &n) {
  setName(mlir::StringAttr::get(&MLIRContextManager::getInstance().getContext(),
                                n));
}

} // namespace mlir_impl
} // namespace morphizen
