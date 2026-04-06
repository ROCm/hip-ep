/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipDNNGraph.h"

#include <backend/hipdnn_backend.h>
#include <frontend/hipdnn_frontend.hpp>
#include <hip/hip_runtime_api.h>

#include <cassert>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#undef interface
#endif
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

namespace hip::graph {

namespace {

// Layout for 4D convolution tensors.
enum class ConvLayout { NCHW, NHWC };

// Compute row-major strides from shape (NCHW / default layout).
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// Compute strides for NHWC layout from an NCHW shape.
// NHWC strides: N-stride = H*W*C, C-stride = 1, H-stride = W*C, W-stride = C.
// The shape is still in NCHW order; only the strides reflect NHWC memory
// layout.
static std::vector<int64_t>
ComputeNHWCStrides(const std::vector<int64_t> &shape) {
  assert(shape.size() == 4 && "ComputeNHWCStrides requires a 4D shape");
  int64_t C = shape[1], H = shape[2], W = shape[3];
  return {H * W * C, 1, W * C, C};
}

// Compute strides for a 4D shape according to the given layout.
static std::vector<int64_t>
ComputeConvStrides(const std::vector<int64_t> &shape, ConvLayout layout) {
  if (layout == ConvLayout::NHWC)
    return ComputeNHWCStrides(shape);
  return ComputeStrides(shape);
}

// Check if a hipDNN data type is a supported floating-point type.
static bool IsFloatDataType(hipdnn_frontend::DataType dtype) {
  return dtype == hipdnn_frontend::DataType::FLOAT ||
         dtype == hipdnn_frontend::DataType::HALF;
}

// Determine compute data type based on input data types.
// For float types with precision <= float32, compute in float32.
static std::optional<hipdnn_frontend::DataType>
GetComputeDataType(hipdnn_frontend::DataType x_dtype,
                   hipdnn_frontend::DataType w_dtype) {
  if (IsFloatDataType(x_dtype) && IsFloatDataType(w_dtype))
    return hipdnn_frontend::DataType::FLOAT;
  return std::nullopt;
}

using TensorAttrPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;

// Return true if the tensor has exactly one element (scalar).
static bool IsScalarAttr(const TensorAttrPtr &attr) {
  int64_t numel = 1;
  for (int64_t d : attr->get_dim())
    numel *= d;
  return numel == 1;
}

// Shape and data type extracted from an MLIR tensor type.
struct TensorInfo {
  std::vector<int64_t> shape;
  hipdnn_frontend::DataType dtype;
};

// Map MLIR element type to hipDNN DataType.
static mlir::FailureOr<hipdnn_frontend::DataType>
MLIRTypeToHipDNNDataType(mlir::Location loc, mlir::Type type) {
  using hipdnn_frontend::DataType;
  if (type.isF32())
    return DataType::FLOAT;
  if (type.isF16())
    return DataType::HALF;
  return mlir::emitError(loc) << "unsupported element type: " << type;
}

// Extract shape and element type from standard RankedTensorType.
static mlir::FailureOr<TensorInfo> GetTensorInfoStandard(mlir::Location loc,
                                                         mlir::Type type) {
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked)
    return mlir::emitError(loc) << "expected RankedTensorType, got: " << type;
  if (!ranked.hasStaticShape())
    return mlir::emitError(loc) << "RankedTensorType has dynamic shape";

  TensorInfo info;
  auto shape = ranked.getShape();
  info.shape.assign(shape.begin(), shape.end());

  auto dtype = MLIRTypeToHipDNNDataType(loc, ranked.getElementType());
  if (mlir::failed(dtype))
    return mlir::failure();
  info.dtype = *dtype;
  return info;
}

// Create TensorAttributes from standard RankedTensorType.
// Uses row-major (NCHW) strides; callers adjust for NHWC if needed.
static mlir::FailureOr<TensorAttrPtr>
CreateTensorAttrFromStandardMLIR(mlir::Location loc, mlir::Type type,
                                 int64_t uid, const std::string &name) {
  using hipdnn_frontend::graph::TensorAttributes;

  auto info = GetTensorInfoStandard(loc, type);
  if (mlir::failed(info))
    return mlir::failure();

  auto attr = std::make_shared<TensorAttributes>();
  attr->set_uid(uid)
      .set_name(name)
      .set_data_type(info->dtype)
      .set_dim(info->shape)
      .set_stride(ComputeStrides(info->shape));
  return attr;
}

// Extract integer values from an ArrayAttr of IntegerAttr.
// Uses getSExtValue() instead of getInt() because ONNX-MLIR encodes
// attributes like strides/pads/dilations as signed si64.
static std::vector<int64_t> ExtractI64Array(mlir::ArrayAttr arr) {
  std::vector<int64_t> result;
  if (!arr)
    return result;
  for (auto a : arr)
    result.push_back(
        mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  return result;
}

// Reshape bias from 1D [C] to 4D [1,C,1,1] for Conv addition.
// hipDNN requires bias shape to match the output rank. If the bias is already
// 4D or pass-by-value (scalar), no reshape is needed. Layout controls stride
// computation (NCHW vs NHWC).
static Status ReshapeBiasForConv(const TensorAttrPtr &bias, ConvLayout layout) {
  if (bias->get_pass_by_value())
    return Status::Success();

  auto bias_dim = bias->get_dim();
  if (bias_dim.size() == 4)
    return Status::Success();

  if (bias_dim.size() == 1) {
    int64_t C = bias_dim[0];
    bias->set_dim({1, C, 1, 1});
    bias->set_stride(ComputeConvStrides({1, C, 1, 1}, layout));
    return Status::Success();
  }

  return Status::Failure("Conv bias has unsupported rank " +
                         std::to_string(bias_dim.size()) +
                         "; expected 1D [C] or 4D [1,C,1,1]");
}

// Add Conv operation from an onnx.Conv op to hipDNN graph.
// Handles optional bias (3rd input) via pointwise ADD after conv_fprop.
static Status
AddConvNodeFromOnnxMLIR(hipdnn_frontend::graph::Graph &graph,
                        mlir::Operation *op,
                        const std::vector<TensorAttrPtr> &input_attrs,
                        TensorAttrPtr &output_attr, int64_t &next_uid) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;
  using hipdnn_frontend::PointwiseMode;

  if (input_attrs.size() < 2)
    return Status::Failure("onnx.Conv requires at least 2 input tensor attrs");

  const auto &x_attr = input_attrs[0];
  const auto &w_attr = input_attrs[1];

  if (IsScalarAttr(x_attr) || IsScalarAttr(w_attr))
    return Status::Failure("onnx.Conv inputs must be tensors, not scalars");

  ConvLayout layout = ConvLayout::NCHW;

  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1};

  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("strides")) {
    auto vals = ExtractI64Array(arr);
    if (vals.size() != 2)
      return Status::Failure("onnx.Conv: expected 2 strides, got " +
                             std::to_string(vals.size()));
    strides = vals;
  }
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
    auto vals = ExtractI64Array(arr);
    if (vals.size() == 4) {
      pads = vals;
    } else if (vals.size() == 2) {
      pads = {vals[0], vals[1], vals[0], vals[1]};
    } else {
      return Status::Failure("onnx.Conv: expected 2 or 4 pads, got " +
                             std::to_string(vals.size()));
    }
  }
  if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("dilations")) {
    auto vals = ExtractI64Array(arr);
    if (vals.size() != 2)
      return Status::Failure("onnx.Conv: expected 2 dilations, got " +
                             std::to_string(vals.size()));
    dilations = vals;
  }

  auto compute_dtype =
      GetComputeDataType(x_attr->get_data_type(), w_attr->get_data_type());
  if (!compute_dtype.has_value())
    return Status::Failure(
        "Unsupported data type combination for Conv compute");

  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({pads[0], pads[1]})
      .set_stride({strides[0], strides[1]})
      .set_dilation({dilations[0], dilations[1]})
      .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
      .set_compute_data_type(compute_dtype.value());

  output_attr = graph.conv_fprop(x_attr, w_attr, conv_attrs);

  if (input_attrs.size() >= 3) {
    auto dtype = compute_dtype.value();
    output_attr->set_data_type(dtype);
    auto bias = input_attrs[2];
    auto reshape_status = ReshapeBiasForConv(bias, layout);
    if (reshape_status.failed())
      return reshape_status;

    PointwiseAttributes add;
    add.set_mode(PointwiseMode::ADD).set_compute_data_type(dtype);
    output_attr = graph.pointwise(output_attr, bias, add);
  }

  return Status::Success();
}

// Dispatch generic ONNX MLIR op to appropriate node builder.
static Status AddNodeFromOnnxMLIR(hipdnn_frontend::graph::Graph &graph,
                                  mlir::Operation *op,
                                  const std::vector<TensorAttrPtr> &input_attrs,
                                  std::vector<TensorAttrPtr> &output_attrs,
                                  int64_t &next_uid) {
  llvm::StringRef op_name = op->getName().getStringRef();

  if (op_name == "onnx.Conv") {
    TensorAttrPtr y_attr;
    auto status =
        AddConvNodeFromOnnxMLIR(graph, op, input_attrs, y_attr, next_uid);
    if (status.failed())
      return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  return Status::Failure("Unsupported ONNX MLIR op: " + op_name.str());
}

} // namespace

//===----------------------------------------------------------------------===//
// HipDNNGraphImpl
//===----------------------------------------------------------------------===//

struct HipDNNGraphImpl {
  explicit HipDNNGraphImpl(hipdnnHandle_t handle) : handle_(handle) {}

  ~HipDNNGraphImpl() {
    if (workspace_ptr_)
      hipFree(workspace_ptr_);
  }

  Status BuildFromOnnxMLIR(mlir::Region &region) {
    using namespace hipdnn_frontend::graph;

    if (region.empty())
      return Status::Failure("Empty region in onnx MLIR graph");

    mlir::Block &block = region.front();
    auto *terminator = block.getTerminator();
    if (!terminator)
      return Status::Failure("Region has no terminator");

    // Store output shapes for workspace sizing at execute time.
    output_shapes_.reserve(terminator->getNumOperands());
    for (mlir::Value output : terminator->getOperands()) {
      auto info = GetTensorInfoStandard(output.getLoc(), output.getType());
      if (mlir::failed(info))
        return Status::Failure("Failed to get tensor info for output");
      output_shapes_.push_back(info->shape);
    }

    graph_ = std::make_unique<Graph>();

    llvm::DenseMap<mlir::Value, TensorAttrPtr> value_map;

    // Process block arguments as graph inputs (standard RankedTensorType).
    for (auto [idx, arg] : llvm::enumerate(block.getArguments())) {
      if (!mlir::isa<mlir::RankedTensorType>(arg.getType()))
        continue;

      std::string name = "input_" + std::to_string(idx);
      auto attr = CreateTensorAttrFromStandardMLIR(arg.getLoc(), arg.getType(),
                                                   next_uid_++, name);
      if (mlir::failed(attr))
        return Status::Failure("Failed to create tensor attr for input " +
                               std::to_string(idx));
      (*attr)->set_is_virtual(false);
      value_map[arg] = *attr;
      input_uids_.push_back((*attr)->get_uid());
    }

    for (mlir::Operation &op : block.without_terminator()) {
      if (op.getNumResults() == 0)
        continue;

      bool has_tensor_result = false;
      for (mlir::Value result : op.getResults()) {
        if (mlir::isa<mlir::RankedTensorType>(result.getType())) {
          has_tensor_result = true;
          break;
        }
      }
      if (!has_tensor_result)
        continue;

      std::vector<TensorAttrPtr> input_attrs;
      for (mlir::Value operand : op.getOperands()) {
        auto it = value_map.find(operand);
        if (it != value_map.end())
          input_attrs.push_back(it->second);
      }

      std::vector<TensorAttrPtr> output_attrs;
      auto status = AddNodeFromOnnxMLIR(*graph_, &op, input_attrs, output_attrs,
                                        next_uid_);
      if (status.failed())
        return status;

      size_t tensor_result_idx = 0;
      for (mlir::Value result : op.getResults()) {
        if (!mlir::isa<mlir::RankedTensorType>(result.getType()))
          continue;
        if (tensor_result_idx >= output_attrs.size())
          break;

        TensorAttrPtr &attr = output_attrs[tensor_result_idx++];
        auto info = GetTensorInfoStandard(result.getLoc(), result.getType());
        if (mlir::failed(info))
          return Status::Failure("Failed to get tensor info for op result");
        attr->set_uid(next_uid_++)
            .set_name("v" + std::to_string(attr->get_uid()))
            .set_data_type(info->dtype)
            .set_dim(info->shape)
            .set_stride(ComputeStrides(info->shape));
        value_map[result] = attr;
      }
    }

    // Mark graph outputs as non-virtual and store their UIDs.
    output_uids_.reserve(terminator->getNumOperands());
    for (mlir::Value output : terminator->getOperands()) {
      auto it = value_map.find(output);
      if (it == value_map.end())
        return Status::Failure("Output value not found in value map");
      it->second->set_is_virtual(false);
      output_uids_.push_back(it->second->get_uid());
    }

    return Status::Success();
  }

  Status Compile() {
    using hipdnn_frontend::HeuristicMode;

    auto error = graph_->validate();
    if (error.is_bad())
      return Status::Failure("hipDNN graph validation failed: " +
                             error.get_message());

    error = graph_->build_operation_graph(handle_);
    if (error.is_bad())
      return Status::Failure("hipDNN build_operation_graph failed: " +
                             error.get_message());

    error = graph_->create_execution_plans({HeuristicMode::FALLBACK});
    if (error.is_bad())
      return Status::Failure("hipDNN create_execution_plans failed: " +
                             error.get_message());

    error = graph_->check_support();
    if (error.is_bad())
      return Status::Failure("hipDNN check_support failed: " +
                             error.get_message());

    error = graph_->build_plans();
    if (error.is_bad())
      return Status::Failure("hipDNN build_plans failed: " +
                             error.get_message());

    error = graph_->get_workspace_size(workspace_size_);
    if (error.is_bad())
      return Status::Failure("hipDNN get_workspace_size failed: " +
                             error.get_message());

    if (workspace_size_ > 0) {
      if (workspace_ptr_) {
        hipFree(workspace_ptr_);
        workspace_ptr_ = nullptr;
      }
      void *ptr = nullptr;
      if (hipMalloc(&ptr, workspace_size_) != hipSuccess)
        return Status::Failure("hipDNN workspace hipMalloc failed for " +
                               std::to_string(workspace_size_) + " bytes");
      workspace_ptr_ = ptr;
    }

    return Status::Success();
  }

  Status Execute(hipdnnHandle_t handle,
                 std::unordered_map<int64_t, void *> &variant_pack,
                 void *workspace) {
    void *ws = workspace_ptr_ ? workspace_ptr_ : workspace;
    auto error = graph_->execute(handle, variant_pack, ws);
    if (error.is_bad())
      return Status::Failure("hipDNN execute failed: " + error.get_message());
    return Status::Success();
  }

  hipdnnHandle_t handle_;
  // hipDNN frontend graph object.
  std::unique_ptr<hipdnn_frontend::graph::Graph> graph_;
  // UIDs assigned to graph inputs/outputs during Build.
  std::vector<int64_t> input_uids_;
  std::vector<int64_t> output_uids_;
  // Output tensor shapes (for workspace sizing at execute time).
  std::vector<std::vector<int64_t>> output_shapes_;
  // UID counter for tensor attributes.
  int64_t next_uid_ = 0;
  // Workspace size in bytes (determined at Compile time).
  int64_t workspace_size_ = 0;
  // GPU workspace buffer, allocated during Compile() and freed in destructor.
  void *workspace_ptr_ = nullptr;
};

//===----------------------------------------------------------------------===//
// HipDNNGraph public API
//===----------------------------------------------------------------------===//

HipDNNGraph::HipDNNGraph(hipdnnHandle_t handle)
    : impl_(std::make_unique<HipDNNGraphImpl>(handle)) {}

HipDNNGraph::~HipDNNGraph() = default;

Status HipDNNGraph::BuildFromOnnxMLIR(mlir::Region &region) {
  return impl_->BuildFromOnnxMLIR(region);
}

Status HipDNNGraph::Compile() { return impl_->Compile(); }

Status HipDNNGraph::Execute(hipdnnHandle_t handle,
                            std::unordered_map<int64_t, void *> &variant_pack,
                            void *workspace) {
  return impl_->Execute(handle, variant_pack, workspace);
}

llvm::ArrayRef<int64_t> HipDNNGraph::getInputUids() const {
  return impl_->input_uids_;
}

llvm::ArrayRef<int64_t> HipDNNGraph::getOutputUids() const {
  return impl_->output_uids_;
}

int64_t HipDNNGraph::getWorkspaceSize() const { return impl_->workspace_size_; }

} // namespace hip::graph
