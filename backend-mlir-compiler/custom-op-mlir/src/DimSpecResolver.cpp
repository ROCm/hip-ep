/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "DimSpecResolver.h"

#include "InferenceState.h"
#include <algorithm>
#include <cstdlib>
#include <glog/logging.h>

namespace mlir_compilation::customop {

bool debugShapesEnabled() {
  static const bool enabled = []() {
    const char *v1 = std::getenv("HIPDNN_EP_DEBUG_SHAPES");
    if (v1 && v1[0] >= '1') return true;
    const char *v2 = std::getenv("HIPDNN_EP_TRACE_SHAPES");
    return v2 && v2[0] >= '1';
  }();
  return enabled;
}

namespace {

const mlir::hip::DimSpecNodeT &nodeAt(const mlir::hip::DimSpecT &spec,
                                      size_t idx) {
  CHECK_LT(idx, spec.nodes.size())
      << "DimSpec node index " << idx << " out of range (size "
      << spec.nodes.size() << ")";
  CHECK(spec.nodes[idx]) << "DimSpec node at index " << idx << " is null";
  return *spec.nodes[idx];
}

// Read an int64 value out of an input tensor at flat element offset
// `flat_offset`. The tensor is assumed to be host-resident i64; this is the
// shape-tensor convention for Range, ConstantOfShape, etc. Returns true on
// success, LOG(FATAL) on out-of-bounds or null data.
int64_t readInputI64(int32_t input_index, int64_t flat_offset,
                     const std::vector<std::vector<int64_t>> &input_shapes,
                     const std::vector<const void *> &input_data) {
  CHECK_GE(input_index, 0);
  CHECK_LT(static_cast<size_t>(input_index), input_shapes.size())
      << "InputValueI64 references input_index " << input_index
      << " but only " << input_shapes.size() << " inputs are marshalled";
  CHECK_LT(static_cast<size_t>(input_index), input_data.size())
      << "InputValueI64 references input_index " << input_index
      << " but only " << input_data.size() << " input data pointers";
  const void *raw = input_data[input_index];
  CHECK(raw) << "InputValueI64 references input_index " << input_index
             << " which is a GPU-resident tensor; the EP must D2H-stage "
                "it before resolving the DimSpec (Category B prerequisite)";

  int64_t numel = 1;
  for (int64_t d : input_shapes[input_index]) {
    numel *= d;
  }
  CHECK_GE(flat_offset, 0);
  CHECK_LT(flat_offset, numel)
      << "InputValueI64 flat_offset " << flat_offset
      << " out of range for input " << input_index << " (numel " << numel
      << ")";
  return reinterpret_cast<const int64_t *>(raw)[flat_offset];
}

bool resolveNode(const mlir::hip::DimSpecT &spec, size_t node_idx,
                 const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<const void *> &input_data,
                 const InferenceState *state, int64_t &out_value) {
  const auto &n = nodeAt(spec, node_idx);
  switch (n.kind) {
  case mlir::hip::DimSpecKind::Static:
    out_value = n.value;
    return true;

  case mlir::hip::DimSpecKind::InputDim: {
    CHECK_GE(n.input_index, 0);
    CHECK_LT(static_cast<size_t>(n.input_index), input_shapes.size())
        << "InputDim references input_index " << n.input_index
        << " but only " << input_shapes.size() << " inputs are marshalled";
    const auto &shape = input_shapes[n.input_index];
    CHECK_GE(n.dim_index, 0);
    CHECK_LT(static_cast<size_t>(n.dim_index), shape.size())
        << "InputDim references dim_index " << n.dim_index << " for input "
        << n.input_index << " (rank " << shape.size() << ")";
    out_value = shape[n.dim_index];
    return true;
  }

  case mlir::hip::DimSpecKind::InputValueI64:
    out_value =
        readInputI64(n.input_index, n.flat_offset, input_shapes, input_data);
    if (debugShapesEnabled()) {
      fprintf(stderr,
              "[Resolver] InputValueI64(input=%d, offset=%lld) = %lld\n",
              (int)n.input_index, (long long)n.flat_offset,
              (long long)out_value);
    }
    return true;

  case mlir::hip::DimSpecKind::RuntimeSlot: {
    if (!state) {
      // Pre-compute pass: signal "deferred".
      return false;
    }
    constexpr int64_t kUnpublished = -1;
    int64_t v = state->read_dim(n.slot_id);
    if (v == kUnpublished) {
      return false;
    }
    out_value = v;
    return true;
  }

  case mlir::hip::DimSpecKind::Add:
  case mlir::hip::DimSpecKind::Sub:
  case mlir::hip::DimSpecKind::Mul:
  case mlir::hip::DimSpecKind::FloorDiv:
  case mlir::hip::DimSpecKind::CeilDiv:
  case mlir::hip::DimSpecKind::Min:
  case mlir::hip::DimSpecKind::Max: {
    CHECK_EQ(n.children.size(), 2u)
        << "Binary DimSpec node must have exactly 2 children";
    int64_t lhs, rhs;
    if (!resolveNode(spec, n.children[0], input_shapes, input_data, state,
                     lhs)) {
      return false;
    }
    if (!resolveNode(spec, n.children[1], input_shapes, input_data, state,
                     rhs)) {
      return false;
    }
    switch (n.kind) {
    case mlir::hip::DimSpecKind::Add:
      out_value = lhs + rhs;
      break;
    case mlir::hip::DimSpecKind::Sub:
      out_value = lhs - rhs;
      break;
    case mlir::hip::DimSpecKind::Mul:
      out_value = lhs * rhs;
      break;
    case mlir::hip::DimSpecKind::FloorDiv:
      CHECK_NE(rhs, 0) << "FloorDiv by zero in DimSpec";
      // Floor-division for signed integers: round toward -inf, matching
      // numpy / Python `//` semantics. The compiler arithmetic also follows
      // these semantics.
      out_value = (lhs / rhs) - (((lhs % rhs) != 0) && ((lhs ^ rhs) < 0));
      break;
    case mlir::hip::DimSpecKind::CeilDiv:
      CHECK_NE(rhs, 0) << "CeilDiv by zero in DimSpec";
      // Standard ceil-division for signed integers, matching the compiler.
      out_value = (lhs / rhs) + (((lhs % rhs) != 0) && ((lhs ^ rhs) >= 0));
      break;
    case mlir::hip::DimSpecKind::Min:
      out_value = std::min(lhs, rhs);
      break;
    case mlir::hip::DimSpecKind::Max:
      out_value = std::max(lhs, rhs);
      break;
    default:
      // Unreachable -- silences the compiler in the outer switch.
      break;
    }
    // NOTE: previously this branch clamped `out_value` to 0 inside the
    // recursion. That was wrong: intermediate values may be legitimately
    // negative (e.g. ONNX `Range(start=10, limit=0, delta=-2)` has the
    // DimSpec `CeilDiv(Sub(limit,start), delta) = CeilDiv(-10, -2) = 5`
    // -- the inner Sub is -10, but the final dim is 5). Saturation only
    // makes sense at the root, where we know the value is the dim size;
    // the public `resolve()` entry-point does that final clamp.
    if (debugShapesEnabled()) {
      const char *opn = "?";
      switch (n.kind) {
        case mlir::hip::DimSpecKind::Add: opn = "Add"; break;
        case mlir::hip::DimSpecKind::Sub: opn = "Sub"; break;
        case mlir::hip::DimSpecKind::Mul: opn = "Mul"; break;
        case mlir::hip::DimSpecKind::FloorDiv: opn = "FloorDiv"; break;
        case mlir::hip::DimSpecKind::CeilDiv: opn = "CeilDiv"; break;
        case mlir::hip::DimSpecKind::Min: opn = "Min"; break;
        case mlir::hip::DimSpecKind::Max: opn = "Max"; break;
        default: break;
      }
      fprintf(stderr, "[Resolver] %s(%lld, %lld) = %lld\n", opn,
              (long long)lhs, (long long)rhs, (long long)out_value);
    }
    return true;
  }
  }
  LOG(FATAL) << "Unhandled DimSpecKind: " << static_cast<int>(n.kind);
  return false;
}

bool containsRuntimeSlotImpl(const mlir::hip::DimSpecT &spec, size_t node_idx) {
  const auto &n = nodeAt(spec, node_idx);
  if (n.kind == mlir::hip::DimSpecKind::RuntimeSlot)
    return true;
  for (int32_t c : n.children) {
    if (containsRuntimeSlotImpl(spec, static_cast<size_t>(c)))
      return true;
  }
  return false;
}

} // namespace

bool containsRuntimeSlot(const mlir::hip::DimSpecT &spec) {
  if (spec.nodes.empty())
    return false;
  return containsRuntimeSlotImpl(spec, 0);
}

bool resolve(const mlir::hip::DimSpecT &spec,
             const std::vector<std::vector<int64_t>> &input_shapes,
             const std::vector<const void *> &input_data,
             const InferenceState *state, int64_t &out_value) {
  if (spec.nodes.empty()) {
    // Empty spec means "no DimSpec encoded for this dim" -- the compiler
    // only emits a node list when it could compose one. Caller is expected
    // to fall back to the legacy proto shape (-1 -> unknown).
    return false;
  }
  if (!resolveNode(spec, /*node_idx=*/0, input_shapes, input_data, state,
                   out_value)) {
    return false;
  }
  // ONNX-side dimension sizes are non-negative. Negative root values can
  // legitimately arise from `Sub(small, large)` in ONNX Range with an
  // empty interval (e.g. Range(0, 0, 1) -> Sub(0,0)=0 / CeilDiv -> 0, or
  // Range(5, 3, 1) -> CeilDiv(-2, 1) = -2 -> clamped to 0). Saturate at
  // the root so downstream consumers see a well-formed dim size.
  if (out_value < 0)
    out_value = 0;
  return true;
}

} // namespace mlir_compilation::customop
