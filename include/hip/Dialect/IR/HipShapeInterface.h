/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIPSHAPEINTERFACE_H
#define HIP_DIALECT_IR_HIPSHAPEINTERFACE_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace hip {

//===----------------------------------------------------------------------===//
// DimSpec: a tree-shaped specification of a single tensor dimension.
//===----------------------------------------------------------------------===//
//
// DimSpec describes how to compute one tensor dimension at runtime. The tree
// is encoded as a flat vector of DimSpecNode in topological order, root at
// index 0. Leaves carry a payload; binary nodes reference two children by
// index.
//
// Mirrors `enum DimSpecKind` in schemas/model_metadata.fbs.

enum class DimSpecKind : uint8_t {
  Static = 0,        // payload: value
  InputDim = 1,      // payload: input_index, dim_index — Category A
  InputValueI64 = 2, // payload: input_index, flat_offset — Category B
  RuntimeSlot = 3,   // payload: slot_id — Category C (POST-compute only)
  // Binary arithmetic nodes (children indices)
  Add = 16,
  Sub = 17,
  Mul = 18,
  FloorDiv = 19,
  CeilDiv = 20,
  Min = 21,
  Max = 22,
};

struct DimSpecNode {
  DimSpecKind kind = DimSpecKind::Static;
  int64_t value = 0;       // Static
  int32_t input_index = 0; // InputDim, InputValueI64
  int32_t dim_index = 0;   // InputDim
  int64_t flat_offset = 0; // InputValueI64
  int32_t slot_id = -1;    // RuntimeSlot
  int32_t lhs = -1;        // binary child index
  int32_t rhs = -1;        // binary child index
};

class DimSpec {
public:
  DimSpec() = default;

  // Convenience leaf builders.
  static DimSpec makeStatic(int64_t value);
  static DimSpec makeInputDim(int32_t input_index, int32_t dim_index);
  static DimSpec makeInputValueI64(int32_t input_index, int64_t flat_offset);
  static DimSpec makeRuntimeSlot(int32_t slot_id);

  // Compose two DimSpecs with a binary operator.
  static DimSpec makeBinary(DimSpecKind op, const DimSpec &lhs,
                            const DimSpec &rhs);

  // Substitute every RuntimeSlot leaf whose slot_id is keyed in
  // `slot_to_subtree` with the matching subtree, returning a new
  // (composed) DimSpec. Used by the ComposeDimSpecs pass.
  DimSpec substituteSlots(
      const llvm::DenseMap<int32_t, DimSpec> &slot_to_subtree) const;

  // Returns true when this DimSpec is a single Static leaf with the given
  // value. Useful as the "compile-time static" predicate.
  bool isStatic() const { return root().kind == DimSpecKind::Static; }
  int64_t staticValue() const { return root().value; }

  // Returns true when the tree contains at least one RuntimeSlot leaf
  // (i.e. requires POST-compute resolution).
  bool needsRuntimeSlot() const;

  // Returns the set of slot_ids referenced anywhere in the tree.
  llvm::SmallVector<int32_t, 4> collectSlotIds() const;

  // Root accessor.
  const DimSpecNode &root() const { return nodes_[0]; }
  const std::vector<DimSpecNode> &nodes() const { return nodes_; }
  std::vector<DimSpecNode> &mutableNodes() { return nodes_; }

  // Serialisation as an MLIR ArrayAttr. Each entry is a DenseI64ArrayAttr
  // holding [kind, value, input_index, dim_index, flat_offset, slot_id,
  // lhs, rhs]. Empty array attr ⇔ empty DimSpec. Used to carry DimSpec
  // payloads as op attributes / func-result attributes through the
  // pipeline.
  mlir::ArrayAttr serializeAsArrayAttr(mlir::MLIRContext *ctx) const;
  static DimSpec parseFromArrayAttr(mlir::ArrayAttr attr);

  // Human-readable form, e.g. "mul(arg[0].shape[0], 16)" or
  // "slot[3]" or "input[1].value[0]". Used by the MLIR attr printer,
  // FlatBuffers inspector, and runtime tracer.
  std::string toString() const;

  // Pretty-print a single ArrayAttr-serialised DimSpec into `os`. Returns
  // false when `attr` is not a valid DimSpec encoding (no-op print).
  static bool printArrayAttr(mlir::ArrayAttr attr, llvm::raw_ostream &os);

  // Validate that the tree is well-formed (no cycles, valid child indices,
  // leaves have no children, binary nodes have two children). Returns true
  // on success; writes a diagnostic into `error` on failure.
  bool verify(std::string &error) const;

private:
  // Recursive toString helper.
  void appendNodeString(int32_t idx, std::string &out) const;
  // Recursive substitution helper. Returns the index in `out` of the
  // copied/substituted subtree root.
  int32_t cloneSubtree(int32_t src_root,
                       const llvm::DenseMap<int32_t, DimSpec> &slot_to_subtree,
                       std::vector<DimSpecNode> &out) const;

  // Flat tree storage; nodes_[0] is the root. May be empty for an
  // unconstructed DimSpec (treat as "no information").
  std::vector<DimSpecNode> nodes_;
};

//===----------------------------------------------------------------------===//
// HipShapeOpInterface: per-op DimSpec contributor.
//===----------------------------------------------------------------------===//
//
// Each HIP op that produces a result whose static MLIR shape contains at
// least one `?` dim should "implement" this interface to declare how each
// dynamic dim is computed.
//
// Implementation pattern (deliberately *not* an ODS-generated interface,
// to keep the surface minimal and avoid invasive TableGen changes across
// many ops): we use plain free functions in `shape_interface::` that
// dispatch on `op->getName().getStringRef()` to per-op handlers.
//
// Op-level composition: `getResultDimSpec` may produce a DimSpec whose
// `InputDim` / `InputValueI64` leaves reference operand indices of the op
// (i.e. "operand[k] dim d"). The ComposeDimSpecs pass walks back through
// producing ops via `resolveDimFromValue` to substitute these operand
// references until every leaf bottoms out at either
//   - a function argument shape access (truly Category-A `InputDim`),
//   - a function argument value access (Category-B `InputValueI64`),
//   - a `Static` constant (Category-D arithmetic over those leaves), or
//   - a `RuntimeSlot` published by a Category-C wrapper.
//
// Slot publishers (Category C): Hip_NonZeroOp publishes a slot via its
// `slot_id` attribute; future Category-C ops (wrap_range and
// wrap_constant_of_shape on intermediate operands) do the same.

namespace shape_interface {
// Return the DimSpec for result `result_index`, dim `dim_index` of `op`.
// Each concrete HIP op that wants to participate registers a builder via
// `registerOpDimSpecBuilder`. Default behavior for unregistered ops:
//   - If the result type is a RankedTensor/RankedMemRef and the dim is
//     static, return DimSpec::makeStatic(dim_size).
//   - If the dim is dynamic, return an empty DimSpec (caller treats this
//     as "unresolvable here — leave the operand-walker to descend further
//     until it hits a known producer").
DimSpec getResultDimSpec(mlir::Operation *op, unsigned result_index,
                         unsigned dim_index);

// Return the list of (slot_id, result_index, dim_index) tuples that `op`
// publishes during compute. Empty for ops that do not publish slots
// (everything except Category-C wrappers).
struct PublishedSlot {
  int32_t slot_id;
  unsigned result_index;
  unsigned dim_index;
};
llvm::SmallVector<PublishedSlot, 1>
getPublishedSlots(mlir::Operation *op);

// Try to derive a DimSpec for `v`'s dim `dim_index` by walking back
// through producing ops. If `v` is a function argument (a `BlockArgument`
// of the parent FuncOp's entry block), returns InputDim. Otherwise,
// recurses into the producer op via `getResultDimSpec`. Returns an empty
// DimSpec if no producer participates or the chain breaks.
DimSpec resolveDimFromValue(mlir::Value v, unsigned dim_index);

// Try to derive a Category-B DimSpec for the i64 value at flat offset
// `flat_offset` of the tensor `v`. If `v` is a function argument that the
// EP will be able to read host-side at runtime, returns InputValueI64.
// Otherwise returns an empty DimSpec (caller should fall back to Category
// C / RuntimeSlot dispatch).
DimSpec resolveValueFromI64Tensor(mlir::Value v, int64_t flat_offset);
} // namespace shape_interface

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIPSHAPEINTERFACE_H
