/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Dialect/IR/HipShapeInterface.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace mlir {
namespace hip {

//===----------------------------------------------------------------------===//
// DimSpec - construction
//===----------------------------------------------------------------------===//

DimSpec DimSpec::makeStatic(int64_t value) {
  DimSpec s;
  DimSpecNode n;
  n.kind = DimSpecKind::Static;
  n.value = value;
  s.nodes_.push_back(n);
  return s;
}

DimSpec DimSpec::makeInputDim(int32_t input_index, int32_t dim_index) {
  DimSpec s;
  DimSpecNode n;
  n.kind = DimSpecKind::InputDim;
  n.input_index = input_index;
  n.dim_index = dim_index;
  s.nodes_.push_back(n);
  return s;
}

DimSpec DimSpec::makeInputValueI64(int32_t input_index, int64_t flat_offset) {
  DimSpec s;
  DimSpecNode n;
  n.kind = DimSpecKind::InputValueI64;
  n.input_index = input_index;
  n.flat_offset = flat_offset;
  s.nodes_.push_back(n);
  return s;
}

DimSpec DimSpec::makeRuntimeSlot(int32_t slot_id) {
  DimSpec s;
  DimSpecNode n;
  n.kind = DimSpecKind::RuntimeSlot;
  n.slot_id = slot_id;
  s.nodes_.push_back(n);
  return s;
}

DimSpec DimSpec::makeBinary(DimSpecKind op, const DimSpec &lhs,
                            const DimSpec &rhs) {
  // Layout: [root, ...lhs_subtree shifted by 1, ...rhs_subtree shifted
  // by 1 + lhs.size()]. Child indices in subtrees are remapped.
  DimSpec out;
  out.nodes_.reserve(1 + lhs.nodes_.size() + rhs.nodes_.size());
  DimSpecNode root;
  root.kind = op;
  out.nodes_.push_back(root);
  // Append lhs starting at index 1.
  const int32_t lhs_base = 1;
  for (const auto &n : lhs.nodes_) {
    DimSpecNode c = n;
    if (c.lhs >= 0)
      c.lhs += lhs_base;
    if (c.rhs >= 0)
      c.rhs += lhs_base;
    out.nodes_.push_back(c);
  }
  const int32_t rhs_base = lhs_base + (int32_t)lhs.nodes_.size();
  for (const auto &n : rhs.nodes_) {
    DimSpecNode c = n;
    if (c.lhs >= 0)
      c.lhs += rhs_base;
    if (c.rhs >= 0)
      c.rhs += rhs_base;
    out.nodes_.push_back(c);
  }
  out.nodes_[0].lhs = lhs_base;
  out.nodes_[0].rhs = rhs_base;
  return out;
}

//===----------------------------------------------------------------------===//
// DimSpec - queries
//===----------------------------------------------------------------------===//

bool DimSpec::needsRuntimeSlot() const {
  for (const auto &n : nodes_)
    if (n.kind == DimSpecKind::RuntimeSlot)
      return true;
  return false;
}

llvm::SmallVector<int32_t, 4> DimSpec::collectSlotIds() const {
  llvm::SmallVector<int32_t, 4> out;
  for (const auto &n : nodes_)
    if (n.kind == DimSpecKind::RuntimeSlot)
      out.push_back(n.slot_id);
  return out;
}

//===----------------------------------------------------------------------===//
// DimSpec - substitution
//===----------------------------------------------------------------------===//

int32_t DimSpec::cloneSubtree(
    int32_t src_root,
    const llvm::DenseMap<int32_t, DimSpec> &slot_to_subtree,
    std::vector<DimSpecNode> &out) const {
  const DimSpecNode &src = nodes_[src_root];
  // Slot substitution: graft the entire subtree from the substitution map.
  if (src.kind == DimSpecKind::RuntimeSlot) {
    auto it = slot_to_subtree.find(src.slot_id);
    if (it != slot_to_subtree.end()) {
      const DimSpec &sub = it->second;
      // Copy sub.nodes_ into `out` and recursively substitute again within
      // the substituted subtree (handles chained slot resolutions).
      int32_t base = (int32_t)out.size();
      for (const auto &n : sub.nodes_) {
        DimSpecNode c = n;
        if (c.lhs >= 0)
          c.lhs += base;
        if (c.rhs >= 0)
          c.rhs += base;
        out.push_back(c);
      }
      // Recurse: re-substitute slot references within the just-copied
      // subtree by performing a second pass on it. Build a temporary
      // DimSpec view and clone it again — simpler than in-place fixup.
      // Cheap because slot chains are typically depth 1.
      DimSpec inner;
      inner.nodes_.assign(out.begin() + base, out.end());
      std::vector<DimSpecNode> inner_out;
      int32_t inner_root =
          inner.cloneSubtree(0, slot_to_subtree, inner_out);
      // Replace the just-appended span with `inner_out` shifted to base.
      out.erase(out.begin() + base, out.end());
      for (auto &n : inner_out) {
        if (n.lhs >= 0)
          n.lhs += base;
        if (n.rhs >= 0)
          n.rhs += base;
        out.push_back(n);
      }
      return base + inner_root;
    }
  }
  // Default: copy the node and recurse on children.
  int32_t my_idx = (int32_t)out.size();
  out.push_back(src);
  out[my_idx].lhs = -1;
  out[my_idx].rhs = -1;
  if (src.lhs >= 0) {
    int32_t new_lhs = cloneSubtree(src.lhs, slot_to_subtree, out);
    out[my_idx].lhs = new_lhs;
  }
  if (src.rhs >= 0) {
    int32_t new_rhs = cloneSubtree(src.rhs, slot_to_subtree, out);
    out[my_idx].rhs = new_rhs;
  }
  return my_idx;
}

DimSpec DimSpec::substituteSlots(
    const llvm::DenseMap<int32_t, DimSpec> &slot_to_subtree) const {
  if (nodes_.empty())
    return DimSpec();
  DimSpec out;
  cloneSubtree(0, slot_to_subtree, out.nodes_);
  return out;
}

//===----------------------------------------------------------------------===//
// DimSpec - serialisation
//===----------------------------------------------------------------------===//

// Encoding: ArrayAttr of DenseI64ArrayAttr, one entry per node. Layout:
//   [kind, value, input_index, dim_index, flat_offset, slot_id, lhs, rhs]
static constexpr unsigned kEncodingArity = 8;

mlir::ArrayAttr
DimSpec::serializeAsArrayAttr(mlir::MLIRContext *ctx) const {
  mlir::Builder b(ctx);
  llvm::SmallVector<mlir::Attribute> entries;
  entries.reserve(nodes_.size());
  for (const auto &n : nodes_) {
    int64_t fields[kEncodingArity] = {
        static_cast<int64_t>(n.kind),
        n.value,
        n.input_index,
        n.dim_index,
        n.flat_offset,
        n.slot_id,
        n.lhs,
        n.rhs,
    };
    entries.push_back(
        b.getDenseI64ArrayAttr(llvm::ArrayRef<int64_t>(fields, kEncodingArity)));
  }
  return b.getArrayAttr(entries);
}

DimSpec DimSpec::parseFromArrayAttr(mlir::ArrayAttr attr) {
  DimSpec out;
  if (!attr)
    return out;
  for (mlir::Attribute e : attr) {
    auto arr = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(e);
    if (!arr || arr.size() < (int64_t)kEncodingArity) {
      out.nodes_.clear();
      return out;
    }
    DimSpecNode n;
    n.kind = static_cast<DimSpecKind>(arr[0]);
    n.value = arr[1];
    n.input_index = static_cast<int32_t>(arr[2]);
    n.dim_index = static_cast<int32_t>(arr[3]);
    n.flat_offset = arr[4];
    n.slot_id = static_cast<int32_t>(arr[5]);
    n.lhs = static_cast<int32_t>(arr[6]);
    n.rhs = static_cast<int32_t>(arr[7]);
    out.nodes_.push_back(n);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// DimSpec - printing
//===----------------------------------------------------------------------===//

static const char *binaryName(DimSpecKind k) {
  switch (k) {
  case DimSpecKind::Add:
    return "add";
  case DimSpecKind::Sub:
    return "sub";
  case DimSpecKind::Mul:
    return "mul";
  case DimSpecKind::FloorDiv:
    return "floordiv";
  case DimSpecKind::CeilDiv:
    return "ceildiv";
  case DimSpecKind::Min:
    return "min";
  case DimSpecKind::Max:
    return "max";
  default:
    return "<?>";
  }
}

void DimSpec::appendNodeString(int32_t idx, std::string &out) const {
  if (idx < 0 || idx >= (int32_t)nodes_.size()) {
    out += "<bad>";
    return;
  }
  const DimSpecNode &n = nodes_[idx];
  std::ostringstream oss;
  switch (n.kind) {
  case DimSpecKind::Static:
    oss << n.value;
    out += oss.str();
    break;
  case DimSpecKind::InputDim:
    oss << "arg[" << n.input_index << "].shape[" << n.dim_index << "]";
    out += oss.str();
    break;
  case DimSpecKind::InputValueI64:
    oss << "arg[" << n.input_index << "].i64[" << n.flat_offset << "]";
    out += oss.str();
    break;
  case DimSpecKind::RuntimeSlot:
    oss << "slot[" << n.slot_id << "]";
    out += oss.str();
    break;
  default: {
    out += binaryName(n.kind);
    out += "(";
    appendNodeString(n.lhs, out);
    out += ", ";
    appendNodeString(n.rhs, out);
    out += ")";
    break;
  }
  }
}

std::string DimSpec::toString() const {
  if (nodes_.empty())
    return "<empty>";
  std::string s;
  appendNodeString(0, s);
  return s;
}

bool DimSpec::printArrayAttr(mlir::ArrayAttr attr, llvm::raw_ostream &os) {
  DimSpec ds = parseFromArrayAttr(attr);
  if (ds.nodes_.empty()) {
    return false;
  }
  os << ds.toString();
  return true;
}

//===----------------------------------------------------------------------===//
// DimSpec - verification
//===----------------------------------------------------------------------===//

bool DimSpec::verify(std::string &error) const {
  if (nodes_.empty())
    return true; // empty is "unset" — consumers should treat appropriately
  const int32_t N = (int32_t)nodes_.size();
  for (int32_t i = 0; i < N; ++i) {
    const auto &n = nodes_[i];
    auto isLeaf = [&]() {
      return n.kind == DimSpecKind::Static ||
             n.kind == DimSpecKind::InputDim ||
             n.kind == DimSpecKind::InputValueI64 ||
             n.kind == DimSpecKind::RuntimeSlot;
    };
    if (isLeaf()) {
      if (n.lhs != -1 || n.rhs != -1) {
        std::ostringstream oss;
        oss << "DimSpec node " << i << " (leaf) has child indices";
        error = oss.str();
        return false;
      }
      if (n.kind == DimSpecKind::RuntimeSlot && n.slot_id < 0) {
        std::ostringstream oss;
        oss << "DimSpec node " << i << " RuntimeSlot has invalid slot_id "
            << n.slot_id;
        error = oss.str();
        return false;
      }
      continue;
    }
    if (n.lhs < 0 || n.lhs >= N || n.rhs < 0 || n.rhs >= N ||
        n.lhs == i || n.rhs == i) {
      std::ostringstream oss;
      oss << "DimSpec node " << i << " has invalid child indices (" << n.lhs
          << ", " << n.rhs << ")";
      error = oss.str();
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// shape_interface dispatcher
//===----------------------------------------------------------------------===//

namespace shape_interface {

namespace {
// Sentinel: return empty DimSpec to signal "no information".
DimSpec emptySpec() { return DimSpec(); }

// Decode a serialized [DenseI64ArrayAttr] of per-result-dim DimSpecs from
// an `output_dim_specs` op attribute. The encoding is one ArrayAttr per
// result, each containing one inner ArrayAttr per dim (the DimSpec
// nodes-array). Returns empty spec if missing.
DimSpec getFromOpDimSpecsAttr(mlir::Operation *op, unsigned result_index,
                              unsigned dim_index) {
  auto outer = op->getAttrOfType<mlir::ArrayAttr>("output_dim_specs");
  if (!outer || result_index >= outer.size())
    return emptySpec();
  auto perResult = llvm::dyn_cast<mlir::ArrayAttr>(outer[result_index]);
  if (!perResult || dim_index >= perResult.size())
    return emptySpec();
  auto specAttr = llvm::dyn_cast<mlir::ArrayAttr>(perResult[dim_index]);
  if (!specAttr)
    return emptySpec();
  return DimSpec::parseFromArrayAttr(specAttr);
}

// True if `v` is a block argument of the entry block of a top-level
// func::FuncOp / FunctionOpInterface op. If so, return its arg index.
bool isFuncEntryBlockArg(mlir::Value v, int32_t &arg_index) {
  auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(v);
  if (!blockArg)
    return false;
  mlir::Block *block = blockArg.getOwner();
  mlir::Operation *parent = block->getParentOp();
  auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(parent);
  if (!funcOp)
    return false;
  if (block != &funcOp.getBody().front())
    return false;
  arg_index = (int32_t)blockArg.getArgNumber();
  return true;
}

// Compiler-side mapping: a HIP context arg occupies a slot in the func
// signature but is not a tensor input from the runtime's perspective. The
// EP-visible input index is the count of non-context args before this
// arg. Returns -1 if `arg_index` is a HIP context arg.
int32_t computeEpInputIndex(mlir::func::FuncOp funcOp, int32_t arg_index);
} // namespace

DimSpec getResultDimSpec(mlir::Operation *op, unsigned result_index,
                         unsigned dim_index) {
  if (!op)
    return emptySpec();
  // Per-op `output_dim_specs` attribute wins.
  DimSpec specFromAttr = getFromOpDimSpecsAttr(op, result_index, dim_index);
  if (!specFromAttr.nodes().empty())
    return specFromAttr;
  // Default: copy the static MLIR type dim when known. Guard against the
  // DPS-after-bufferize case: ops like `hip.range` lose their results
  // after `BufferResultsToOutParams` (writes into an out-param memref
  // instead), so `op->getResult(result_index)` would assert. In that
  // case `result_index` is requested by ComposeDimSpecs scanning the
  // out-param users and we have nothing more to add.
  if (result_index >= op->getNumResults())
    return emptySpec();
  mlir::Value result = op->getResult(result_index);
  if (auto rankedType =
          llvm::dyn_cast<mlir::ShapedType>(result.getType())) {
    if (rankedType.hasRank() && dim_index < rankedType.getRank()) {
      int64_t d = rankedType.getDimSize(dim_index);
      if (!mlir::ShapedType::isDynamic(d))
        return DimSpec::makeStatic(d);
    }
  }
  return emptySpec();
}

llvm::SmallVector<PublishedSlot, 1> getPublishedSlots(mlir::Operation *op) {
  llvm::SmallVector<PublishedSlot, 1> out;
  if (!op)
    return out;
  // Convention: an op publishes one or more slots when it carries either a
  // singular `slot_id` (IntegerAttr) or a plural `slot_ids`
  // (DenseI32ArrayAttr) attribute, AND an `output_dim_specs` attribute
  // whose per-(result,dim) entry is a RuntimeSlot referencing one of those
  // slot ids. The plural form is used by multi-dim publishers such as
  // ConstantOfShape (one slot per output axis); the singular form covers
  // single-dim publishers such as NonZero / Range / Size.
  llvm::SmallVector<int32_t, 8> declared_slots;
  if (auto slotAttr = op->getAttrOfType<mlir::IntegerAttr>("slot_id")) {
    int32_t s = static_cast<int32_t>(slotAttr.getInt());
    if (s >= 0)
      declared_slots.push_back(s);
  }
  if (auto arrAttr = op->getAttrOfType<mlir::DenseI32ArrayAttr>("slot_ids")) {
    for (int32_t s : arrAttr.asArrayRef()) {
      if (s >= 0)
        declared_slots.push_back(s);
    }
  }
  if (declared_slots.empty())
    return out;
  auto outer = op->getAttrOfType<mlir::ArrayAttr>("output_dim_specs");
  if (!outer)
    return out;
  llvm::SmallDenseSet<int32_t, 8> declared_set;
  for (int32_t s : declared_slots)
    declared_set.insert(s);
  for (unsigned r = 0; r < outer.size(); ++r) {
    auto perResult = llvm::dyn_cast<mlir::ArrayAttr>(outer[r]);
    if (!perResult)
      continue;
    for (unsigned d = 0; d < perResult.size(); ++d) {
      auto specAttr = llvm::dyn_cast<mlir::ArrayAttr>(perResult[d]);
      if (!specAttr)
        continue;
      DimSpec ds = DimSpec::parseFromArrayAttr(specAttr);
      if (!ds.nodes().empty() && ds.root().kind == DimSpecKind::RuntimeSlot &&
          declared_set.count(ds.root().slot_id)) {
        PublishedSlot p;
        p.slot_id = ds.root().slot_id;
        p.result_index = r;
        p.dim_index = d;
        out.push_back(p);
      }
    }
  }
  return out;
}

namespace {
int32_t computeEpInputIndex(mlir::func::FuncOp funcOp, int32_t arg_index) {
  int32_t epIdx = 0;
  for (int32_t i = 0; i < arg_index; ++i) {
    if (llvm::isa<mlir::hip::ContextType>(
            funcOp.getFunctionType().getInput(i)))
      continue;
    ++epIdx;
  }
  // If arg itself is ContextType, return -1.
  if (llvm::isa<mlir::hip::ContextType>(
          funcOp.getFunctionType().getInput(arg_index)))
    return -1;
  return epIdx;
}
} // namespace

DimSpec resolveDimFromValue(mlir::Value v, unsigned dim_index) {
  if (!v)
    return emptySpec();
  int32_t arg_index = -1;
  if (isFuncEntryBlockArg(v, arg_index)) {
    auto blockArg = llvm::cast<mlir::BlockArgument>(v);
    auto funcOp = llvm::cast<mlir::func::FuncOp>(
        blockArg.getOwner()->getParentOp());
    int32_t epIdx = computeEpInputIndex(funcOp, arg_index);
    if (epIdx < 0)
      return emptySpec(); // ContextType arg
    // If the func arg dim is statically known, prefer Static (cheaper
    // evaluation, no shape table indirection on the host side).
    if (auto rankedType =
            llvm::dyn_cast<mlir::ShapedType>(v.getType())) {
      if (rankedType.hasRank() && dim_index < rankedType.getRank()) {
        int64_t d = rankedType.getDimSize(dim_index);
        if (!mlir::ShapedType::isDynamic(d))
          return DimSpec::makeStatic(d);
      }
    }
    return DimSpec::makeInputDim(epIdx, (int32_t)dim_index);
  }
  // Producer chain: defer to the producing op's getResultDimSpec.
  mlir::Operation *producer = v.getDefiningOp();
  if (!producer)
    return emptySpec();
  unsigned res_idx = 0;
  for (unsigned i = 0; i < producer->getNumResults(); ++i) {
    if (producer->getResult(i) == v) {
      res_idx = i;
      break;
    }
  }
  return getResultDimSpec(producer, res_idx, dim_index);
}

DimSpec resolveValueFromI64Tensor(mlir::Value v, int64_t flat_offset) {
  if (!v)
    return emptySpec();
  // Only handle func entry-block args (the host-readable case). If `v` is
  // a func arg, mark it as InputValueI64 — the EP evaluator will read it
  // from the host-side ORT input tensor.
  int32_t arg_index = -1;
  if (isFuncEntryBlockArg(v, arg_index)) {
    auto blockArg = llvm::cast<mlir::BlockArgument>(v);
    auto funcOp = llvm::cast<mlir::func::FuncOp>(
        blockArg.getOwner()->getParentOp());
    int32_t epIdx = computeEpInputIndex(funcOp, arg_index);
    if (epIdx < 0)
      return emptySpec();
    return DimSpec::makeInputValueI64(epIdx, flat_offset);
  }
  return emptySpec();
}

} // namespace shape_interface

} // namespace hip
} // namespace mlir
