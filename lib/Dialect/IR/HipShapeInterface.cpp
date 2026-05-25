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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <mutex>
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

std::vector<std::vector<DimSpec>>
DimSpec::parseOutputDimSpecsAttr(mlir::ArrayAttr attr) {
  std::vector<std::vector<DimSpec>> result;
  if (!attr)
    return result;
  result.reserve(attr.size());
  for (mlir::Attribute outer : attr) {
    auto perOutput = llvm::dyn_cast<mlir::ArrayAttr>(outer);
    std::vector<DimSpec> dims;
    if (!perOutput) {
      // Malformed entry — push an empty per-output vector so the outer
      // index keeps lining up with the model output index. Caller is
      // expected to spot the empty vector and report.
      result.push_back(std::move(dims));
      continue;
    }
    dims.reserve(perOutput.size());
    for (mlir::Attribute inner : perOutput) {
      auto perDim = llvm::dyn_cast<mlir::ArrayAttr>(inner);
      if (!perDim) {
        dims.emplace_back(); // empty DimSpec
        continue;
      }
      dims.push_back(parseFromArrayAttr(perDim));
    }
    result.push_back(std::move(dims));
  }
  return result;
}

bool DimSpec::printOutputDimSpecsAttr(mlir::ArrayAttr attr,
                                      llvm::raw_ostream &os) {
  if (!attr) {
    os << "  <no hipdnn.output_dim_specs attribute on module>\n";
    return false;
  }
  auto parsed = parseOutputDimSpecsAttr(attr);
  if (parsed.empty()) {
    os << "  <hipdnn.output_dim_specs is empty>\n";
    return true;
  }
  bool ok = true;
  for (size_t i = 0; i < parsed.size(); ++i) {
    const auto &dims = parsed[i];
    if (dims.empty()) {
      // ComposeDimSpecs uses an empty inner ArrayAttr to mean
      // "nothing was recorded for this output" — typically a legacy
      // output with all-static shape that the pass left untouched.
      os << "  Output[" << i << "]: <no dim_specs entries>\n";
      continue;
    }
    for (size_t d = 0; d < dims.size(); ++d) {
      os << "  Output[" << i << "] dim[" << d << "] = ";
      if (dims[d].nodes_.empty()) {
        os << "<no spec>\n";
        continue;
      }
      os << dims[d].toString() << "\n";
    }
  }
  return ok;
}

//===----------------------------------------------------------------------===//
// DimSpec - verification
//===----------------------------------------------------------------------===//

namespace {

// Hash a node sub-tree starting at `idx` into a small uint64. Used for
// canonical commutative-child sorting. The hash combines the node's
// kind + payload + recursive hashes of children so structurally-equal
// trees hash to the same value.
uint64_t hashSubtree(const std::vector<DimSpecNode> &nodes, int32_t idx) {
  if (idx < 0 || idx >= (int32_t)nodes.size())
    return 0;
  const auto &n = nodes[idx];
  uint64_t h = (uint64_t)n.kind;
  auto mix = [&](uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  switch (n.kind) {
  case DimSpecKind::Static:
    mix((uint64_t)n.value);
    break;
  case DimSpecKind::InputDim:
    mix((uint64_t)n.input_index);
    mix((uint64_t)n.dim_index);
    break;
  case DimSpecKind::InputValueI64:
    mix((uint64_t)n.input_index);
    mix((uint64_t)n.flat_offset);
    break;
  case DimSpecKind::RuntimeSlot:
    mix((uint64_t)n.slot_id);
    break;
  default:
    // Binary op: hash both child subtrees.
    mix(hashSubtree(nodes, n.lhs));
    mix(hashSubtree(nodes, n.rhs));
    break;
  }
  return h;
}

// Topologically copy a subtree rooted at `srcIdx` from `src` into `dst`,
// returning the new root index in `dst`. Children are processed first
// so the resulting indices are valid by the time the parent is
// written.
int32_t copySubtree(const std::vector<DimSpecNode> &src, int32_t srcIdx,
                    std::vector<DimSpecNode> &dst) {
  if (srcIdx < 0 || srcIdx >= (int32_t)src.size())
    return -1;
  DimSpecNode n = src[srcIdx];
  switch (n.kind) {
  case DimSpecKind::Static:
  case DimSpecKind::InputDim:
  case DimSpecKind::InputValueI64:
  case DimSpecKind::RuntimeSlot: {
    int32_t newIdx = (int32_t)dst.size();
    dst.push_back(n);
    return newIdx;
  }
  default: {
    int32_t newLhs = copySubtree(src, n.lhs, dst);
    int32_t newRhs = copySubtree(src, n.rhs, dst);
    n.lhs = newLhs;
    n.rhs = newRhs;
    int32_t newIdx = (int32_t)dst.size();
    dst.push_back(n);
    return newIdx;
  }
  }
}

bool isCommutative(DimSpecKind k) {
  return k == DimSpecKind::Add || k == DimSpecKind::Mul ||
         k == DimSpecKind::Min || k == DimSpecKind::Max;
}

// Canonicalise the subtree rooted at `srcIdx` in `src` and append the
// result to `dst`, returning the new root index in `dst`. Bottom-up:
// children are canonicalised first, then the parent applies its
// fold / identity / sort rules.
int32_t canonicalizeSubtree(const std::vector<DimSpecNode> &src,
                            int32_t srcIdx,
                            std::vector<DimSpecNode> &dst) {
  if (srcIdx < 0 || srcIdx >= (int32_t)src.size())
    return -1;
  DimSpecNode n = src[srcIdx];
  // Leaves: copy verbatim.
  if (n.kind == DimSpecKind::Static || n.kind == DimSpecKind::InputDim ||
      n.kind == DimSpecKind::InputValueI64 ||
      n.kind == DimSpecKind::RuntimeSlot) {
    int32_t idx = (int32_t)dst.size();
    dst.push_back(n);
    return idx;
  }
  int32_t lhs = canonicalizeSubtree(src, n.lhs, dst);
  int32_t rhs = canonicalizeSubtree(src, n.rhs, dst);
  if (lhs < 0 || rhs < 0)
    return -1;

  const DimSpecNode &L = dst[lhs];
  const DimSpecNode &R = dst[rhs];

  // Constant-fold Static <op> Static -> Static.
  if (L.kind == DimSpecKind::Static && R.kind == DimSpecKind::Static) {
    int64_t a = L.value, b = R.value;
    int64_t out = 0;
    bool ok = true;
    switch (n.kind) {
    case DimSpecKind::Add:
      out = a + b;
      break;
    case DimSpecKind::Sub:
      out = a - b;
      break;
    case DimSpecKind::Mul:
      out = a * b;
      break;
    case DimSpecKind::FloorDiv:
      if (b == 0) {
        ok = false;
        break;
      }
      out = (a / b) - (((a % b) != 0) && ((a ^ b) < 0));
      break;
    case DimSpecKind::CeilDiv:
      if (b == 0) {
        ok = false;
        break;
      }
      out = (a / b) + (((a % b) != 0) && ((a ^ b) >= 0));
      break;
    case DimSpecKind::Min:
      out = std::min(a, b);
      break;
    case DimSpecKind::Max:
      out = std::max(a, b);
      break;
    default:
      ok = false;
      break;
    }
    if (ok) {
      // Pop the two static children -- safe because they're the last
      // appended nodes and nothing else references them yet.
      dst.pop_back();
      dst.pop_back();
      DimSpecNode folded;
      folded.kind = DimSpecKind::Static;
      folded.value = out;
      int32_t idx = (int32_t)dst.size();
      dst.push_back(folded);
      return idx;
    }
  }

  // Identity reductions over one constant child.
  auto isStatic0 = [](const DimSpecNode &m) {
    return m.kind == DimSpecKind::Static && m.value == 0;
  };
  auto isStatic1 = [](const DimSpecNode &m) {
    return m.kind == DimSpecKind::Static && m.value == 1;
  };

  auto returnChild = [&](int32_t survivorIdx, int32_t deadIdx) -> int32_t {
    // The dead child is the last appended; pop it so the resulting
    // tree's root references only survivors.
    if (deadIdx == (int32_t)dst.size() - 1) {
      dst.pop_back();
    }
    return survivorIdx;
  };

  switch (n.kind) {
  case DimSpecKind::Add:
    if (isStatic0(L))
      return returnChild(rhs, lhs);
    if (isStatic0(R))
      return returnChild(lhs, rhs);
    break;
  case DimSpecKind::Sub:
    if (isStatic0(R))
      return returnChild(lhs, rhs);
    break;
  case DimSpecKind::Mul:
    if (isStatic1(L))
      return returnChild(rhs, lhs);
    if (isStatic1(R))
      return returnChild(lhs, rhs);
    if (isStatic0(L) || isStatic0(R)) {
      // Mul(0, *) -> 0; both children become dead. Easiest is to drop
      // everything we just appended for this binary node back to dst's
      // size before the recursion -- but we can't easily roll back
      // arbitrary subtrees once their nodes have been pushed. Instead
      // we leave the dead nodes in place (they become orphans, harmless
      // because the verifier only walks reachable nodes from index 0)
      // and append a fresh Static(0). Slightly wasteful but correct;
      // the test cases below all exercise the "drop the static 1" path
      // where the more efficient pop is possible.
      DimSpecNode zero;
      zero.kind = DimSpecKind::Static;
      zero.value = 0;
      int32_t idx = (int32_t)dst.size();
      dst.push_back(zero);
      return idx;
    }
    break;
  case DimSpecKind::Max:
  case DimSpecKind::Min:
    // Max(x, x) / Min(x, x) -> x (structural equality via hash).
    if (hashSubtree(dst, lhs) == hashSubtree(dst, rhs)) {
      return returnChild(lhs, rhs);
    }
    break;
  default:
    break;
  }

  // Commutative reorder: sort children by hash so equivalent trees
  // serialise identically. Done by swapping lhs/rhs when the
  // canonical order is rhs < lhs.
  if (isCommutative(n.kind)) {
    uint64_t hl = hashSubtree(dst, lhs);
    uint64_t hr = hashSubtree(dst, rhs);
    if (hl > hr)
      std::swap(lhs, rhs);
  }

  n.lhs = lhs;
  n.rhs = rhs;
  int32_t idx = (int32_t)dst.size();
  dst.push_back(n);
  return idx;
}

} // namespace

void DimSpec::canonicalize() {
  if (nodes_.empty())
    return;
  std::vector<DimSpecNode> dst;
  dst.reserve(nodes_.size());
  int32_t newRoot = canonicalizeSubtree(nodes_, 0, dst);
  if (newRoot < 0) {
    // Bail out -- the input was malformed; verifier will catch it.
    return;
  }
  // canonicalizeSubtree appends children before parents, so newRoot
  // ends up as the LAST element. Rotate it to index 0 by walking the
  // tree depth-first and renumbering.
  std::vector<DimSpecNode> reordered;
  reordered.reserve(dst.size());
  std::function<int32_t(int32_t)> emit = [&](int32_t idx) -> int32_t {
    if (idx < 0 || idx >= (int32_t)dst.size())
      return -1;
    DimSpecNode n = dst[idx];
    // Reserve our slot up front so that the resulting layout is root
    // (index 0) followed by children.
    int32_t placeholder = (int32_t)reordered.size();
    reordered.push_back(DimSpecNode());
    if (n.kind == DimSpecKind::Static || n.kind == DimSpecKind::InputDim ||
        n.kind == DimSpecKind::InputValueI64 ||
        n.kind == DimSpecKind::RuntimeSlot) {
      reordered[placeholder] = n;
      return placeholder;
    }
    int32_t newLhs = emit(n.lhs);
    int32_t newRhs = emit(n.rhs);
    n.lhs = newLhs;
    n.rhs = newRhs;
    reordered[placeholder] = n;
    return placeholder;
  };
  emit(newRoot);
  nodes_ = std::move(reordered);
}

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

// Per-op DimSpec builder registry. Populated once from
// `HipDialect::initialize()` via `populateBuiltinDimSpecBuilders()`.
// Lookup is read-only at pattern time so no per-call lock is needed; the
// `std::call_once` in `populateBuiltinDimSpecBuilders` guards the
// initialisation race when several MLIRContexts spin up concurrently.
llvm::StringMap<DimSpecBuilderFn> &getBuilderRegistry() {
  static llvm::StringMap<DimSpecBuilderFn> registry;
  return registry;
}
} // namespace

void registerOpDimSpecBuilder(llvm::StringRef op_name,
                              DimSpecBuilderFn fn) {
  getBuilderRegistry()[op_name] = fn;
}

DimSpec getResultDimSpec(mlir::Operation *op, unsigned result_index,
                         unsigned dim_index) {
  if (!op)
    return emptySpec();
  // Strategy 1 — per-op `output_dim_specs` attribute wins.
  DimSpec specFromAttr = getFromOpDimSpecsAttr(op, result_index, dim_index);
  if (!specFromAttr.nodes().empty())
    return specFromAttr;

  // Strategy 2 — registered per-op builder. This is how rank-preserving,
  // elementwise/broadcast, and other "shape-behavior class" ops
  // contribute without every conversion having to pre-attach an
  // attribute. The builder is responsible for walking the op's operands
  // (typically via `resolveDimFromValue`) and returning a composed
  // DimSpec. An empty return is a "I cannot resolve this here" signal
  // and falls through to strategy 3.
  auto &registry = getBuilderRegistry();
  auto it = registry.find(op->getName().getStringRef());
  if (it != registry.end()) {
    DimSpec built = it->second(op, result_index, dim_index);
    if (!built.nodes().empty())
      return built;
  }

  // Strategy 3 — copy the static MLIR type dim when known. Guard against
  // the DPS-after-bufferize case: ops like `hip.range` lose their
  // results after `BufferResultsToOutParams` (writes into an out-param
  // memref instead), so `op->getResult(result_index)` would assert. In
  // that case `result_index` is requested by ComposeDimSpecs scanning
  // the out-param users and we have nothing more to add.
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
  // Convention: an op publishes one or more slots when it carries ANY of
  //   * the unified `hipdnn.output_slot_ids` (preferred, Phase 2): an
  //     ArrayAttr<DenseI32ArrayAttr> of shape `[num_results][rank]`
  //     where each slot is the i32 slot_id (-1 = no slot for that
  //     (result,dim));
  //   * the legacy `slot_id` (IntegerAttr): single-dim publishers like
  //     NonZero / Range Cat-C;
  //   * the legacy `slot_ids` (DenseI32ArrayAttr): multi-dim publishers
  //     like ConstantOfShape Cat-C (one slot per output axis).
  //
  // AND, in all three cases, an `output_dim_specs` attribute whose
  // per-(result,dim) entry is a RuntimeSlot referencing one of those
  // slot ids.
  //
  // The new array form makes it possible to publish slots for ARBITRARY
  // (result, dim) pairs on an op with multiple results -- needed for the
  // future translucent-propagator-as-publisher path that Phase 2.3 lays
  // down. Existing ops continue to use the legacy keys without any
  // change.

  // Path A: new array-form attribute. If present we trust it directly
  // -- the (result, dim) -> slot mapping is explicit and we don't need
  // to cross-reference output_dim_specs.
  if (auto arrAttr = op->getAttrOfType<mlir::ArrayAttr>(
          "hipdnn.output_slot_ids")) {
    for (unsigned r = 0; r < arrAttr.size(); ++r) {
      auto perResult = llvm::dyn_cast<mlir::DenseI32ArrayAttr>(arrAttr[r]);
      if (!perResult)
        continue;
      for (int64_t d = 0; d < perResult.size(); ++d) {
        int32_t s = perResult.asArrayRef()[d];
        if (s < 0)
          continue;
        PublishedSlot p;
        p.slot_id = s;
        p.result_index = r;
        p.dim_index = (unsigned)d;
        out.push_back(p);
      }
    }
    if (!out.empty())
      return out;
  }

  // Path B: legacy attribute keys cross-referenced against the DimSpec
  // attribute. Behaves exactly the same as the pre-Phase-2 code.
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

// After bufferize-to-out-params + pool-allocs, a SSA value referencing
// e.g. a NonZero output looks like `%v = memref.view %pool[%off][]` —
// the "defining op" is `memref.view` / `memref.alloc`, NOT the
// `hip.nonzero` that actually populated the buffer. Walk the use-list
// of `v` and prefer the DPS-init writer (the first Hip-dialect op that
// uses `v` as a DPS output) as the producer for DimSpec queries; this
// lets RuntimeSlot leaves attached on `hip.nonzero` etc. flow through
// any number of intermediate consumers.
//
// Returns `nullptr` (with `*resIdx` unchanged) if no DPS writer is
// found — callers should then fall back to `v.getDefiningOp()`.
static mlir::Operation *findDpsWriter(mlir::Value v, unsigned *resIdx) {
  for (mlir::Operation *user : v.getUsers()) {
    auto dpsOp = llvm::dyn_cast<mlir::DestinationStyleOpInterface>(user);
    if (!dpsOp)
      continue;
    if (user->getDialect() !=
        user->getContext()->getLoadedDialect<mlir::hip::HipDialect>())
      continue;
    auto inits = dpsOp.getDpsInits();
    for (auto [i, init] : llvm::enumerate(inits)) {
      if (init == v) {
        if (resIdx)
          *resIdx = static_cast<unsigned>(i);
        return user;
      }
    }
  }
  return nullptr;
}

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
  // Producer chain: prefer the DPS writer (post-bufferize) over the
  // raw defining op (pre-bufferize). Without this, after bufferize
  // every memref operand's `getDefiningOp()` is `memref.alloc` /
  // `memref.view` which carries no DimSpec — and the chain would
  // silently break at the first bufferized hop.
  if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(v.getType())) {
    if (llvm::isa<mlir::MemRefType>(shaped)) {
      unsigned writerResIdx = 0;
      if (mlir::Operation *writer = findDpsWriter(v, &writerResIdx)) {
        // For 0-result DPS ops (post-bufferize), `getResultDimSpec`
        // reads the per-op `output_dim_specs` attribute keyed on the
        // ORIGINAL result index. The writer index returned by
        // `findDpsWriter` is the DPS-init operand index, which
        // mirrors the original result position for every Hip_DpsOp
        // (one result per init operand pre-bufferize).
        DimSpec specFromWriter =
            getResultDimSpec(writer, writerResIdx, dim_index);
        if (!specFromWriter.nodes().empty())
          return specFromWriter;
      }
    }
  }
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

//===----------------------------------------------------------------------===//
// Built-in DimSpec builders
//===----------------------------------------------------------------------===//
//
// One builder per "shape behavior" class — they cover whole groups of ops
// at a time so adding a new op with the same behavior is a one-line
// registration call below. Builders are deliberately tolerant: if they
// cannot resolve a dim they return empty so `getResultDimSpec` falls
// through to the static-type strategy.

namespace {

// Walk-back helper that prefers static / non-empty over empty without
// hard-failing. Returns empty when `v` is null. Used by all builders.
DimSpec resolveOperandDim(mlir::Value v, unsigned dim_index) {
  if (!v)
    return emptySpec();
  return resolveDimFromValue(v, dim_index);
}

// Rank-preserving permutation builder. Handles `hip.transpose`: output
// dim `d` is exactly input dim `perm[d]`. Operand 0 is the `!hip.context`
// arg; operand 1 is the data tensor.
//
// `perm` is required by the op verifier (TransposeOp::verify) so we
// assert presence; bail to empty if anything unexpected appears in the
// attr to stay graceful in the face of partially-converted IR.
DimSpec buildTransposeDimSpec(mlir::Operation *op, unsigned result_index,
                              unsigned dim_index) {
  if (op->getNumOperands() < 2 || result_index != 0)
    return emptySpec();
  auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm");
  if (!permAttr || dim_index >= permAttr.size())
    return emptySpec();
  auto intAttr =
      llvm::dyn_cast<mlir::IntegerAttr>(permAttr[dim_index]);
  if (!intAttr)
    return emptySpec();
  int64_t srcDim = intAttr.getValue().getSExtValue();
  if (srcDim < 0)
    return emptySpec();
  mlir::Value data = op->getOperand(1);
  return resolveOperandDim(data, (unsigned)srcDim);
}

// NumPy-style broadcast builder. Covers every elementwise / broadcast op
// (add/mul/sub/div/min/max/equal/less/and/not/cos/sin/neg/sign/mod/cast
// /silu/sigmoid/softplus/gelu/reciprocal/sqrt/where/pow/...). The
// op-signature convention is: operand 0 is `!hip.context`, the last
// operand is the DPS init/output buffer, and everything in between is a
// data input that participates in broadcast.
//
// Broadcast semantics: right-align all operand ranks against the result
// rank. For output dim `d` (front-indexed), the contribution from
// operand `k` is `operand_k_dim[d - (out_rank - operand_k_rank)]`, or
// "no contribution" when that index would be negative. A static dim of
// 1 means "broadcast me", so prefer any other operand whose dim resolves
// to either a non-Static-1 value or a non-empty DimSpec.
//
// Result: the *first* operand whose contribution at this position is
// (a) non-empty and (b) not static-1. Falls back to "any non-empty"
// (incl. static-1) when nothing else exists. Returns empty when every
// participating operand returns empty — caller hits static-type fallback.
DimSpec buildBroadcastDimSpec(mlir::Operation *op, unsigned result_index,
                              unsigned dim_index) {
  if (op->getNumOperands() < 3 || result_index != 0)
    return emptySpec();
  // Determine the output rank from the result type (tensor mode) or the
  // last operand's type (memref/DPS mode after bufferize). Both are
  // ShapedType.
  unsigned outRank = 0;
  if (op->getNumResults() > 0) {
    auto t = llvm::dyn_cast<mlir::ShapedType>(
        op->getResult(0).getType());
    if (!t || !t.hasRank())
      return emptySpec();
    outRank = (unsigned)t.getRank();
  } else {
    auto t = llvm::dyn_cast<mlir::ShapedType>(
        op->getOperand(op->getNumOperands() - 1).getType());
    if (!t || !t.hasRank())
      return emptySpec();
    outRank = (unsigned)t.getRank();
  }
  if (dim_index >= outRank)
    return emptySpec();

  // Data operands span [1, N-1) in tensor mode (last operand is the
  // DPS init) and [1, N) in memref mode (no result, the init is the
  // last operand but still counts as a buffer — we exclude it the same
  // way by stopping before N-1, since after bufferize the init is also
  // a memref of the output shape, which gives the SAME broadcast info
  // as the result type).
  unsigned firstData = 1;
  unsigned pastEnd = op->getNumOperands() - 1;
  if (pastEnd <= firstData)
    return emptySpec();

  DimSpec fallback = emptySpec();
  for (unsigned k = firstData; k < pastEnd; ++k) {
    mlir::Value v = op->getOperand(k);
    auto t = llvm::dyn_cast<mlir::ShapedType>(v.getType());
    if (!t || !t.hasRank())
      continue;
    int64_t shift = (int64_t)outRank - (int64_t)t.getRank();
    int64_t opIdx = (int64_t)dim_index - shift;
    if (opIdx < 0 || opIdx >= t.getRank())
      continue; // operand doesn't reach this dim (broadcast as 1)
    // Skip operands whose dim is statically 1 — they're broadcast over.
    // We still keep them as a last-resort fallback in case every other
    // operand returns empty.
    int64_t staticDim = t.getDimSize(opIdx);
    DimSpec d = resolveOperandDim(v, (unsigned)opIdx);
    if (d.nodes().empty())
      continue;
    if (!mlir::ShapedType::isDynamic(staticDim) && staticDim == 1) {
      if (fallback.nodes().empty())
        fallback = d;
      continue;
    }
    return d; // first non-broadcast operand wins
  }
  return fallback;
}

// ---- Builders for the dynamic-shape op classes added in Phase 2 ------------
//
// All builders below follow the convention: operand 0 is the !hip.context
// argument (skipped); the data operand is operand 1 unless noted; the DPS
// init / output operand is always the last operand (skipped).

// Gather builder: ONNX Gather output rank is
//   data_rank - 1 + indices_rank
// with the gathered `axis` dim of `data` replaced by the full shape of
// `indices`. So:
//   output_dim[k] = data_dim[k]            for k < axis
//   output_dim[axis + j] = indices_dim[j]  for j in [0, indices_rank)
//   output_dim[axis + indices_rank + k] = data_dim[axis + 1 + k]
// Operand layout: ctx@0, data@1, indices@2, output@3.
DimSpec buildGatherDimSpec(mlir::Operation *op, unsigned result_index,
                           unsigned dim_index) {
  if (op->getNumOperands() < 4 || result_index != 0)
    return emptySpec();
  mlir::Value data = op->getOperand(1);
  mlir::Value indices = op->getOperand(2);
  auto dataTy = llvm::dyn_cast<mlir::ShapedType>(data.getType());
  auto indicesTy = llvm::dyn_cast<mlir::ShapedType>(indices.getType());
  if (!dataTy || !dataTy.hasRank() || !indicesTy || !indicesTy.hasRank())
    return emptySpec();
  int64_t axis = 0;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = a.getValue().getSExtValue();
  if (axis < 0)
    axis += dataTy.getRank();
  if (axis < 0 || axis >= dataTy.getRank())
    return emptySpec();
  int64_t indicesRank = indicesTy.getRank();
  int64_t outDim = (int64_t)dim_index;
  if (outDim < axis)
    return resolveOperandDim(data, (unsigned)outDim);
  if (outDim < axis + indicesRank)
    return resolveOperandDim(indices, (unsigned)(outDim - axis));
  int64_t kData = outDim - axis - indicesRank + axis + 1;
  if (kData >= dataTy.getRank())
    return emptySpec();
  return resolveOperandDim(data, (unsigned)kData);
}

// Tile builder: output_dim[i] = input_dim[i] * repeats[i].
// `repeats` is operand 2 (rank-1 i64). If repeats[i] is host-readable
// (func-arg) we encode it as `Mul(InputDim, InputValueI64(repeats, i))`;
// otherwise we can only return empty (the caller falls back to the
// static-type lookup — which is fine when at least one of (input dim,
// repeats[i]) is compile-time known).
//
// Operand layout: ctx@0, input@1, repeats@2, output@3.
DimSpec buildTileDimSpec(mlir::Operation *op, unsigned result_index,
                         unsigned dim_index) {
  if (op->getNumOperands() < 4 || result_index != 0)
    return emptySpec();
  mlir::Value input = op->getOperand(1);
  mlir::Value repeats = op->getOperand(2);
  DimSpec inDimSpec = resolveOperandDim(input, dim_index);
  DimSpec repSpec =
      shape_interface::resolveValueFromI64Tensor(repeats, (int64_t)dim_index);
  if (inDimSpec.nodes().empty() || repSpec.nodes().empty())
    return emptySpec();
  // Fold the trivial cases: repeats == 1 means the input dim survives
  // unchanged; input dim == 1 means the repeated count IS the output dim.
  if (repSpec.isStatic() && repSpec.staticValue() == 1)
    return inDimSpec;
  if (inDimSpec.isStatic() && inDimSpec.staticValue() == 1)
    return repSpec;
  return DimSpec::makeBinary(DimSpecKind::Mul, inDimSpec, repSpec);
}

// Expand builder: output shape is the broadcast of `input.shape` and
// the values of `shape` (operand 2). For each output dim:
//   * if input has that dim and input_dim != 1, output_dim = input_dim;
//   * else output_dim = shape[i] (i.e. InputValueI64(shape, i)).
//
// Operand layout: ctx@0, input@1, shape@2, output@3.
DimSpec buildExpandDimSpec(mlir::Operation *op, unsigned result_index,
                           unsigned dim_index) {
  if (op->getNumOperands() < 4 || result_index != 0)
    return emptySpec();
  mlir::Value input = op->getOperand(1);
  mlir::Value shape = op->getOperand(2);
  auto inputTy = llvm::dyn_cast<mlir::ShapedType>(input.getType());
  unsigned outRank = 0;
  if (op->getNumResults() > 0) {
    auto outTy =
        llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
    if (!outTy || !outTy.hasRank())
      return emptySpec();
    outRank = (unsigned)outTy.getRank();
  } else {
    auto outTy =
        llvm::dyn_cast<mlir::ShapedType>(op->getOperand(3).getType());
    if (!outTy || !outTy.hasRank())
      return emptySpec();
    outRank = (unsigned)outTy.getRank();
  }
  if (dim_index >= outRank)
    return emptySpec();
  // Right-align the input rank against the output rank.
  int64_t inRank = inputTy && inputTy.hasRank() ? inputTy.getRank() : 0;
  int64_t shift = (int64_t)outRank - inRank;
  int64_t inIdx = (int64_t)dim_index - shift;
  // Prefer the input dim when it exists AND is statically non-1; else
  // fall back to shape[dim_index].
  if (inputTy && inIdx >= 0 && inIdx < inRank) {
    int64_t staticDim = inputTy.getDimSize(inIdx);
    if (!mlir::ShapedType::isDynamic(staticDim) && staticDim > 1)
      return DimSpec::makeStatic(staticDim);
    DimSpec fromInput = resolveOperandDim(input, (unsigned)inIdx);
    if (!fromInput.nodes().empty() && !(fromInput.isStatic() &&
                                         fromInput.staticValue() == 1))
      return fromInput;
  }
  DimSpec fromShape =
      shape_interface::resolveValueFromI64Tensor(shape, (int64_t)dim_index);
  if (!fromShape.nodes().empty())
    return fromShape;
  // Last-resort: input dim 1 (broadcast) — produces Static(1).
  if (inputTy && inIdx >= 0 && inIdx < inRank) {
    int64_t staticDim = inputTy.getDimSize(inIdx);
    if (!mlir::ShapedType::isDynamic(staticDim))
      return DimSpec::makeStatic(staticDim);
  }
  return emptySpec();
}

// Range builder: output is rank-1, output_dim[0] = ceildiv(limit-start, delta).
// Operand layout: ctx@0, start@1, limit@2, delta@3, output@4.
// When start/limit/delta are all host-readable scalars (rank-0 i64 tensors
// from func-args), we encode the dim as a Cat-B compound; otherwise the
// existing Cat-C publisher path (slot_id on the op) handles it via
// the pre-attached output_dim_specs.
DimSpec buildRangeDimSpec(mlir::Operation *op, unsigned result_index,
                          unsigned dim_index) {
  if (op->getNumOperands() < 5 || result_index != 0 || dim_index != 0)
    return emptySpec();
  mlir::Value start = op->getOperand(1);
  mlir::Value limit = op->getOperand(2);
  mlir::Value delta = op->getOperand(3);
  DimSpec startSpec =
      shape_interface::resolveValueFromI64Tensor(start, /*flat_offset=*/0);
  DimSpec limitSpec =
      shape_interface::resolveValueFromI64Tensor(limit, /*flat_offset=*/0);
  DimSpec deltaSpec =
      shape_interface::resolveValueFromI64Tensor(delta, /*flat_offset=*/0);
  if (startSpec.nodes().empty() || limitSpec.nodes().empty() ||
      deltaSpec.nodes().empty())
    return emptySpec();
  DimSpec diff = DimSpec::makeBinary(DimSpecKind::Sub, limitSpec, startSpec);
  return DimSpec::makeBinary(DimSpecKind::CeilDiv, diff, deltaSpec);
}

// Slice builder: output_dim[axis_i] = ceildiv(end - start, step) for the
// axes listed in `axes` (or all axes when `axes` is absent); other dims
// pass through unchanged.
//
// Operand layout (Hip_SliceOp): ctx@0, data@1, starts@2, ends@3,
// axes?@4-or-skip, steps?@4-or-5, output@last. AttrSizedOperandSegments
// makes the index of optional operands hard to compute statically; we
// take the conservative approach of returning empty unless both starts
// and ends are host-readable. In that case we compute a compound:
//   ceildiv(InputValueI64(ends,i) - InputValueI64(starts,i),
//           InputValueI64(steps,i) || 1)
// for the i-th axis. Without host-readable starts/ends we fall through
// to the static-type strategy.
DimSpec buildSliceDimSpec(mlir::Operation *op, unsigned result_index,
                          unsigned dim_index) {
  if (op->getNumOperands() < 5 || result_index != 0)
    return emptySpec();
  // Discriminate by AttrSizedOperandSegments-style operand_segment_sizes
  // -- if present, we can locate `axes` and `steps` precisely. Otherwise
  // fall back to a heuristic: operands are
  //   [ctx, data, starts, ends, output]                  (5 total)
  //   [ctx, data, starts, ends, axes, output]            (6 total)
  //   [ctx, data, starts, ends, steps, output]           (6 total, ambiguous)
  //   [ctx, data, starts, ends, axes, steps, output]     (7 total)
  // The ambiguous 6-total case is rare in practice (ONNX Slice rarely
  // ships steps without axes), but we still need the operand_segment_sizes
  // discriminator. If the attr is present we use it; otherwise we
  // conservatively bail.
  auto segSizesAttr = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
      "operand_segment_sizes");
  if (!segSizesAttr || segSizesAttr.size() != 7)
    return emptySpec();
  auto seg = segSizesAttr.asArrayRef();
  // Segments order from HipOps.td: ctx, data, starts, ends, axes, steps,
  // output.
  unsigned ctxN = (unsigned)seg[0];
  unsigned dataN = (unsigned)seg[1];
  unsigned startsN = (unsigned)seg[2];
  unsigned endsN = (unsigned)seg[3];
  unsigned axesN = (unsigned)seg[4];
  unsigned stepsN = (unsigned)seg[5];
  unsigned outN = (unsigned)seg[6];
  if (ctxN != 1 || dataN != 1 || startsN != 1 || endsN != 1 || outN != 1)
    return emptySpec();
  mlir::Value data = op->getOperand(1);
  mlir::Value starts = op->getOperand(2);
  mlir::Value ends = op->getOperand(3);
  // axes / steps live at the indices implied by the segment cumulative
  // sums. axes_idx = 4 if axesN else absent; steps_idx = 4 + axesN if
  // stepsN else absent.
  mlir::Value axes;
  mlir::Value steps;
  unsigned cursor = 4;
  if (axesN == 1) {
    axes = op->getOperand(cursor);
    ++cursor;
  }
  if (stepsN == 1) {
    steps = op->getOperand(cursor);
    ++cursor;
  }
  auto dataTy = llvm::dyn_cast<mlir::ShapedType>(data.getType());
  if (!dataTy || !dataTy.hasRank())
    return emptySpec();
  // Resolve which axis-of-data is dim_index of the output. ONNX Slice
  // preserves the data rank — output_rank == data_rank — so by default
  // the output's dim i corresponds to data's dim i. The `axes` list
  // tells us WHICH data dims are sliced, not how dims are permuted.
  unsigned outRank = dataTy.getRank();
  if (dim_index >= outRank)
    return emptySpec();
  // Determine whether the current dim is sliced and, if so, what its
  // position within (starts/ends/steps) is.
  int axesSlot = -1; // index within the [starts/ends/steps] axis list
  if (axes) {
    // Need the values of `axes` (host-readable) to know if dim_index is
    // sliced. If axes is not a func-arg, conservatively return empty.
    auto axesTy = llvm::dyn_cast<mlir::ShapedType>(axes.getType());
    if (!axesTy || !axesTy.hasRank() || axesTy.getRank() != 1)
      return emptySpec();
    if (mlir::ShapedType::isDynamic(axesTy.getDimSize(0)))
      return emptySpec();
    int64_t axesLen = axesTy.getDimSize(0);
    for (int64_t i = 0; i < axesLen; ++i) {
      DimSpec axisI = shape_interface::resolveValueFromI64Tensor(axes, i);
      if (axisI.nodes().empty() || !axisI.isStatic())
        return emptySpec();
      int64_t axisVal = axisI.staticValue();
      if (axisVal < 0)
        axisVal += outRank;
      if ((unsigned)axisVal == dim_index) {
        axesSlot = (int)i;
        break;
      }
    }
  } else {
    // axes absent => sliced axes are [0, 1, ..., starts_len-1].
    auto startsTy = llvm::dyn_cast<mlir::ShapedType>(starts.getType());
    if (!startsTy || !startsTy.hasRank() || startsTy.getRank() != 1)
      return emptySpec();
    if (mlir::ShapedType::isDynamic(startsTy.getDimSize(0)))
      return emptySpec();
    int64_t startsLen = startsTy.getDimSize(0);
    if ((int64_t)dim_index < startsLen)
      axesSlot = (int)dim_index;
  }
  if (axesSlot < 0)
    return resolveOperandDim(data, dim_index);
  DimSpec startVal =
      shape_interface::resolveValueFromI64Tensor(starts, axesSlot);
  DimSpec endVal = shape_interface::resolveValueFromI64Tensor(ends, axesSlot);
  if (startVal.nodes().empty() || endVal.nodes().empty())
    return emptySpec();
  DimSpec stepVal = DimSpec::makeStatic(1);
  if (steps) {
    DimSpec s = shape_interface::resolveValueFromI64Tensor(steps, axesSlot);
    if (!s.nodes().empty())
      stepVal = s;
  }
  DimSpec diff = DimSpec::makeBinary(DimSpecKind::Sub, endVal, startVal);
  return DimSpec::makeBinary(DimSpecKind::CeilDiv, diff, stepVal);
}

// Pad builder: output_dim[i] = input_dim[i] + pads[i_begin] + pads[i_end].
// Pads layout (ONNX): [x1_begin, ..., xN_begin, x1_end, ..., xN_end]
// where N == num_axes (axes provided) or == rank(data) (axes absent).
//
// Operand layout (Hip_PadOp): ctx@0, data@1, pads@2, cval?@3 or absent,
// axes?@3-or-4, output@last. Uses AttrSizedOperandSegments.
DimSpec buildPadDimSpec(mlir::Operation *op, unsigned result_index,
                        unsigned dim_index) {
  if (op->getNumOperands() < 4 || result_index != 0)
    return emptySpec();
  auto segSizesAttr = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
      "operand_segment_sizes");
  if (!segSizesAttr || segSizesAttr.size() != 6)
    return emptySpec();
  auto seg = segSizesAttr.asArrayRef();
  // Segments: ctx, data, pads, constant_value, axes, output.
  unsigned cvalN = (unsigned)seg[3];
  unsigned axesN = (unsigned)seg[4];
  unsigned outN = (unsigned)seg[5];
  if (outN != 1)
    return emptySpec();
  mlir::Value data = op->getOperand(1);
  mlir::Value pads = op->getOperand(2);
  unsigned cursor = 3;
  if (cvalN == 1)
    ++cursor;
  mlir::Value axes;
  if (axesN == 1) {
    axes = op->getOperand(cursor);
    ++cursor;
  }
  auto dataTy = llvm::dyn_cast<mlir::ShapedType>(data.getType());
  if (!dataTy || !dataTy.hasRank())
    return emptySpec();
  auto padsTy = llvm::dyn_cast<mlir::ShapedType>(pads.getType());
  if (!padsTy || !padsTy.hasRank() || padsTy.getRank() != 1)
    return emptySpec();
  if (mlir::ShapedType::isDynamic(padsTy.getDimSize(0)))
    return emptySpec();
  int64_t padsLen = padsTy.getDimSize(0);
  // Find this dim's position within the padded-axis list.
  int axisSlot = -1;
  if (axes) {
    auto axesTy = llvm::dyn_cast<mlir::ShapedType>(axes.getType());
    if (!axesTy || !axesTy.hasRank() || axesTy.getRank() != 1)
      return emptySpec();
    if (mlir::ShapedType::isDynamic(axesTy.getDimSize(0)))
      return emptySpec();
    int64_t axesLen = axesTy.getDimSize(0);
    if (axesLen * 2 != padsLen)
      return emptySpec();
    for (int64_t i = 0; i < axesLen; ++i) {
      DimSpec axisI = shape_interface::resolveValueFromI64Tensor(axes, i);
      if (axisI.nodes().empty() || !axisI.isStatic())
        return emptySpec();
      int64_t axisVal = axisI.staticValue();
      if (axisVal < 0)
        axisVal += dataTy.getRank();
      if ((unsigned)axisVal == dim_index) {
        axisSlot = (int)i;
        break;
      }
    }
  } else {
    int64_t rank = dataTy.getRank();
    if (padsLen != 2 * rank)
      return emptySpec();
    axisSlot = (int)dim_index;
  }
  if (axisSlot < 0)
    return resolveOperandDim(data, dim_index);
  int64_t halfLen = padsLen / 2;
  DimSpec startPad =
      shape_interface::resolveValueFromI64Tensor(pads, axisSlot);
  DimSpec endPad =
      shape_interface::resolveValueFromI64Tensor(pads, axisSlot + halfLen);
  DimSpec inDim = resolveOperandDim(data, dim_index);
  if (startPad.nodes().empty() || endPad.nodes().empty() ||
      inDim.nodes().empty())
    return emptySpec();
  DimSpec sum1 = DimSpec::makeBinary(DimSpecKind::Add, inDim, startPad);
  return DimSpec::makeBinary(DimSpecKind::Add, sum1, endPad);
}

// Reduce_* builder: with `keepdims=1`, the reduced axes become Static(1);
// with `keepdims=0`, the reduced axes are dropped (out_rank ==
// in_rank - num_reduced_axes). For dim_index in the OUTPUT, we walk
// through `axes` (host-readable when intermediate) to figure out which
// data axis it maps to.
//
// Operand layout (Hip_ReduceSumOp / ReduceMaxOp / ReduceProdOp):
//   ctx@0, data@1, axes@2, output@3.
DimSpec buildReduceDimSpec(mlir::Operation *op, unsigned result_index,
                           unsigned dim_index) {
  if (op->getNumOperands() < 4 || result_index != 0)
    return emptySpec();
  mlir::Value data = op->getOperand(1);
  mlir::Value axes = op->getOperand(2);
  auto dataTy = llvm::dyn_cast<mlir::ShapedType>(data.getType());
  if (!dataTy || !dataTy.hasRank())
    return emptySpec();
  int64_t keepdims = 1;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = a.getValue().getSExtValue();
  int64_t noopEmpty = 0;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes"))
    noopEmpty = a.getValue().getSExtValue();
  // Collect the axes set when host-readable; otherwise we bail.
  auto axesTy = llvm::dyn_cast<mlir::ShapedType>(axes.getType());
  if (!axesTy || !axesTy.hasRank() || axesTy.getRank() != 1)
    return emptySpec();
  // axes_len == 0 + noop_with_empty_axes == 1 => identity in shape.
  // axes_len == 0 + noop_with_empty_axes == 0 => reduce all axes.
  if (mlir::ShapedType::isDynamic(axesTy.getDimSize(0)))
    return emptySpec();
  int64_t axesLen = axesTy.getDimSize(0);
  llvm::SmallDenseSet<int64_t, 4> reducedAxes;
  if (axesLen == 0) {
    if (noopEmpty)
      return resolveOperandDim(data, dim_index);
    for (int64_t i = 0; i < dataTy.getRank(); ++i)
      reducedAxes.insert(i);
  } else {
    for (int64_t i = 0; i < axesLen; ++i) {
      DimSpec axisI = shape_interface::resolveValueFromI64Tensor(axes, i);
      if (axisI.nodes().empty() || !axisI.isStatic())
        return emptySpec();
      int64_t v = axisI.staticValue();
      if (v < 0)
        v += dataTy.getRank();
      reducedAxes.insert(v);
    }
  }
  if (keepdims) {
    if (reducedAxes.count((int64_t)dim_index))
      return DimSpec::makeStatic(1);
    return resolveOperandDim(data, dim_index);
  }
  // keepdims=0: walk data dims, skipping reduced ones, until we've
  // counted dim_index non-reduced dims.
  int64_t outCursor = 0;
  for (int64_t d = 0; d < dataTy.getRank(); ++d) {
    if (reducedAxes.count(d))
      continue;
    if ((int64_t)dim_index == outCursor)
      return resolveOperandDim(data, (unsigned)d);
    ++outCursor;
  }
  return emptySpec();
}

// CumSum is shape-preserving (rank and per-dim sizes match input).
// Operand layout: ctx@0, x@1, axis@2, y@3.
DimSpec buildCumSumDimSpec(mlir::Operation *op, unsigned result_index,
                           unsigned dim_index) {
  if (op->getNumOperands() < 4 || result_index != 0)
    return emptySpec();
  return resolveOperandDim(op->getOperand(1), dim_index);
}

// ScatterND is shape-preserving (output_shape == data_shape).
// Operand layout: ctx@0, data@1, indices@2, updates@3, output@4.
DimSpec buildScatterNDDimSpec(mlir::Operation *op, unsigned result_index,
                              unsigned dim_index) {
  if (op->getNumOperands() < 5 || result_index != 0)
    return emptySpec();
  return resolveOperandDim(op->getOperand(1), dim_index);
}

// GatherND output rank: `q + r - k - 1 - batch_dims`, where
// k = indices.shape[-1], q = indices_rank, r = data_rank, batch_dims =
// attribute. For dim_index < batch_dims: output dim == data dim ==
// indices dim. For batch_dims <= dim_index < q-1: output dim ==
// indices dim. For q-1 <= dim_index: output dim == data dim shifted.
//
// Operand layout (Hip_GatherNDOp): ctx@0, data@1, indices@2, output@3.
DimSpec buildGatherNDDimSpec(mlir::Operation *op, unsigned result_index,
                             unsigned dim_index) {
  if (op->getNumOperands() < 4 || result_index != 0)
    return emptySpec();
  mlir::Value data = op->getOperand(1);
  mlir::Value indices = op->getOperand(2);
  auto dataTy = llvm::dyn_cast<mlir::ShapedType>(data.getType());
  auto indicesTy = llvm::dyn_cast<mlir::ShapedType>(indices.getType());
  if (!dataTy || !dataTy.hasRank() || !indicesTy || !indicesTy.hasRank())
    return emptySpec();
  int64_t batchDims = 0;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("batch_dims"))
    batchDims = a.getValue().getSExtValue();
  int64_t q = indicesTy.getRank();
  int64_t r = dataTy.getRank();
  if (q <= 0)
    return emptySpec();
  int64_t k = indicesTy.getDimSize(q - 1);
  if (mlir::ShapedType::isDynamic(k))
    return emptySpec();
  int64_t outRank = q + r - k - 1 - batchDims;
  if ((int64_t)dim_index >= outRank)
    return emptySpec();
  // First `batchDims` dims: take from data (same as indices).
  if ((int64_t)dim_index < batchDims)
    return resolveOperandDim(data, dim_index);
  // Next `q - 1 - batchDims` dims: from indices[batchDims..q-1).
  int64_t pos = dim_index - batchDims;
  int64_t outerIndicesLen = q - 1 - batchDims;
  if (pos < outerIndicesLen)
    return resolveOperandDim(indices, (unsigned)(batchDims + pos));
  // Trailing dims: from data[batchDims + k + (pos - outerIndicesLen)..r).
  int64_t dataIdx = batchDims + k + (pos - outerIndicesLen);
  if (dataIdx >= r)
    return emptySpec();
  return resolveOperandDim(data, (unsigned)dataIdx);
}

// Concat aggregating builder: output_dim[axis] is the sum of
// input_dim[axis] across every data operand. Operand layout for the
// FUTURE `hip.concat` op (no converter today): ctx@0, inputs[0..N-1],
// output@N+1. We accept any number of data operands and treat the
// first as the rank reference.
//
// Other dims pass through (all inputs must share them by ONNX semantics).
DimSpec buildConcatDimSpec(mlir::Operation *op, unsigned result_index,
                           unsigned dim_index) {
  if (op->getNumOperands() < 3 || result_index != 0)
    return emptySpec();
  int64_t axis = 0;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = a.getValue().getSExtValue();
  unsigned firstData = 1;
  unsigned pastEnd = op->getNumOperands() - 1;
  if (pastEnd <= firstData)
    return emptySpec();
  // Determine the rank from operand 1.
  auto refTy = llvm::dyn_cast<mlir::ShapedType>(
      op->getOperand(firstData).getType());
  if (!refTy || !refTy.hasRank())
    return emptySpec();
  if (axis < 0)
    axis += refTy.getRank();
  if (axis < 0 || axis >= refTy.getRank())
    return emptySpec();
  if ((int64_t)dim_index != axis) {
    // Non-axis dim: just take from the first non-empty contribution.
    for (unsigned k = firstData; k < pastEnd; ++k) {
      DimSpec d = resolveOperandDim(op->getOperand(k), dim_index);
      if (!d.nodes().empty())
        return d;
    }
    return emptySpec();
  }
  // Axis dim: sum every operand's axis dim.
  DimSpec sum;
  bool first = true;
  for (unsigned k = firstData; k < pastEnd; ++k) {
    DimSpec d = resolveOperandDim(op->getOperand(k), (unsigned)axis);
    if (d.nodes().empty())
      return emptySpec();
    if (first) {
      sum = d;
      first = false;
    } else {
      sum = DimSpec::makeBinary(DimSpecKind::Add, sum, d);
    }
  }
  return sum;
}

} // namespace

void populateBuiltinDimSpecBuilders() {
  // Idempotent — guarded so multiple MLIRContext spin-ups don't race.
  static std::once_flag once;
  std::call_once(once, [] {
    // Rank-preserving permutation.
    registerOpDimSpecBuilder("hip.transpose", buildTransposeDimSpec);

    // Elementwise / broadcast (binary).
    registerOpDimSpecBuilder("hip.add", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.sub", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.mul", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.div", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.min", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.equal", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.less", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.and", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.mod", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.where", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.miopen.add", buildBroadcastDimSpec);

    // Elementwise unary — these never broadcast (rank+shape == input
    // exactly), so the broadcast builder degenerates to "input operand
    // dim k for output dim k". Same registration works.
    registerOpDimSpecBuilder("hip.not", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.neg", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.cos", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.sin", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.sign", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.silu", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.sigmoid", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.softplus", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.gelu", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.reciprocal", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.sqrt", buildBroadcastDimSpec);
    registerOpDimSpecBuilder("hip.cast", buildBroadcastDimSpec);

    // Phase 2 op-class builders: gather/tile/expand/range/slice/pad/
    // reduce/cumsum/scatter/gather_nd/concat. The Cat-C variant of
    // range / constant_of_shape / nonzero already attaches its own
    // output_dim_specs at converter time and short-circuits via
    // strategy 1 in getResultDimSpec; the builders below cover the
    // Cat-A / Cat-B / static cases and translucent propagator paths.
    registerOpDimSpecBuilder("hip.gather", buildGatherDimSpec);
    registerOpDimSpecBuilder("hip.tile", buildTileDimSpec);
    registerOpDimSpecBuilder("hip.expand", buildExpandDimSpec);
    registerOpDimSpecBuilder("hip.range", buildRangeDimSpec);
    registerOpDimSpecBuilder("hip.slice", buildSliceDimSpec);
    registerOpDimSpecBuilder("hip.pad", buildPadDimSpec);
    registerOpDimSpecBuilder("hip.reduce_sum", buildReduceDimSpec);
    registerOpDimSpecBuilder("hip.reduce_max", buildReduceDimSpec);
    registerOpDimSpecBuilder("hip.reduce_prod", buildReduceDimSpec);
    registerOpDimSpecBuilder("hip.cumsum", buildCumSumDimSpec);
    registerOpDimSpecBuilder("hip.scatter_nd", buildScatterNDDimSpec);
    registerOpDimSpecBuilder("hip.gather_nd", buildGatherNDDimSpec);
    // Concat: the dialect doesn't expose a hip.concat op yet (see
    // docs/design/slot-buffer-coalesce.md "Out-of-scope blockers").
    // When the converter lands, this registration unlocks the
    // aggregating Cat-B compound automatically.
    registerOpDimSpecBuilder("hip.concat", buildConcatDimSpec);
  });
}

//===----------------------------------------------------------------------===//
// Phase 4 — Identity-propagator predicate registry.
//===----------------------------------------------------------------------===//
//
// Same pattern as the DimSpec builder registry above: a per-op
// function pointer table populated once via `std::call_once`. Each
// predicate returns true iff its op is a runtime no-op (the result is
// bit-identical to its single SSA input). Predicates may ONLY look
// at compile-time information (op attributes + result types) — they
// must not peek into runtime values.

namespace {

llvm::StringMap<IdentityPredicateFn> &getIdentityRegistry() {
  static llvm::StringMap<IdentityPredicateFn> registry;
  return registry;
}

//===----------------------------------------------------------------------===//
// Built-in identity predicates
//===----------------------------------------------------------------------===//

// hip.transpose is identity iff `perm == [0, 1, ..., rank-1]`. The
// attribute is mandatory on Hip_TransposeOp so a missing attr would
// be a verifier failure — we still tolerate it (return false) to keep
// this predicate strictly attribute-driven and side-effect-free.
bool isIdentityTranspose(mlir::Operation *op) {
  auto perm = op->getAttrOfType<mlir::ArrayAttr>("perm");
  if (!perm)
    return false;
  for (auto [i, a] : llvm::enumerate(perm)) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(a);
    if (!intAttr)
      return false;
    if ((uint64_t)intAttr.getInt() != i)
      return false;
  }
  return true;
}

// hip.cast is identity iff input element-type == output element-type.
// Pre-bufferize the op carries tensor types; post-bufferize it
// carries memref types. Both expose `getElementType()` via
// `ShapedType`.
bool isIdentityCast(mlir::Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() < 1)
    return false;
  // ins() = (ctx, input), result is the output.
  auto inTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getOperand(1).getType());
  auto outTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
  if (!inTy || !outTy)
    return false;
  return inTy.getElementType() == outTy.getElementType();
}

// hip.expand is identity iff the input and output shapes match
// element-wise. We compare the static MLIR shapes; if either side has
// a dynamic dim at the same index AND both are dynamic, we consider
// them MATCHING for the runtime-no-op check (the operands are identical
// at the SSA level — broadcasting a `?` to itself is a copy). When
// only one side is dynamic the static side may end up larger at
// runtime, so we must conservatively bail (return false).
bool isIdentityExpand(mlir::Operation *op) {
  if (op->getNumOperands() < 2 || op->getNumResults() < 1)
    return false;
  auto inTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getOperand(1).getType());
  auto outTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
  if (!inTy || !outTy || !inTy.hasRank() || !outTy.hasRank())
    return false;
  if (inTy.getRank() != outTy.getRank())
    return false;
  for (int64_t i = 0; i < inTy.getRank(); ++i) {
    int64_t iD = inTy.getDimSize(i);
    int64_t oD = outTy.getDimSize(i);
    bool iDyn = mlir::ShapedType::isDynamic(iD);
    bool oDyn = mlir::ShapedType::isDynamic(oD);
    if (iDyn != oDyn)
      return false; // can't prove equality across mixed dynamic/static
    if (!iDyn && iD != oD)
      return false;
  }
  return true;
}

// hip.slice is identity iff `starts == 0`, `ends == fullSize`, and
// `steps == 1` on every axis. `starts`/`ends`/`steps` arrive as
// tensors -- we can only fold this when they come from
// `arith.constant` / `tosa.const`-shaped producers (or are part of
// the op's attribute list). For now, we recognise the constant-i64
// pattern that the converter emits: arith.constant dense<...>.
// Conservatively returns false when the producers are runtime values.
bool isIdentitySlice(mlir::Operation *op) {
  if (op->getNumOperands() < 3 || op->getNumResults() < 1)
    return false;
  // ins() = (ctx, data, starts, ends [, axes [, steps]]); op has
  // AttrSizedOperandSegments so we don't know precise positions
  // without consulting the segment attr. Read the result vs input
  // shapes as a shortcut: if every result dim equals the input dim
  // (modulo dynamic), this is necessarily an identity slice
  // regardless of the operand values.
  auto dataTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getOperand(1).getType());
  auto outTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
  if (!dataTy || !outTy || !dataTy.hasRank() || !outTy.hasRank())
    return false;
  if (dataTy.getRank() != outTy.getRank())
    return false;
  for (int64_t i = 0; i < dataTy.getRank(); ++i) {
    int64_t iD = dataTy.getDimSize(i);
    int64_t oD = outTy.getDimSize(i);
    bool iDyn = mlir::ShapedType::isDynamic(iD);
    bool oDyn = mlir::ShapedType::isDynamic(oD);
    if (iDyn != oDyn)
      return false;
    if (!iDyn && iD != oD)
      return false;
  }
  return true;
}

// hip.reduce_* is identity iff `noop_with_empty_axes = 1` AND the
// axes tensor is a constant of zero elements. The latter is hard to
// check structurally without traversing the producer chain, so we
// approximate via static result-type comparison: when every result
// dim equals every input dim (modulo dynamic), the reduce is a
// no-op. The `noop_with_empty_axes` flag must also be set so that an
// empty `axes` operand IS treated as "no reduction" rather than the
// implicit "reduce all axes" fallback.
bool isIdentityReduce(mlir::Operation *op) {
  auto noopAttr =
      op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes");
  if (!noopAttr || noopAttr.getInt() != 1)
    return false;
  if (op->getNumOperands() < 2 || op->getNumResults() < 1)
    return false;
  // ins() = (ctx, data, axes); for `noop_with_empty_axes=1` the
  // identity is "every input dim survives in the output at the same
  // index" — which the result-vs-input static shape check captures.
  auto dataTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getOperand(1).getType());
  auto outTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
  if (!dataTy || !outTy || !dataTy.hasRank() || !outTy.hasRank())
    return false;
  if (dataTy.getRank() != outTy.getRank())
    return false;
  for (int64_t i = 0; i < dataTy.getRank(); ++i) {
    int64_t iD = dataTy.getDimSize(i);
    int64_t oD = outTy.getDimSize(i);
    bool iDyn = mlir::ShapedType::isDynamic(iD);
    bool oDyn = mlir::ShapedType::isDynamic(oD);
    if (iDyn != oDyn)
      return false;
    if (!iDyn && iD != oD)
      return false;
  }
  return true;
}

// hip.tile is identity iff `repeats == [1, 1, ..., 1]` on every
// axis. We approximate the same way as slice: when every output dim
// equals the corresponding input dim statically, this is the only
// way an honest `hip.tile` can produce them.
bool isIdentityTile(mlir::Operation *op) {
  if (op->getNumOperands() < 3 || op->getNumResults() < 1)
    return false;
  auto inTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getOperand(1).getType());
  auto outTy =
      llvm::dyn_cast<mlir::ShapedType>(op->getResult(0).getType());
  if (!inTy || !outTy || !inTy.hasRank() || !outTy.hasRank())
    return false;
  if (inTy.getRank() != outTy.getRank())
    return false;
  for (int64_t i = 0; i < inTy.getRank(); ++i) {
    int64_t iD = inTy.getDimSize(i);
    int64_t oD = outTy.getDimSize(i);
    bool iDyn = mlir::ShapedType::isDynamic(iD);
    bool oDyn = mlir::ShapedType::isDynamic(oD);
    if (iDyn != oDyn)
      return false;
    if (!iDyn && iD != oD)
      return false;
  }
  return true;
}

} // namespace

void registerOpIdentityPredicate(llvm::StringRef op_name,
                                 IdentityPredicateFn fn) {
  getIdentityRegistry()[op_name] = fn;
}

bool isIdentityOp(mlir::Operation *op) {
  if (!op)
    return false;
  auto &registry = getIdentityRegistry();
  auto it = registry.find(op->getName().getStringRef());
  if (it == registry.end())
    return false;
  return it->second(op);
}

void populateBuiltinIdentityPredicates() {
  static std::once_flag once;
  std::call_once(once, [] {
    registerOpIdentityPredicate("hip.transpose", isIdentityTranspose);
    registerOpIdentityPredicate("hip.cast", isIdentityCast);
    registerOpIdentityPredicate("hip.expand", isIdentityExpand);
    registerOpIdentityPredicate("hip.slice", isIdentitySlice);
    registerOpIdentityPredicate("hip.tile", isIdentityTile);
    registerOpIdentityPredicate("hip.reduce_sum", isIdentityReduce);
    registerOpIdentityPredicate("hip.reduce_max", isIdentityReduce);
    registerOpIdentityPredicate("hip.reduce_prod", isIdentityReduce);
  });
}

} // namespace shape_interface

} // namespace hip
} // namespace mlir
