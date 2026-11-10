/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- IndexerRotaryFold.cpp - Fold MRoPE tables into the QSA indexer ----===//
//
// A qsa SparseAttentionIndexer with MRoPE positions, one rotary_position_ids
// row per section, gets its cos/sin caches from this subgraph, with constant
// per-frequency masks:
//
//   pos_k = Gather(rotary_position_ids [M, B, X], k)            [B, X]
//   g_k   = Gather(base [Y, R/2], pos_k)                       [B, X, R/2]
//   w     = Where(mask_2, g_2, Where(mask_1, g_1, g_0))        [B, X, R/2]
//   cache = Concat(w, w, axis = -1)                            [B, X, R]
//
// - The subgraph runs once per call even when several indexers share it, but
//   it writes and reads B * X * R elements per table, while each indexer only
//   rotates its S present rows. onnxruntime-genai binds X to the generation's
//   max_length, so the shared rebuild scales with max_length instead of S.
//   Gathering the present rows inside each indexer costs less, even though
//   every indexer repeats it.
// - The masks fix each frequency's section, so the fold passes the base tables,
//   the ids and that section map instead. The kernels gather only the rows they
//   rotate. Every node is a copy, so the result is bit-identical.
// - The rewritten onnx.Custom has the base tables in the cos_cache / sin_cache
//   slots, rotary_position_ids at kRotaryPositionIdsIdx and the map in
//   kRotarySectionsAttr. SparseAttentionIndexerConversion is its only reader.
// - Any other cache producer keeps the materialized tables.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

namespace mlir {
namespace hip {
namespace {

constexpr unsigned kCosCacheIdx = 4;
constexpr unsigned kSinCacheIdx = 5;
// The exporter's tree has one Where per extra section; this only bounds the
// recursion.
constexpr unsigned kMaxWhereDepth = 8;

int64_t axisOf(Operation *op) {
  auto attr = op->getAttrOfType<IntegerAttr>("axis");
  return attr ? attr.getValue().getSExtValue() : 0;
}

DenseElementsAttr constantValue(Value v) {
  Operation *op = onnxProducer(v, "onnx.Constant");
  return op ? op->getAttrOfType<DenseElementsAttr>("value") : nullptr;
}

/// One of the cos / sin subgraphs: the base table, the position ids and the
/// MRoPE section of every frequency.
struct GatheredTable {
  Value base;
  Value ids;
  SmallVector<int32_t> sections;
};

/// Returns the section of each of the `half` frequencies of `v`, a tree of
/// Where(constant mask, a, b) over leaves Gather(base, Gather(ids, k)) with a
/// constant scalar k. Fails unless all leaves share `table.base` and
/// `table.ids`, which the first leaf sets.
///   Where([0, 1], Gather(base, Gather(ids, 2)), Gather(base, Gather(ids, 0)))
///   -> [0, 2]
FailureOr<SmallVector<int32_t>>
sectionsOf(Value v, int64_t half, GatheredTable &table, unsigned depth) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getRank() != 3 || type.getDimSize(2) != half) {
    return failure();
  }
  if (Operation *where = onnxProducer(v, "onnx.Where")) {
    DenseElementsAttr mask = constantValue(where->getOperand(0));
    auto maskType = mask ? dyn_cast<ShapedType>(mask.getType()) : nullptr;
    if (depth == kMaxWhereDepth || !maskType ||
        !isa<IntegerType>(maskType.getElementType()) ||
        maskType.getRank() == 0 || mask.getNumElements() != half ||
        maskType.getShape().back() != half) {
      return failure();
    }
    auto picked = sectionsOf(where->getOperand(1), half, table, depth + 1);
    if (failed(picked)) {
      return failure();
    }
    auto other = sectionsOf(where->getOperand(2), half, table, depth + 1);
    if (failed(other)) {
      return failure();
    }
    SmallVector<int32_t> sections;
    for (auto [f, bit] : llvm::enumerate(mask.getValues<APInt>())) {
      sections.push_back(bit.isZero() ? (*other)[f] : (*picked)[f]);
    }
    return sections;
  }

  Operation *gather = onnxProducer(v, "onnx.Gather");
  Operation *position =
      gather ? onnxProducer(gather->getOperand(1), "onnx.Gather") : nullptr;
  if (!position || axisOf(gather) != 0 || axisOf(position) != 0) {
    return failure();
  }
  Value base = gather->getOperand(0);
  Value ids = position->getOperand(0);
  DenseElementsAttr index = constantValue(position->getOperand(1));
  auto baseType = dyn_cast<RankedTensorType>(base.getType());
  auto idsType = dyn_cast<RankedTensorType>(ids.getType());
  if (!index || cast<ShapedType>(index.getType()).getRank() != 0 ||
      !isa<IntegerType>(index.getElementType()) || !baseType ||
      baseType.getRank() != 2 || baseType.getDimSize(1) != half || !idsType ||
      idsType.getRank() != 3 || !idsType.getElementType().isInteger(64) ||
      (table.base && table.base != base) || (table.ids && table.ids != ids)) {
    return failure();
  }
  table.base = base;
  table.ids = ids;
  const int64_t numSections = idsType.getDimSize(0);
  const bool known = !ShapedType::isDynamic(numSections);
  int64_t k = (*index.getValues<APInt>().begin()).getSExtValue();
  if (k < 0 && known) {
    k += numSections;
  }
  if (k < 0 || k >= kMaxRotarySections || (known && k >= numSections)) {
    return failure();
  }
  return SmallVector<int32_t>(half, static_cast<int32_t>(k));
}

/// Returns the table behind `cache = Concat(w, w, axis = -1)` over a Where
/// tree `w`, or std::nullopt when `cache` has another producer.
std::optional<GatheredTable> matchTable(Value cache) {
  Operation *concat = onnxProducer(cache, "onnx.Concat");
  auto type = dyn_cast<RankedTensorType>(cache.getType());
  if (!concat || !type || type.getRank() != 3 || type.isDynamicDim(2) ||
      type.getDimSize(2) % 2 != 0 || concat->getNumOperands() != 2 ||
      concat->getOperand(0) != concat->getOperand(1) ||
      (axisOf(concat) != -1 && axisOf(concat) != 2)) {
    return std::nullopt;
  }
  GatheredTable table;
  auto sections =
      sectionsOf(concat->getOperand(0), type.getDimSize(2) / 2, table, 0);
  if (failed(sections)) {
    return std::nullopt;
  }
  table.sections = std::move(*sections);
  return table;
}

/// Erases the Concat, Where and Gather ops of the table subgraphs rooted at
/// `roots` once they have no uses. The constants stay for lowerOnnxConstants,
/// which owns them.
void eraseDeadTables(PatternRewriter &rewriter, ArrayRef<Value> roots) {
  llvm::SetVector<Operation *> worklist;
  for (Value v : roots) {
    if (Operation *op = v.getDefiningOp()) {
      worklist.insert(op);
    }
  }
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    StringRef name = op->getName().getStringRef();
    if (!op->use_empty() || (name != "onnx.Concat" && name != "onnx.Where" &&
                             name != "onnx.Gather")) {
      continue;
    }
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        worklist.insert(def);
      }
    }
    rewriter.eraseOp(op);
  }
}

struct IndexerRotaryFold : public RewritePattern {
  IndexerRotaryFold(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isQsaSparseAttentionIndexer(op) || op->hasAttr(kRotarySectionsAttr)) {
      return rewriter.notifyMatchFailure(
          op, "not an unfolded qsa SparseAttentionIndexer");
    }
    const unsigned numOps = op->getNumOperands();
    if (numOps <= kSinCacheIdx || numOps > kRotaryPositionIdsIdx ||
        !isa<NoneType>(op->getOperand(1).getType())) {
      return rewriter.notifyMatchFailure(
          op, "expects packed QK and at most 14 inputs");
    }
    Value cos = op->getOperand(kCosCacheIdx);
    Value sin = op->getOperand(kSinCacheIdx);
    std::optional<GatheredTable> cosTable = matchTable(cos);
    std::optional<GatheredTable> sinTable = matchTable(sin);
    if (!cosTable || !sinTable) {
      return rewriter.notifyMatchFailure(
          op, "cos/sin caches are not gathered MRoPE tables");
    }
    auto query = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto baseType = cast<RankedTensorType>(cosTable->base.getType());
    if (cosTable->ids != sinTable->ids ||
        cosTable->sections != sinTable->sections ||
        sinTable->base.getType() != baseType || !query ||
        baseType.getElementType() != query.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "cos/sin tables must share ids, sections, shape and the query "
              "element type");
    }

    // Absent trailing inputs take input 1's none value, so the ids land at
    // kRotaryPositionIdsIdx.
    SmallVector<Value> operands = llvm::to_vector(op->getOperands());
    operands.resize(kRotaryPositionIdsIdx, op->getOperand(1));
    operands[kCosCacheIdx] = cosTable->base;
    operands[kSinCacheIdx] = sinTable->base;
    operands.push_back(cosTable->ids);
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    state.addAttribute(kRotarySectionsAttr,
                       rewriter.getDenseI32ArrayAttr(cosTable->sections));
    Operation *folded = rewriter.create(state);
    rewriter.replaceOp(op, folded->getResults());
    eraseDeadTables(rewriter, {cos, sin});
    return success();
  }
};

} // namespace

void populateIndexerRotaryFoldPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<IndexerRotaryFold>(ctx);
}

} // namespace hip
} // namespace mlir
