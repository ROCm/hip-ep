/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::hip {
namespace {

// All ranges refer to the one private runtime scratch allocation. Other
// allocations cannot alias it. Host accesses to other storage are rejected.
struct ByteRange {
  int64_t begin;
  int64_t end;

  bool overlaps(ByteRange other) const {
    return begin < end && other.begin < other.end && begin < other.end &&
           other.begin < end;
  }
  bool operator==(ByteRange other) const {
    return begin == other.begin && end == other.end;
  }
};

struct Descriptor {
  Type elementType;
  int64_t width;
  int64_t base;
  int64_t offset;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
};

// Match the current 64-bit index ABI. Reject padded and unsupported widths.
static std::optional<int64_t> storageWidth(Type type) {
  if (!type.isIntOrIndexOrFloat())
    return std::nullopt;
  unsigned bits = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  if (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;
  return (bits + 7) / 8;
}

static bool getConstants(ArrayRef<OpFoldResult> values,
                         SmallVectorImpl<int64_t> &result) {
  for (OpFoldResult value : values) {
    auto constant = getConstantIntValue(value);
    if (!constant)
      return false;
    result.push_back(*constant);
  }
  return true;
}

// Keep the aligned pointer and descriptor offset separate. Reinterpretation
// replaces the offset; it does not add to the previous descriptor's offset.
class ScratchRanges {
public:
  explicit ScratchRanges(int64_t capacity) : capacity(capacity) {}

  DenseMap<Value, Descriptor> descriptors;

  std::optional<ByteRange> range(const Descriptor &d) const {
    int64_t lo = d.offset, hi = d.offset;
    for (auto [size, stride] : llvm::zip(d.sizes, d.strides)) {
      if (size < 0)
        return std::nullopt;
      if (size == 0)
        return ByteRange{0, 0};
      int64_t delta;
      if (llvm::MulOverflow(size - 1, stride, delta) ||
          llvm::AddOverflow(lo, std::min(int64_t{0}, delta), lo) ||
          llvm::AddOverflow(hi, std::max(int64_t{0}, delta), hi))
        return std::nullopt;
    }
    int64_t begin, end;
    if (llvm::AddOverflow(hi, int64_t{1}, hi) ||
        llvm::MulOverflow(lo, d.width, begin) ||
        llvm::MulOverflow(hi, d.width, end) ||
        llvm::AddOverflow(d.base, begin, begin) ||
        llvm::AddOverflow(d.base, end, end) || begin < 0 || end > capacity ||
        begin % d.width != 0)
      return std::nullopt;
    return ByteRange{begin, end};
  }

  std::optional<ByteRange> element(Value value, ValueRange indices) const {
    auto it = descriptors.find(value);
    if (it == descriptors.end() || indices.size() != it->second.sizes.size())
      return std::nullopt;
    Descriptor scalar = it->second;
    for (auto [index, size, stride] :
         llvm::zip(indices, scalar.sizes, scalar.strides)) {
      auto constant = getConstantIntValue(index);
      int64_t delta;
      if (!constant || *constant < 0 || *constant >= size ||
          llvm::MulOverflow(*constant, stride, delta) ||
          llvm::AddOverflow(scalar.offset, delta, scalar.offset))
        return std::nullopt;
    }
    scalar.sizes.clear();
    scalar.strides.clear();
    return range(scalar);
  }

  std::optional<ByteRange> scalar(Value value, Type type) const {
    auto it = descriptors.find(value);
    if (it == descriptors.end() || it->second.elementType != type ||
        llvm::any_of(it->second.sizes, [](int64_t size) { return size != 1; }))
      return std::nullopt;
    return range(it->second);
  }

  // Return failure for an unsupported scratch alias. Descriptors alone do
  // not access payload bytes, but their later users must have proven ranges.
  LogicalResult track(Operation *op) {
    Value source;
    if (auto view = dyn_cast<memref::ViewOp>(op))
      source = view.getSource();
    else if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(op))
      source = metadata.getSource();
    else if (auto alias = dyn_cast<memref::ReinterpretCastOp>(op))
      source = alias.getSource();
    else
      return failure();
    auto it = descriptors.find(source);
    if (it == descriptors.end())
      return success();
    Descriptor d = it->second;
    Value result;
    if (auto view = dyn_cast<memref::ViewOp>(op)) {
      auto shift = getConstantIntValue(view.getByteShift());
      auto width = storageWidth(view.getType().getElementType());
      if (!shift || !width || !view.getType().hasStaticShape() ||
          !view.getType().getLayout().isIdentity() ||
          llvm::AddOverflow(d.base, *shift, d.base))
        return failure();
      d.elementType = view.getType().getElementType();
      d.width = *width;
      d.sizes.assign(view.getType().getShape().begin(),
                     view.getType().getShape().end());
      d.offset = 0;
      d.strides.resize(d.sizes.size());
      int64_t stride = 1;
      for (size_t i = d.sizes.size(); i > 0; --i) {
        d.strides[i - 1] = stride;
        if (llvm::MulOverflow(stride, d.sizes[i - 1], stride))
          return failure();
      }
      result = view.getResult();
    } else if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(op)) {
      d.offset = 0;
      d.sizes.clear();
      d.strides.clear();
      result = metadata.getBaseBuffer();
    } else {
      auto alias = cast<memref::ReinterpretCastOp>(op);
      SmallVector<int64_t> offsets;
      d.sizes.clear();
      d.strides.clear();
      if (!getConstants(alias.getMixedOffsets(), offsets) ||
          !getConstants(alias.getMixedSizes(), d.sizes) ||
          !getConstants(alias.getMixedStrides(), d.strides))
        return failure();
      d.offset = offsets.front();
      result = alias.getResult();
    }
    // Metadata can name the end of an empty view. Check payload bounds at
    // each access, not when an unused descriptor is constructed.
    descriptors[result] = std::move(d);
    return success();
  }

private:
  int64_t capacity;
};

// These lowerings use this context's stream for payload accesses. Their
// internal host reads use an ordered copy and wait. Do not infer this contract
// from a dialect name or from MemoryEffectsOpInterface alone. Add operations
// here only after checking their lowering and runtime implementation.
static bool hasOrderedPayloadAccesses(Operation *op) {
  return isa<CastOp, AddOp, MulOp, MinOp, MaxOp, SubOp, GqaOp, PadOp, SliceOp,
             NonZeroOp, AndOp, OrOp, NotOp, EqualOp, LessOp, WhereOp, ExpandOp,
             GatherOp, GatherElementsOp, ScatterElementsOp, CumSumOp, GeluOp,
             SigmoidOp, TanhOp, SoftplusOp, SwishOp, SqrtOp, ReciprocalOp,
             ReduceMeanOp, ReduceSumOp, RmsNormOp, SkipRmsNormOp, RopeOp,
             MatmulOp, MatMulNBitsOp, QMoEOp, MiopenSoftmaxOp>(op);
}

struct Accesses {
  Operation *op;
  SmallVector<ByteRange> reads;
  SmallVector<ByteRange> writes;
  SmallVector<ByteRange> earlyHostReads;
  bool host = false;
  bool readback = false;
  Value stored;
  Value source;
  Type requestedType;
};

static LogicalResult summarize(Operation *op, Value ctx, ScratchRanges &ranges,
                               Accesses &accesses) {
  if (op->getNumRegions() || isa<CallOpInterface>(op))
    return failure();
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    auto bytes = ranges.element(store.getMemRef(), store.getIndices());
    if (!bytes)
      return failure();
    accesses.host = true;
    accesses.writes.push_back(*bytes);
    accesses.stored = store.getValue();
    return success();
  }
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    auto bytes = ranges.element(load.getMemRef(), load.getIndices());
    if (!bytes)
      return failure();
    accesses.host = true;
    accesses.reads.push_back(*bytes);
    return success();
  }
  if (isa<memref::ViewOp, memref::ExtractStridedMetadataOp,
          memref::ReinterpretCastOp>(op))
    return ranges.track(op);
  if (isa<memref::DimOp>(op))
    return success();

  // Escaping a scratch descriptor loses the private-allocation guarantee.
  if (isMemoryEffectFree(op)) {
    return success(llvm::none_of(op->getOperands(), [&](Value value) {
      return ranges.descriptors.contains(value);
    }));
  }
  if (isa<memref::AllocOp>(op))
    return success();

  auto addRange = [&](Value value, bool write, bool contiguous) {
    auto it = ranges.descriptors.find(value);
    if (it == ranges.descriptors.end())
      return success();
    auto type = cast<MemRefType>(value.getType());
    // Most compute wrappers pass the aligned pointer without the descriptor
    // offset. Copies and explicit readbacks use the offset-aware helper.
    if (contiguous &&
        (!type.getLayout().isIdentity() || it->second.offset != 0))
      return failure();
    auto bytes = ranges.range(it->second);
    if (!bytes)
      return failure();
    (write ? accesses.writes : accesses.reads).push_back(*bytes);
    return success();
  };
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    if (!hasFlatRuntimeCopyLowering(
            dyn_cast<MemRefType>(copy.getSource().getType()),
            dyn_cast<MemRefType>(copy.getTarget().getType())))
      return failure();
    return success(succeeded(addRange(copy.getSource(), false, false)) &&
                   succeeded(addRange(copy.getTarget(), true, false)));
  }
  if (op->getNumOperands() == 0 || op->getOperand(0) != ctx)
    return failure();
  // These requests do not access existing scratch payloads. Their possible
  // growth waits are not guaranteed completion points for other buffers.
  if (isa<GetHostScratchOp, GetPoolOp, GetConstantOp, AllocOutputOp, AllocOp>(
          op))
    return success();
  if (auto read = dyn_cast<ReadbackScalarOp>(op)) {
    accesses.readback = true;
    accesses.source = read.getScalar();
    accesses.requestedType = read.getValue().getType();
    return addRange(read.getScalar(), false, false);
  }
  if (auto read = dyn_cast<ReadbackDimOp>(op)) {
    accesses.readback = true;
    accesses.source = read.getScalar();
    accesses.requestedType = IntegerType::get(op->getContext(), 32);
    return addRange(read.getScalar(), false, false);
  }
  // TopK copies K before its context-stream wait. Do not depend on default
  // stream ordering. Require an unchanged constant or a checked scratch read.
  if (auto topk = dyn_cast<TopKOp>(op)) {
    Value k = topk.getK();
    if (auto it = ranges.descriptors.find(k); it != ranges.descriptors.end()) {
      auto bytes = ranges.range(it->second);
      if (!bytes)
        return failure();
      accesses.earlyHostReads.push_back(*bytes);
    } else if (!k.getDefiningOp<GetConstantOp>()) {
      return failure();
    }
  } else if (!hasOrderedPayloadAccesses(op)) {
    return failure();
  }
  if (op->getNumResults())
    return failure();
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return failure();
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  for (const auto &effect : instances) {
    if (!effect.getValue() ||
        !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()) ||
        failed(addRange(effect.getValue(),
                        isa<MemoryEffects::Write>(effect.getEffect()), true)))
      return failure();
  }
  if (auto gqa = dyn_cast<GqaOp>(op)) {
    auto type = dyn_cast<MemRefType>(gqa.getSeqlensK().getType());
    if (type && type.getLayout().isIdentity() &&
        type.getShape() == ArrayRef<int64_t>{1} &&
        type.getElementType().isInteger(32)) {
      accesses.source = gqa.getSeqlensK();
      accesses.requestedType = type.getElementType();
    }
  }
  return success();
}

struct Fact {
  ByteRange bytes;
  Value value;
};

static bool conflicts(ArrayRef<ByteRange> a, ArrayRef<ByteRange> b) {
  return llvm::any_of(a, [&](ByteRange x) {
    return llvm::any_of(b, [&](ByteRange y) { return x.overlaps(y); });
  });
}

} // namespace

DenseMap<Operation *, Value> collectHostScalarValues(ModuleOp module) {
  SmallVector<GetHostScratchOp> scratches;
  module.walk([&](GetHostScratchOp op) { scratches.push_back(op); });
  // Scratch belongs to the runtime state. Another request can resize it.
  if (scratches.size() != 1)
    return DenseMap<Operation *, Value>();
  GetHostScratchOp scratch = scratches.front();
  auto func = scratch->getParentOfType<func::FuncOp>();
  auto capacity = getConstantIntValue(scratch.getScratchSize());
  if (!func || !llvm::hasSingleElement(func.getBody()) ||
      scratch->getBlock() != &func.front() || func.getNumArguments() == 0 ||
      scratch.getCtx() != func.getArgument(0) || !capacity || *capacity <= 0)
    return DenseMap<Operation *, Value>();
  auto uses = SymbolTable::getSymbolUses(func, module);
  if (!uses || !uses->empty())
    return DenseMap<Operation *, Value>();

  ScratchRanges ranges(*capacity);
  ranges.descriptors[scratch.getResult()] = Descriptor{
      IntegerType::get(module.getContext(), 8), 1, 0, 0, {*capacity}, {1}};
  SmallVector<Accesses> events;
  for (Operation &op : func.front()) {
    Accesses event{&op};
    if (failed(summarize(&op, scratch.getCtx(), ranges, event)))
      return DenseMap<Operation *, Value>();
    events.push_back(std::move(event));
  }

  // First propose replacements from exact stored bits. A retained scalar
  // readback supplies a new witness. A removed readback never supplies one.
  DenseMap<Operation *, Value> replacements;
  SmallVector<Fact> facts;
  for (const Accesses &event : events) {
    auto bytes = event.source ? ranges.scalar(event.source, event.requestedType)
                              : std::nullopt;
    if (bytes) {
      for (const Fact &fact : facts) {
        if (fact.bytes == *bytes &&
            fact.value.getType() == event.requestedType) {
          replacements[event.op] = fact.value;
          break;
        }
      }
    }
    llvm::erase_if(facts, [&](const Fact &fact) {
      return llvm::any_of(event.writes, [&](ByteRange write) {
        return write.overlaps(fact.bytes);
      });
    });
    if (event.stored)
      facts.push_back({event.writes.front(), event.stored});
    if (bytes && isa<ReadbackScalarOp>(event.op) &&
        !replacements.contains(event.op))
      facts.push_back({*bytes, event.op->getResult(0)});
  }

  // Validate all removals together. Two individually removable waits can
  // protect the same later host access. Neither removed wait clears work.
  SmallVector<ByteRange> pendingReads, pendingWrites;
  for (const Accesses &event : events) {
    if (conflicts(event.earlyHostReads, pendingWrites))
      return DenseMap<Operation *, Value>();
    if (event.readback) {
      if (!replacements.contains(event.op)) {
        pendingReads.clear();
        pendingWrites.clear();
      }
    } else if (event.host) {
      if (conflicts(event.reads, pendingWrites) ||
          conflicts(event.writes, pendingReads) ||
          conflicts(event.writes, pendingWrites))
        return DenseMap<Operation *, Value>();
    } else {
      llvm::append_range(pendingReads, event.reads);
      llvm::append_range(pendingWrites, event.writes);
    }
  }
  // The invocation's completion wait protects the remaining device users.
  // No stack payloads, releases, escapes, or calls were accepted above.
  return replacements;
}

} // namespace mlir::hip
