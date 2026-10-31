/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <limits>

namespace mlir {
namespace hip {
namespace {

// Record host lengths before dialect conversion replaces their stores.
//
// Before:
//   memref.store %length, %slot[%c0] : memref<1xi32>
//   hip.gqa ... ins(..., %slot, ...) ...
// After:
//   llvm.store %length, %slot_ptr : i32, !llvm.ptr
//   llvm.call @wrap_group_query_attention_host_length(..., %length)
//
// Keep the buffer for GPU readers. Require disjoint scratch slots with all
// host stores before their readers. No slot can change while a GPU reader is
// pending. Unknown uses and control flow keep the readback path.
static DenseMap<Operation *, Value> collectHostLengths(ModuleOp module) {
  DenseMap<Operation *, Value> lengths;
  SmallVector<GetHostScratchOp> scratches;
  module.walk([&](GetHostScratchOp op) { scratches.push_back(op); });
  // Host scratch is owned by the runtime state, not by each function. A
  // second request can reuse or resize it, including in an outlined body.
  if (scratches.size() != 1)
    return lengths;
  GetHostScratchOp scratch = scratches.front();
  auto func = scratch->getParentOfType<func::FuncOp>();
  if (!func || !llvm::hasSingleElement(func.getBody()) ||
      scratch->getBlock() != &func.front())
    return lengths;
  // An outlined or called body can run again before its GPU readers finish.
  auto symbolUses = SymbolTable::getSymbolUses(func, module);
  if (!symbolUses || !symbolUses->empty())
    return lengths;
  for (Operation &op : func.front()) {
    if (op.getNumRegions() || isa<CallOpInterface, LoopOp, IfOp>(op))
      return lengths;
  }

  struct Slot {
    memref::ViewOp view;
    int64_t begin;
    int64_t end;
    SmallVector<memref::StoreOp> stores;
  };
  SmallVector<Slot> slots;
  SmallVector<Operation *> aliasReaders;
  for (Operation *user : scratch.getResult().getUsers()) {
    auto view = dyn_cast<memref::ViewOp>(user);
    if (!view || view->getBlock() != &func.front())
      return lengths;
    MemRefType type = view.getType();
    auto offset = getConstantIntValue(view.getByteShift());
    if (!type.hasStaticShape() || !type.getLayout().isIdentity() || !offset ||
        *offset < 0 || !type.getElementType().isIntOrIndexOrFloat())
      return lengths;
    unsigned bits = type.getElementType().isIndex()
                        ? 64
                        : type.getElementType().getIntOrFloatBitWidth();
    // Other widths can have padding between adjacent elements.
    if (bits != 1 && bits != 8 && bits != 16 && bits != 32 && bits != 64)
      return lengths;
    int64_t bytesPerElement = (bits + 7) / 8;
    int64_t count = type.getNumElements();
    if (count <= 0 || count > (std::numeric_limits<int64_t>::max() - *offset) /
                                  bytesPerElement)
      return lengths;
    Slot slot{view, *offset, *offset + count * bytesPerElement, {}};
    SmallVector<Operation *> readers;
    SmallVector<Value> worklist{view.getResult()};
    DenseSet<Value> visited;
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      if (!visited.insert(value).second)
        continue;
      for (Operation *use : value.getUsers()) {
        if (use->getBlock() != &func.front())
          return lengths;
        if (auto store = dyn_cast<memref::StoreOp>(use)) {
          if (value != view.getResult() || store.getMemRef() != value)
            return lengths;
          slot.stores.push_back(store);
          continue;
        }
        if (isa<memref::DimOp>(use))
          continue;
        // These operations build descriptors. They do not read memory.
        if (auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(use)) {
          worklist.push_back(metadata.getBaseBuffer());
          continue;
        }
        if (auto alias = dyn_cast<memref::ReinterpretCastOp>(use)) {
          worklist.push_back(alias.getResult());
          continue;
        }
        if (auto copy = dyn_cast<memref::CopyOp>(use)) {
          if (copy.getSource() != value || copy.getTarget() == value)
            return lengths;
        } else if (!isa<memref::LoadOp>(use)) {
          // Buffer-form DPS operations do not return aliases. Their effects
          // must say that they only read this value.
          auto effects = dyn_cast<MemoryEffectOpInterface>(use);
          if (!isa<DestinationStyleOpInterface>(use) || use->getNumResults() ||
              use->getNumRegions() || !effects)
            return lengths;
          SmallVector<MemoryEffects::EffectInstance> instances;
          effects.getEffects(instances);
          for (const auto &effect : instances) {
            if ((!effect.getValue() || effect.getValue() == value) &&
                !isa<MemoryEffects::Read>(effect.getEffect()))
              return lengths;
          }
        }
        readers.push_back(use);
        if (value != view.getResult())
          aliasReaders.push_back(use);
      }
    }
    if (slot.stores.empty())
      return lengths;
    for (memref::StoreOp store : slot.stores) {
      for (Operation *reader : readers) {
        if (!store->isBeforeInBlock(reader))
          return lengths;
      }
    }
    slots.push_back(std::move(slot));
  }
  llvm::sort(slots,
             [](const Slot &a, const Slot &b) { return a.begin < b.begin; });
  for (size_t i = 1; i < slots.size(); ++i) {
    if (slots[i].begin < slots[i - 1].end)
      return lengths;
  }
  // An extracted base can address other slots in the backing allocation.
  // Without a range proof, require all stores before every aliased reader.
  for (Slot &slot : slots) {
    for (memref::StoreOp store : slot.stores) {
      for (Operation *reader : aliasReaders) {
        if (!store->isBeforeInBlock(reader))
          return lengths;
      }
    }
  }
  for (Slot &slot : slots) {
    if (slot.stores.size() != 1 ||
        slot.view.getType().getShape() != ArrayRef<int64_t>{1} ||
        !slot.view.getType().getElementType().isInteger(32) ||
        !matchPattern(slot.stores.front().getIndices().front(), m_Zero()))
      continue;
    for (Operation *user : slot.view.getResult().getUsers()) {
      if (auto gqa = dyn_cast<GqaOp>(user)) {
        if (gqa.getSeqlensK() == slot.view.getResult())
          lengths[gqa] = slot.stores.front().getValue();
      }
    }
  }
  return lengths;
}

// hip.gqa(ctx, query, key, value, past_key, past_value, seqlens_k,
// total_seq_len,
//         output, present_key, present_value) {attributes...}
struct GqaOpLowering : public ConvertOpToLLVMPattern<GqaOp> {
  GqaOpLowering(const LLVMTypeConverter &converter, ModuleOp module)
      : ConvertOpToLLVMPattern<GqaOp>(converter),
        hostLengths(collectHostLengths(module)) {}

  DenseMap<Operation *, Value> hostLengths;

  LogicalResult
  matchAndRewrite(GqaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    // Helper: create nullptr
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    // Helper: extract memref pointer or nullptr for optional operands
    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref)
        return nullPtr;
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };

    // === Extract all inputs (required + optional) ===

    Value statePtr = adaptor.getCtx();

    // Required inputs
    Value queryPtr =
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc);
    Value seqlensKPtr =
        extractContiguousMemRefPtr(adaptor.getSeqlensK(), rewriter, loc);
    Value totalSeqLenPtr =
        extractContiguousMemRefPtr(adaptor.getTotalSeqLen(), rewriter, loc);

    // Optional inputs (may be nullptr)
    Value keyPtr = getMemRefPtrOrNull(adaptor.getKey());
    Value valuePtr = getMemRefPtrOrNull(adaptor.getValue());
    Value pastKeyPtr = getMemRefPtrOrNull(adaptor.getPastKey());
    Value pastValuePtr = getMemRefPtrOrNull(adaptor.getPastValue());
    Value cosCachePtr = getMemRefPtrOrNull(adaptor.getCosCache());
    Value sinCachePtr = getMemRefPtrOrNull(adaptor.getSinCache());
    Value positionIdsPtr = getMemRefPtrOrNull(adaptor.getPositionIds());
    Value attentionBiasPtr = getMemRefPtrOrNull(adaptor.getAttentionBias());
    Value headSinkPtr = getMemRefPtrOrNull(adaptor.getHeadSink());
    Value kScalePtr = getMemRefPtrOrNull(adaptor.getKScale());
    Value vScalePtr = getMemRefPtrOrNull(adaptor.getVScale());

    // Output pointers
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value presentKeyPtr =
        extractContiguousMemRefPtr(adaptor.getPresentKey(), rewriter, loc);
    Value presentValuePtr =
        extractContiguousMemRefPtr(adaptor.getPresentValue(), rewriter, loc);
    Value outputQkPtr = getMemRefPtrOrNull(adaptor.getOutputQk());

    // === Extract attributes ===

    Value numHeads = createI64Const(op.getNumHeads());
    Value kvNumHeads = createI64Const(op.getKvNumHeads());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value doRotary = createI64Const(op.getDoRotary());
    Value rotaryInterleaved = createI64Const(op.getRotaryInterleaved());
    Value softcap = createF32Const(op.getSoftcap().convertToFloat());
    Value localWindowSize = createI64Const(op.getLocalWindowSize());
    Value smoothSoftmax = createI64Const(op.getSmoothSoftmax());
    Value qkOutput = createI64Const(op.getQkOutput());
    Value kvCacheBitWidth = createI64Const(op.getKvCacheBitWidth());

    // String attributes need to be converted to enum integers
    // "NONE"=0, "PER_TENSOR"=1, "PER_CHANNEL"=2
    auto quantTypeToEnum = [](llvm::StringRef str) -> int64_t {
      if (str == "NONE")
        return 0;
      if (str == "PER_TENSOR")
        return 1;
      if (str == "PER_CHANNEL")
        return 2;
      return 0; // default to NONE
    };
    Value kQuantType = createI64Const(quantTypeToEnum(op.getKQuantType()));
    Value vQuantType = createI64Const(quantTypeToEnum(op.getVQuantType()));

    // no_causal: emit as i32 (matches wrap_group_query_attention signature).
    Value noCausal = LLVM::ConstantOp::create(
        rewriter, loc, i32Type,
        rewriter.getI32IntegerAttr(op.getNoCausal() ? 1 : 0));

    // Extract shape info from query memref: [batch, seq_q, num_heads *
    // head_dim]. Uses getMemRefDimSize() to handle both static and dynamic
    // dimensions — static dims become LLVM constants, dynamic dims are
    // extracted from the memref descriptor at runtime.
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    Value batchSizeVal =
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc);
    Value seqLenQVal =
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc);
    Value queryHiddenVal =
        getMemRefDimSize(queryType, 2, adaptor.getQuery(), rewriter, loc);

    // Packed QKV: query shape is [B, S, (H + 2*G)*d] instead of [B, S, H*d].
    // Derive head_dim accordingly: d = hidden / (H + 2*G) vs hidden / H.
    bool packedQKV = !op.getKey();
    int64_t headDimDivisor = packedQKV
                                 ? (op.getNumHeads() + 2 * op.getKvNumHeads())
                                 : op.getNumHeads();
    // Compile-time sanity: when query_hidden is static, it must be evenly
    // divisible by the divisor. A mismatch means num_heads/kv_num_heads is
    // wrong for this model — fail loudly here rather than silently producing
    // a truncated head_dim at runtime via integer SDiv.
    //
    // When query_hidden is DYNAMIC, divisibility cannot be checked here.
    // The lowering proceeds with SDiv as before, but emits a one-time
    // build-time advisory so the warning shows up in the compile log;
    // a runtime mismatch (model with wrong num_heads/kv_num_heads) would
    // silently truncate head_dim via integer division — there is no
    // runtime helper to abort cleanly, and the locked design choice for
    // this PR is "warn and continue" rather than "assert and abort".
    if (!queryType.isDynamicDim(2)) {
      int64_t queryHidden = queryType.getDimSize(2);
      if (headDimDivisor <= 0 || queryHidden % headDimDivisor != 0)
        return op.emitOpError()
               << "query_hidden (" << queryHidden << ") not divisible by "
               << (packedQKV ? "(H + 2*G)" : "H") << " = " << headDimDivisor
               << " (num_heads=" << op.getNumHeads()
               << ", kv_num_heads=" << op.getKvNumHeads()
               << (packedQKV ? ", packed QKV)" : ")");
    } else {
      op.emitWarning()
          << "hip.gqa: dynamic query_hidden — divisibility by "
          << (packedQKV ? "(H + 2*G)" : "H") << " = " << headDimDivisor
          << " cannot be verified at compile time; runtime SDiv will "
          << "silently truncate on a model with mismatched num_heads "
          << "(num_heads=" << op.getNumHeads()
          << ", kv_num_heads=" << op.getKvNumHeads()
          << (packedQKV ? ", packed QKV)" : ")");
    }
    Value headDimVal = LLVM::SDivOp::create(rewriter, loc, queryHiddenVal,
                                            createI64Const(headDimDivisor));
    unsigned elementSizeBytes =
        queryType.getElementType().getIntOrFloatBitWidth() / 8;

    // Extract seq_len_kv from present_key shape.
    // ONNX GQA uses BNSD layout: [batch, kv_num_heads, total_seq, head_dim]
    auto presentKeyType = cast<MemRefType>(op.getPresentKey().getType());
    unsigned pkSeqDim = (presentKeyType.getRank() == 4) ? 2 : 1;
    Value seqLenKVVal = getMemRefDimSize(
        presentKeyType, pkSeqDim, adaptor.getPresentKey(), rewriter, loc);

    // past_buf_seq: buffer dimension of past_key (may be max_length for
    // pre-allocated caches, which is larger than actual valid past tokens).
    // Needed so gqa_forward can distinguish buffer stride from valid length.
    Value pastBufSeqVal = createI64Const(0);
    if (op.getPastKey()) {
      auto pastKeyType = cast<MemRefType>(op.getPastKey().getType());
      unsigned pastSeqDim = (pastKeyType.getRank() == 4) ? 2 : 1;
      pastBufSeqVal = getMemRefDimSize(pastKeyType, pastSeqDim,
                                       adaptor.getPastKey(), rewriter, loc);
    }

    // kv_bnsd: which layout the key/value operands are in. hip.gqa accepts both
    // rank-3 BSHD [B, S, G*d] (the onnx.Attention and GroupQueryAttention
    // lowerings) and rank-4 BNSD [B, G, S, d] (the MultiHeadAttention
    // cross-attn lowering, which forwards the encoder KV untouched). The
    // runtime needs this on the bidirectional no-past path, where key/value are
    // the whole KV and get staged into the BNSD present cache: the two layouts
    // coincide only at G == 1, so the shapes alone cannot distinguish them.
    // Absent key means packed QKV, which is always BSHD.
    Value kvBnsdVal = createI64Const(0);
    // seq_len_new_kv: tokens carried by key/value. A KV-sharing layer passes a
    // zero-length key/value next to a past cache produced by another layer,
    // and the runtime must then attend over the past without appending.
    Value seqLenNewKVVal = seqLenQVal;
    if (op.getKey()) {
      auto keyMemRefType = cast<MemRefType>(op.getKey().getType());
      kvBnsdVal = createI64Const(keyMemRefType.getRank() == 4 ? 1 : 0);
      unsigned keySeqDim = (keyMemRefType.getRank() == 4) ? 2 : 1;
      seqLenNewKVVal = getMemRefDimSize(keyMemRefType, keySeqDim,
                                        adaptor.getKey(), rewriter, loc);
    }

    Value elemSizeVal = createI64Const(elementSizeBytes);

    // attention_bias broadcast extents: ONNX masks are often [B,1,S,T].
    Value attnBiasBatchVal = createI64Const(1);
    Value attnBiasNumHeadsVal = createI64Const(1);
    if (op.getAttentionBias()) {
      auto biasType = cast<MemRefType>(op.getAttentionBias().getType());
      if (biasType.getRank() == 4) {
        attnBiasBatchVal = getMemRefDimSize(
            biasType, 0, adaptor.getAttentionBias(), rewriter, loc);
        attnBiasNumHeadsVal = getMemRefDimSize(
            biasType, 1, adaptor.getAttentionBias(), rewriter, loc);
      }
    }

    // Function signature matches wrap_group_query_attention() in gqa.cpp
    SmallVector<Type, 42> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        // Inputs (14 pointers - some may be nullptr)
        ptrType, // query
        ptrType, // key
        ptrType, // value
        ptrType, // past_key
        ptrType, // past_value
        ptrType, // seqlens_k
        ptrType, // total_seq_len
        ptrType, // cos_cache
        ptrType, // sin_cache
        ptrType, // position_ids
        ptrType, // attention_bias
        ptrType, // head_sink
        ptrType, // k_scale
        ptrType, // v_scale
        // Outputs (4 pointers - output_qk may be nullptr)
        ptrType, // output
        ptrType, // present_key
        ptrType, // present_value
        ptrType, // output_qk
        // Attributes (12 values)
        i64Type, // num_heads
        i64Type, // kv_num_heads
        f32Type, // scale
        i64Type, // do_rotary
        i64Type, // rotary_interleaved
        f32Type, // softcap
        i64Type, // local_window_size
        i64Type, // smooth_softmax
        i64Type, // qk_output
        i64Type, // k_quant_type
        i64Type, // v_quant_type
        i64Type, // kv_cache_bit_width
        i32Type, // no_causal
        // Shape info (10 values)
        i64Type, // batch_size
        i64Type, // seq_len_q
        i64Type, // seq_len_kv
        i64Type, // past_buf_seq
        i64Type, // head_dim
        i64Type, // element_size_bytes
        i64Type, // attn_bias_batch
        i64Type, // attn_bias_num_heads
        i64Type, // kv_bnsd
        i64Type  // seq_len_new_kv
    };

    Value hostLength;
    if (Value original = hostLengths.lookup(op)) {
      hostLength = rewriter.getRemappedValue(original);
      if (!hostLength)
        return rewriter.notifyMatchFailure(op, "cannot convert host length");
      paramTypes.push_back(i32Type);
    }
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, hostLength ? kWrapGQAHostLength : kWrapGQA,
        paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 42> args = {
        statePtr,
        // Per-instance op-state slot (threaded by --assign-op-state-slots)
        getOpStateSlotValue(op, rewriter, loc),
        // Inputs (14 pointers)
        queryPtr, keyPtr, valuePtr, pastKeyPtr, pastValuePtr, seqlensKPtr,
        totalSeqLenPtr, cosCachePtr, sinCachePtr, positionIdsPtr,
        attentionBiasPtr, headSinkPtr, kScalePtr, vScalePtr,
        // Outputs (4 pointers)
        outputPtr, presentKeyPtr, presentValuePtr, outputQkPtr,
        // Attributes (13 values)
        numHeads, kvNumHeads, scale, doRotary, rotaryInterleaved, softcap,
        localWindowSize, smoothSoftmax, qkOutput, kQuantType, vQuantType,
        kvCacheBitWidth, noCausal,
        // Shape info (10 values)
        batchSizeVal, seqLenQVal, seqLenKVVal, pastBufSeqVal, headDimVal,
        elemSizeVal, attnBiasBatchVal, attnBiasNumHeadsVal, kvBnsdVal,
        seqLenNewKVVal};
    if (hostLength)
      args.push_back(hostLength);

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGqaLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns, ModuleOp module) {
  patterns.add<GqaOpLowering>(converter, module);
}

} // namespace hip
} // namespace mlir
