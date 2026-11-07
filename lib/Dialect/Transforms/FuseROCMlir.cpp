/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "GemmGemm.h"
#include "QLinearPlan.h"
#include "QuantizedEpilogue.h"
#include "RebiasUnsignedQuantize.h"
#include "RocMlirFusionPredicates.h"
#include "SinkQuantize.h"

#include "hip/Conversion/HipToTosa/HipToTosaLegality.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir::hip {
#define GEN_PASS_DEF_FUSEROCMLIRPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

#define DEBUG_TYPE "fuse-rocmlir"

namespace {

// Rank-0 inline hip.constant used by fused pointwise ops (Relu zero, Clip
// bounds, residual Mul scale). File- and memory-backed carriers have no dense
// attr for in-kernel tosa.const and stay kernel arguments.
static bool isInlineScalarConstant(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getRank() != 0)
    return false;
  auto constant = v.getDefiningOp<ConstantOp>();
  return constant && constant.getValueAttr();
}

// The kernel ABI rocMLIR expects for a scalar: MIGraphX gives one the shape
// {1}, and rock-flatten-tosa-func-args leaves rank-1 boundaries alone. A rank-0
// argument reaches rock.transforms_to_ptr with no coordinate to linearize
// (`affine_map -> ()`), which fails as "Transforms are not well formed";
// tensor<1xT> still carries the single index 0.
static Type promoteRankZero(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || tensorType.getRank() != 0)
    return type;
  return RankedTensorType::get({1}, tensorType.getElementType(),
                               tensorType.getEncoding());
}

/// rocMLIR's arithmetic uses signless integers. ONNX BOOL is ui8 in the
/// importer, but a Where condition is a one-byte 0/1 mask, so the outlined
/// kernel can safely view that argument as i8. The dispatch continues to pass
/// the original ui8 buffer; only the private kernel signature is normalized.
static Type getRocMlirKernelArgumentType(Value operand,
                                         const SetVector<Operation *> &ops) {
  Type type = promoteRankZero(operand.getType());
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getElementType().isUnsignedInteger(8))
    return type;

  for (Operation *op : ops) {
    auto where = dyn_cast<WhereOp>(op);
    if (where && where.getCondition() == operand)
      return tensorType.clone(IntegerType::get(type.getContext(), 8));
  }
  return type;
}

static bool isStaticTensor(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  return type && type.hasStaticShape();
}

// rock.blockwise_load and rock.blockwise_store take integers of at most 32
// bits, so a wider integer tensor cannot be a kernel argument or result.
static constexpr unsigned kMaxKernelIntegerWidth = 32;

static bool isKernelIOType(Type type) {
  auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
  return !intType || intType.getWidth() <= kMaxKernelIntegerWidth;
}

static bool isKernelIOValue(Value v) { return isKernelIOType(v.getType()); }

// A quantize on the way out lets the kernel write the narrow type. A
// dequantize there would only widen the output again, so it waits to be the
// prologue of whichever kernel reads it.
static bool isFusableQuantizeEpilogue(Operation *op) {
  return isa_and_present<QuantizeLinearOp>(op) && isTosaLowerableQdq(op);
}

// A dequantize on the way in lets the kernel read the narrow type, and a
// quantize feeding it folds the round trip into the load.
static bool isFusablePrologue(Operation *op) {
  return isFusablePointwise(op) || isTosaLowerableQdq(op);
}

// Which neighbours of an anchor may join its kernel, beyond the forward
// pointwise chain every anchor takes.
struct FusionPolicy {
  // ViewKind bits the input absorption may take.
  unsigned inputViewKinds = 0;
  // ViewKind bits the forward chain past the anchor may take.
  unsigned forwardViewKinds = 0;
  // A unary pointwise producer of an input root that allows one.
  bool activationPrologue = false;
  // A quantize on the forward chain.
  bool quantizeEpilogue = false;
  // The forward chain may pass a value that is also read outside the kernel;
  // the kernel then returns that value as an extra result.
  bool multiOutput = false;

  bool allowsEpilogue(Operation *op) const {
    return isFusablePointwise(op) ||
           (quantizeEpilogue && isFusableQuantizeEpilogue(op));
  }
  bool allowsInputView(Operation *op) const {
    return hasKind(op, inputViewKinds);
  }
  bool allowsForwardView(Operation *op) const {
    return hasKind(op, forwardViewKinds);
  }
  bool absorbsInputs() const {
    return inputViewKinds != 0 || activationPrologue;
  }

private:
  static bool hasKind(Operation *op, unsigned kinds) {
    std::optional<ViewKind> kind = getFusableViewKind(op);
    return kind && (kinds & static_cast<unsigned>(*kind));
  }
};

// rocMLIR reads every kernel input through rock.transform maps, so a view on
// one becomes an index remap instead of a separate copy: a transpose next to
// a convolution folds into its input_layout / output_layout, and a reshape,
// slice, or broadcast on any input folds into that input's map.
//
// Past the anchor, rocMLIR has to invert the view to map each output element
// back to the GEMM tile that computes it. A transpose or reshape inverts; a
// broadcast does not ("cannot regularize"). A slice would compute output that
// the kernel then drops, so it stays outside too.
//
// A value on the chain that something outside also reads is written out as
// one more kernel result instead of ending the chain: it is stored either way,
// and the epilogue no longer has to load it back.
//
// Convolution, matmul and gemm anchors share these rules; only where input
// absorption starts differs (getInputRoots).
static FusionPolicy getFusionPolicy(bool fuseViews) {
  FusionPolicy policy;
  if (fuseViews) {
    policy.forwardViewKinds = static_cast<unsigned>(ViewKind::Transpose) |
                              static_cast<unsigned>(ViewKind::Reshape);
    policy.inputViewKinds = policy.forwardViewKinds |
                            static_cast<unsigned>(ViewKind::Slice) |
                            static_cast<unsigned>(ViewKind::Broadcast);
    policy.activationPrologue = true;
    policy.quantizeEpilogue = true;
    policy.multiOutput = true;
  }
  return policy;
}

// Where input absorption starts, with whether each root may take a pointwise
// prologue. A conv's filter does not: an elementwise op on it would rerun in
// every workgroup that loads it. Both matmul operands may, since in attention
// both are activations. A gemm's C is added once per output element, so it
// takes views only.
template <typename AnchorOp>
SmallVector<std::pair<Value, bool>> getInputRoots(AnchorOp anchorOp) {
  SmallVector<std::pair<Value, bool>> roots;
  if constexpr (std::is_same_v<AnchorOp, ConvOp>) {
    roots.push_back({anchorOp.getInput(), true});
    roots.push_back({anchorOp.getWeights(), false});
    if (Value bias = anchorOp.getBias())
      roots.push_back({bias, false});
  } else if constexpr (std::is_same_v<AnchorOp, MatmulOp>) {
    roots.push_back({anchorOp.getA(), true});
    roots.push_back({anchorOp.getB(), true});
  } else if constexpr (std::is_same_v<AnchorOp, GemmOp>) {
    roots.push_back({anchorOp.getInputA(), true});
    roots.push_back({anchorOp.getInputB(), true});
    if (Value c = anchorOp.getInputC())
      roots.push_back({c, false});
  } else if constexpr (std::is_same_v<AnchorOp, QuantConvOp> ||
                       std::is_same_v<AnchorOp, QLinearConvOp>) {
    roots.push_back({anchorOp.getInput(), true});
    roots.push_back({anchorOp.getWeights(), false});
  } else if constexpr (std::is_same_v<AnchorOp, QuantMatmulOp> ||
                       std::is_same_v<AnchorOp, QLinearMatMulOp>) {
    roots.push_back({anchorOp.getA(), true});
    roots.push_back({anchorOp.getB(), false});
  }
  return roots;
}

// Anchors that own a forward chain once they are in the graph.
static bool isAnchorOp(Operation *op) {
  return isa<MatmulOp, ConvOp, GemmOp, QuantConvOp, QuantMatmulOp>(op);
}

// Whether DecomposeQLinearConv, DecomposeQLinearMatMul or
// DecomposeDequantizedConv will turn `op` into an integer anchor.

// Anchors whose result is computed in integers, so the dequantize, pointwise,
// and quantize ops after it can run in its epilogue.
static bool isQuantizedAnchor(Operation *op) {
  return isa<QuantConvOp, QuantMatmulOp, QLinearConvOp, QLinearMatMulOp>(op) ||
         (isa<ConvOp>(op) && isPlannedQLinear(op));
}

// A rank-0 value the kernel can recompute from inline scalar constants: such a
// constant, or a dequantize or reciprocal of them (a QLinearAdd's constant
// operand, a Div's divisor).
static bool isConstantScalarExpr(Value v) {
  if (isInlineScalarConstant(v))
    return true;
  Operation *def = v.getDefiningOp();
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!def || !type || type.getRank() != 0)
    return false;
  bool isDequantize = isa<DequantizeLinearOp>(def) && isTosaLowerableQdq(def);
  if (!isDequantize && !isa<ReciprocalOp>(def))
    return false;
  return llvm::all_of(
      cast<DestinationStyleOpInterface>(def).getDpsInputs(), [](Value input) {
        return isa<ContextType>(input.getType()) || isConstantScalarExpr(input);
      });
}

// The ops isConstantScalarExpr walked, which the kernel clones.
static void collectConstantScalarExpr(Value v, SetVector<Operation *> &ops) {
  Operation *def = v.getDefiningOp();
  if (!ops.insert(def) || isInlineScalarConstant(v))
    return;
  for (Value input : cast<DestinationStyleOpInterface>(def).getDpsInputs())
    if (!isa<ContextType>(input.getType()))
      collectConstantScalarExpr(input, ops);
}

// The ops one rock.kernel holds and the values it reads from outside, grown
// outward from an anchor. The steps run in the order matchAndRewrite calls
// them; kernel argument order follows insertion order, so reordering them
// changes the kernel signature.
class KernelRegion {
public:
  KernelRegion(Operation *anchor, const FusionPolicy &policy)
      : anchor(anchor), policy(policy) {}

  // The last op of the forward chain; a DPS op, so its init becomes the
  // dispatch's last outs.
  DestinationStyleOpInterface getEnd() const { return end; }
  // What the kernel returns, in chain order: each chain value read outside
  // the kernel, ending with the end's result. Each comes from a DPS op.
  ArrayRef<Value> getResults() const { return results; }
  Value getContext() const { return context; }
  SetVector<Value> &getOperands() { return operands; }
  SetVector<Operation *> &getOps() { return ops; }

  // Follows pointwise ops, and views the policy allows, from the anchor
  // forward. Without the multi-output policy every op on the chain has one
  // reader. With it, a chain value may have readers outside; those readers
  // can feed a later op's other operand, so each candidate is checked for a
  // path back into the chain, which would make the kernel read its own output.
  void growForward();
  // Drops the leading hip.context the anchor's inputs carry.
  bool takeContext();
  // Moves inline scalar constants into the kernel.
  void cloneInlineScalars();
  // A bitsandbytes dequant that feeds only this anchor belongs in the same
  // rock kernel. HipToTosa turns it into a nibble lookup and a multiply, and
  // the matmul into tosa.matmul. A dynamic activation never reaches this
  // pattern, so that case stays a HIP dequant plus hipBLASLt.
  void absorbDequantBnb4();
  void absorbInputProducers(ArrayRef<std::pair<Value, bool>> roots);
  void absorbOperandFreeProducers();
  // Whatever is still read from outside has to be passed in, which costs one
  // more kernel argument and is what any init producer that cannot be
  // rematerialized -- a caller-supplied buffer, say -- falls back on.
  void collectRemainingOperands();
  std::optional<SmallVector<Operation *>> readersToMove() const;

private:
  Operation *nextOnChain(const SetVector<Operation *> &chain) const;
  void growQuantizedEpilogue(SetVector<Operation *> &chain) const;

  Operation *anchor;
  const FusionPolicy &policy;
  DestinationStyleOpInterface end;
  SmallVector<Value> results;
  Value context;
  SetVector<Value> operands;
  SetVector<Operation *> ops;
  // What the forward chain past the anchor reads from outside: the residuals,
  // biases, and scales of the epilogue.
  SmallVector<Value> epilogueInputs;
};

// A pointwise op that broadcasts the chain value maps many output elements
// back to one GEMM element, which rocMLIR cannot invert ("cannot
// regularize"), so the chain value must keep its shape through it.
//
//   %m = hip.matmul ... -> tensor<1x1x1536xf16>
//   %a = hip.add ins(%r, %m : tensor<1x2x1536xf16>, tensor<1x1x1536xf16>)
//   // %a stays outside the kernel
static bool keepsChainShape(Operation *user, Value value) {
  return llvm::all_of(user->getResultTypes(), [&](Type type) {
    auto result = dyn_cast<ShapedType>(type);
    auto chain = dyn_cast<ShapedType>(value.getType());
    return result && chain && result.getShape() == chain.getShape();
  });
}

Operation *
KernelRegion::nextOnChain(const SetVector<Operation *> &chain) const {
  Operation *tail = chain.back();
  if (tail->getNumResults() != 1)
    return nullptr;
  Value value = tail->getResult(0);
  auto canTake = [&](Operation *user) {
    auto loadable = [&](Value v) {
      return isKernelIOValue(v) || isInlineScalarConstant(v) ||
             chain.contains(v.getDefiningOp());
    };
    if (!llvm::all_of(user->getOperands(), loadable))
      return false;
    return (policy.allowsEpilogue(user) && keepsChainShape(user, value)) ||
           policy.allowsForwardView(user);
  };
  if (!policy.multiOutput) {
    if (!value.hasOneUse())
      return nullptr;
    Operation *user = *value.user_begin();
    return canTake(user) ? user : nullptr;
  }

  SetVector<Operation *> users(value.user_begin(), value.user_end());
  SmallVector<Operation *> candidates;
  for (Operation *user : users)
    if (user->getBlock() == anchor->getBlock() && canTake(user))
      candidates.push_back(user);
  llvm::sort(candidates,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *user : candidates) {
    bool closesCycle = llvm::any_of(user->getOperands(), [&](Value operand) {
      return !chain.contains(operand.getDefiningOp()) &&
             dependsOnChain(operand, anchor, chain);
    });
    if (!closesCycle)
      return user;
  }
  return nullptr;
}

// Past the requantize of a quantized anchor, its readers form a graph of
// dequantizes, pointwise ops, and quantizes, not a chain. The kernel takes the
// closure of that graph: every op whose operands are kernel values, inline
// scalars, values the kernel can recompute from them, or inputs that do not
// depend on the kernel. A dequantize outside that only the kernel reads (the
// other side of a residual add) joins too, so the kernel reads its 8-bit
// input. A value that a dequantize widened is never written out: whatever
// would return it, or reads it outside, stays outside, and each result is an
// 8-bit value or one from before the first dequantize.
//
// Before:
//   %y  = <quant_conv, requantize>             -> tensor<1x16x8x8xui8>
//   %d0 = hip.dequantize_linear ins(%y, ...)
//   %c  = hip.dequantize_linear ins(%three, ...) : tensor<f32>
//   %a  = hip.add ins(%d0, %c)
//   %t  = hip.quantize_linear ins(%a, ...)
//   %d1 = hip.dequantize_linear ins(%t, ...)
//   %r  = hip.reciprocal ins(%six)
//   %m  = hip.mul ins(%d1, %r)
//   %t2 = hip.quantize_linear ins(%m, ...)
//   %d2 = hip.dequantize_linear ins(%y, ...)
//   %d3 = hip.dequantize_linear ins(%t2, ...)
//   %h  = hip.mul ins(%d2, %d3)
//   %z  = hip.quantize_linear ins(%h, ...)    // read by the next conv
// After:
//   %z = hip.rocmlir @rocMlir0(...)            // all of the above
void KernelRegion::growQuantizedEpilogue(SetVector<Operation *> &chain) const {
  Block *block = anchor->getBlock();
  auto admits = [&](Operation *op) {
    return op->getBlock() == block && op->getNumResults() == 1 &&
           isa<DestinationStyleOpInterface>(op) &&
           (policy.allowsEpilogue(op) ||
            (isa<DequantizeLinearOp>(op) && isTosaLowerableQdq(op)));
  };
  auto pullableDequantize = [&](Value value, Operation *user) -> Operation * {
    auto dequantize = value.getDefiningOp<DequantizeLinearOp>();
    if (!dequantize || dequantize->getBlock() != block ||
        !isTosaLowerableQdq(dequantize) ||
        !isStaticTensor(dequantize.getInput()) ||
        !isKernelIOValue(dequantize.getInput()))
      return nullptr;
    bool onlyKernelReads = llvm::all_of(dequantize->getUsers(), [&](auto *u) {
      return u == user || chain.contains(u);
    });
    return onlyKernelReads ? dequantize.getOperation() : nullptr;
  };

  bool grew = true;
  while (grew) {
    grew = false;
    SmallVector<Operation *> users;
    for (Operation *op : chain)
      for (Operation *user : op->getUsers())
        if (!chain.contains(user) && admits(user))
          users.push_back(user);
    llvm::sort(users, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    users.erase(llvm::unique(users), users.end());
    for (Operation *user : users) {
      if (chain.contains(user))
        continue;
      SmallVector<Operation *> pulled;
      bool joins = true;
      for (Value input :
           cast<DestinationStyleOpInterface>(user).getDpsInputs()) {
        if (isa<ContextType>(input.getType()))
          continue;
        Operation *def = input.getDefiningOp();
        if (def && chain.contains(def)) {
          joins = keepsChainShape(user, input);
        } else if (!isConstantScalarExpr(input)) {
          joins = isStaticTensor(input) && isKernelIOValue(input) &&
                  !dependsOnChain(input, anchor, chain);
          if (Operation *dequantize = pullableDequantize(input, user))
            pulled.push_back(dequantize);
        }
        if (!joins)
          break;
      }
      if (!joins)
        continue;
      chain.insert_range(pulled);
      chain.insert(user);
      grew = true;
    }
  }

  auto cut = [&](Operation *root) {
    SmallVector<Operation *> worklist = {root};
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (op == anchor || !chain.remove(op))
        continue;
      llvm::append_range(worklist, op->getUsers());
    }
  };
  auto byBlockOrder = [&] {
    SmallVector<Operation *> ordered(chain.begin(), chain.end());
    llvm::sort(ordered, [&](Operation *a, Operation *b) {
      return a == anchor || (b != anchor && a->isBeforeInBlock(b));
    });
    return ordered;
  };
  bool cutAny = true;
  while (cutAny) {
    cutAny = false;
    DenseSet<Operation *> dequantized;
    for (Operation *op : byBlockOrder()) {
      if (isa<DequantizeLinearOp>(op) ||
          (!isa<QuantizeLinearOp>(op) &&
           llvm::any_of(op->getOperands(), [&](Value v) {
             return dequantized.contains(v.getDefiningOp());
           })))
        dequantized.insert(op);
    }
    for (Operation *op : byBlockOrder()) {
      if (!dequantized.contains(op) ||
          !isa<FloatType>(getElementTypeOrSelf(op->getResult(0).getType())))
        continue;
      bool written =
          op->use_empty() || llvm::any_of(op->getUsers(), [&](Operation *user) {
            return !chain.contains(user);
          });
      if (written) {
        cut(op);
        cutAny = true;
        break;
      }
    }
  }
  SmallVector<Operation *> ordered = byBlockOrder();
  chain.clear();
  chain.insert_range(ordered);
}

void KernelRegion::growForward() {
  SetVector<Operation *> chain;
  chain.insert(anchor);
  while (Operation *next = nextOnChain(chain))
    chain.insert(next);
  if (isQuantizedAnchor(anchor) && policy.multiOutput &&
      policy.quantizeEpilogue)
    growQuantizedEpilogue(chain);
  // A chain value read outside becomes a kernel result, which needs an init,
  // so a view with an outside reader ends the chain. A view whose readers all
  // joined the chain stays inside, as in an erf GELU after a matmul:
  //
  //   %v = tensor.expand_shape %matmul        // read by %s and %g
  //   %s = hip.mul ins(%v, %scale)
  //   %e = hip.erf ins(%s)
  //   %a = hip.add ins(%e, %one)
  //   %g = hip.mul ins(%v, %a)                // all of it in one kernel
  auto escapingView = llvm::find_if(chain, [&](Operation *op) {
    return !isa<DestinationStyleOpInterface>(op) &&
           llvm::any_of(op->getUsers(),
                        [&](Operation *user) { return !chain.contains(user); });
  });
  if (escapingView != chain.end()) {
    Operation *cut = *escapingView;
    while (chain.back() != cut)
      chain.pop_back();
  }
  // A view outside the kernel is free, and one with no init cannot supply the
  // dispatch's outs. The end's result is stored, so it must be a type the
  // kernel can store; a wider integer may still live inside the chain.
  while (!isa<DestinationStyleOpInterface>(chain.back()) ||
         (chain.back() != anchor &&
          !llvm::all_of(chain.back()->getResultTypes(), isKernelIOType)))
    chain.pop_back();

  for (Operation *op : chain) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    SmallVector<Value> inputs = dps ? SmallVector<Value>(dps.getDpsInputs())
                                    : SmallVector<Value>(op->getOperands());
    if (op != anchor) {
      llvm::erase_if(inputs, [&](Value input) {
        return chain.contains(input.getDefiningOp());
      });
      llvm::append_range(epilogueInputs, inputs);
    }
    operands.insert_range(inputs);
    if (dps) {
      for (Value init : dps.getDpsInits()) {
        // An init with no producer is a block argument, which cannot be
        // cloned. It is picked up as a kernel argument later instead.
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
      }
      end = dps;
    }
    ops.insert(op);
  }

  for (Operation *op : chain) {
    if (op == end.getOperation())
      break;
    bool readOutside = llvm::any_of(
        op->getUsers(), [&](Operation *user) { return !chain.contains(user); });
    if (readOutside)
      results.push_back(op->getResult(0));
  }
  llvm::append_range(results, end->getResults());
}

bool KernelRegion::takeContext() {
  if (operands.empty() || !isa<ContextType>(operands.front().getType()))
    return false;
  context = operands.front();
  operands.erase(operands.begin());
  return true;
}

// Pointwise scalar literals belong in the outlined subgraph, not the kernel
// ABI. A rank-0 buffer argument has no index for rock.transforms_to_ptr. Clone
// only inline constants, and for a quantized anchor the dequantizes and
// reciprocals of them its epilogue reads: a dynamic Clip bound remains an
// argument.
void KernelRegion::cloneInlineScalars() {
  SetVector<Value> kernelOperands;
  bool quantized = isQuantizedAnchor(anchor);
  for (Value operand : operands) {
    if (isInlineScalarConstant(operand) ||
        (quantized && isConstantScalarExpr(operand))) {
      collectConstantScalarExpr(operand, ops);
      continue;
    }
    kernelOperands.insert(operand);
  }
  operands = std::move(kernelOperands);
}

void KernelRegion::absorbDequantBnb4() {
  SmallVector<Value> dequantResults;
  for (Value operand : operands) {
    auto dequant = operand.getDefiningOp<DequantBnb4Op>();
    if (!dequant || !dequant->hasOneUse())
      continue;
    dequantResults.push_back(operand);
  }
  for (Value value : dequantResults) {
    auto dequant = value.getDefiningOp<DequantBnb4Op>();
    ops.insert(dequant);
    operands.remove(value);
    operands.insert(dequant.getPacked());
    operands.insert(dequant.getAbsmax());
  }
}

// The values an absorbed producer reads as data, which become kernel arguments
// unless absorbed in turn. A hip.expand's target shape is not data: HipToTosa
// takes the broadcast from the result type, so the shape only has to resolve
// inside the kernel, and absorbOperandFreeProducers clones a constant one.
// Returns nullopt for an expand whose shape is computed at run time.
static std::optional<SmallVector<Value>> getDataInputs(Operation *producer,
                                                       Value context) {
  if (auto expand = dyn_cast<ExpandOp>(producer)) {
    Operation *shape = expand.getShape().getDefiningOp();
    if (!shape || shape->getNumOperands() != 0 || !isMemoryEffectFree(shape))
      return std::nullopt;
    return SmallVector<Value>{expand.getInput()};
  }
  auto dps = dyn_cast<DestinationStyleOpInterface>(producer);
  SmallVector<Value> inputs = dps ? SmallVector<Value>(dps.getDpsInputs())
                                  : SmallVector<Value>(producer->getOperands());
  llvm::erase(inputs, context);
  return inputs;
}

// Whether `op` is on the forward chain of an anchor that is still in the
// graph. Such an op belongs to that anchor's epilogue: there it runs once per
// output element, while as a prologue of the next kernel it would rerun on
// every load of that kernel's input. Checking the claim keeps the result
// independent of the order the greedy driver visits anchors in: epilogues
// first, then the leftover pointwise ops as prologues.
//
//   %c0 = hip.conv ins(%x, %w0, %b0)
//   %r  = hip.max ins(%c0, %zero)       // claimed by %c0's epilogue
//   %c1 = hip.conv ins(%r, %w1, %b1)    // so %c1 does not take %r
//
// A quantized anchor's epilogue reaches past dequantizes and quantizes, so the
// walk does too, and an op that will become an integer anchor already claims
// its epilogue.
static bool isClaimedByForwardChain(Operation *op, const FusionPolicy &policy) {
  SmallVector<Operation *> worklist = {op};
  DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (Value operand : current->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || def->getBlock() != op->getBlock())
        continue;
      if (isAnchorOp(def) || isPlannedQLinear(def)) {
        if (!llvm::all_of(def->getOperands(), [](Value v) {
              auto type = dyn_cast<TensorType>(v.getType());
              return !type || type.hasStaticShape();
            }))
          continue;
        KernelRegion region(def, policy);
        region.growForward();
        if (region.getOps().contains(op))
          return true;
        continue;
      }
      if (policy.allowsEpilogue(def) || policy.allowsForwardView(def) ||
          isTosaLowerableQdq(def))
        worklist.push_back(def);
    }
  }
  return false;
}

// Pulls the producers of kernel inputs into the kernel: any view the policy
// allows, on the anchor's inputs and on the epilogue's, and on a root that
// permits it a unary pointwise op. A producer is absorbed only when the kernel
// is its only reader. One that keeps running for another reader saves nothing
// -- a hip.transpose outside rocMLIR is a real copy -- and the kernel would
// become a second reader of the producer's input, which stops that input's
// own kernel from growing forward through the producer.
//
// Before:
//   %t0 = hip.transpose ins(%x : tensor<1x96x96x512xf16>) {perm = [0,3,1,2]}
//   %c  = hip.conv ins(%t0, %w, %b) -> tensor<1x512x96x96xf16>
//   %t1 = hip.transpose ins(%c) {perm = [0,2,3,1]}
//   %rb = hip.expand ins(%r : tensor<1x1x1x512xf16>, %shape)
//   %a  = hip.add ins(%t1, %rb) -> tensor<1x96x96x512xf16>
// After:
//   %a = hip.rocmlir @rocMlir0(%x, %w, %b, %r)
//   func @rocMlir0 { transpose -> conv -> transpose, expand -> add }
void KernelRegion::absorbInputProducers(
    ArrayRef<std::pair<Value, bool>> roots) {
  if (!policy.absorbsInputs())
    return;
  auto onlyReadInKernel = [&](Operation *op) {
    return llvm::all_of(op->getUsers(),
                        [&](Operation *user) { return ops.contains(user); });
  };
  SmallVector<Value> original(operands.begin(), operands.end());
  DenseMap<Value, SmallVector<Value>> replacedBy;
  SmallVector<std::pair<Value, bool>> worklist(roots.begin(), roots.end());
  for (Value input : epilogueInputs)
    worklist.push_back({input, false});

  while (!worklist.empty()) {
    auto [value, allowPointwise] = worklist.pop_back_val();
    if (!operands.contains(value))
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer || ops.contains(producer) || !onlyReadInKernel(producer))
      continue;
    std::optional<SmallVector<Value>> dataInputs =
        getDataInputs(producer, context);
    if (!dataInputs)
      continue;
    SmallVector<Value> inputs;
    for (Value input : *dataInputs)
      if (!isInlineScalarConstant(input))
        inputs.push_back(input);
    bool absorb = policy.allowsInputView(producer) ||
                  (allowPointwise && policy.activationPrologue &&
                   isFusablePrologue(producer) && inputs.size() == 1 &&
                   !isClaimedByForwardChain(producer, policy));
    if (!absorb || !llvm::all_of(inputs, isStaticTensor) ||
        !llvm::all_of(inputs, isKernelIOValue))
      continue;

    ops.insert(producer);
    operands.remove(value);
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(producer))
      for (Value init : dps.getDpsInits())
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
    for (Value input : *dataInputs) {
      if (isInlineScalarConstant(input)) {
        ops.insert(input.getDefiningOp());
        continue;
      }
      operands.insert(input);
      replacedBy[value].push_back(input);
      worklist.push_back({input, allowPointwise});
    }
  }

  // An absorbed producer's inputs take the place of the value it produced,
  // so the kernel arguments keep the order of the anchor's operands.
  SetVector<Value> ordered;
  std::function<void(Value)> place = [&](Value value) {
    if (operands.contains(value))
      ordered.insert(value);
    for (Value input : replacedBy.lookup(value))
      place(input);
  };
  for (Value value : original)
    place(value);
  assert(ordered.size() == operands.size() && "kernel argument lost");
  operands = std::move(ordered);
}

// `ops` is cloned into an IsolatedFromAbove func, so every value it reads has
// to resolve inside that func. The DPS inputs above cover the data, but they do
// not cover what the inits are built from: an init is only a bare tensor.empty
// when the anchor already has the rank rocMLIR wants. A 1-D convolution does
// not -- convert-onnx-to-hip widens it to 2-D by wrapping the operands *and the
// init* in tensor.expand_shape -- so the init's producer is a reshape whose own
// operand is the empty. Cloning just the reshape leaves it pointing at an empty
// back in main_graph.
//
// Absorb producers that carry no operands and no side effects, so a
// materialised destination travels with the ops that use it instead of
// becoming a kernel argument. That keeps the signature rocMLIR sees unchanged
// for the shapes that already worked. Values already headed for the argument
// list stay there: a non-scalar weight is passed in even though hip.constant
// would otherwise qualify to be cloned.
void KernelRegion::absorbOperandFreeProducers() {
  SmallVector<Operation *> worklist(ops.begin(), ops.end());
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      if (operands.contains(operand))
        continue;
      Operation *producer = operand.getDefiningOp();
      if (!producer || ops.contains(producer))
        continue;
      if (producer->getNumOperands() != 0 || !isMemoryEffectFree(producer))
        continue;
      ops.insert(producer);
      worklist.push_back(producer);
    }
  }
}

void KernelRegion::collectRemainingOperands() {
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      if (operand != context && !ops.contains(operand.getDefiningOp()))
        operands.insert(operand);
  // Cloning follows this order, so a producer absorbed above has to come
  // before the op that reads it.
  ops = topologicalSort(ops);
}

// The dispatch goes where the end is, so an outside reader of an earlier
// result that sits above the end has to move below the dispatch, together
// with whatever reads it in turn. Returns them in block order, or nullopt if
// one feeds the kernel, which growForward's cycle check rules out.
//
// Before:
//   %c = hip.conv ins(%x, %w, %b)
//   %m = hip.reduce_mean ins(%c, %axes)       <- outside reader of %c
//   %s = hip.sigmoid ins(%c)
// After:
//   %r:2 = hip.rocmlir @rocMlir0 ins(%x, %w, %b) outs(%e0, %e1)
//   %m = hip.reduce_mean ins(%r#0, %axes)
std::optional<SmallVector<Operation *>> KernelRegion::readersToMove() const {
  SmallVector<Operation *> moved;
  if (results.size() == end->getNumResults())
    return moved;
  DenseSet<Value> escaped(results.begin(),
                          results.end() - end->getNumResults());
  auto readsEscaped = [&](Operation *op) {
    WalkResult walk = op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (escaped.contains(operand))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return walk.wasInterrupted();
  };
  Operation *first = results.front().getDefiningOp();
  for (Operation &op :
       llvm::make_range(std::next(first->getIterator()), end->getIterator())) {
    if (ops.contains(&op) || !readsEscaped(&op))
      continue;
    moved.push_back(&op);
    escaped.insert(op.result_begin(), op.result_end());
  }
  for (Value operand : operands)
    if (escaped.contains(operand))
      return std::nullopt;
  return moved;
}

// Grows `region` from its anchor, in the order that fixes the kernel
// signature. Only reads the IR. Returns why the anchor cannot be outlined, or
// null when it can; `moved` then holds the readers to move below the dispatch.
static const char *growRegion(KernelRegion &region,
                              ArrayRef<std::pair<Value, bool>> roots,
                              std::optional<SmallVector<Operation *>> &moved) {
  region.growForward();
  if (!region.takeContext())
    return "first operand not hip.context";
  region.cloneInlineScalars();
  region.absorbDequantBnb4();
  region.absorbInputProducers(roots);
  region.absorbOperandFreeProducers();
  region.collectRemainingOperands();
  if (!llvm::all_of(region.getOperands(), isKernelIOValue) ||
      !llvm::all_of(region.getResults(), isKernelIOValue))
    return "a kernel argument or result has an integer wider than 32 bits";
  moved = region.readersToMove();
  if (!moved)
    return "an outside reader of a kernel result feeds the kernel";
  return nullptr;
}

template <typename AnchorOp>
class FuseAnchorPointwise : public OpRewritePattern<AnchorOp> {
public:
  FuseAnchorPointwise(MLIRContext *context, int *counter, bool fuseViews,
                      StringRef arch, PatternBenefit benefit = 1)
      : OpRewritePattern<AnchorOp>(context, benefit), counter(counter),
        policy(getFusionPolicy(fuseViews)), arch(arch.str()) {}

  LogicalResult matchAndRewrite(AnchorOp anchorOp,
                                PatternRewriter &rewriter) const override {
    // An i32 matmul or convolution has an exact HIP kernel. Outlining it
    // asks rocMLIR to compile a non-quantized integer tosa op, which this
    // pipeline does not establish. Leave the op in main_graph. The i32
    // accumulator of a quantized anchor is its init, not an input.
    for (Value input : anchorOp.getDpsInputs()) {
      auto shaped = dyn_cast<ShapedType>(input.getType());
      if (shaped && shaped.getElementType().isInteger(32))
        return rewriter.notifyMatchFailure(
            anchorOp, "i32 anchor stays on the HIP kernel");
    }

    // Fail on dynamic shapes
    for (Value operand : anchorOp->getOperands()) {
      if (auto tensorType = dyn_cast<TensorType>(operand.getType());
          tensorType && !tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(anchorOp,
                                           "only static shapes are supported");
    }

    KernelRegion region(anchorOp, policy);
    std::optional<SmallVector<Operation *>> moved;
    if (const char *reason = growRegion(region, getInputRoots(anchorOp), moved))
      return rewriter.notifyMatchFailure(anchorOp, reason);

    DestinationStyleOpInterface endOp = region.getEnd();
    Value context = region.getContext();
    SetVector<Value> &operands = region.getOperands();
    ArrayRef<Value> results = region.getResults();
    auto resultTypes = llvm::map_to_vector(
        results, [](Value v) -> Type { return v.getType(); });

    auto parentModule = anchorOp->template getParentOfType<ModuleOp>();
    func::FuncOp newFunc;
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      auto funcType = rewriter.getFunctionType(
          llvm::map_to_vector(
              operands,
              [&region](Value v) {
                return getRocMlirKernelArgumentType(v, region.getOps());
              }),
          resultTypes);

      newFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                     "rocMlir" + std::to_string((*counter)++),
                                     funcType);
      newFunc->setAttr("rock.kernel", rewriter.getUnitAttr());
      if (!arch.empty())
        newFunc->setAttr("rock.arch", rewriter.getStringAttr(arch));
      auto *funcBlock = newFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(funcBlock);
      IRMapping mapping;

      // Map the hip.context to ub.poison
      mapping.map(context,
                  ub::PoisonOp::create(rewriter, rewriter.getUnknownLoc(),
                                       context.getType()));

      // Map other operands
      for (auto [idx, operand] : llvm::enumerate(operands)) {
        mapping.map(operand, funcBlock->getArgument(idx));
      }

      for (auto *op : region.getOps()) {
        rewriter.clone(*op, mapping);
      }
      auto returns = llvm::map_to_vector(
          results, [&mapping](Value v) { return mapping.lookup(v); });
      func::ReturnOp::create(rewriter, rewriter.getUnknownLoc(), returns);
    }

    Operation *last = endOp;
    for (Operation *op : *moved) {
      rewriter.moveOpAfter(op, last);
      last = op;
    }
    rewriter.setInsertionPointAfter(endOp);

    // The graph outside keeps its rank-0 value; only what crosses into the
    // kernel is reshaped, so the dispatch matches the signature built above.
    SmallVector<Value> dispatchOperands;
    dispatchOperands.reserve(operands.size());
    Value rankOneShape;
    for (Value operand : operands) {
      Type promoted = promoteRankZero(operand.getType());
      if (promoted == operand.getType()) {
        dispatchOperands.push_back(operand);
        continue;
      }
      if (!rankOneShape) {
        auto shapeType = RankedTensorType::get({1}, rewriter.getIndexType());
        rankOneShape = arith::ConstantOp::create(
            rewriter, rewriter.getUnknownLoc(), shapeType,
            DenseIntElementsAttr::get(shapeType, ArrayRef<int64_t>{1}));
      }
      dispatchOperands.push_back(tensor::ReshapeOp::create(
          rewriter, rewriter.getUnknownLoc(), promoted, operand, rankOneShape));
    }

    auto inits = llvm::map_to_vector(results, [](Value v) {
      auto result = cast<OpResult>(v);
      return cast<DestinationStyleOpInterface>(result.getOwner())
          .getTiedOpOperand(result)
          ->get();
    });
    auto rocMlirOp = RocMlirOp::create(rewriter, rewriter.getUnknownLoc(),
                                       resultTypes, SymbolRefAttr::get(newFunc),
                                       context, dispatchOperands, inits);

    SetVector<Operation *> &kernelOps = region.getOps();
    size_t numEscaped = results.size() - endOp->getNumResults();
    for (auto [value, replacement] :
         llvm::zip(results.take_front(numEscaped), rocMlirOp.getResults())) {
      rewriter.replaceUsesWithIf(value, replacement, [&](OpOperand &use) {
        return !kernelOps.contains(use.getOwner());
      });
    }
    rewriter.replaceOp(endOp, rocMlirOp.getResults().drop_front(numEscaped));
    return success();
  }

private:
  int *counter = nullptr;
  FusionPolicy policy;
  std::string arch;
};

// With rocMLIR on, a QLinearConv that planQLinearConv accepts becomes an
// i8 x i8 -> i32 convolution with zero points 0, a constant correction, and an
// f32 requantize. FuseAnchorPointwise<QuantConvOp> then outlines the chain as
// one rocMLIR kernel; a quantize whose scale is 1 rounds half to even, adds the
// zero point, and saturates, as ORT does.
//
// Before:
//   %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b)
//          outs(%e : tensor<1x32x8x8xui8>) {pads = [1, 1, 1, 1], ...}
// After:
//   %acc = hip.quant_conv(%ctx) ins(%x, %w) outs(%e0 : tensor<1x32x8x8xi32>)
//   %a   = hip.add(%ctx) ins(%acc, %bias : ..., tensor<1x32x8x8xi32>)
//   %f   = hip.cast(%ctx) ins(%a) outs(%e1 : tensor<1x32x8x8xf32>)
//   %s   = hip.mul(%ctx) ins(%f, %m : ..., tensor<f32>)
//   %y   = hip.quantize_linear(%ctx) ins(%s, %one) zero_point(%yz)
//            outs(%e : tensor<1x32x8x8xui8>)
//
// The rewrite runs only when the original op would outline, so a declined
// QLinearConv leaves the IR unchanged and keeps its HIP kernel.
class DecomposeQLinearConv : public OpRewritePattern<QLinearConvOp> {
public:
  DecomposeQLinearConv(MLIRContext *context, bool fuseViews)
      : OpRewritePattern<QLinearConvOp>(context),
        policy(getFusionPolicy(fuseViews)) {}

  LogicalResult matchAndRewrite(QLinearConvOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    std::optional<QLinearPlan> plan = planQLinearConv(op);
    if (!plan)
      return rewriter.notifyMatchFailure(
          op, "not an inline-constant QLinearConv with a zero weight zp");
    KernelRegion region(op, policy);
    std::optional<SmallVector<Operation *>> moved;
    if (const char *reason = growRegion(region, getInputRoots(op), moved))
      return rewriter.notifyMatchFailure(op, reason);

    Location loc = op.getLoc();
    auto accType = cast<RankedTensorType>(op.getOutput().getType())
                       .clone(rewriter.getI32Type());
    Value init = tensor::EmptyOp::create(rewriter, loc, accType.getShape(),
                                         accType.getElementType());
    Value acc =
        QuantConvOp::create(rewriter, loc, TypeRange{accType}, op.getCtx(),
                            op.getInput(), op.getWeights(), init,
                            op.getKernelShape(), op.getStrides(), op.getPads(),
                            op.getDilations(), op.getGroupAttr())
            .getResult(0);
    rewriter.replaceOp(op, emitRequantizeEpilogue(rewriter, loc, op.getCtx(),
                                                  acc, *plan, op.getOutput()));
    return success();
  }

private:
  FusionPolicy policy;
};

// With rocMLIR on, a conv that planDequantizedConv accepts reads the integer
// operands of its dequantizes through hip.quant_conv. The i32 sum is exact,
// then the kernel scales it by x_scale * w_scale and adds the bias in f32, so
// the activation and quantize after the conv form its epilogue.
//
// Before:
//   %dx = hip.dequantize_linear(%ctx) ins(%x, %xs) zero_point(%xz)
//   %dw = hip.dequantize_linear(%ctx) ins(%w, %ws) zero_point(%wz)
//   %db = hip.dequantize_linear(%ctx) ins(%b, %bs) zero_point(%bz)
//   %c  = hip.conv(%ctx) ins(%dx, %dw, %db) outs(%e : tensor<1x32x8x8xf32>)
// After:
//   %acc = hip.quant_conv(%ctx) ins(%x, %w) outs(%e0 : tensor<1x32x8x8xi32>)
//   %f   = hip.cast(%ctx) ins(%acc) outs(%e1 : tensor<1x32x8x8xf32>)
//   %s   = hip.mul(%ctx) ins(%f, %m : ..., tensor<f32>)
//   %c   = hip.add(%ctx) ins(%s, %bias : ..., tensor<1x32x1x1xf32>)
//
// The rewrite runs only when the conv would outline, so a declined conv leaves
// the IR unchanged and keeps its float kernel.
class DecomposeDequantizedConv : public OpRewritePattern<ConvOp> {
public:
  DecomposeDequantizedConv(MLIRContext *context, bool fuseViews)
      : OpRewritePattern<ConvOp>(context, /*benefit=*/2),
        policy(getFusionPolicy(fuseViews)) {}

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<DequantizedConvPlan> plan = planDequantizedConv(op);
    if (!plan)
      return rewriter.notifyMatchFailure(op, "not a conv of dequantizes");
    KernelRegion region(op, policy);
    std::optional<SmallVector<Operation *>> moved;
    if (const char *reason = growRegion(region, getInputRoots(op), moved))
      return rewriter.notifyMatchFailure(op, reason);

    Location loc = op.getLoc();
    Value ctx = op.getCtx();
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    auto accType = resultType.clone(rewriter.getI32Type());
    Value acc =
        QuantConvOp::create(
            rewriter, loc, TypeRange{accType}, ctx, plan->input, plan->weights,
            emitTensorEmpty(rewriter, loc, accType), op.getKernelShape(),
            op.getStrides(), op.getPads(), op.getDilations(), op.getGroupAttr())
            .getResult(0);
    Value result = emitScaledAccumulator(rewriter, loc, ctx, acc,
                                         plan->correction, plan->multiplier);
    if (plan->bias)
      result = AddOp::create(rewriter, loc, TypeRange{resultType}, ctx, result,
                             emitConstant(rewriter, loc, plan->bias),
                             emitTensorEmpty(rewriter, loc, resultType))
                   .getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  FusionPolicy policy;
};

// The QLinearMatMul form of DecomposeQLinearConv.
//
// Before:
//   %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz)
//          outs(%e : tensor<4x16xui8>)
// After:
//   %acc = hip.quant_matmul(%ctx) ins(%a, %b) outs(%e0 : tensor<4x16xi32>)
//   %c   = hip.add(%ctx) ins(%acc, %bias : ..., tensor<1x16xi32>)
//   %f   = hip.cast(%ctx) ins(%c) outs(%e1 : tensor<4x16xf32>)
//   %s   = hip.mul(%ctx) ins(%f, %m : ..., tensor<f32>)
//   %y   = hip.quantize_linear(%ctx) ins(%s, %one) zero_point(%yz)
//            outs(%e : tensor<4x16xui8>)
class DecomposeQLinearMatMul : public OpRewritePattern<QLinearMatMulOp> {
public:
  DecomposeQLinearMatMul(MLIRContext *context, bool fuseViews)
      : OpRewritePattern<QLinearMatMulOp>(context),
        policy(getFusionPolicy(fuseViews)) {}

  LogicalResult matchAndRewrite(QLinearMatMulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    std::optional<QLinearPlan> plan = planQLinearMatMul(op);
    if (!plan)
      return rewriter.notifyMatchFailure(
          op, "not an inline-constant QLinearMatMul with a zero B zp");
    KernelRegion region(op, policy);
    std::optional<SmallVector<Operation *>> moved;
    if (const char *reason = growRegion(region, getInputRoots(op), moved))
      return rewriter.notifyMatchFailure(op, reason);

    Location loc = op.getLoc();
    auto accType = cast<RankedTensorType>(op.getOutput().getType())
                       .clone(rewriter.getI32Type());
    Value init = tensor::EmptyOp::create(rewriter, loc, accType.getShape(),
                                         accType.getElementType());
    Value acc = QuantMatmulOp::create(rewriter, loc, TypeRange{accType},
                                      op.getCtx(), op.getA(), op.getB(), init)
                    .getResult(0);
    rewriter.replaceOp(op, emitRequantizeEpilogue(rewriter, loc, op.getCtx(),
                                                  acc, *plan, op.getOutput()));
    return success();
  }

private:
  FusionPolicy policy;
};

class FuseROCMlirPass : public impl::FuseROCMlirPassBase<FuseROCMlirPass> {
public:
  using impl::FuseROCMlirPassBase<FuseROCMlirPass>::FuseROCMlirPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    impl::FuseROCMlirPassBase<FuseROCMlirPass>::getDependentDialects(registry);
    registerElementMoveExternalModels(registry);
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;
    // Before outlining, so an anchor's 8-bit result feeds the moved ops and a
    // requantize left on one input joins that anchor's epilogue.
    sinkQuantizeThroughElementMoves(funcOp);
    // After the sinking, which fixes each tree's final shape, and before the
    // QLinear and dequantized-conv rewrites, which then plan on i8 inputs.
    rebiasUnsignedQuantizedTrees(funcOp, isPlannedQLinear);

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    int counter = 0;
    patterns.add<FuseAnchorPointwise<MatmulOp>, FuseAnchorPointwise<ConvOp>,
                 FuseAnchorPointwise<GemmOp>, FuseAnchorPointwise<QuantConvOp>,
                 FuseAnchorPointwise<QuantMatmulOp>>(ctx, &counter, fuseViews,
                                                     arch);
    patterns.add<DecomposeQLinearConv, DecomposeQLinearMatMul,
                 DecomposeDequantizedConv>(ctx, fuseViews);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // hip.quant_conv and hip.quant_matmul have no lowering outside a rocMLIR
    // kernel.
    WalkResult leftover = funcOp.walk([](Operation *op) {
      if (!isa<QuantConvOp, QuantMatmulOp>(op))
        return WalkResult::advance();
      op->emitOpError("was not outlined into a rocMLIR kernel");
      return WalkResult::interrupt();
    });
    if (leftover.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    for (RocMlirOp dispatch : funcOp.getOps<RocMlirOp>())
      if (auto kernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
              dispatch, dispatch.getKernelAttr()))
        rewriteQuantizedBinaryOps(kernel);
    if (fuseGemmGemm)
      fuseGemmGemmKernels(funcOp, counter, gemmGemmAccelTypes);
  }
};

} // namespace

}; // namespace mlir::hip
