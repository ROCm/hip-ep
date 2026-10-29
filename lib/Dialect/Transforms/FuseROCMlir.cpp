/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

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

// Elementwise ops HipToTosa lowers to TOSA elementwise ops, which rocMLIR
// fuses into the anchor's epilogue or prologue.
static bool isFusablePointwise(Operation *op) {
  return isa_and_present<MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp, TanhOp,
                         SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp,
                         LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
                         AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp,
                         RoundOp, AtanOp, FloorOp, ExpOp, LogOp, AbsOp, NegOp,
                         SubOp, CastOp, LessOp, SignOp, ModOp, WhereOp>(op);
}

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

// View ops that rocMLIR folds into rock.transform maps instead of running as a
// copy. The legality is HipToTosa's own, so an absorbed view always lowers.
enum class ViewKind : unsigned {
  Transpose = 1u << 0,
  Reshape = 1u << 1,
  Slice = 1u << 2,
  Broadcast = 1u << 3,
};

static std::optional<ViewKind> getFusableViewKind(Operation *op) {
  if (isa_and_present<TransposeOp>(op))
    return ViewKind::Transpose;
  if (isTosaLowerableReshape(op))
    return ViewKind::Reshape;
  if (auto slice = dyn_cast_if_present<tensor::ExtractSliceOp>(op);
      slice && isTosaLowerableSlice(slice))
    return ViewKind::Slice;
  if (auto expand = dyn_cast_if_present<ExpandOp>(op);
      expand && isTosaLowerableExpand(expand))
    return ViewKind::Broadcast;
  return std::nullopt;
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

// Whether `value` is computed, through ops outside the kernel, from a chain
// op. Only ops between the anchor and the reader can be on such a path, so
// the walk stops at anything defined before the anchor.
static bool dependsOnChain(Value value, Operation *anchor,
                           const SetVector<Operation *> &chain) {
  Block *block = anchor->getBlock();
  SmallVector<Value> worklist = {value};
  DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (def)
      def = block->findAncestorOpInBlock(*def);
    if (!def || def->isBeforeInBlock(anchor))
      continue;
    if (chain.contains(def))
      return true;
    if (!visited.insert(def).second)
      continue;
    // A region can read a chain value without passing it as an operand.
    def->walk([&](Operation *nested) {
      llvm::append_range(worklist, nested->getOperands());
    });
  }
  return false;
}

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

void KernelRegion::growForward() {
  SetVector<Operation *> chain;
  chain.insert(anchor);
  while (Operation *next = nextOnChain(chain))
    chain.insert(next);
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
// only inline constants: a dynamic Clip bound remains an argument.
void KernelRegion::cloneInlineScalars() {
  SetVector<Value> kernelOperands;
  for (Value operand : operands) {
    if (isInlineScalarConstant(operand)) {
      ops.insert(operand.getDefiningOp());
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
      if (isAnchorOp(def)) {
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
      if (policy.allowsEpilogue(def) || policy.allowsForwardView(def))
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
    // pipeline does not establish. Leave the op in main_graph.
    for (Type type : anchorOp->getOperandTypes()) {
      auto shaped = dyn_cast<ShapedType>(type);
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
              operands, [](Value v) { return promoteRankZero(v.getType()); }),
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

static DenseElementsAttr getInlineDense(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return {};
  return dyn_cast_if_present<DenseElementsAttr>(constant.getValueAttr());
}

static SmallVector<int64_t> getIntValues(DenseElementsAttr attr) {
  bool isUnsigned = attr.getElementType().isUnsignedInteger();
  return llvm::map_to_vector(attr.getValues<APInt>(), [&](const APInt &v) {
    return isUnsigned ? static_cast<int64_t>(v.getZExtValue())
                      : v.getSExtValue();
  });
}

static bool isQuantizedByte(Type type) {
  return type.isUnsignedInteger(8) || type.isSignlessInteger(8);
}

// A correction that varies over the output plane is one i32 per output
// element of one image; past this size the conv keeps its HIP kernel.
static constexpr int64_t kMaxQLinearConvCorrection = int64_t{1} << 22;

// The constants that turn a hip.qlinear_conv into hip.quant_conv and an
// epilogue. With z' = x_zp - offset (offset is getSignedRebiasOffset of the
// input type: 128 for ui8, 0 for i8):
//
//   conv(x - x_zp, w) = conv(x - offset, w) - z' * sum(w over the taps that
//                       read the input, not the padding)
//
// so the epilogue adds `bias` = B - z' * sum(w), then requantizes with
// `multiplier` = x_scale * w_scale / y_scale in f32, the order ORT CPU
// uses. Without padding every output element reads every tap, and `bias`
// holds one value per channel; otherwise it has one value per output element.
// A hip.qlinear_matmul uses the same plan with sums over the columns of B.
struct QLinearPlan {
  // i32, broadcastable to the result: [1, K, 1, 1] or [1, K, OH, OW] for a
  // conv, [1, N] for a matmul. Null when every entry is 0.
  DenseElementsAttr bias;
  // f32, rank 0 or broadcastable to the result along the channel or column.
  DenseElementsAttr multiplier;
  // Rank 0, the output element type.
  DenseElementsAttr outputZeroPoint;
};

// The per-tensor input scale times each weight scale, over the output scale,
// in f32 and in ORT CPU's order. `channelShape` is the broadcast shape of a
// per-channel multiplier.
static DenseElementsAttr getRequantMultiplier(DenseElementsAttr inputScale,
                                              ArrayRef<float> weightScales,
                                              DenseElementsAttr outputScale,
                                              ArrayRef<int64_t> channelShape) {
  float xScale = *inputScale.getValues<float>().begin();
  float yScale = *outputScale.getValues<float>().begin();
  SmallVector<float> multipliers = llvm::map_to_vector(
      weightScales, [&](float wScale) { return xScale * wScale / yScale; });
  if (!llvm::all_of(multipliers, [](float m) { return std::isfinite(m); }))
    return {};
  Type f32 = Float32Type::get(inputScale.getContext());
  if (llvm::all_equal(multipliers))
    return DenseElementsAttr::get(RankedTensorType::get({}, f32),
                                  multipliers.front());
  return DenseElementsAttr::get(RankedTensorType::get(channelShape, f32),
                                ArrayRef<float>(multipliers));
}

static DenseElementsAttr getScalarZeroPoint(DenseElementsAttr zp,
                                            Type elementType) {
  return DenseElementsAttr::get(RankedTensorType::get({}, elementType),
                                *zp.getValues<APInt>().begin());
}

// The rocMLIR route needs every quantization parameter, the weights, and the
// bias as inline constants, a weight zero point of 0 (rocMLIR has no
// zero-point operand), and static shapes. Anything else keeps the HIP kernel.
static std::optional<QLinearPlan> planQLinearConv(QLinearConvOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(op.getWeights().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
      !weightType.hasStaticShape() || !resultType.hasStaticShape())
    return std::nullopt;
  Type inputElement = inputType.getElementType();
  Type outputElement = resultType.getElementType();
  if (!isQuantizedByte(inputElement) || !isQuantizedByte(outputElement) ||
      !weightType.getElementType().isSignlessInteger(8))
    return std::nullopt;

  DenseElementsAttr inputScale = getInlineDense(op.getInputScale());
  DenseElementsAttr inputZp = getInlineDense(op.getInputZeroPoint());
  DenseElementsAttr weightScale = getInlineDense(op.getWeightScale());
  DenseElementsAttr weightZp = getInlineDense(op.getWeightZeroPoint());
  DenseElementsAttr outputScale = getInlineDense(op.getOutputScale());
  DenseElementsAttr outputZp = getInlineDense(op.getOutputZeroPoint());
  DenseElementsAttr weights = getInlineDense(op.getWeights());
  DenseElementsAttr bias;
  if (Value biasValue = op.getBias()) {
    bias = getInlineDense(biasValue);
    if (!bias)
      return std::nullopt;
  }
  if (!inputScale || !inputZp || !weightScale || !weightZp || !outputScale ||
      !outputZp || !weights || inputScale.getNumElements() != 1 ||
      inputZp.getNumElements() != 1 || outputScale.getNumElements() != 1 ||
      outputZp.getNumElements() != 1)
    return std::nullopt;
  if (llvm::any_of(getIntValues(weightZp), [](int64_t v) { return v != 0; }))
    return std::nullopt;

  ArrayRef<int64_t> inShape = inputType.getShape();
  ArrayRef<int64_t> wShape = weightType.getShape();
  ArrayRef<int64_t> outShape = resultType.getShape();
  int64_t channels = wShape[0];
  int64_t outH = outShape[2], outW = outShape[3];
  SmallVector<float> weightScales =
      llvm::to_vector(weightScale.getValues<float>());
  if (weightScales.size() != 1 &&
      static_cast<int64_t>(weightScales.size()) != channels)
    return std::nullopt;
  SmallVector<int64_t> biasValues(channels, 0);
  if (bias) {
    if (bias.getNumElements() != channels)
      return std::nullopt;
    biasValues = getIntValues(bias);
  }

  auto attrValue = [](ArrayAttr attr, unsigned index) {
    return cast<IntegerAttr>(attr[index]).getInt();
  };
  int64_t strideH = attrValue(op.getStrides(), 0);
  int64_t strideW = attrValue(op.getStrides(), 1);
  int64_t dilationH = attrValue(op.getDilations(), 0);
  int64_t dilationW = attrValue(op.getDilations(), 1);
  int64_t padTop = attrValue(op.getPads(), 0);
  int64_t padLeft = attrValue(op.getPads(), 1);

  int64_t offset = getSignedRebiasOffset(inputElement);
  int64_t zpShift = getIntValues(inputZp).front() - offset;

  // The taps along one axis that read the input at output position `o`; a
  // contiguous range because the taps are equally spaced.
  auto validTaps = [](int64_t o, int64_t stride, int64_t pad, int64_t dilation,
                      int64_t kernel, int64_t size) {
    int64_t lo = kernel, hi = -1;
    for (int64_t k : llvm::seq<int64_t>(kernel)) {
      int64_t at = o * stride - pad + k * dilation;
      if (at >= 0 && at < size) {
        lo = std::min(lo, k);
        hi = std::max(hi, k);
      }
    }
    return std::make_pair(lo, hi);
  };
  SmallVector<std::pair<int64_t, int64_t>> rowTaps, colTaps;
  for (int64_t oh : llvm::seq<int64_t>(outH))
    rowTaps.push_back(
        validTaps(oh, strideH, padTop, dilationH, wShape[2], inShape[2]));
  for (int64_t ow : llvm::seq<int64_t>(outW))
    colTaps.push_back(
        validTaps(ow, strideW, padLeft, dilationW, wShape[3], inShape[3]));

  // tapSums[k][r][s] = sum over the input channels of w[k, c, r, s].
  int64_t taps = wShape[2] * wShape[3];
  SmallVector<int64_t> tapSums(channels * taps, 0);
  if (zpShift != 0) {
    int64_t index = 0;
    for (const APInt &v : weights.getValues<APInt>()) {
      int64_t k = index / (wShape[1] * taps);
      tapSums[k * taps + index % taps] += v.getSExtValue();
      ++index;
    }
  }
  auto correctedBias = [&](int64_t k, int64_t oh, int64_t ow) {
    auto [r0, r1] = rowTaps[oh];
    auto [s0, s1] = colTaps[ow];
    int64_t sum = 0;
    for (int64_t r = r0; r <= r1; ++r)
      for (int64_t s = s0; s <= s1; ++s)
        sum += tapSums[k * taps + r * wShape[3] + s];
    return biasValues[k] - zpShift * sum;
  };

  bool uniform =
      zpShift == 0 || (llvm::all_equal(rowTaps) && llvm::all_equal(colTaps));
  int64_t planeSize = uniform ? 1 : outH * outW;
  if (channels * planeSize > kMaxQLinearConvCorrection)
    return std::nullopt;
  SmallVector<int32_t> biasData;
  biasData.reserve(channels * planeSize);
  bool allZero = true;
  for (int64_t k : llvm::seq<int64_t>(channels))
    for (int64_t position : llvm::seq<int64_t>(planeSize)) {
      int64_t value = correctedBias(k, position / outW, position % outW);
      if (value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max())
        return std::nullopt;
      allZero &= value == 0;
      biasData.push_back(static_cast<int32_t>(value));
    }

  QLinearPlan plan;
  if (!allZero) {
    SmallVector<int64_t> shape = {1, channels, uniform ? 1 : outH,
                                  uniform ? 1 : outW};
    plan.bias = DenseElementsAttr::get(
        RankedTensorType::get(shape, IntegerType::get(op.getContext(), 32)),
        ArrayRef<int32_t>(biasData));
  }
  plan.multiplier = getRequantMultiplier(inputScale, weightScales, outputScale,
                                         {1, channels, 1, 1});
  if (!plan.multiplier)
    return std::nullopt;
  plan.outputZeroPoint = getScalarZeroPoint(outputZp, outputElement);
  return plan;
}

// The matmul form of the plan: matmul(a - a_zp, b) = matmul(a - offset, b)
// - z' * colsum(b), with one bias value per output column. B must be an inline
// constant only when z' is not 0.
static std::optional<QLinearPlan> planQLinearMatMul(QLinearMatMulOp op) {
  auto aType = cast<RankedTensorType>(op.getA().getType());
  auto bType = cast<RankedTensorType>(op.getB().getType());
  auto resultType = cast<RankedTensorType>(op.getOutput().getType());
  Type aElement = aType.getElementType();
  Type outputElement = resultType.getElementType();
  if (!isQuantizedByte(aElement) || !isQuantizedByte(outputElement) ||
      !bType.getElementType().isSignlessInteger(8))
    return std::nullopt;

  DenseElementsAttr aScale = getInlineDense(op.getAScale());
  DenseElementsAttr aZp = getInlineDense(op.getAZeroPoint());
  DenseElementsAttr bScale = getInlineDense(op.getBScale());
  DenseElementsAttr bZp = getInlineDense(op.getBZeroPoint());
  DenseElementsAttr yScale = getInlineDense(op.getYScale());
  DenseElementsAttr yZp = getInlineDense(op.getYZeroPoint());
  if (!aScale || !aZp || !bScale || !bZp || !yScale || !yZp)
    return std::nullopt;
  if (llvm::any_of(getIntValues(bZp), [](int64_t v) { return v != 0; }))
    return std::nullopt;

  int64_t n = bType.getDimSize(1);
  int64_t offset = getSignedRebiasOffset(aElement);
  int64_t zpShift = getIntValues(aZp).front() - offset;

  QLinearPlan plan;
  if (zpShift != 0) {
    DenseElementsAttr b = getInlineDense(op.getB());
    if (!b)
      return std::nullopt;
    SmallVector<int64_t> columnSums(n, 0);
    int64_t index = 0;
    for (const APInt &v : b.getValues<APInt>())
      columnSums[index++ % n] += v.getSExtValue();
    SmallVector<int32_t> biasData;
    biasData.reserve(n);
    for (int64_t sum : columnSums) {
      int64_t value = -zpShift * sum;
      if (value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max())
        return std::nullopt;
      biasData.push_back(static_cast<int32_t>(value));
    }
    if (llvm::any_of(biasData, [](int32_t v) { return v != 0; }))
      plan.bias = DenseElementsAttr::get(
          RankedTensorType::get({1, n}, IntegerType::get(op.getContext(), 32)),
          ArrayRef<int32_t>(biasData));
  }

  SmallVector<float> bScales = llvm::to_vector(bScale.getValues<float>());
  plan.multiplier = getRequantMultiplier(aScale, bScales, yScale, {1, n});
  if (!plan.multiplier)
    return std::nullopt;
  plan.outputZeroPoint = getScalarZeroPoint(yZp, outputElement);
  return plan;
}

// The epilogue that both QLinear rewrites share: add the corrected bias, cast
// to f32, scale, and quantize with unit scale, which rounds half to even,
// adds the zero point, and saturates, as ORT CPU does.
static Value emitRequantizeEpilogue(PatternRewriter &rewriter, Location loc,
                                    Value ctx, Value acc,
                                    const QLinearPlan &plan, Value output) {
  auto accType = cast<RankedTensorType>(acc.getType());
  auto resultType = cast<RankedTensorType>(output.getType());
  auto floatType = accType.clone(rewriter.getF32Type());
  auto empty = [&](RankedTensorType type) -> Value {
    return tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                   type.getElementType());
  };
  auto constant = [&](DenseElementsAttr value) -> Value {
    return ConstantOp::create(rewriter, loc,
                              cast<RankedTensorType>(value.getType()), value);
  };
  if (plan.bias)
    acc = AddOp::create(rewriter, loc, TypeRange{accType}, ctx, acc,
                        constant(plan.bias), empty(accType))
              .getResult(0);
  Value scaled = CastOp::create(rewriter, loc, TypeRange{floatType}, ctx, acc,
                                empty(floatType),
                                rewriter.getI64IntegerAttr(/*ONNX FLOAT=*/1))
                     .getResult(0);
  scaled = MulOp::create(rewriter, loc, TypeRange{floatType}, ctx, scaled,
                         constant(plan.multiplier), empty(floatType))
               .getResult(0);
  Value one = constant(DenseElementsAttr::get(
      RankedTensorType::get({}, rewriter.getF32Type()), 1.0f));
  return QuantizeLinearOp::create(
             rewriter, loc, TypeRange{resultType}, ctx, scaled, one,
             constant(plan.outputZeroPoint), output,
             rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(0),
             rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(1))
      .getResult(0);
}

// With rocMLIR on, a QLinearConv that planQLinearConv accepts becomes the
// MIGraphX form: an i8 x i8 -> i32 convolution with zero points 0, a constant
// correction, and an f32 requantize. FuseAnchorPointwise<QuantConvOp> then
// outlines the chain as one rocMLIR kernel; a quantize whose scale is 1
// rounds half to even, adds the zero point, and saturates, as ORT does.
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

// The QLinearMatMul form of DecomposeQLinearConv, as MIGraphX's quant_dot.
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

// The kernel's one matmul or gemm, if it has exactly one and no convolution.
static Operation *getSoleGemmAnchor(func::FuncOp kernel) {
  Operation *anchor = nullptr;
  for (Operation &op : kernel.getBody().front()) {
    if (isa<ConvOp, QuantConvOp, QuantMatmulOp>(op))
      return nullptr;
    if (isa<MatmulOp, GemmOp>(op)) {
      if (anchor)
        return nullptr;
      anchor = &op;
    }
  }
  return anchor;
}

static OpOperand &getGemmAOperand(Operation *anchor) {
  if (auto matmul = dyn_cast<MatmulOp>(anchor))
    return matmul.getAMutable();
  return cast<GemmOp>(anchor).getInputAMutable();
}

// (m, n, k) of `A[m, k] @ B[k, n]`, after the gemm's own transposes.
static std::optional<std::array<int64_t, 3>> getGemmDims(Operation *anchor) {
  Value a = getGemmAOperand(anchor).get();
  Value b;
  bool transA = false, transB = false;
  if (auto matmul = dyn_cast<MatmulOp>(anchor)) {
    b = matmul.getB();
  } else {
    auto gemm = cast<GemmOp>(anchor);
    b = gemm.getInputB();
    transA = gemm.getTransA() != 0;
    transB = gemm.getTransB() != 0;
  }
  auto aType = dyn_cast<RankedTensorType>(a.getType());
  auto bType = dyn_cast<RankedTensorType>(b.getType());
  if (!aType || !bType || aType.getRank() < 2 || bType.getRank() < 2)
    return std::nullopt;
  ArrayRef<int64_t> aShape = aType.getShape(), bShape = bType.getShape();
  int64_t aRows = aShape[aShape.size() - 2], aCols = aShape.back();
  int64_t bRows = bShape[bShape.size() - 2], bCols = bShape.back();
  return std::array<int64_t, 3>{transA ? aCols : aRows, transB ? bRows : bCols,
                                transA ? aRows : aCols};
}

// A gemm-gemm kernel without matrix acceleration for its element types falls
// back to a slow lowering, so both gemms must read types the target
// accelerates. `accelTypes` holds them as printed (f16, bf16, ...).
static bool isGemmGemmSupported(Operation *anchor,
                                ArrayRef<std::string> accelTypes) {
  Value b = isa<MatmulOp>(anchor) ? cast<MatmulOp>(anchor).getB()
                                  : cast<GemmOp>(anchor).getInputB();
  auto isAccelerated = [&](Value operand) {
    std::string name;
    llvm::raw_string_ostream os(name);
    getElementTypeOrSelf(operand.getType()).print(os);
    return llvm::is_contained(accelTypes, name);
  };
  return isAccelerated(getGemmAOperand(anchor).get()) && isAccelerated(b);
}

// The shape gate for gemm0 (m x k) @ (k x n) followed by gemm1
// (m x n) @ (n x g). Fusing keeps the
// m x n intermediate on chip, which pays only for a column-vector
// intermediate or a skinny-k MLP bottom over a large batch. Both gates cap g
// at 256, so rocMLIR's LDS estimate for its conservative default perf config,
// (32*32 + 32*32 + 32*pow2ceil(g)) elements, always fits in 64 KiB.
static bool passesGemmGemmHeuristic(std::array<int64_t, 3> first, int64_t g) {
  auto [m, n, k] = first;
  if (m <= 0 || n <= 0 || k <= 0 || g <= 1)
    return false;
  if (m <= 8 && n >= 512)
    return false;
  if (n == 1 && k <= 128 && m <= 1024 && g <= 64)
    return true;
  return m >= 98304 && n >= 512 && k <= 13 && g <= 256;
}

// What may lie between the two gemms: pointwise ops, and a reshape that only
// regroups the rows (batch and m) and keeps n. A transpose there makes the
// second gemm's reduction dimension disagree with the first gemm's n.
static bool isGemmGemmLink(Operation *op) {
  if (isFusablePointwise(op))
    return true;
  if (getFusableViewKind(op) != ViewKind::Reshape)
    return false;
  auto in = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return in && out && in.getRank() > 0 && out.getRank() > 0 &&
         in.getShape().back() == out.getShape().back();
}

// Whether only links lie between the anchor and the value the kernel
// returns, so that value is the gemm-elementwise result rocMLIR expects
// between the two gemms. Views of epilogue operands (a reshaped bias) do not
// read the anchor and may sit anywhere.
static bool endsInGemmPointwise(func::FuncOp kernel, Operation *anchor) {
  DenseSet<Value> derived(anchor->result_begin(), anchor->result_end());
  for (Operation &op : llvm::make_range(std::next(anchor->getIterator()),
                                        kernel.getBody().front().end())) {
    if (isa<func::ReturnOp>(op) ||
        llvm::none_of(op.getOperands(),
                      [&](Value v) { return derived.contains(v); }))
      continue;
    if (!isGemmGemmLink(&op))
      return false;
    derived.insert(op.result_begin(), op.result_end());
  }
  return true;
}

// Whether `arg` reaches the anchor's A operand through single-use links only
// (a prologue or row reshape the second kernel absorbed).
static bool feedsGemmA(BlockArgument arg, Operation *anchor) {
  Value value = arg;
  while (value.hasOneUse()) {
    OpOperand &use = *value.use_begin();
    Operation *user = use.getOwner();
    if (user == anchor)
      return use.getOperandNumber() ==
             getGemmAOperand(anchor).getOperandNumber();
    if (!isGemmGemmLink(user) || user->getNumResults() != 1)
      return false;
    value = user->getResult(0);
  }
  return false;
}

// Gemm-pointwise-gemm: merge two outlined kernels when the first kernel's
// only result feeds the second kernel's gemm A operand, so rocMLIR lowers the
// pair to one rock.gemm_elementwise_gemm. Convolutions stay out.
//
// Before:
//   %h = hip.rocmlir @rocMlir0 ins(%x, %w0) outs(%e0)   // matmul -> sigmoid
//   %y = hip.rocmlir @rocMlir1 ins(%h, %w1) outs(%e1)   // matmul
// After:
//   %y = hip.rocmlir @rocMlir2 ins(%x, %w0, %w1) outs(%e1)
//   // @rocMlir2: matmul -> sigmoid -> matmul
static bool fuseGemmGemmPair(RocMlirOp second, int &counter,
                             ArrayRef<std::string> accelTypes) {
  auto module = second->getParentOfType<ModuleOp>();
  auto secondKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      second, second.getKernelAttr());
  if (!secondKernel)
    return false;
  Operation *secondAnchor = getSoleGemmAnchor(secondKernel);
  if (!secondAnchor || !isGemmGemmSupported(secondAnchor, accelTypes))
    return false;

  for (auto [index, input] : llvm::enumerate(second.getInputs())) {
    auto first = input.getDefiningOp<RocMlirOp>();
    if (!first || first->getNumResults() != 1 || !input.hasOneUse() ||
        first->getBlock() != second->getBlock())
      continue;
    auto firstKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        first, first.getKernelAttr());
    if (!firstKernel)
      continue;
    Operation *firstAnchor = getSoleGemmAnchor(firstKernel);
    if (!firstAnchor || !endsInGemmPointwise(firstKernel, firstAnchor) ||
        !isGemmGemmSupported(firstAnchor, accelTypes))
      continue;
    if (!feedsGemmA(secondKernel.getArgument(index), secondAnchor))
      continue;
    std::optional<std::array<int64_t, 3>> firstDims = getGemmDims(firstAnchor);
    std::optional<std::array<int64_t, 3>> secondDims =
        getGemmDims(secondAnchor);
    if (!firstDims || !secondDims ||
        !passesGemmGemmHeuristic(*firstDims, (*secondDims)[1]))
      continue;
    // Another input computed from the first kernel's result would make the
    // merged kernel read its own output.
    SetVector<Operation *> firstOnly;
    firstOnly.insert(first);
    bool closesCycle =
        llvm::any_of(llvm::enumerate(second.getInputs()), [&](auto other) {
          return other.index() != index &&
                 dependsOnChain(other.value(), first, firstOnly);
        });
    if (closesCycle)
      continue;

    OpBuilder builder(second);
    SmallVector<Value> operands(first.getInputs());
    for (auto [otherIndex, other] : llvm::enumerate(second.getInputs()))
      if (otherIndex != index)
        operands.push_back(other);

    func::FuncOp merged;
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto argTypes = llvm::map_to_vector(
          operands, [](Value v) -> Type { return v.getType(); });
      merged = func::FuncOp::create(
          builder, secondKernel.getLoc(), "rocMlir" + std::to_string(counter++),
          builder.getFunctionType(argTypes,
                                  secondKernel.getFunctionType().getResults()));
      merged->setAttrs(secondKernel->getDiscardableAttrDictionary());
      Block *body = merged.addEntryBlock();
      builder.setInsertionPointToStart(body);

      IRMapping mapping;
      unsigned numFirst = first.getInputs().size();
      for (auto [arg, newArg] :
           llvm::zip(firstKernel.getArguments(),
                     body->getArguments().take_front(numFirst)))
        mapping.map(arg, newArg);
      Value intermediate;
      for (Operation &op : firstKernel.getBody().front()) {
        if (auto ret = dyn_cast<func::ReturnOp>(op)) {
          intermediate = mapping.lookup(ret.getOperand(0));
          break;
        }
        builder.clone(op, mapping);
      }
      unsigned next = numFirst;
      for (BlockArgument arg : secondKernel.getArguments())
        mapping.map(arg, arg.getArgNumber() == index
                             ? intermediate
                             : body->getArgument(next++));
      for (Operation &op : secondKernel.getBody().front())
        builder.clone(op, mapping);
    }

    auto fused =
        RocMlirOp::create(builder, second.getLoc(), second->getResultTypes(),
                          SymbolRefAttr::get(merged), second.getCtx(), operands,
                          SmallVector<Value>(second.getOutputs()));
    second->replaceAllUsesWith(fused);
    second->erase();
    first->erase();
    firstKernel->erase();
    secondKernel->erase();
    return true;
  }
  return false;
}

static void fuseGemmGemmKernels(func::FuncOp funcOp, int &counter,
                                ArrayRef<std::string> accelTypes) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto op : llvm::make_early_inc_range(funcOp.getOps<RocMlirOp>()))
      if (fuseGemmGemmPair(op, counter, accelTypes)) {
        changed = true;
        break;
      }
  }
}

class FuseROCMlirPass : public impl::FuseROCMlirPassBase<FuseROCMlirPass> {
public:
  using impl::FuseROCMlirPassBase<FuseROCMlirPass>::FuseROCMlirPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    int counter = 0;
    patterns.add<FuseAnchorPointwise<MatmulOp>, FuseAnchorPointwise<ConvOp>,
                 FuseAnchorPointwise<GemmOp>, FuseAnchorPointwise<QuantConvOp>,
                 FuseAnchorPointwise<QuantMatmulOp>>(ctx, &counter, fuseViews,
                                                     arch);
    patterns.add<DecomposeQLinearConv, DecomposeQLinearMatMul>(ctx, fuseViews);

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
    if (fuseGemmGemm)
      fuseGemmGemmKernels(funcOp, counter, gemmGemmAccelTypes);
  }
};

} // namespace

}; // namespace mlir::hip
