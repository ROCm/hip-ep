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
  }
  return roots;
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
  // An escaping value becomes a kernel result, which needs an init.
  if (users.size() > 1 && !isa<DestinationStyleOpInterface>(tail))
    return nullptr;
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
  // A view outside the kernel is free, and one with no init cannot supply the
  // dispatch's outs.
  while (!isa<DestinationStyleOpInterface>(chain.back()))
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
// independent of the order the greedy driver visits anchors in, which is
// MIGraphX's order: epilogues first (find_mlir_fused_ops), then the leftover
// pointwise ops as prologues (find_pointwise_mlir).
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
      if (isa<MatmulOp, ConvOp, GemmOp>(def)) {
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
    if (!absorb || !llvm::all_of(inputs, isStaticTensor))
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

template <typename AnchorOp>
class FuseAnchorPointwise : public OpRewritePattern<AnchorOp> {
public:
  FuseAnchorPointwise(MLIRContext *context, int *counter, bool fuseViews,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<AnchorOp>(context, benefit), counter(counter),
        policy(getFusionPolicy(fuseViews)) {}

  LogicalResult matchAndRewrite(AnchorOp anchorOp,
                                PatternRewriter &rewriter) const override {
    // Fail on dynamic shapes
    for (Value operand : anchorOp->getOperands()) {
      if (auto tensorType = dyn_cast<TensorType>(operand.getType());
          tensorType && !tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(anchorOp,
                                           "only static shapes are supported");
    }

    KernelRegion region(anchorOp, policy);
    region.growForward();
    if (!region.takeContext())
      return rewriter.notifyMatchFailure(anchorOp,
                                         "first operand not hip.context");
    region.cloneInlineScalars();
    region.absorbDequantBnb4();
    region.absorbInputProducers(getInputRoots(anchorOp));
    region.absorbOperandFreeProducers();
    region.collectRemainingOperands();

    std::optional<SmallVector<Operation *>> moved = region.readersToMove();
    if (!moved)
      return rewriter.notifyMatchFailure(
          anchorOp, "an outside reader of a kernel result feeds the kernel");

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
      newFunc->setAttr("rock.arch", rewriter.getStringAttr("gfx1151"));
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
};

// The kernel's one matmul or gemm, if it has exactly one and no convolution.
static Operation *getSoleGemmAnchor(func::FuncOp kernel) {
  Operation *anchor = nullptr;
  for (Operation &op : kernel.getBody().front()) {
    if (isa<ConvOp>(op))
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

// WMMA on gfx11 and gfx12 has no fp32 path, so an fp32 gemm-gemm kernel would
// fall back to a slow non-matrix-core lowering.
static bool isGemmGemmSupported(Operation *anchor, StringRef arch) {
  auto type = cast<ShapedType>(anchor->getResult(0).getType());
  bool wmma = arch.starts_with("gfx11") || arch.starts_with("gfx12");
  return !(wmma && type.getElementType().isF32());
}

// MIGraphX's find_mlir_fused_geg_ops::check_heuristic for gemm0
// (m x k) @ (k x n) followed by gemm1 (m x n) @ (n x g). Fusing keeps the
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
// pair to one rock.gemm_elementwise_gemm. Mirrors MIGraphX's
// find_mlir_fused_geg_ops; convolutions stay out (its CEG mode is opt-in).
//
// Before:
//   %h = hip.rocmlir @rocMlir0 ins(%x, %w0) outs(%e0)   // matmul -> sigmoid
//   %y = hip.rocmlir @rocMlir1 ins(%h, %w1) outs(%e1)   // matmul
// After:
//   %y = hip.rocmlir @rocMlir2 ins(%x, %w0, %w1) outs(%e1)
//   // @rocMlir2: matmul -> sigmoid -> matmul
static bool fuseGemmGemmPair(RocMlirOp second, int &counter) {
  auto module = second->getParentOfType<ModuleOp>();
  auto secondKernel = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      second, second.getKernelAttr());
  if (!secondKernel)
    return false;
  Operation *secondAnchor = getSoleGemmAnchor(secondKernel);
  if (!secondAnchor)
    return false;
  StringRef arch = "";
  if (auto archAttr = secondKernel->getAttrOfType<StringAttr>("rock.arch"))
    arch = archAttr.getValue();

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
        !isGemmGemmSupported(firstAnchor, arch) ||
        !isGemmGemmSupported(secondAnchor, arch))
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

static void fuseGemmGemmKernels(func::FuncOp funcOp, int &counter) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto op : llvm::make_early_inc_range(funcOp.getOps<RocMlirOp>()))
      if (fuseGemmGemmPair(op, counter)) {
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
                 FuseAnchorPointwise<GemmOp>>(ctx, &counter, fuseViews);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    if (fuseGemmGemm)
      fuseGemmGemmKernels(funcOp, counter);
  }
};

} // namespace

}; // namespace mlir::hip
