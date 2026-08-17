
void AllocOutputOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Two requirements, both load-bearing:
  //   (1) NOT MemoryEffects::Allocate. The returned buffer is EP/runtime-owned
  //       (a graph output): buffer-deallocation must never free it and
  //       hip-pool-allocs must never pool it. Ownership is keyed on the
  //       Allocate effect, so we deliberately omit it (unlike AllocOp/GetPoolOp
  //       above).
  //   (2) A generic Write effect (no associated value) marks the side effect of
  //       calling into the EP output allocator, which mutates external runtime
  //       state. This is what keeps the op alive: an op carrying a Write is
  //       never trivially dead (DCE / canonicalize) even when its result is
  //       unused, and CSE never merges side-effecting ops. (Contrast: an
  //       Allocate-on-result op IS removed when its result is unused -- a
  //       second reason to avoid Allocate here.)
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  }
  
  LogicalResult AllocOutputOp::verify() {
  // Operand convention matches hip.alloc: exactly one Index per dynamic dim of
  // the result memref (static dims come from the type, dynamic dims from
  // operands). Guards against malformed alloc_output before lowering.
  auto memrefTy = cast<MemRefType>(getMemref().getType());
  if (static_cast<int64_t>(getDynamicSizes().size()) !=
      memrefTy.getNumDynamicDims())
    return emitOpError("expected ")
           << memrefTy.getNumDynamicDims() << " dynamic size operand(s), got "
           << getDynamicSizes().size();
  return success();
  }