//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_OP_STATE_INTERFACE
#define HIP_OP_STATE_INTERFACE

include "mlir/IR/OpBase.td"

// Interface for HIP operators that own per-session state (see
// docs/design/op-state-slots-design.md). A stateful op implements this
// interface; --assign-op-state-slots gives each instance its own dense slot,
// and --generate-op-state-init asks each op to emit the IR that constructs its
// own state into that slot. Construction arguments are taken from the op's
// own compile-time attributes, so two instances of the same kind can build
// independent state from different attributes.
//
// The op provides `generateOpStateInit` directly (typically a one-liner that
// reads its attributes and calls `mlir::hip::emitOpStateConstruct` with its
// construct symbol). It does NOT store into the slot itself: --generate-op-
// state-init stores the returned pointer via hipdnn_ep_op_state_set, keeping
// the RuntimeState layout opaque to generated IR.
def OpStateOpInterface : OpInterface<"OpStateOpInterface"> {
  let description = [{
    Marker + code-gen interface for stateful HIP ops. Implementers emit the IR
    that constructs their own per-instance state object during inference_init.
  }];

  let cppNamespace = "::mlir::hip";

  let methods = [
    InterfaceMethod<
      [{
        Emit IR (into `builder`'s current insertion point, inside the generated
        op-state init function) that constructs this op's state object. Read
        the op's own attributes, declare the construct symbol with whatever
        parameter types are needed, and emit the call. Returns the constructed
        OpState pointer (a null value at runtime means construction failed);
        the caller stores it into op_states[slot].
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"generateOpStateInit",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$statePtr, "int32_t":$slot)>
  ];
}

#endif // HIP_OP_STATE_INTERFACE
