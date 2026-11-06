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
// construct symbol and slot). The emitted construct symbol stores the built
// state into op_states[slot] itself (via hipdnn_ep_op_state_set, called inside
// the runtime construct fn -- not the slot array directly), keeping the
// RuntimeState layout opaque to generated IR. It returns an i8 that is
// vestigial (always 0): hipdnn_ep_op_state_set has no failure path, so the
// init pass keeps the result for a stable signature but does not branch on it.
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
        parameter types are needed, and emit the call passing `slot`. The
        construct symbol stores the built state into op_states[slot] itself and
        returns an i8 that is vestigial (always 0): the store has no failure
        path, so the init pass does not branch on it.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"generateOpStateInit",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$statePtr, "int32_t":$slot)>
  ];
}

#endif // HIP_OP_STATE_INTERFACE
