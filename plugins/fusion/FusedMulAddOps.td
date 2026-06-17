//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Out-of-tree plugin op: hip.fused_mul_add.
//
// This op is contributed by a dynamically-loaded plugin (.so) rather than the
// in-tree dialect. It is a *real* hip.* op (registered into the existing `hip`
// dialect at load time via HipDialect::registerPluginOps), so it reuses the
// in-tree bufferization machinery (HipDstBufferizableModel). It is deliberately
// distinct from the in-tree ops:
//
//   hip.fused_mul_add  ==  (mul_operand * x) + add_operand   ==  b*x + a
//
// to prove plugin-contributed ops do not collide with in-tree ops.
//===----------------------------------------------------------------------===//

#ifndef HIP_PLUGIN_FUSED_MUL_ADD_OPS
#define HIP_PLUGIN_FUSED_MUL_ADD_OPS

include "hip/Dialect/IR/HipDialect.td"
include "hip/Dialect/IR/HipTypes.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// Accepts ranked tensor (pre-bufferization) or memref (post-bufferization),
// mirroring the in-tree Hip_TensorOrMemRef constraint.
def HipPlugin_TensorOrMemRef
    : AnyTypeOf<[AnyRankedTensor, AnyMemRef], "ranked tensor or memref">;

def FusedMulAddOp : Op<Hip_Dialect, "fused_mul_add", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DestinationStyleOpInterface,
                              ["getDpsInitsMutable"]>
  ]> {
  let summary = "Fused elementwise (mul_operand * x) + add_operand (plugin op)";
  let description = [{
    Performs the fused elementwise computation:
      output = (mul_operand * x) + add_operand

    Produced by the plugin `--hip-fuse-mul-add` pass when an `hip.add` consumes
    the single-use result of an `hip.mul`. Destination-passing style: the output
    buffer is provided as an argument.

    Example:
    ```mlir
    %r = hip.fused_mul_add(%ctx)
        ins(%x, %b, %a : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
        outs(%out : tensor<4xf32>) -> tensor<4xf32>
    ```
  }];

  let arguments = (ins
    Hip_ContextType:$ctx,
    HipPlugin_TensorOrMemRef:$x,
    HipPlugin_TensorOrMemRef:$mul_operand,
    HipPlugin_TensorOrMemRef:$add_operand,
    HipPlugin_TensorOrMemRef:$output
  );

  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let assemblyFormat = [{
    `(` $ctx `)` `ins` `(` $x `,` $mul_operand `,` $add_operand `:`
        type($x) `,` type($mul_operand) `,` type($add_operand) `)`
    `outs` `(` $output `:` type($output) `)` attr-dict
    (`->` type($result_tensors)^)?
  }];
}

#endif // HIP_PLUGIN_FUSED_MUL_ADD_OPS
