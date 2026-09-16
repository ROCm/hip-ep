//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_BitShiftOp
//===----------------------------------------------------------------------===//

def Onnx_BitShiftOp : Onnx_Op<"BitShift", [Pure]> {
  let summary = "ONNX BitShift, schema version 11";

  let description = [{
    Bitwise shift operator performs element-wise operation. For each input element, if the
    attribute "direction" is "RIGHT", this operator moves its binary representation toward
    the right side so that the input value is effectively decreased. If the attribute "direction"
    is "LEFT", bits of binary representation moves toward the left side, which results the
    increase of its actual value. The input X is the tensor to be shifted and another input
    Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
    and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
    X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

    Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
    not necessarily identical.
    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>
    ]>:$Y,
    StrAttr:$direction
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>
    ]>:$Z
  );
}
