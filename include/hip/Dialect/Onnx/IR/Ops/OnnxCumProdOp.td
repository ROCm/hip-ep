//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_CumProdOp
//===----------------------------------------------------------------------===//

def Onnx_CumProdOp : Onnx_Op<"CumProd", [Pure]> {
  let summary = "ONNX CumProd, schema version 26";

  let description = [{
    Performs cumulative product of the input elements along the given axis.
    By default, it will do the product inclusively meaning the first element is copied as is.
    Through an `exclusive` attribute, this behavior can change to exclude the first element.
    It can also perform product in the opposite direction of the axis. For that, set `reverse` attribute to 1.

    Example:
    ```
    input_x = [1, 2, 3]
    axis=0
    output = [1, 2, 6]
    exclusive=1
    output = [1, 1, 2]
    exclusive=0
    reverse=1
    output = [6, 6, 3]
    exclusive=1
    reverse=1
    output = [6, 3, 1]
    ```

  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$x,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$axis,
    DefaultValuedAttr<SI64Attr, "0">:$exclusive,
    DefaultValuedAttr<SI64Attr, "0">:$reverse
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$y
  );
}
