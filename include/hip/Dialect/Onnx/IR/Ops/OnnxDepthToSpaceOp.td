//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_DepthToSpaceOp
//===----------------------------------------------------------------------===//

def Onnx_DepthToSpaceOp : Onnx_Op<"DepthToSpace", [Pure]> {
  let summary = "ONNX DepthToSpace, schema version 13";

  let description = [{
    DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
    This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
    the input tensor where values from the depth dimension are moved in spatial blocks to the height
    and width dimensions. By default, `mode` = `DCR`.
    In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
    following order: depth, column, and then row. The output y is computed from the input x as below:

    ```
    b, c, h, w = x.shape
    tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
    tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
    ```

    In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
    following order: column, row, and the depth. The output y is computed from the input x as below:

    ```
    b, c, h, w = x.shape
    tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
    tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
    y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    SI64Attr:$blocksize,
    DefaultValuedStrAttr<StrAttr, "DCR">:$mode
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$output
  );
}
