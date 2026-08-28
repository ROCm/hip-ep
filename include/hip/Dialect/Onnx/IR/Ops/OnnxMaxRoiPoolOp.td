//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MaxRoiPoolOp
//===----------------------------------------------------------------------===//

def Onnx_MaxRoiPoolOp : Onnx_Op<"MaxRoiPool", [Pure]> {
  let summary = "ONNX MaxRoiPool, schema version 22";

  let description = [{
     ROI max pool consumes an input tensor X and region of interests (RoIs) to
     apply max pooling across each RoI, to produce output 4-D tensor of shape
     (num_rois, channels, pooled_shape[0], pooled_shape[1]).
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$rois,
    I64ArrayAttr:$pooled_shape,
    DefaultValuedAttr<F32Attr, "1.0">:$spatial_scale
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
