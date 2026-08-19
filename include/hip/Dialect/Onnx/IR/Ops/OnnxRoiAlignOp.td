//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_RoiAlignOp
//===----------------------------------------------------------------------===//

def Onnx_RoiAlignOp : Onnx_Op<"RoiAlign", [Pure]> {
  let summary = "ONNX RoiAlign, schema version 22";

  let description = [{
    Region of Interest (RoI) align operation described in the
    [Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
    RoiAlign consumes an input tensor X and region of interests (rois)
    to apply pooling across each RoI; it produces a 4-D tensor of shape
    (num_rois, C, output_height, output_width).

    RoiAlign is proposed to avoid the misalignment by removing
    quantizations while converting from original image into feature
    map and from feature map into RoI feature; in each ROI bin,
    the value of the sampled locations are computed directly
    through bilinear interpolation.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$rois,
    TensorOf<[I64]>:$batch_indices,
    DefaultValuedStrAttr<StrAttr, "half_pixel">:$coordinate_transformation_mode,
    DefaultValuedStrAttr<StrAttr, "avg">:$mode,
    DefaultValuedAttr<SI64Attr, "1">:$output_height,
    DefaultValuedAttr<SI64Attr, "1">:$output_width,
    DefaultValuedAttr<SI64Attr, "0">:$sampling_ratio,
    DefaultValuedAttr<F32Attr, "1.0">:$spatial_scale
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
