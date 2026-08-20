//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_NonMaxSuppressionOp
//===----------------------------------------------------------------------===//

def Onnx_NonMaxSuppressionOp : Onnx_Op<"NonMaxSuppression", [Pure]> {
  let summary = "ONNX NonMaxSuppression, schema version 11";

  let description = [{
    Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
    Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
    Boxes are suppressed if their IOU with a previously selected box is strictly greater than iou_threshold (i.e., boxes with IOU exactly equal to the threshold are kept).
    Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
    orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
    result in the same boxes being selected by the algorithm.
    The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
    The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
  }];

  let arguments = (ins
    TensorOf<[F32]>:$boxes,
    TensorOf<[F32]>:$scores,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$max_output_boxes_per_class,
    AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$iou_threshold,
    AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$score_threshold,
    DefaultValuedAttr<SI64Attr, "0">:$center_point_box
  );

  let results = (outs
    TensorOf<[I64]>:$selected_indices
  );
}
