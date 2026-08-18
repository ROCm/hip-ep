//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_DynamicQuantizeLinearOp
//===----------------------------------------------------------------------===//

def Onnx_DynamicQuantizeLinearOp : Onnx_Op<"DynamicQuantizeLinear", [Pure]> {
  let summary = "ONNX DynamicQuantizeLinear, schema version 11";

  let description = [{
    A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
    Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
    Scale is calculated as:
    ```
    y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
    ```

    * where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
    * data range is adjusted to include 0.

    Zero point is calculated as:
    ```
    intermediate_zero_point = qmin - min(x)/y_scale
    y_zero_point = cast(round(saturate(intermediate_zero_point)))
    ```

    * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
    * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
    * rounding to nearest ties to even.

    Data quantization formula is:
    ```
    y = saturate (round (x / y_scale) + y_zero_point)
    ```

    * for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
    * rounding to nearest ties to even.
  }];

  let arguments = (ins
    TensorOf<[F32]>:$x
  );

  let results = (outs
    TensorOf<[UI8]>:$y,
    TensorOf<[F32]>:$y_scale,
    TensorOf<[UI8]>:$y_zero_point
  );
}
