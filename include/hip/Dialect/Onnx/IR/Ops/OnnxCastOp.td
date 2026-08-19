//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_CastOp
//===----------------------------------------------------------------------===//

def Onnx_CastOp : Onnx_Op<"Cast", [Pure]> {
  let summary = "ONNX Cast, schema version 25";

  let description = [{
    The operator casts the elements of a given input tensor to a data type
    specified by the 'to' argument and returns an output tensor of the same size in
    the converted type. The 'to' argument must be one of the data types specified
    in the 'DataType' enum field in the TensorProto message.

    Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
    (e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
    yield result 100. There are some string literals reserved for special floating-point values;
    "+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
    Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
    this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
    to string tensors, plain floating-point representation (such as "314.15926") would be used.
    Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
    of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

    Conversion from a numerical type to any numerical type is always allowed.
    User must be aware of precision loss and value change caused by range difference between two types.
    For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
    an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

    In more detail, the conversion among numerical types should follow these rules
    if the destination type is not a float 8 type.

    * Casting from floating point to:
      * floating point: +/- infinity if OOR (out of range).
      * fixed point: undefined if OOR.
      * bool: +/- 0.0 to False; all else to True.
    * Casting from fixed point to:
      * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
      * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
        signed types). For example, 200 (int16) -> -56 (int8).
      * bool: zero to False; nonzero to True.
    * Casting from bool to:
      * floating point: `{1.0, 0.0}`.
      * fixed point: `{1, 0}`.
      * bool: no change.

    Float 8 types (E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ) were introduced to speed up the training of
    deep models. By default the conversion of a float *x* obeys
    to the following rules. `[x]` means the value rounded to
    the target mantissa width.

    | x                 | E4M3FN   | E4M3FNUZ | E5M2     | E5M2FNUZ |
    | ----------------- | -------- | -------- | -------- | -------- |
    | 0                 | 0        | 0        | 0        | 0        |
    | -0                | -0       | 0        | -0       | 0        |
    | NaN               | NaN      | NaN      | NaN      | NaN      |
    | Inf               | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
    | -Inf              | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
    | \[x\] > FLT_MAX   | FLT_MAX  | FLT_MAX  | FLT_MAX  | FLT_MAX  |
    | \[x\] \< -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
    | else              | RNE      | RNE      | RNE      | RNE      |

    The behavior changes if the parameter 'saturate' is set to False.
    The rules then become:

    | x                 | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
    | ----------------- | ------ | -------- | ---- | -------- |
    | 0                 | 0      | 0        | 0    | 0        |
    | -0                | -0     | 0        | -0   | 0        |
    | NaN               | NaN    | NaN      | NaN  | NaN      |
    | -NaN              | -NaN   | NaN      | -NaN | NaN      |
    | Inf               | NaN    | NaN      | Inf  | NaN      |
    | -Inf              | -NaN   | NaN      | -Inf | NaN      |
    | \[x\] > FLT_MAX   | NaN    | NaN      | Inf  | NaN      |
    | \[x\] \< -FLT_MAX | NaN    | NaN      | -Inf | NaN      |
    | else              | RNE    | RNE      | RNE  | RNE      |

    FLOAT8E8M0 type was introduced to enable [Microscaling (MX) formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
    When casting to FLOAT8E8M0, the rounding behavior can be specified using the `round_mode` and `saturate` attributes.
    The current CUDA behavior is to round up and saturate. Casting negative values to FLOAT8E8M0 gives undefined behavior.
    The following table describes the casting behavior of special values to FLOAT8E8M0 in the two most common cases.

    | x                 | saturate + up | non-saturate + nearest |
    | ----------------- | ------------- | ---------------------  |
    | 0                 | 0             | NaN                    |
    | -0                | Unspecified   | Unspecified            |
    | NaN               | NaN           | NaN                    |
    | Inf               | E8M0_MAX      | NaN                    |
    | x > E8M0_MAX      | E8M0_MAX      | NaN                    |
    | x \< E8M0_MIN     | E8M0_MIN      | NaN                    |
    | x \< 0            | Unspecified   | Unspecified            |
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>,
      TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>,
      TensorOf<[UI<2>]>, TensorOf<[I<2>]>
    ]>:$input,
    DefaultValuedStrAttr<StrAttr, "up">:$round_mode,
    DefaultValuedAttr<SI64Attr, "1">:$saturate,
    TypeAttr:$to
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>,
      TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>,
      TensorOf<[I<4>]>, TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>,
      TensorOf<[UI<2>]>, TensorOf<[I<2>]>
    ]>:$output
  );
}
