//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_EinsumOp
//===----------------------------------------------------------------------===//

def Onnx_EinsumOp : Onnx_Op<"Einsum", [Pure]> {
  let summary = "ONNX Einsum, schema version 12";

  let description = [{
    An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation

    ```
    output[output-term] = reduce-sum( input1[term1] * input2[term2] )
    ```

    where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
    that do not occur in the output-term.

    The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
    convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
    an operand tensor, and the characters within the terms correspond to operands dimensions.

    This sequence may be followed by "->" to separate the left and right hand side of the equation.
    If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
    summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
    output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
    equation.

    When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

    The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
    Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
    The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
    beginning of the output. The equation string may contain space (U+0020) character.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>>:$Inputs,
    StrAttr:$equation
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Output
  );
}
