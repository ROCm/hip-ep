//********************************************************
//   MiniONNX Operation Definitions
//   Auto-generated from ONNX schemas
//
//   To regenerate: python3 gen_mini_onnx.py --ops all
//********************************************************

def MiniONNX_AbsOp : MiniONNX_Op<"Abs",
  [Pure]> {
  let summary = "ONNX Abs operation";
  let description = [{
  Absolute takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where absolute value, y = abs(x), is applied to
  the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AcosOp : MiniONNX_Op<"Acos",
  [Pure]> {
  let summary = "ONNX Acos operation";
  let description = [{
  Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AcoshOp : MiniONNX_Op<"Acosh",
  [Pure]> {
  let summary = "ONNX Acosh operation";
  let description = [{
  Calculates the hyperbolic arccosine of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AddOp : MiniONNX_Op<"Add",
  [Pure]> {
  let summary = "ONNX Add operation";
  let description = [{
  Performs element-wise binary addition (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AffineGridOp : MiniONNX_Op<"AffineGrid",
  [Pure]> {
  let summary = "ONNX AffineGrid operation";
  let description = [{
  Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
  (https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
  An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
  ```
  [r00, r01, r02, t0]   [x]   [x']
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$theta,
    TensorOf<[I64]>:$size,
    OptionalAttr<SI64Attr>:$align_corners
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$grid
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AndOp : MiniONNX_Op<"And",
  [Pure]> {
  let summary = "ONNX And operation";
  let description = [{
  Returns the tensor resulted from performing the `and` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    TensorOf<[I1]>:$A,
    TensorOf<[I1]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ArgMaxOp : MiniONNX_Op<"ArgMax",
  [Pure]> {
  let summary = "ONNX ArgMax operation";
  let description = [{
  Computes the indices of the max elements of the input tensor's element along the
  provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
  If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
  If select_last_index is True (default False), the index of the last occurrence of the max
  is selected if the max appears more than once in the input. Otherwise the index of the
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$select_last_index
  );
  let results = (outs
    TensorOf<[I64]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ArgMinOp : MiniONNX_Op<"ArgMin",
  [Pure]> {
  let summary = "ONNX ArgMin operation";
  let description = [{
  Computes the indices of the min elements of the input tensor's element along the
  provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
  If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
  If select_last_index is True (default False), the index of the last occurrence of the min
  is selected if the min appears more than once in the input. Otherwise the index of the
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$select_last_index
  );
  let results = (outs
    TensorOf<[I64]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AsinOp : MiniONNX_Op<"Asin",
  [Pure]> {
  let summary = "ONNX Asin operation";
  let description = [{
  Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AsinhOp : MiniONNX_Op<"Asinh",
  [Pure]> {
  let summary = "ONNX Asinh operation";
  let description = [{
  Calculates the hyperbolic arcsine of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AtanOp : MiniONNX_Op<"Atan",
  [Pure]> {
  let summary = "ONNX Atan operation";
  let description = [{
  Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AtanhOp : MiniONNX_Op<"Atanh",
  [Pure]> {
  let summary = "ONNX Atanh operation";
  let description = [{
  Calculates the hyperbolic arctangent of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_AveragePoolOp : MiniONNX_Op<"AveragePool",
  [Pure]> {
  let summary = "ONNX AveragePool operation";
  let description = [{
  AveragePool consumes an input tensor X and applies average pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   average pooling consisting of computing the average on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape is calculated differently
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<SI64Attr>:$ceil_mode,
    OptionalAttr<SI64Attr>:$count_include_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    I64ArrayAttr:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BatchNormalizationOp : MiniONNX_Op<"BatchNormalization",
  [Pure]> {
  let summary = "ONNX BatchNormalization operation";
  let description = [{
  Carries out batch normalization as described in the paper
  https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
  There are five required inputs 'X', 'scale', 'B', 'input_mean' and
  'input_var'.
  Note that 'input_mean' and 'input_var' are expected to be the estimated
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scale,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input_mean,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input_var,
    OptionalAttr<F32Attr>:$epsilon,
    OptionalAttr<F32Attr>:$momentum,
    OptionalAttr<SI64Attr>:$training_mode
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$running_mean,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType, NoneType]>:$running_var
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BernoulliOp : MiniONNX_Op<"Bernoulli",
  [Pure]> {
  let summary = "ONNX Bernoulli operation";
  let description = [{
  Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
  containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
  where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).
  
  This operator is non-deterministic and may not produce the same values in different
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<F32Attr>:$seed
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BitShiftOp : MiniONNX_Op<"BitShift",
  [Pure]> {
  let summary = "ONNX BitShift operation";
  let description = [{
  Bitwise shift operator performs element-wise operation. For each input element, if the
  attribute \"direction\" is \"RIGHT\", this operator moves its binary representation toward
  the right side so that the input value is effectively decreased. If the attribute \"direction\"
  is \"LEFT\", bits of binary representation moves toward the left side, which results the
  increase of its actual value. The input X is the tensor to be shifted and another input
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>:$X,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>:$Y,
    StrAttr:$direction
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>]>:$Z
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BitwiseAndOp : MiniONNX_Op<"BitwiseAnd",
  [Pure]> {
  let summary = "ONNX BitwiseAnd operation";
  let description = [{
  Returns the tensor resulting from performing the bitwise `and` operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BitwiseNotOp : MiniONNX_Op<"BitwiseNot",
  [Pure]> {
  let summary = "ONNX BitwiseNot operation";
  let description = [{
  Returns the bitwise not of the input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BitwiseOrOp : MiniONNX_Op<"BitwiseOr",
  [Pure]> {
  let summary = "ONNX BitwiseOr operation";
  let description = [{
  Returns the tensor resulting from performing the bitwise `or` operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BitwiseXorOp : MiniONNX_Op<"BitwiseXor",
  [Pure]> {
  let summary = "ONNX BitwiseXor operation";
  let description = [{
  Returns the tensor resulting from performing the bitwise `xor` operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_BlackmanWindowOp : MiniONNX_Op<"BlackmanWindow",
  [Pure]> {
  let summary = "ONNX BlackmanWindow operation";
  let description = [{
  Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$size,
    OptionalAttr<SI64Attr>:$output_datatype,
    OptionalAttr<SI64Attr>:$periodic
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CastOp : MiniONNX_Op<"Cast",
  [Pure]> {
  let summary = "ONNX Cast operation";
  let description = [{
  The operator casts the elements of a given input tensor to a data type
  specified by the 'to' argument and returns an output tensor of the same size in
  the converted type. The 'to' argument must be one of the data types specified
  in the 'DataType' enum field in the TensorProto message.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$input,
    OptionalAttr<SI64Attr>:$saturate,
    SI64Attr:$to
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CastLikeOp : MiniONNX_Op<"CastLike",
  [Pure]> {
  let summary = "ONNX CastLike operation";
  let description = [{
  The operator casts the elements of a given input tensor (the first input) to
  the same data type as the elements of the second input tensor.
  See documentation of the Cast operator for further details.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$input,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$target_type,
    OptionalAttr<SI64Attr>:$saturate
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CeilOp : MiniONNX_Op<"Ceil",
  [Pure]> {
  let summary = "ONNX Ceil operation";
  let description = [{
  Ceil takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the ceil is, y = ceil(x), is applied to
  the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CeluOp : MiniONNX_Op<"Celu",
  [Pure]> {
  let summary = "ONNX Celu operation";
  let description = [{
  Continuously Differentiable Exponential Linear Units:
  Perform the linear unit element-wise on the input tensor X
  using formula:
  
  ```
  }];
  let arguments = (ins
    TensorOf<[F32]>:$X,
    OptionalAttr<F32Attr>:$alpha
  );
  let results = (outs
    TensorOf<[F32]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CenterCropPadOp : MiniONNX_Op<"CenterCropPad",
  [Pure]> {
  let summary = "ONNX CenterCropPad operation";
  let description = [{
  Center crop or pad an input to given dimensions.
  
  The crop/pad dimensions can be specified for a subset of the `axes`. Non-specified dimensions will not be
  cropped or padded.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input_data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$shape,
    OptionalAttr<I64ArrayAttr>:$axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output_data
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ClipOp : MiniONNX_Op<"Clip",
  [Pure]> {
  let summary = "ONNX Clip operation";
  let description = [{
  Clip operator limits the given input within an interval. The interval is
  specified by the inputs 'min' and 'max'. They default to
  numeric_limits::lowest() and numeric_limits::max(), respectively.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$min,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType, NoneType]>:$max
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType, NoneType]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_Col2ImOp : MiniONNX_Op<"Col2Im",
  [Pure]> {
  let summary = "ONNX Col2Im operation";
  let description = [{
  The operator rearranges column blocks back into a multidimensional image
  
  Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
  but it only supports *batched* multi-dimensional image tensors.
  Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    TensorOf<[I64]>:$image_shape,
    TensorOf<[I64]>:$block_shape,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CompressOp : MiniONNX_Op<"Compress",
  [Pure]> {
  let summary = "ONNX Compress operation";
  let description = [{
  Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
      In case axis is not provided, input is flattened before elements are selected.
      Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
      
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    TensorOf<[I1]>:$condition,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ConcatOp : MiniONNX_Op<"Concat",
  [Pure]> {
  let summary = "ONNX Concat operation";
  let description = [{
  Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>>:$inputs,
    SI64Attr:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$concat_result
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ConstantOp : MiniONNX_Op<"Constant",
  [Pure]> {
  let summary = "ONNX Constant operation";
  let description = [{
  This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
  or value_* must be specified.
  }];
  let arguments = (ins
    OptionalAttr<AnyAttr>:$sparse_value,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<F32Attr>:$value_float,
    OptionalAttr<F32ArrayAttr>:$value_floats,
    OptionalAttr<SI64Attr>:$value_int,
    OptionalAttr<I64ArrayAttr>:$value_ints,
    OptionalAttr<StrAttr>:$value_string,
    OptionalAttr<StrArrayAttr>:$value_strings
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ConstantOfShapeOp : MiniONNX_Op<"ConstantOfShape",
  [Pure]> {
  let summary = "ONNX ConstantOfShape operation";
  let description = [{
  Generate a tensor with given value and shape.
  }];
  let arguments = (ins
    TensorOf<[I64]>:$input,
    OptionalAttr<AnyAttr>:$value
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>, TensorOf<[BF16]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ConvOp : MiniONNX_Op<"Conv",
  [Pure]> {
  let summary = "ONNX Conv operation";
  let description = [{
  The convolution operator consumes an input tensor and a filter, and
  computes the output.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<SI64Attr>:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ConvIntegerOp : MiniONNX_Op<"ConvInteger",
  [Pure]> {
  let summary = "ONNX ConvInteger operation";
  let description = [{
  The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
  and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$x_zero_point,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$w_zero_point,
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<SI64Attr>:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    TensorOf<[I32]>:$y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ConvTransposeOp : MiniONNX_Op<"ConvTranspose",
  [Pure]> {
  let summary = "ONNX ConvTranspose operation";
  let description = [{
  The convolution transpose operator consumes an input tensor and a filter,
  and computes the output.
  
  If the pads parameter is provided the shape of the output is calculated via the following equation:
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<SI64Attr>:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$output_padding,
    OptionalAttr<I64ArrayAttr>:$output_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CosOp : MiniONNX_Op<"Cos",
  [Pure]> {
  let summary = "ONNX Cos operation";
  let description = [{
  Calculates the cosine of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CoshOp : MiniONNX_Op<"Cosh",
  [Pure]> {
  let summary = "ONNX Cosh operation";
  let description = [{
  Calculates the hyperbolic cosine of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_CumSumOp : MiniONNX_Op<"CumSum",
  [Pure]> {
  let summary = "ONNX CumSum operation";
  let description = [{
  Performs cumulative sum of the input elements along the given axis.
  By default, it will do the sum inclusively meaning the first element is copied as is.
  Through an `exclusive` attribute, this behavior can change to exclude the first element.
  It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$x,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$axis,
    OptionalAttr<SI64Attr>:$exclusive,
    OptionalAttr<SI64Attr>:$reverse
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DFTOp : MiniONNX_Op<"DFT",
  [Pure]> {
  let summary = "ONNX DFT operation";
  let description = [{
  Computes the discrete Fourier Transform (DFT) of the input.
  
  Assuming the input has shape `[M, N]`, where `N` is the dimension over which the
  DFT is computed and `M` denotes the conceptual \"all other dimensions,\"
  the DFT `y[m, k]` of shape `[M, N]` is defined as
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$dft_length,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axis,
    OptionalAttr<SI64Attr>:$inverse,
    OptionalAttr<SI64Attr>:$onesided
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DeformConvOp : MiniONNX_Op<"DeformConv",
  [Pure]> {
  let summary = "ONNX DeformConv operation";
  let description = [{
  Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
  This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$offset,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType]>:$mask,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<SI64Attr>:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<SI64Attr>:$offset_group,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DepthToSpaceOp : MiniONNX_Op<"DepthToSpace",
  [Pure]> {
  let summary = "ONNX DepthToSpace operation";
  let description = [{
  DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
  This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
  the input tensor where values from the depth dimension are moved in spatial blocks to the height
  and width dimensions. By default, `mode` = `DCR`.
  In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    SI64Attr:$blocksize,
    OptionalAttr<StrAttr>:$mode
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DequantizeLinearOp : MiniONNX_Op<"DequantizeLinear",
  [Pure]> {
  let summary = "ONNX DequantizeLinear operation";
  let description = [{
  The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
  The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
  for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
  `x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
  there's no zero point (zero point is supposed to be 0).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$x,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$x_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[I32]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$x_zero_point,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DetOp : MiniONNX_Op<"Det",
  [Pure]> {
  let summary = "ONNX Det operation";
  let description = [{
  Det calculates determinant of a square matrix or batches of square matrices.
  Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
  and the inner-most 2 dimensions form square matrices.
  The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
  e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DivOp : MiniONNX_Op<"Div",
  [Pure]> {
  let summary = "ONNX Div operation";
  let description = [{
  Performs element-wise binary division (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DropoutOp : MiniONNX_Op<"Dropout",
  [Pure]> {
  let summary = "ONNX Dropout operation";
  let description = [{
  Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
  output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
  Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
  the user can simply not pass `training_mode` input or set it to false.
  ```
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$ratio,
    AnyTypeOf<[TensorOf<[I1]>, NoneType]>:$training_mode,
    OptionalAttr<SI64Attr>:$seed
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output,
    AnyTypeOf<[TensorOf<[I1]>, NoneType, NoneType]>:$mask
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_DynamicQuantizeLinearOp : MiniONNX_Op<"DynamicQuantizeLinear",
  [Pure]> {
  let summary = "ONNX DynamicQuantizeLinear operation";
  let description = [{
  A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
  Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
  Scale is calculated as:
  ```
  y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
  }];
  let arguments = (ins
    TensorOf<[F32]>:$x
  );
  let results = (outs
    TensorOf<[UI8]>:$y,
    TensorOf<[F32]>:$y_scale,
    TensorOf<[UI8]>:$y_zero_point
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_EinsumOp : MiniONNX_Op<"Einsum",
  [Pure]> {
  let summary = "ONNX Einsum operation";
  let description = [{
  An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation
  
  ```
  output[output-term] = reduce-sum( input1[term1] * input2[term2] )
  ```
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>>:$Inputs,
    StrAttr:$equation
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_EluOp : MiniONNX_Op<"Elu",
  [Pure]> {
  let summary = "ONNX Elu operation";
  let description = [{
  Elu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
  0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<F32Attr>:$alpha
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_EqualOp : MiniONNX_Op<"Equal",
  [Pure]> {
  let summary = "ONNX Equal operation";
  let description = [{
  Returns the tensor resulted from performing the `equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[I1]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ErfOp : MiniONNX_Op<"Erf",
  [Pure]> {
  let summary = "ONNX Erf operation";
  let description = [{
  Computes the error function of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ExpOp : MiniONNX_Op<"Exp",
  [Pure]> {
  let summary = "ONNX Exp operation";
  let description = [{
  Calculates the exponential of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ExpandOp : MiniONNX_Op<"Expand",
  [Pure]> {
  let summary = "ONNX Expand operation";
  let description = [{
  Broadcast the input tensor following the given shape and the broadcast rule.
  The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
  Dimensions are right alignment;
  Two corresponding dimensions must have the same value, or one of them is equal to 1.
  Also, this operator is similar to numpy.broadcast_to(input, shape),
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    TensorOf<[I64]>:$shape
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_EyeLikeOp : MiniONNX_Op<"EyeLike",
  [Pure]> {
  let summary = "ONNX EyeLike operation";
  let description = [{
  Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
  tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
  same as the input tensor. The data type can be specified by the 'dtype' argument. If
  'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
  is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<SI64Attr>:$k
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_FlattenOp : MiniONNX_Op<"Flatten",
  [Pure]> {
  let summary = "ONNX Flatten operation";
  let description = [{
  Flattens the input tensor into a 2D matrix. If input tensor has shape
  (d_0, d_1, ... d_n) then the output will have shape
  (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_FloorOp : MiniONNX_Op<"Floor",
  [Pure]> {
  let summary = "ONNX Floor operation";
  let description = [{
  Floor takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the floor is, y = floor(x), is applied to
  the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GRUOp : MiniONNX_Op<"GRU",
  [Pure]> {
  let summary = "ONNX GRU operation";
  let description = [{
  Computes an one-layer GRU. This operator is usually supported via some custom
  implementation such as CuDNN.
  
  Notations:
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType]>:$initial_h,
    OptionalAttr<F32ArrayAttr>:$activation_alpha,
    OptionalAttr<F32ArrayAttr>:$activation_beta,
    OptionalAttr<StrArrayAttr>:$activations,
    OptionalAttr<F32Attr>:$clip,
    OptionalAttr<StrAttr>:$direction,
    OptionalAttr<SI64Attr>:$hidden_size,
    OptionalAttr<SI64Attr>:$layout,
    OptionalAttr<SI64Attr>:$linear_before_reset
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType]>:$Y,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType, NoneType]>:$Y_h
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GatherOp : MiniONNX_Op<"Gather",
  [Pure]> {
  let summary = "ONNX Gather operation";
  let description = [{
  Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
  entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
  them in an output tensor of rank q + (r - 1).
  
  If `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1\}\]`
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GatherElementsOp : MiniONNX_Op<"GatherElements",
  [Pure]> {
  let summary = "ONNX GatherElements operation";
  let description = [{
  GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
  and an optional attribute `axis` that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0). It is an indexing operation
  that produces its output by indexing into the input data tensor at index
  positions determined by elements of the `indices` tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GatherNDOp : MiniONNX_Op<"GatherND",
  [Pure]> {
  let summary = "ONNX GatherND operation";
  let description = [{
  Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
  slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.
  
  `indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
  where each element defines a slice of `data`
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    TensorOf<[I64]>:$indices,
    OptionalAttr<SI64Attr>:$batch_dims
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GeluOp : MiniONNX_Op<"Gelu",
  [Pure]> {
  let summary = "ONNX Gelu operation";
  let description = [{
  Gelu takes one input data (Tensor<T>) and produces one
  output data (Tensor<T>) where the gaussian error linear units function,
  $y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
  If the attribute \"approximate\" is set to \"tanh\", the function estimation,
  $y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
    OptionalAttr<StrAttr>:$approximate
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GemmOp : MiniONNX_Op<"Gemm",
  [Pure]> {
  let summary = "ONNX Gemm operation";
  let description = [{
  General Matrix multiplication:
  https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
  
  * A' = transpose(A) if transA else A
  * B' = transpose(B) if transB else B
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$B,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, NoneType]>:$C,
    OptionalAttr<F32Attr>:$alpha,
    OptionalAttr<F32Attr>:$beta,
    OptionalAttr<SI64Attr>:$transA,
    OptionalAttr<SI64Attr>:$transB
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, NoneType]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GlobalAveragePoolOp : MiniONNX_Op<"GlobalAveragePool",
  [Pure]> {
  let summary = "ONNX GlobalAveragePool operation";
  let description = [{
  GlobalAveragePool consumes an input tensor X and applies average pooling across
   the values in the same channel. This is equivalent to AveragePool with kernel size
   equal to the spatial dimension of input tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GlobalLpPoolOp : MiniONNX_Op<"GlobalLpPool",
  [Pure]> {
  let summary = "ONNX GlobalLpPool operation";
  let description = [{
  GlobalLpPool consumes an input tensor X and applies lp pool pooling across
   the values in the same channel. This is equivalent to LpPool with kernel size
   equal to the spatial dimension of input tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<SI64Attr>:$p
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GlobalMaxPoolOp : MiniONNX_Op<"GlobalMaxPool",
  [Pure]> {
  let summary = "ONNX GlobalMaxPool operation";
  let description = [{
  GlobalMaxPool consumes an input tensor X and applies max pooling across
   the values in the same channel. This is equivalent to MaxPool with kernel size
   equal to the spatial dimension of input tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GreaterOp : MiniONNX_Op<"Greater",
  [Pure]> {
  let summary = "ONNX Greater operation";
  let description = [{
  Returns the tensor resulted from performing the `greater` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GreaterOrEqualOp : MiniONNX_Op<"GreaterOrEqual",
  [Pure]> {
  let summary = "ONNX GreaterOrEqual operation";
  let description = [{
  Returns the tensor resulted from performing the `greater_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GridSampleOp : MiniONNX_Op<"GridSample",
  [Pure]> {
  let summary = "ONNX GridSample operation";
  let description = [{
  Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
  For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
  the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
  the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
  More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$grid,
    OptionalAttr<SI64Attr>:$align_corners,
    OptionalAttr<StrAttr>:$mode,
    OptionalAttr<StrAttr>:$padding_mode
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_GroupNormalizationOp : MiniONNX_Op<"GroupNormalization",
  [Pure]> {
  let summary = "ONNX GroupNormalization operation";
  let description = [{
  A GroupNormalization function. Carries out group normalization as described in
  the paper https://arxiv.org/abs/1803.08494
  
  This operator transforms input according to
  ```
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scale,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$bias,
    OptionalAttr<F32Attr>:$epsilon,
    SI64Attr:$num_groups
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_HammingWindowOp : MiniONNX_Op<"HammingWindow",
  [Pure]> {
  let summary = "ONNX HammingWindow operation";
  let description = [{
  Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$size,
    OptionalAttr<SI64Attr>:$output_datatype,
    OptionalAttr<SI64Attr>:$periodic
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_HannWindowOp : MiniONNX_Op<"HannWindow",
  [Pure]> {
  let summary = "ONNX HannWindow operation";
  let description = [{
  Generates a Hann window as described in the paper https://ieeexplore.ieee.org/document/1455106.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$size,
    OptionalAttr<SI64Attr>:$output_datatype,
    OptionalAttr<SI64Attr>:$periodic
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_HardSigmoidOp : MiniONNX_Op<"HardSigmoid",
  [Pure]> {
  let summary = "ONNX HardSigmoid operation";
  let description = [{
  HardSigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
  is applied to the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<F32Attr>:$alpha,
    OptionalAttr<F32Attr>:$beta
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_HardSwishOp : MiniONNX_Op<"HardSwish",
  [Pure]> {
  let summary = "ONNX HardSwish operation";
  let description = [{
  HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
  the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
  where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_HardmaxOp : MiniONNX_Op<"Hardmax",
  [Pure]> {
  let summary = "ONNX Hardmax operation";
  let description = [{
  The operator computes the hardmax values for the given input:
  
   Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise
  
  The \"axis\" attribute indicates the dimension along which Hardmax
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_IdentityOp : MiniONNX_Op<"Identity",
  [Pure]> {
  let summary = "ONNX Identity operation";
  let description = [{
  Identity operator
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ImageDecoderOp : MiniONNX_Op<"ImageDecoder",
  [Pure]> {
  let summary = "ONNX ImageDecoder operation";
  let description = [{
  Loads and decodes and image from a file. If it can't decode for any reason (e.g. corrupted encoded
  stream, invalid format, it will return an empty matrix).
  The following image formats are supported:
  * BMP
  * JPEG (note: Lossless JPEG support is optional)
  }];
  let arguments = (ins
    TensorOf<[UI8]>:$encoded_stream,
    OptionalAttr<StrAttr>:$pixel_format
  );
  let results = (outs
    TensorOf<[UI8]>:$image
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_InstanceNormalizationOp : MiniONNX_Op<"InstanceNormalization",
  [Pure]> {
  let summary = "ONNX InstanceNormalization operation";
  let description = [{
  Carries out instance normalization as described in the paper
  https://arxiv.org/abs/1607.08022.
  
  y = scale * (x - mean) / sqrt(variance + epsilon) + B,
  where mean and variance are computed per instance per channel.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$scale,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$B,
    OptionalAttr<F32Attr>:$epsilon
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_IsInfOp : MiniONNX_Op<"IsInf",
  [Pure]> {
  let summary = "ONNX IsInf operation";
  let description = [{
  Map infinity to true and other values to false.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$X,
    OptionalAttr<SI64Attr>:$detect_negative,
    OptionalAttr<SI64Attr>:$detect_positive
  );
  let results = (outs
    TensorOf<[I1]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_IsNaNOp : MiniONNX_Op<"IsNaN",
  [Pure]> {
  let summary = "ONNX IsNaN operation";
  let description = [{
  Returns which elements of the input are NaN.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$X
  );
  let results = (outs
    TensorOf<[I1]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LRNOp : MiniONNX_Op<"LRN",
  [Pure]> {
  let summary = "ONNX LRN operation";
  let description = [{
  Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
  It normalizes over local input regions.
  The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
  of shape `(N x C x D1 x D2, ..., Dk)`, its region is
  `{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
    OptionalAttr<F32Attr>:$alpha,
    OptionalAttr<F32Attr>:$beta,
    OptionalAttr<F32Attr>:$bias,
    SI64Attr:$size
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LSTMOp : MiniONNX_Op<"LSTM",
  [Pure]> {
  let summary = "ONNX LSTM operation";
  let description = [{
  Computes an one-layer LSTM. This operator is usually supported via some
  custom implementation such as CuDNN.
  
  Notations:
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType]>:$initial_h,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType]>:$initial_c,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType, NoneType]>:$P,
    OptionalAttr<F32ArrayAttr>:$activation_alpha,
    OptionalAttr<F32ArrayAttr>:$activation_beta,
    OptionalAttr<StrArrayAttr>:$activations,
    OptionalAttr<F32Attr>:$clip,
    OptionalAttr<StrAttr>:$direction,
    OptionalAttr<SI64Attr>:$hidden_size,
    OptionalAttr<SI64Attr>:$input_forget,
    OptionalAttr<SI64Attr>:$layout
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType, NoneType, NoneType]>:$Y,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType, NoneType, NoneType, NoneType]>:$Y_h,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType, NoneType, NoneType, NoneType, NoneType]>:$Y_c
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LayerNormalizationOp : MiniONNX_Op<"LayerNormalization",
  [Pure]> {
  let summary = "ONNX LayerNormalization operation";
  let description = [{
  This is layer normalization defined in ONNX as function.
        The overall computation can be split into two stages.
        The first stage is standardization, which makes the
        normalized elements have zero mean and unit variances.
        The computation required by standardization can be
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Scale,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$B,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<F32Attr>:$epsilon,
    OptionalAttr<SI64Attr>:$stash_type
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$Y,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>, NoneType]>:$Mean,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>, NoneType, NoneType]>:$InvStdDev
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LeakyReluOp : MiniONNX_Op<"LeakyRelu",
  [Pure]> {
  let summary = "ONNX LeakyRelu operation";
  let description = [{
  LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<F32Attr>:$alpha
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LessOp : MiniONNX_Op<"Less",
  [Pure]> {
  let summary = "ONNX Less operation";
  let description = [{
  Returns the tensor resulted from performing the `less` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LessOrEqualOp : MiniONNX_Op<"LessOrEqual",
  [Pure]> {
  let summary = "ONNX LessOrEqual operation";
  let description = [{
  Returns the tensor resulted from performing the `less_equal` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LogOp : MiniONNX_Op<"Log",
  [Pure]> {
  let summary = "ONNX Log operation";
  let description = [{
  Calculates the natural log of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LogSoftmaxOp : MiniONNX_Op<"LogSoftmax",
  [Pure]> {
  let summary = "ONNX LogSoftmax operation";
  let description = [{
  The operator computes the log of softmax values for the given input:
  
   LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
  
  The \"axis\" attribute indicates the dimension along which LogSoftmax
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LpNormalizationOp : MiniONNX_Op<"LpNormalization",
  [Pure]> {
  let summary = "ONNX LpNormalization operation";
  let description = [{
  Given a matrix, apply Lp-normalization along the provided axis.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<SI64Attr>:$p
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_LpPoolOp : MiniONNX_Op<"LpPool",
  [Pure]> {
  let summary = "ONNX LpPool operation";
  let description = [{
  LpPool consumes an input tensor X and applies Lp pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   Lp pooling consisting of computing the Lp norm on all values of a subset
   of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape will be following:
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<SI64Attr>:$ceil_mode,
    OptionalAttr<I64ArrayAttr>:$dilations,
    I64ArrayAttr:$kernel_shape,
    OptionalAttr<SI64Attr>:$p,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MatMulOp : MiniONNX_Op<"MatMul",
  [Pure]> {
  let summary = "ONNX MatMul operation";
  let description = [{
  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MatMulIntegerOp : MiniONNX_Op<"MatMulInteger",
  [Pure]> {
  let summary = "ONNX MatMulInteger operation";
  let description = [{
  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
  The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$A,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$B,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$a_zero_point,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, NoneType]>:$b_zero_point
  );
  let results = (outs
    TensorOf<[I32]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MaxOp : MiniONNX_Op<"Max",
  [Pure]> {
  let summary = "ONNX Max operation";
  let description = [{
  Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$max
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MaxPoolOp : MiniONNX_Op<"MaxPool",
  [Pure]> {
  let summary = "ONNX MaxPool operation";
  let description = [{
  MaxPool consumes an input tensor X and applies max pooling across
   the tensor according to kernel sizes, stride sizes, and pad lengths.
   max pooling consisting of computing the max on all values of a
   subset of the input tensor according to the kernel size and downsampling the
   data into the output tensor Y for further processing. The output spatial shape is calculated differently
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>:$X,
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<SI64Attr>:$ceil_mode,
    OptionalAttr<I64ArrayAttr>:$dilations,
    I64ArrayAttr:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<SI64Attr>:$storage_order,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I8]>, TensorOf<[UI8]>]>:$Y,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$Indices
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MaxRoiPoolOp : MiniONNX_Op<"MaxRoiPool",
  [Pure]> {
  let summary = "ONNX MaxRoiPool operation";
  let description = [{
  ROI max pool consumes an input tensor X and region of interests (RoIs) to
   apply max pooling across each RoI, to produce output 4-D tensor of shape
   (num_rois, channels, pooled_shape[0], pooled_shape[1]).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$rois,
    I64ArrayAttr:$pooled_shape,
    OptionalAttr<F32Attr>:$spatial_scale
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MaxUnpoolOp : MiniONNX_Op<"MaxUnpool",
  [Pure]> {
  let summary = "ONNX MaxUnpool operation";
  let description = [{
  MaxUnpool essentially computes the partial inverse of the MaxPool op.
   The input information to this op is typically the output information from a MaxPool op. The first
   input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
   from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
   to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I64]>:$I,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$output_shape,
    I64ArrayAttr:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MeanOp : MiniONNX_Op<"Mean",
  [Pure]> {
  let summary = "ONNX Mean operation";
  let description = [{
  Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$mean
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MeanVarianceNormalizationOp : MiniONNX_Op<"MeanVarianceNormalization",
  [Pure]> {
  let summary = "ONNX MeanVarianceNormalization operation";
  let description = [{
  A MeanVarianceNormalization Function: Perform mean variance normalization
        on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
    OptionalAttr<I64ArrayAttr>:$axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MelWeightMatrixOp : MiniONNX_Op<"MelWeightMatrix",
  [Pure]> {
  let summary = "ONNX MelWeightMatrix operation";
  let description = [{
  Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
  This function defines the mel scale in terms of a frequency in hertz according to the following formula:
  
      mel(f) = 2595 * log10(1 + f/700)
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$num_mel_bins,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$dft_length,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$sample_rate,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$lower_edge_hertz,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$upper_edge_hertz,
    OptionalAttr<SI64Attr>:$output_datatype
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MinOp : MiniONNX_Op<"Min",
  [Pure]> {
  let summary = "ONNX Min operation";
  let description = [{
  Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$min
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MishOp : MiniONNX_Op<"Mish",
  [Pure]> {
  let summary = "ONNX Mish operation";
  let description = [{
  Mish: A Self Regularized Non-Monotonic Neural Activation Function.
  
  Perform the linear unit element-wise on the input tensor X using formula:
  
  ```
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ModOp : MiniONNX_Op<"Mod",
  [Pure]> {
  let summary = "ONNX Mod operation";
  let description = [{
  Performs element-wise binary modulus (with Numpy-style broadcasting support).
    The sign of the remainder is the same as that of the Divisor.
  
    Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
    (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B,
    OptionalAttr<SI64Attr>:$fmod
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MulOp : MiniONNX_Op<"Mul",
  [Pure]> {
  let summary = "ONNX Mul operation";
  let description = [{
  Performs element-wise binary multiplication (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_MultinomialOp : MiniONNX_Op<"Multinomial",
  [Pure]> {
  let summary = "ONNX Multinomial operation";
  let description = [{
  Generate a tensor of samples from a multinomial distribution according to the probabilities
  of each of the possible outcomes.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<SI64Attr>:$sample_size,
    OptionalAttr<F32Attr>:$seed
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_NegOp : MiniONNX_Op<"Neg",
  [Pure]> {
  let summary = "ONNX Neg operation";
  let description = [{
  Neg takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where each element flipped sign, y = -x, is applied to
  the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_NegativeLogLikelihoodLossOp : MiniONNX_Op<"NegativeLogLikelihoodLoss",
  [Pure]> {
  let summary = "ONNX NegativeLogLikelihoodLoss operation";
  let description = [{
  A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
  Its \"input\" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
  The \"input\" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
  The operator's \"target\" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
  or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$target,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$weight,
    OptionalAttr<SI64Attr>:$ignore_index,
    OptionalAttr<StrAttr>:$reduction
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$loss
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_NonMaxSuppressionOp : MiniONNX_Op<"NonMaxSuppression",
  [Pure]> {
  let summary = "ONNX NonMaxSuppression operation";
  let description = [{
  Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
  Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
  Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
  orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
  result in the same boxes being selected by the algorithm.
  }];
  let arguments = (ins
    TensorOf<[F32]>:$boxes,
    TensorOf<[F32]>:$scores,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$max_output_boxes_per_class,
    AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$iou_threshold,
    AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$score_threshold,
    OptionalAttr<SI64Attr>:$center_point_box
  );
  let results = (outs
    TensorOf<[I64]>:$selected_indices
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_NonZeroOp : MiniONNX_Op<"NonZero",
  [Pure]> {
  let summary = "ONNX NonZero operation";
  let description = [{
  Returns the indices of the elements that are non-zero
      (in row-major order - by dimension).
      NonZero behaves similar to numpy.nonzero:
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
      but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$X
  );
  let results = (outs
    TensorOf<[I64]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_NotOp : MiniONNX_Op<"Not",
  [Pure]> {
  let summary = "ONNX Not operation";
  let description = [{
  Returns the negation of the input tensor element-wise.
  }];
  let arguments = (ins
    TensorOf<[I1]>:$X
  );
  let results = (outs
    TensorOf<[I1]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_OneHotOp : MiniONNX_Op<"OneHot",
  [Pure]> {
  let summary = "ONNX OneHot operation";
  let description = [{
  Produces a one-hot tensor based on inputs.
      The locations represented by the index values in the 'indices' input tensor will have 'on_value'
      and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
      are specified as part of required input argument 'values', which is a two-element tensor of format
      [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$indices,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$depth,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$values,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_OptionalGetElementOp : MiniONNX_Op<"OptionalGetElement",
  [Pure]> {
  let summary = "ONNX OptionalGetElement operation";
  let description = [{
  If the input is a tensor or sequence type, it returns the input.
  If the input is an optional type, it outputs the element in the input.
  It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_OptionalHasElementOp : MiniONNX_Op<"OptionalHasElement",
  [Pure]> {
  let summary = "ONNX OptionalHasElement operation";
  let description = [{
  Returns true if (1) the input is an optional-type and contains an element,
  or, (2) the input is a tensor or sequence type.
  If the input is not provided or is an empty optional-type, this op returns false.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, NoneType]>:$input
  );
  let results = (outs
    TensorOf<[I1]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_OrOp : MiniONNX_Op<"Or",
  [Pure]> {
  let summary = "ONNX Or operation";
  let description = [{
  Returns the tensor resulted from performing the `or` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    TensorOf<[I1]>:$A,
    TensorOf<[I1]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_PReluOp : MiniONNX_Op<"PRelu",
  [Pure]> {
  let summary = "ONNX PRelu operation";
  let description = [{
  PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
  output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
  `f(x) = x for x >= 0`., is applied to the data tensor elementwise.
  This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$X,
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$slope
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_PadOp : MiniONNX_Op<"Pad",
  [Pure]> {
  let summary = "ONNX Pad operation";
  let description = [{
  Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
  a padded tensor (`output`) is generated.
  
  The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    TensorOf<[I64]>:$pads,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, NoneType]>:$constant_value,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<StrAttr>:$mode
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, NoneType]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_PowOp : MiniONNX_Op<"Pow",
  [Pure]> {
  let summary = "ONNX Pow operation";
  let description = [{
  Pow takes input data (Tensor<T>) and exponent Tensor, and
  produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
  is applied to the data tensor elementwise.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Z
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_QLinearConvOp : MiniONNX_Op<"QLinearConv",
  [Pure]> {
  let summary = "ONNX QLinearConv operation";
  let description = [{
  The convolution operator consumes a quantized input tensor, its scale and zero point,
  a quantized filter, its scale and zero point, and output's scale and zero point,
  and computes the quantized output. Each scale and zero-point pair must have same shape.
  It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
  Each input or output and its related zero point must have same type.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x,
    TensorOf<[F32]>:$x_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$x_zero_point,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w,
    TensorOf<[F32]>:$w_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$w_zero_point,
    TensorOf<[F32]>:$y_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y_zero_point,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$B,
    OptionalAttr<StrAttr>:$auto_pad,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<SI64Attr>:$group,
    OptionalAttr<I64ArrayAttr>:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_QLinearMatMulOp : MiniONNX_Op<"QLinearMatMul",
  [Pure]> {
  let summary = "ONNX QLinearMatMul operation";
  let description = [{
  Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
  It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
  and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
  For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
  Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$a,
    TensorOf<[F32]>:$a_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$a_zero_point,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$b,
    TensorOf<[F32]>:$b_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$b_zero_point,
    TensorOf<[F32]>:$y_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y_zero_point
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>]>:$y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_QuantizeLinearOp : MiniONNX_Op<"QuantizeLinear",
  [Pure]> {
  let summary = "ONNX QuantizeLinear operation";
  let description = [{
  The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
  The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
  The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
  For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
  For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$x,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>, TensorOf<[I32]>]>:$y_scale,
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$y_zero_point,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<SI64Attr>:$saturate
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[I8]>, TensorOf<[UI8]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>, NoneType]>:$y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RNNOp : MiniONNX_Op<"RNN",
  [Pure]> {
  let summary = "ONNX RNN operation";
  let description = [{
  Computes an one-layer simple RNN. This operator is usually supported
  via some custom implementation such as CuDNN.
  
  Notations:
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$W,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$R,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$B,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType]>:$initial_h,
    OptionalAttr<F32ArrayAttr>:$activation_alpha,
    OptionalAttr<F32ArrayAttr>:$activation_beta,
    OptionalAttr<StrArrayAttr>:$activations,
    OptionalAttr<F32Attr>:$clip,
    OptionalAttr<StrAttr>:$direction,
    OptionalAttr<SI64Attr>:$hidden_size,
    OptionalAttr<SI64Attr>:$layout
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType]>:$Y,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType, NoneType, NoneType, NoneType]>:$Y_h
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RandomNormalOp : MiniONNX_Op<"RandomNormal",
  [Pure]> {
  let summary = "ONNX RandomNormal operation";
  let description = [{
  Generate a tensor with random values drawn from a normal distribution. The shape
  of the tensor is specified by the `shape` argument and the parameter of the normal distribution
  specified by `mean` and `scale`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  }];
  let arguments = (ins
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<F32Attr>:$mean,
    OptionalAttr<F32Attr>:$scale,
    OptionalAttr<F32Attr>:$seed,
    I64ArrayAttr:$shape
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RandomNormalLikeOp : MiniONNX_Op<"RandomNormalLike",
  [Pure]> {
  let summary = "ONNX RandomNormalLike operation";
  let description = [{
  Generate a tensor with random values drawn from a normal distribution.
  The shape of the output tensor is copied from the shape of the input tensor,
  and the parameters of the normal distribution are specified by `mean` and `scale`.
  
  The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<F32Attr>:$mean,
    OptionalAttr<F32Attr>:$scale,
    OptionalAttr<F32Attr>:$seed
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RandomUniformOp : MiniONNX_Op<"RandomUniform",
  [Pure]> {
  let summary = "ONNX RandomUniform operation";
  let description = [{
  Generate a tensor with random values drawn from a uniform distribution. The shape
  of the tensor is specified by the `shape` argument and the range by `low` and `high`.
  
  The data type is specified by the 'dtype' argument. The 'dtype' argument must
  be one of the data types specified in the 'DataType' enum field in the
  }];
  let arguments = (ins
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<F32Attr>:$high,
    OptionalAttr<F32Attr>:$low,
    OptionalAttr<F32Attr>:$seed,
    I64ArrayAttr:$shape
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RandomUniformLikeOp : MiniONNX_Op<"RandomUniformLike",
  [Pure]> {
  let summary = "ONNX RandomUniformLike operation";
  let description = [{
  Generate a tensor with random values drawn from a uniform distribution.
  The shape of the output tensor is copied from the shape of the input tensor,
  and the parameters of the uniform distribution are specified by `low` and `high`.
  
  The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    OptionalAttr<SI64Attr>:$dtype,
    OptionalAttr<F32Attr>:$high,
    OptionalAttr<F32Attr>:$low,
    OptionalAttr<F32Attr>:$seed
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RangeOp : MiniONNX_Op<"Range",
  [Pure]> {
  let summary = "ONNX Range operation";
  let description = [{
  Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
  up to `limit` (exclusive).
  
  The number of elements in the output of range is computed as below:
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$start,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$limit,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$delta
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReciprocalOp : MiniONNX_Op<"Reciprocal",
  [Pure]> {
  let summary = "ONNX Reciprocal operation";
  let description = [{
  Reciprocal takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the reciprocal is, y = 1/x, is applied to
  the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceL1Op : MiniONNX_Op<"ReduceL1",
  [Pure]> {
  let summary = "ONNX ReduceL1 operation";
  let description = [{
  Computes the L1 norm of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields 0.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceL2Op : MiniONNX_Op<"ReduceL2",
  [Pure]> {
  let summary = "ONNX ReduceL2 operation";
  let description = [{
  Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields 0.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceLogSumOp : MiniONNX_Op<"ReduceLogSum",
  [Pure]> {
  let summary = "ONNX ReduceLogSum operation";
  let description = [{
  Computes the log sum of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceLogSumExpOp : MiniONNX_Op<"ReduceLogSumExp",
  [Pure]> {
  let summary = "ONNX ReduceLogSumExp operation";
  let description = [{
  Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceMaxOp : MiniONNX_Op<"ReduceMax",
  [Pure]> {
  let summary = "ONNX ReduceMax operation";
  let description = [{
  Computes the max of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceMeanOp : MiniONNX_Op<"ReduceMean",
  [Pure]> {
  let summary = "ONNX ReduceMean operation";
  let description = [{
  Computes the mean of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields undefined.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceMinOp : MiniONNX_Op<"ReduceMin",
  [Pure]> {
  let summary = "ONNX ReduceMin operation";
  let description = [{
  Computes the min of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, TensorOf<[UI8]>, TensorOf<[I8]>, TensorOf<[I1]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceProdOp : MiniONNX_Op<"ReduceProd",
  [Pure]> {
  let summary = "ONNX ReduceProd operation";
  let description = [{
  Computes the product of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields 1.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceSumOp : MiniONNX_Op<"ReduceSum",
  [Pure]> {
  let summary = "ONNX ReduceSum operation";
  let description = [{
  Computes the sum of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields 0.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReduceSumSquareOp : MiniONNX_Op<"ReduceSumSquare",
  [Pure]> {
  let summary = "ONNX ReduceSumSquare operation";
  let description = [{
  Computes the sum square of the input tensor's elements along the provided axes. The resulting
  tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
  the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
  valid. Reduction over an empty set of values yields 0.
  
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes,
    OptionalAttr<SI64Attr>:$keepdims,
    OptionalAttr<SI64Attr>:$noop_with_empty_axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$reduced
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReluOp : MiniONNX_Op<"Relu",
  [Pure]> {
  let summary = "ONNX Relu operation";
  let description = [{
  Relu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
  the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[I32]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReshapeOp : MiniONNX_Op<"Reshape",
  [Pure]> {
  let summary = "ONNX Reshape operation";
  let description = [{
  Reshape the input tensor similar to numpy.reshape.
  First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
  At most one dimension of the new shape can be -1. In this case, the value is
  inferred from the size of the tensor and the remaining dimensions. A dimension
  could also be 0, in which case the actual dimension value is unchanged (i.e. taken
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$data,
    TensorOf<[I64]>:$shape,
    OptionalAttr<SI64Attr>:$allowzero
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$reshaped
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ResizeOp : MiniONNX_Op<"Resize",
  [Pure]> {
  let summary = "ONNX Resize operation";
  let description = [{
  Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
  Each dimension value of the output tensor is:
  ```
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
  ```
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$roi,
    AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$scales,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$sizes,
    OptionalAttr<SI64Attr>:$antialias,
    OptionalAttr<I64ArrayAttr>:$axes,
    OptionalAttr<StrAttr>:$coordinate_transformation_mode,
    OptionalAttr<F32Attr>:$cubic_coeff_a,
    OptionalAttr<SI64Attr>:$exclude_outside,
    OptionalAttr<F32Attr>:$extrapolation_value,
    OptionalAttr<StrAttr>:$keep_aspect_ratio_policy,
    OptionalAttr<StrAttr>:$mode,
    OptionalAttr<StrAttr>:$nearest_mode
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ReverseSequenceOp : MiniONNX_Op<"ReverseSequence",
  [Pure]> {
  let summary = "ONNX ReverseSequence operation";
  let description = [{
  Reverse batch of sequences having different lengths specified by `sequence_lens`.
  
  For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
  and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
  sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    TensorOf<[I64]>:$sequence_lens,
    OptionalAttr<SI64Attr>:$batch_axis,
    OptionalAttr<SI64Attr>:$time_axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RoiAlignOp : MiniONNX_Op<"RoiAlign",
  [Pure]> {
  let summary = "ONNX RoiAlign operation";
  let description = [{
  Region of Interest (RoI) align operation described in the
  [Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
  RoiAlign consumes an input tensor X and region of interests (rois)
  to apply pooling across each RoI; it produces a 4-D tensor of shape
  (num_rois, C, output_height, output_width).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$rois,
    TensorOf<[I64]>:$batch_indices,
    OptionalAttr<StrAttr>:$coordinate_transformation_mode,
    OptionalAttr<StrAttr>:$mode,
    OptionalAttr<SI64Attr>:$output_height,
    OptionalAttr<SI64Attr>:$output_width,
    OptionalAttr<SI64Attr>:$sampling_ratio,
    OptionalAttr<F32Attr>:$spatial_scale
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_RoundOp : MiniONNX_Op<"Round",
  [Pure]> {
  let summary = "ONNX Round operation";
  let description = [{
  Round takes one input Tensor and rounds the values, element-wise, meaning
  it finds the nearest integer for each value.
  In case of halves, the rule is to round them to the nearest even integer.
  If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
  The output tensor has the same shape and type as the input.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_STFTOp : MiniONNX_Op<"STFT",
  [Pure]> {
  let summary = "ONNX STFT operation";
  let description = [{
  Computes the Short-time Fourier Transform of the signal.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$signal,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$frame_step,
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$window,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$frame_length,
    OptionalAttr<SI64Attr>:$onesided
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ScatterOp : MiniONNX_Op<"Scatter",
  [Pure]> {
  let summary = "ONNX Scatter operation";
  let description = [{
  This operator is deprecated. Please use ScatterElements, which provides the same functionality.
  
  Scatter takes three inputs `data`, `updates`, and `indices` of the same
  rank r >= 1 and an optional attribute axis that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0). The output of the operation
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$updates,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ScatterElementsOp : MiniONNX_Op<"ScatterElements",
  [Pure]> {
  let summary = "ONNX ScatterElements operation";
  let description = [{
  ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
  rank r >= 1 and an optional attribute axis that identifies an axis of `data`
  (by default, the outer-most axis, that is axis 0). The output of the operation
  is produced by creating a copy of the input `data`, and then updating its value
  to values specified by `updates` at specific index positions specified by
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$indices,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$updates,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<StrAttr>:$reduction
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ScatterNDOp : MiniONNX_Op<"ScatterND",
  [Pure]> {
  let summary = "ONNX ScatterND operation";
  let description = [{
  ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
  and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
  is produced by creating a copy of the input `data`, and then updating its value to values
  specified by `updates` at specific index positions specified by `indices`. Its output shape
  is the same as the shape of `data`.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    TensorOf<[I64]>:$indices,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$updates,
    OptionalAttr<StrAttr>:$reduction
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SeluOp : MiniONNX_Op<"Selu",
  [Pure]> {
  let summary = "ONNX Selu operation";
  let description = [{
  Selu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the scaled exponential linear unit function,
  `y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
  is applied to the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<F32Attr>:$alpha,
    OptionalAttr<F32Attr>:$gamma
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ShapeOp : MiniONNX_Op<"Shape",
  [Pure]> {
  let summary = "ONNX Shape operation";
  let description = [{
  Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  Optional attributes start and end can be used to compute a slice of the input tensor's shape.
  If start axis is omitted, the slice starts from axis 0.
  The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
  If the end axis is omitted, the axes upto the last one will be included.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$data,
    OptionalAttr<SI64Attr>:$end,
    OptionalAttr<SI64Attr>:$start
  );
  let results = (outs
    TensorOf<[I64]>:$shape
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ShrinkOp : MiniONNX_Op<"Shrink",
  [Pure]> {
  let summary = "ONNX Shrink operation";
  let description = [{
  Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
  having same datatype and shape with input. It has two attributes, lambd and
  bias. The formula of this operator is: If x < -lambd, y = x + bias;
  If x > lambd, y = x - bias; Otherwise, y = 0.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input,
    OptionalAttr<F32Attr>:$bias,
    OptionalAttr<F32Attr>:$lambd
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SigmoidOp : MiniONNX_Op<"Sigmoid",
  [Pure]> {
  let summary = "ONNX Sigmoid operation";
  let description = [{
  Sigmoid takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
  tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SignOp : MiniONNX_Op<"Sign",
  [Pure]> {
  let summary = "ONNX Sign operation";
  let description = [{
  Calculate the sign of the given input tensor element-wise.
  If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SinOp : MiniONNX_Op<"Sin",
  [Pure]> {
  let summary = "ONNX Sin operation";
  let description = [{
  Calculates the sine of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SinhOp : MiniONNX_Op<"Sinh",
  [Pure]> {
  let summary = "ONNX Sinh operation";
  let description = [{
  Calculates the hyperbolic sine of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SizeOp : MiniONNX_Op<"Size",
  [Pure]> {
  let summary = "ONNX Size operation";
  let description = [{
  Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>, TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>, TensorOf<[F8E5M2FNUZ]>]>:$data
  );
  let results = (outs
    TensorOf<[I64]>:$size
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SliceOp : MiniONNX_Op<"Slice",
  [Pure]> {
  let summary = "ONNX Slice operation";
  let description = [{
  Produces a slice of the input tensor along multiple axes. Similar to numpy:
  https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding
  
  Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
  of its input `data` tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$starts,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$ends,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType]>:$axes,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>, NoneType, NoneType]>:$steps
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SoftmaxOp : MiniONNX_Op<"Softmax",
  [Pure]> {
  let summary = "ONNX Softmax operation";
  let description = [{
  The operator computes the normalized exponential values for the given input:
  
   Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) 
  
  The \"axis\" attribute indicates the dimension along which Softmax
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input,
    OptionalAttr<SI64Attr>:$axis
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SoftmaxCrossEntropyLossOp : MiniONNX_Op<"SoftmaxCrossEntropyLoss",
  [Pure]> {
  let summary = "ONNX SoftmaxCrossEntropyLoss operation";
  let description = [{
  Loss function that measures the softmax cross entropy
  between 'scores' and 'labels'.
  This operator first computes a loss tensor whose shape is identical to the labels input.
  If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
  If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$scores,
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$labels,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$weights,
    OptionalAttr<SI64Attr>:$ignore_index,
    OptionalAttr<StrAttr>:$reduction
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType]>:$output,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>, NoneType, NoneType]>:$log_prob
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SoftplusOp : MiniONNX_Op<"Softplus",
  [Pure]> {
  let summary = "ONNX Softplus operation";
  let description = [{
  Softplus takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
  the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SoftsignOp : MiniONNX_Op<"Softsign",
  [Pure]> {
  let summary = "ONNX Softsign operation";
  let description = [{
  Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SpaceToDepthOp : MiniONNX_Op<"SpaceToDepth",
  [Pure]> {
  let summary = "ONNX SpaceToDepth operation";
  let description = [{
  SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
  this op outputs a copy of the input tensor where values from the height and width dimensions
  are moved to the depth dimension.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    SI64Attr:$blocksize
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SplitOp : MiniONNX_Op<"Split",
  [Pure]> {
  let summary = "ONNX Split operation";
  let description = [{
  Split a tensor into a list of tensors, along the specified 'axis'.
  Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
  If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
  If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
  If the input 'split' is specified, it indicates the sizes of each output in the split.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$split,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<SI64Attr>:$num_outputs
  );
  let results = (outs
    Variadic<AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>>:$outputs
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SqrtOp : MiniONNX_Op<"Sqrt",
  [Pure]> {
  let summary = "ONNX Sqrt operation";
  let description = [{
  Square root takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the square root is, y = x^0.5, is applied to
  the tensor elementwise. If x is negative, then it will return NaN.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$X
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SqueezeOp : MiniONNX_Op<"Squeeze",
  [Pure]> {
  let summary = "ONNX Squeeze operation";
  let description = [{
  Remove single-dimensional entries from the shape of a tensor.
  Takes an input `axes` with a list of axes to squeeze.
  If `axes` is not provided, all the single dimensions will be removed from
  the shape. If an axis is selected with shape entry not equal to one, an error is raised.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$squeezed
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SubOp : MiniONNX_Op<"Sub",
  [Pure]> {
  let summary = "ONNX Sub operation";
  let description = [{
  Performs element-wise binary subtraction (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  
  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_SumOp : MiniONNX_Op<"Sum",
  [Pure]> {
  let summary = "ONNX Sum operation";
  let description = [{
  Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
  All inputs and outputs must have the same data type.
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>>:$data_0
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$sum
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_TanOp : MiniONNX_Op<"Tan",
  [Pure]> {
  let summary = "ONNX Tan operation";
  let description = [{
  Calculates the tangent of the given input tensor, element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_TanhOp : MiniONNX_Op<"Tanh",
  [Pure]> {
  let summary = "ONNX Tanh operation";
  let description = [{
  Calculates the hyperbolic tangent of the given input tensor element-wise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$input
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_TfIdfVectorizerOp : MiniONNX_Op<"TfIdfVectorizer",
  [Pure]> {
  let summary = "ONNX TfIdfVectorizer operation";
  let description = [{
  This transform extracts n-grams from the input sequence and save them as a vector. Input can
  be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
  For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
  More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
  If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$X,
    SI64Attr:$max_gram_length,
    SI64Attr:$max_skip_count,
    SI64Attr:$min_gram_length,
    StrAttr:$mode,
    I64ArrayAttr:$ngram_counts,
    I64ArrayAttr:$ngram_indexes,
    OptionalAttr<I64ArrayAttr>:$pool_int64s,
    OptionalAttr<StrArrayAttr>:$pool_strings,
    OptionalAttr<F32ArrayAttr>:$weights
  );
  let results = (outs
    TensorOf<[F32]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_ThresholdedReluOp : MiniONNX_Op<"ThresholdedRelu",
  [Pure]> {
  let summary = "ONNX ThresholdedRelu operation";
  let description = [{
  ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
  (Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
  is applied to the tensor elementwise.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    OptionalAttr<F32Attr>:$alpha
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_TileOp : MiniONNX_Op<"Tile",
  [Pure]> {
  let summary = "ONNX Tile operation";
  let description = [{
  Constructs a tensor by tiling a given tensor.
  This is the same as function `tile` in Numpy, but no broadcast.
  For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    TensorOf<[I64]>:$repeats
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_TopKOp : MiniONNX_Op<"TopK",
  [Pure]> {
  let summary = "ONNX TopK operation";
  let description = [{
  Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
  shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  
  * Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    TensorOf<[I64]>:$K,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<SI64Attr>:$largest,
    OptionalAttr<SI64Attr>:$sorted
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Values,
    TensorOf<[I64]>:$Indices
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_TransposeOp : MiniONNX_Op<"Transpose",
  [Pure]> {
  let summary = "ONNX Transpose operation";
  let description = [{
  Transpose the input tensor similar to numpy.transpose. For example, when
  perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
  will be (2, 1, 3).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    OptionalAttr<I64ArrayAttr>:$perm
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$transposed
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_TriluOp : MiniONNX_Op<"Trilu",
  [Pure]> {
  let summary = "ONNX Trilu operation";
  let description = [{
  Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
  The attribute \"upper\" determines whether the upper or lower part is retained. If set to true,
  the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
  Default value for the \"upper\" attribute is true.
  Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$input,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$k,
    OptionalAttr<SI64Attr>:$upper
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_UniqueOp : MiniONNX_Op<"Unique",
  [Pure]> {
  let summary = "ONNX Unique operation";
  let description = [{
  Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
  Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.
  
  This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
  The first output tensor 'Y' contains all unique values or subtensors of the input.
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$X,
    OptionalAttr<SI64Attr>:$axis,
    OptionalAttr<SI64Attr>:$sorted
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$Y,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$indices,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$inverse_indices,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$counts
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_UnsqueezeOp : MiniONNX_Op<"Unsqueeze",
  [Pure]> {
  let summary = "ONNX Unsqueeze operation";
  let description = [{
  Insert single-dimensional entries to the shape of an input tensor (`data`).
  Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).
  
  For example, given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$data,
    TensorOf<[I64]>:$axes
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$expanded
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_UpsampleOp : MiniONNX_Op<"Upsample",
  [Pure]> {
  let summary = "ONNX Upsample operation";
  let description = [{
  Upsample the input tensor.
  Each dimension value of the output tensor is:
    output_dimension = floor(input_dimension * scale).
  }];
  let arguments = (ins
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$X,
    TensorOf<[F32]>:$scales,
    OptionalAttr<StrAttr>:$mode
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$Y
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_WhereOp : MiniONNX_Op<"Where",
  [Pure]> {
  let summary = "ONNX Where operation";
  let description = [{
  Return elements, either from X or Y, depending on condition.
  Where behaves like
  [numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
  with three parameters.
  
  }];
  let arguments = (ins
    TensorOf<[I1]>:$condition,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$X,
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$Y
  );
  let results = (outs
    AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[I1]>]>:$output
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MiniONNX_XorOp : MiniONNX_Op<"Xor",
  [Pure]> {
  let summary = "ONNX Xor operation";
  let description = [{
  Returns the tensor resulted from performing the `xor` logical operation
  elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
  
  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];
  let arguments = (ins
    TensorOf<[I1]>:$A,
    TensorOf<[I1]>:$B
  );
  let results = (outs
    TensorOf<[I1]>:$C
  );
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

