//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT_GGML_IR_GGMLOPS_TD
#define HIP_DIALECT_GGML_IR_GGMLOPS_TD

include "hip/Dialect/Ggml/IR/GgmlDialect.td"
include "hip/Dialect/Ggml/IR/GgmlAttrs.td"
include "hip/Dialect/Ggml/IR/GgmlShapeRule.td"
include "hip/Dialect/Ggml/IR/GgmlSource.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

/// One GGML operation, as the dialect has it.
///
/// Every operation states what it is and what shape its result has -- the
/// verifier and the shape rule are not opt-in, because an operation that
/// answers neither is an operation nobody checked. One that cannot answer the
/// second says so, in a sentence, where the answer would have been.
class Ggml_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ggml_Dialect, mnemonic,
       !listconcat(traits, [DeclareOpInterfaceMethods<Ggml_ShapeRule>])> {
  let hasVerifier = 1;
  // Keep operand and result types explicit: GGML views and repeated batch
  // axes can change shapes and layouts without changing element types.
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  /// The GGML operation this one is, spelled as `ggml_op_name` gives it.
  /// Empty means the capture has no typed semantics for it, and a graph that
  /// uses it is refused.
  string ggmlOperation = "";
  /// The struct the capture records this operation's configuration in.
  /// Operations that share a configuration share a struct; empty means the
  /// operation is configured with nothing.
  string ggmlParameters = "";
  /// Where each configured value comes from. Empty beside a non-empty struct
  /// means the operation is not regular enough to generate, and its traversals
  /// are written out.
  list<GgmlParam> ggmlSources = [];
  /// The operand slot the result reads through, or -1 when the result owns its
  /// own bytes. This is what `ViewLikeOpInterface` and
  /// `DestinationStyleOpInterface` say about the operation, said where the
  /// emitter can reach it before the operation exists.
  int ggmlAliasedOperand = -1;
  /// Other GGML operations that are this one. Each is a spelling GGML keeps
  /// separately because it fixes part of the configuration.
  list<GgmlSpelling> ggmlAlsoSpelled = [];
  /// Arguments whose value is which optional operands GGML was given, rather
  /// than anything GGML stored. `operandSegmentSizes` is not among them: an
  /// operation that declares `AttrSizedOperandSegments` needs it, and its own
  /// argument list says what it holds.
  list<FromOperandPresence> ggmlOperandArguments = [];
}

class Ggml_BinaryTensorOp<string mnemonic, list<Trait> traits = []> :
    Ggml_Op<mnemonic, !listconcat(traits, [Pure])> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);

  // GGML repeats the second operand into the first, which says something
  // about the first's extents where the two were captured equal.
  let extraClassDeclaration = [{
    void constrainOperands(::mlir::ggml::ShapeConstraints &constraints);
  }];
}

// A GGML view does not copy. It hands back the same bytes read through
// different extents, strides, or a different starting offset, which is what
// makes aliasing in a captured graph something the operations themselves say
// rather than something a side table has to record.
class Ggml_ViewLikeOp<string mnemonic, list<Trait> traits = []> :
    Ggml_Op<mnemonic, !listconcat(traits, [
      Pure, DeclareOpInterfaceMethods<ViewLikeOpInterface>])> {
  let extraClassDefinition = [{
    ::mlir::Value $cppClass::getViewSource() { return getInput(); }
  }];
}

def Ggml_GetRowsOp : Ggml_Op<"get_rows", [Pure]> {
  let summary = "Gather logical rows with GGML batch semantics";
  let arguments = (ins AnyRankedTensor:$data, AnyRankedTensor:$indices);
  let results = (outs AnyRankedTensor:$result);

  // Every other operation is content with the interface's silence; this one
  // asserts something about its operands that GGML asserts too, and the
  // declaration is what lets it say so.
  let extraClassDeclaration = [{
    void constrainOperands(::mlir::ggml::ShapeConstraints &constraints);
  }];

  let ggmlOperation = "GET_ROWS";
}

def Ggml_RmsNormOp : Ggml_Op<"rms_norm", [Pure]> {
  let summary = "Apply unscaled RMS normalization along GGML dimension zero";
  let arguments = (ins AnyRankedTensor:$input, F32Attr:$epsilon);
  let results = (outs AnyRankedTensor:$result);

  let ggmlOperation = "RMS_NORM";
  let ggmlParameters = "NormParameters";
  let ggmlSources = [
    FromF32<"epsilon", 0>
  ];
}

def Ggml_MulMatOp : Ggml_Op<"mul_mat", [Pure]> {
  let summary = "Apply GGML matrix multiplication with repeated batch axes";
  let arguments = (ins
    AnyRankedTensor:$weight,
    AnyRankedTensor:$input,
    Ggml_ComputePrecisionAttr:$precision,
    I64Attr:$hint
  );
  let results = (outs AnyRankedTensor:$result);

  // GGML repeats the weight across the input's batch axes, which says
  // something about the input's where the two were captured equal.
  let extraClassDeclaration = [{
    void constrainOperands(::mlir::ggml::ShapeConstraints &constraints);
  }];

  let ggmlOperation = "MUL_MAT";
  let ggmlParameters = "MatmulParameters";
  let ggmlSources = [
    FromI32Enum<"precision", 0, "compute_precision_name">,
    FromI32<"hint", 1>
  ];
}

def Ggml_AddOp : Ggml_BinaryTensorOp<"add"> {
  let summary = "Add tensors using GGML repeat semantics";

  let ggmlOperation = "ADD";
}
def Ggml_MulOp : Ggml_BinaryTensorOp<"mul"> {
  let summary = "Multiply tensors using GGML repeat semantics";

  let ggmlOperation = "MUL";
}
def Ggml_ReshapeOp : Ggml_ViewLikeOp<"reshape", []> {
  let summary = "Reinterpret a tensor with the same logical element count";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);

  let ggmlOperation = "RESHAPE";
  let ggmlAliasedOperand = 0;
}

def Ggml_ViewOp : Ggml_ViewLikeOp<"view", []> {
  let summary = "Read an existing buffer through different extents or strides";
  let description = [{
    The strides this walks and the byte it starts at are the result's
    `#ggml.layout`: they are what the result *is*, and for a view those are the
    same thing. GGML moves no bytes here.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    OptionalAttr<AnyAttr>:$byte_offset,
    OptionalAttr<ArrayAttr>:$byte_strides
  );
  let results = (outs AnyRankedTensor:$result);

  let ggmlOperation = "VIEW";
  let ggmlAliasedOperand = 0;
  let ggmlParameters = "ViewParameters";
  let ggmlSources = [
    FromTensor<"byte_offset", "int64_t", "viewOffs", "byteOffset">
  ];
}

def Ggml_PermuteOp : Ggml_ViewLikeOp<"permute", []> {
  let summary = "Permute four raw GGML dimensions";
  let description = [{
    Applies the source GGML axis permutation to a rank-4 tensor. The verifier
    checks both that `axes` is a permutation and that the result type has the
    corresponding permuted shape and unchanged element type.
  }];
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$axes);
  let results = (outs AnyRankedTensor:$result);

  let ggmlOperation = "PERMUTE";
  let ggmlAliasedOperand = 0;
  let ggmlParameters = "PermuteParameters";
  let ggmlSources = [
    FromI32Run<"axes", 0, 4>
  ];
  // GGML spells the swap of the first two axes separately, and stores no axes
  // for it because the name is the axes.
  let ggmlAlsoSpelled = [
    GgmlSpelling<"TRANSPOSE", [Fixed<"axes", "{1, 0, 2, 3}">]>
  ];
}

def Ggml_RopeOp : Ggml_Op<"rope", [Pure]> {
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$positions,
    Optional<AnyRankedTensor>:$frequency_factors,
    DefaultValuedAttr<I64Attr, "0">:$n_past,
    I64Attr:$dimensions,
    Ggml_RopeModeAttr:$mode,
    DefaultValuedAttr<I64Attr, "0">:$legacy_context,
    I64Attr:$original_context,
    F32Attr:$frequency_base,
    F32Attr:$frequency_scale,
    F32Attr:$extension_factor,
    F32Attr:$attention_factor,
    F32Attr:$beta_fast,
    F32Attr:$beta_slow,
    OptionalAttr<I64ArrayAttr>:$sections,
    DefaultValuedAttr<I64Attr, "0">:$n_offs
  );
  let results = (outs AnyRankedTensor:$result);
  // What GGML asserts of its operands, said where the two were captured
  // equal.
  let extraClassDeclaration = [{
    void constrainOperands(::mlir::ggml::ShapeConstraints &constraints);
  }];

  let ggmlOperation = "ROPE";
  let ggmlParameters = "RopeParameters";
  let ggmlSources = [
    FromI32<"n_past", 0, "pastTokenCount">,
    FromI32<"dimensions", 1>,
    FromI32Enum<"mode", 2, "rope_mode_name">,
    FromI32<"legacy_context", 3>,
    FromI32<"original_context", 4>,
    FromF32<"frequency_base", 5>,
    FromF32<"frequency_scale", 6>,
    FromF32<"extension_factor", 7>,
    FromF32<"attention_factor", 8>,
    FromF32<"beta_fast", 9>,
    FromF32<"beta_slow", 10>,
    FromI32Run<"sections", 11, 4>,
    FromI32<"n_offs", 15, "offsetCount">
  ];
}

def Ggml_SetRowsOp : Ggml_Op<"set_rows", [Pure, DestinationStyleOpInterface]> {
  let summary = "Functionally updates selected rows of a destination tensor";
  let description = [{
    GGML writes the selected rows into the destination's own buffer. The import
    turns that into a value, but the destination is still where the result
    lives, which is what makes this a destination-style operation and what
    lets the state donation proof ask whether the buffer may be reused.
  }];
  let arguments = (ins
    AnyRankedTensor:$updates,
    AnyRankedTensor:$indices,
    AnyRankedTensor:$destination
  );
  let results = (outs AnyRankedTensor:$result);
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDestinationMutable();
    }
    // What GGML asserts of its operands, said where the two were captured
    // equal.
    void constrainOperands(::mlir::ggml::ShapeConstraints &constraints);
  }];

  let ggmlOperation = "SET_ROWS";
  let ggmlAliasedOperand = 2;
}

def Ggml_FlashAttentionExtOp : Ggml_Op<"flash_attention_ext", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    Optional<AnyRankedTensor>:$mask,
    Optional<AnyRankedTensor>:$sinks,
    F32Attr:$scale,
    F32Attr:$max_bias,
    F32Attr:$logit_softcap,
    Ggml_ComputePrecisionAttr:$precision
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $query `,` $key `,` $value (`mask` $mask^)? (`sinks` $sinks^)? attr-dict
    `:` functional-type(operands, results)
  }];
  // What GGML asserts of its operands, said where the two were captured
  // equal.
  let extraClassDeclaration = [{
    void constrainOperands(::mlir::ggml::ShapeConstraints &constraints);
  }];

  let ggmlOperation = "FLASH_ATTN_EXT";
  let ggmlParameters = "FlashAttentionParameters";
  let ggmlSources = [
    FromF32<"scale", 0>,
    FromF32<"max_bias", 1, "maxBias">,
    FromF32<"logit_softcap", 2, "logitSoftcap">,
    FromI32Enum<"precision", 3, "compute_precision_name">
  ];
}

def Ggml_GluOp : Ggml_Op<"glu", [Pure]> {
  let arguments = (ins
    AnyRankedTensor:$gate,
    Optional<AnyRankedTensor>:$up,
    Ggml_GluKindAttr:$glu_kind,
    DefaultValuedAttr<BoolAttr, "true">:$split,
    DefaultValuedAttr<BoolAttr, "false">:$swapped,
    DefaultValuedAttr<F32Attr, "0.0">:$alpha,
    DefaultValuedAttr<F32Attr, "0.0">:$limit
  );
  let results = (outs AnyRankedTensor:$result);
  // What GGML asserts of its operands, said where the two were captured
  // equal.
  let extraClassDeclaration = [{
    void constrainOperands(::mlir::ggml::ShapeConstraints &constraints);
  }];

  let ggmlOperation = "GLU";
  let ggmlParameters = "GluParameters";
  // A GLU splits one operand in half unless it was given two.
  let ggmlOperandArguments = [FromOperandPresence<"split", 1>];
  let ggmlSources = [
    FromI32Enum<"glu_kind", 0, "glu_kind_name", "kind">,
    FromI32Flag<"swapped", 1>,
    FromF32<"alpha", 2>,
    FromF32<"limit", 3>
  ];
}

#endif // HIP_DIALECT_GGML_IR_GGMLOPS_TD
