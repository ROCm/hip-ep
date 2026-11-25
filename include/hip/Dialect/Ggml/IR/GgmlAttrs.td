//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT_GGML_IR_GGMLATTRS_TD
#define HIP_DIALECT_GGML_IR_GGMLATTRS_TD

include "hip/Dialect/Ggml/IR/GgmlDialect.td"
include "hip/Dialect/Ggml/IR/GgmlEnums.td"
include "hip/Dialect/Ggml/IR/GgmlShapeExpr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/TensorEncoding.td"

def Ggml_ComputePrecisionAttr :
    EnumAttr<Ggml_Dialect, Ggml_ComputePrecision, "compute_precision"> {
  let assemblyFormat = "`<` $value `>`";
}

def Ggml_RopeModeAttr :
    EnumAttr<Ggml_Dialect, Ggml_RopeMode, "rope_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def Ggml_GluKindAttr :
    EnumAttr<Ggml_Dialect, Ggml_GluKind, "glu_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Ggml_AffineTerm : AttrDef<Ggml_Dialect, "AffineTerm"> {
  let mnemonic = "affine_term";
  let summary = "One symbol scaled by a nonzero integer coefficient";
  let parameters = (ins
    StringRefParameter<"symbol name">:$symbol,
    "int64_t":$coefficient
  );
  let assemblyFormat = "`<` $coefficient `*` $symbol `>`";
  let genVerifyDecl = 1;
}

def Ggml_AffineAttr : AttrDef<Ggml_Dialect, "Affine", [
    DeclareAttrInterfaceMethods<Ggml_ShapeExprAttrInterface>
  ]> {
  let mnemonic = "affine";
  let summary = "An integer offset plus a sum of scaled symbols";
  let description = [{
    Prints as `#ggml.affine<16384*token_count>`, or with several terms and a
    nonzero offset as `#ggml.affine<3*token_count + 5*output_count + 7>`.

    The form is canonical: terms carry nonzero coefficients, appear once per
    symbol, and are sorted by name, so two attributes are equal exactly when
    they denote the same function. An expression with no terms is rejected --
    a constant belongs in an `IntegerAttr`, which keeps one spelling per value.
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::ggml::AffineTermAttr">:$terms,
    "int64_t":$offset
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the coefficient on `symbol`, or zero when absent.
    ///
    /// Build one of these through `makeShapeExpr`, which sums repeated symbols
    /// and returns an integer attribute when every term cancels, or null if
    /// coefficient addition overflows.
    int64_t getCoefficient(::llvm::StringRef symbol) const;
  }];
}

def Ggml_LayoutAttr : AttrDef<Ggml_Dialect, "Layout", [
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>
  ]> {
  let mnemonic = "layout";
  let summary = "Where a GGML tensor's elements sit in its buffer";
  let description = [{
    The encoding of a `tensor` in the `ggml` dialect. It carries layout
    information beyond the logical shape and element type:

      - `ne`, the four extents **innermost first**, as GGML orders them. Where
        the shape has an anonymous `?`, this says which symbol it is;
      - `nb`, the four byte strides in the same order, which say whether the
        elements are dense, permuted, or a strided window on something larger;
      - `offset`, where these bytes begin inside the buffer they belong to;
      - `storage`, the ggml type those bytes are in, when it is not the one
        the element type implies. A quantized value is logically `f32` and
        physically a run of blocks -- q8_0 puts thirty-two elements in
        thirty-four bytes -- and the element type can only say the first. Absent
        means the two agree, which is every dense value and every value in a
        graph that quantizes nothing.

    A `ggml.parameter` leaves it absent even when it is packed: it already says
    what the bytes it reads are, on the operation, as a fact about the file it
    reads them from.

    ```mlir
    tensor<1x1x?x4096xf32, #ggml.layout<
      ne = [4096, #ggml.affine<1*token_count>, 1, 1],
      nb = [4, 16384, #ggml.affine<16384*token_count>,
            #ggml.affine<16384*token_count>],
      offset = 0>>
    ```

    Being the encoding rather than an attribute beside the operation has two
    consequences the dialect relies on. A type-preserving rewrite retains it;
    a conversion that removes it must account for its semantics. Type equality
    includes extents, strides, offsets and layout storage encodings, so a
    verifier that compares types also compares layouts. It does not establish
    equal tensor contents or shared storage; those require separate proofs.

    `ne` is the reverse of the shape, and that is checked rather than assumed:
    an extent that names a symbol must meet a `?`, and a constant extent must
    meet that same constant.
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::Attribute", "extents, innermost first">:$extents,
    ArrayRefParameter<"::mlir::Attribute", "byte strides, same order">:$strides,
    "::mlir::Attribute":$offset,
    "::mlir::StringAttr":$storage
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// The extent along one GGML dimension, innermost first.
    ::mlir::Attribute getExtent(unsigned dimension) const {
      return getExtents()[dimension];
    }
    /// The byte stride along one GGML dimension, innermost first.
    ::mlir::Attribute getStride(unsigned dimension) const {
      return getStrides()[dimension];
    }
    /// The same layout without a packed encoding, for a pass that rebuilds the
    /// extents and has no opinion about the bytes.
    static LayoutAttr get(::mlir::MLIRContext *context,
                          ::llvm::ArrayRef<::mlir::Attribute> extents,
                          ::llvm::ArrayRef<::mlir::Attribute> strides,
                          ::mlir::Attribute offset) {
      return get(context, extents, strides, offset, ::mlir::StringAttr());
    }
  }];
}

#endif // HIP_DIALECT_GGML_IR_GGMLATTRS_TD
