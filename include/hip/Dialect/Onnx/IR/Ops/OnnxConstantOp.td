//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The onnx.Constant op. A schema does describe it, but the importer also writes
// an external-data form naming a byte range that no schema attribute covers, so
// this definition is hand-written rather than following the schema.
//

//===----------------------------------------------------------------------===//
// Onnx_ConstantOp
//===----------------------------------------------------------------------===//

def Onnx_ConstantOp : Onnx_Op<"Constant"> {
  let summary = "A graph initializer or an inline constant";
  let description = [{
    Holds a constant in one of two forms. The inline form puts the data in
    `value` (or in one of the scalar/list variants the ONNX schema allows). The
    external form names a byte range instead: `location` is a file path, or the
    tag the ORT bridge uses for a host address, and `offset`/`size` delimit the
    range within it.

    `offset` and `size` are signless `i64` rather than the `si64` an ONNX
    integer attribute uses, because they are not ONNX attributes -- they
    describe where the importer put the bytes.

    Not `Pure`: in the external form the value is a byte range this operation
    does not carry, so the IR alone does not determine the result.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$location,
    OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$size,
    OptionalAttr<AnyAttr>:$sparse_value,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<F32Attr>:$value_float,
    OptionalAttr<F32ArrayAttr>:$value_floats,
    OptionalAttr<SI64Attr>:$value_int,
    OptionalAttr<I64ArrayAttr>:$value_ints,
    OptionalAttr<StrAttr>:$value_string,
    OptionalAttr<StrArrayAttr>:$value_strings
  );

  let results = (outs AnyTensor:$output);
}
