//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Hipsr_ConstantOp
//===----------------------------------------------------------------------===//

def Hipsr_ConstantOp : Hipsr_Op<"constant", [ConstantLike, Pure]> {
  let summary = "Constant tensor buffer";
  let description = [{
    Produces a constant buffer.

    - `value` : inline `DenseElementsAttr` carrying the bytes, or
                `DenseResourceElementsAttr` naming a blob keyed
                `file|<path>|<offset>` or `mem|<address-hex>`.
    - `offset`/`size`/`index` : offset+size = location in the constants file,
                index = ordinal in the runtime array.
  }];

  let arguments = (ins
    ElementsAttr:$value,
    OptionalAttr<I64Attr>:$index,   // Constant index (after externalization)
    OptionalAttr<I64Attr>:$offset,  // Offset in constants.bin (after externalization)
    OptionalAttr<I64Attr>:$size     // Size in bytes (after externalization)
  );

  let results = (outs Hipsr_DeviceTensorOrMemRef:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// True if the constant has been externalized (offset/size into constants file).
    bool isExternalized() { return getOffsetAttr() != nullptr; }
  }];
}
