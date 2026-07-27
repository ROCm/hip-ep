//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.constant op: a constant tensor buffer. Holds its data in one of
// three forms (inline / external source / externalized). Foundation only --
// the onnx->hipsr conversion, externalization, and LLVM lowering that produce
// and consume these forms live in later PRs.
//

//===----------------------------------------------------------------------===//
// Hipsr_ConstantOp
//===----------------------------------------------------------------------===//

def Hipsr_ConstantOp : Hipsr_Op<"constant", [Pure]> {
  let summary = "Constant tensor buffer (inline value or external source)";
  let description = [{
    Produces a constant buffer.

    - `value`  : inline `DenseElementsAttr` carrying the bytes.
    - `source` : `#hipsr.file_source<...>` (on-disk file) or
                 `#hipsr.mem_source<...>` (a raw ORT pointer).
    - `offset`/`size`/`index` : offset+size = location in the constants file,
                 index = ordinal in the runtime array.

    Valid states: `{value}`, `{source}`, or either plus `{offset, size, index}`.
  }];

  let arguments = (ins
    OptionalAttr<ElementsAttr>:$value,
    OptionalAttr<AnyAttrOf<[Hipsr_FileSourceAttr, Hipsr_MemSourceAttr]>>:$source,
    OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$size,
    OptionalAttr<I64Attr>:$index
  );

  let results = (outs Hipsr_TensorOrDeviceMemRef:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// True if the constant carries an inline dense value.
    bool isInline() { return getValueAttr() != nullptr; }
    /// True if the constant references an external file/memory source.
    bool hasExternalSource() { return getSourceAttr() != nullptr; }
    /// True if the constant has been externalized (offset/size into constants file).
    bool isExternalized() { return getOffsetAttr() != nullptr; }

    /// Returns a typed view of the constant data (inline dense / mem source /
    /// file source via the dialect cache). MorphiZen-only contract; works both
    /// pre- and post-externalization. See the CONTRACT comment on the
    /// definition in HipsrConstantOp.h. Defined out-of-line there.
    template <typename T>
    ::llvm::ArrayRef<T> getDataValues();
  }];
}
