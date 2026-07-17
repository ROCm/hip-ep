//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.constant op: a constant tensor buffer. Holds its data in one of
// three forms (inline / external source / externalized). Foundation only --
// the onnx->hipsr conversion, externalization, and LLVM lowering that produce
// and consume these forms live in later PRs.
//

#ifndef HIPSR_CONSTANT_OP
#define HIPSR_CONSTANT_OP

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hip/Dialect/Hipsr/IR/HipsrOps.td"
include "hip/Dialect/Hipsr/IR/HipsrAttrs.td"

//===----------------------------------------------------------------------===//
// Hipsr_ConstantOp
//===----------------------------------------------------------------------===//

def Hipsr_ConstantOp : Hipsr_Op<"constant", [Pure]> {
  let summary = "Constant tensor buffer (inline value or external source)";
  let description = [{
    Produces a constant buffer. The data source is exactly one of:

    - inline     : `value` is a `DenseElementsAttr` carrying the bytes.
    - ext source : `source` references a `#hipsr.file_source<...>` (on-disk
                   file) or `#hipsr.mem_source<...>` (a raw ORT pointer).

    Externalization only *adds* `offset` + `size` (the location in the
    compiler-produced constants sidecar); it never removes `value` / `source`,
    so `getDataValues()` keeps working afterwards. Valid states:
    `{value}`, `{source}`, `{value, offset, size}`, `{source, offset, size}`.

    `externalize` (default true) is an orthogonal opt-out flag; a downstream pass
    calls `disableExternalize()` to keep a constant inline (compile-time value).

    The result is a ranked tensor (pre-bufferization) or a device `memref`
    (`#hipsr.mem<device>`, post-bufferization); constants reside in VRAM. The
    `Hipsr_TensorOrDeviceMemRef` ODS constraint enforces this.

    Example:
    ```mlir
    %w = hipsr.constant {value = dense<1.0> : tensor<4xf16>}
       : memref<4xf16, #hipsr.mem<device>>
    ```
  }];

  let arguments = (ins
    OptionalAttr<ElementsAttr>:$value,
    OptionalAttr<AnyAttrOf<[Hipsr_FileSourceAttr, Hipsr_MemSourceAttr]>>:$source,
    OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$size,
    OptionalAttr<BoolAttr>:$externalize
  );

  let results = (outs Hipsr_TensorOrDeviceMemRef:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// True if the constant carries an inline dense value.
    bool isInline() { return getValueAttr() != nullptr; }
    /// True if the constant references an external file/memory source.
    bool hasExternalSource() { return getSourceAttr() != nullptr; }
    /// True if the constant has been externalized (offset/size into sidecar).
    bool isExternalized() { return getOffsetAttr() != nullptr; }
    /// Whether this constant should be externalized (defaults to true).
    bool shouldExternalize() {
      ::mlir::BoolAttr a = getExternalizeAttr();
      return a ? a.getValue() : true;
    }
    /// Opt out of externalization -- keep inline. One-way decision.
    void disableExternalize() {
      (*this)->setAttr("externalize",
                       ::mlir::BoolAttr::get(getContext(), false));
    }

    /// Returns a typed view of the constant data (inline dense / mem source /
    /// file source via the dialect cache). MorphiZen-only contract; works both
    /// pre- and post-externalization. See the CONTRACT comment on the
    /// definition in HipsrConstantOp.h. Defined out-of-line there.
    template <typename T>
    ::llvm::ArrayRef<T> getDataValues();
  }];
}

#endif // HIPSR_CONSTANT_OP
