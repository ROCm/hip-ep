//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The onnx.Custom op. No ONNX schema describes it, because it stands for an
// operation from a domain outside the ONNX standard.
//

//===----------------------------------------------------------------------===//
// Onnx_CustomOp
//===----------------------------------------------------------------------===//

def Onnx_CustomOp : Onnx_Op<"Custom"> {
  let summary = "An operation from a domain outside the ONNX standard";
  let description = [{
    Carries a contrib-domain operation, with `function_name` selecting which one
    and `domain_name` naming the domain it came from. Every other attribute the
    operation needs rides along in the attribute dictionary, because the set
    differs per `function_name`.

    Not `Pure`: `function_name` selects the semantics, so the dialect cannot
    claim the operation is side-effect free.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$inputs,
    OptionalAttr<StrAttr>:$function_name,
    OptionalAttr<StrAttr>:$domain_name
  );

  let results = (outs Variadic<AnyTypeOf<[AnyTensor, NoneType]>>:$outputs);
}
