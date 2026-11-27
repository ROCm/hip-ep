//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIP_DIALECT_GGML_IR_GGMLDIALECT_TD
#define HIP_DIALECT_GGML_IR_GGMLDIALECT_TD

include "mlir/IR/DialectBase.td"

def Ggml_Dialect : Dialect {
  let name = "ggml";
  let cppNamespace = "::mlir::ggml";
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
  let summary = "Target-independent semantics imported from a GGML compute graph";
  let description = [{
    The GGML dialect is the stable compiler boundary between llama.cpp graph
    construction and target lowering. Operations preserve GGML operand order,
    typed operation parameters, logical tensor shapes, and explicit state
    updates. Physical storage, aliases, and strides are represented by
    operation and argument attributes.

    The dialect contains no model-specific operations. Architecture knowledge
    remains in llama.cpp; reusable operation semantics lower independently.
  }];
}

#endif // HIP_DIALECT_GGML_IR_GGMLDIALECT_TD
