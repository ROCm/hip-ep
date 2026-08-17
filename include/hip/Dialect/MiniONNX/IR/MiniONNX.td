//===- MiniONNX.td - MiniONNX Dialect Definition -----------*- tablegen -*-===//
//
// Minimal ONNX dialect for hip-ep conversion.
// Operations are auto-generated from ONNX schemas.
//
//===----------------------------------------------------------------------===//

#ifndef MINI_ONNX_DIALECT
#define MINI_ONNX_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// MiniONNX Dialect Definition
//===----------------------------------------------------------------------===//

def MiniONNX_Dialect : Dialect {
  let name = "onnx";
  let summary = "Minimal ONNX dialect for hip-ep";
  let description = [{
    Lightweight ONNX dialect containing only operations needed for hip-ep
    conversion, without heavy dependencies on shape inference infrastructure.
    
    Unregistered ONNX operations are allowed and handled as generic Operation*.
  }];
  
  let cppNamespace = "::mlir::onnx";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["func::FuncDialect"];
  
  let extraClassDeclaration = [{
    void initialize();
  }];
}

//===----------------------------------------------------------------------===//
// Base Operation Class
//===----------------------------------------------------------------------===//

class MiniONNX_Op<string mnemonic, list<Trait> traits = []> :
  Op<MiniONNX_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Operation Definitions (auto-generated)
//===----------------------------------------------------------------------===//

include "hip/Dialect/MiniONNX/IR/MiniONNXOps.td.inc"

#endif // MINI_ONNX_DIALECT
