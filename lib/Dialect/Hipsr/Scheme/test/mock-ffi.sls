#!r6rs

(library (test mock-ffi)
  (export mlir-operation-name
          mlir-operation-num-operands
          mlir-operation-num-results
          mlir-tensor-attach-address-space)
  (import (rnrs (6)))
  
  (define (mlir-operation-name op)
    (cond
      [(= op 1) "test.op"]
      [(= op 2) "onnx.Cast"]
      [else "unknown.op"]))
  
  (define (mlir-operation-num-operands op) 2)
  (define (mlir-operation-num-results op) 1)
  (define (mlir-tensor-attach-address-space type)
    (string-append (symbol->string type) "+device")))

