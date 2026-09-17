#!r6rs
;;===----------------------------------------------------------------------===;;
;; Mock FFI for testing - provides stub implementations
;;===----------------------------------------------------------------------===;;

(library (test mock-ffi)
  (export mlir-operation-name
          mlir-operation-num-operands
          mlir-operation-num-results)

  (import (rnrs (6)))

  ;; Mock mlir-operation-name: returns a test op name based on op pointer
  (define (mlir-operation-name op)
    (cond
      [(= op 1) "test.op"]
      [(= op 2) "onnx.Cast"]
      [else "unknown.op"]))

  ;; Mock mlir-operation-num-operands
  (define (mlir-operation-num-operands op)
    1)

  ;; Mock mlir-operation-num-results
  (define (mlir-operation-num-results op)
    1)

) ;; end library
