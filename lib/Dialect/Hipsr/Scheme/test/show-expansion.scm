#!r6rs

(import (rnrs (6))
        (prefix (test mock-ffi) ffi:)
        (mlir pattern-macro))

;; Mock FFI functions
(define mlir-operation-name ffi:mlir-operation-name)
(define mlir-operation-get-operand (lambda (op idx) idx))
(define mlir-operation-get-result (lambda (op idx) 999))
(define mlir-value-get-type (lambda (val) 'f32))
(define mlir-operation-get-attr (lambda (op name) 'f16))
(define mlir-create-operation (lambda (name operands attrs) (list 'op name operands attrs)))
(define mlir-tensor-attach-address-space ffi:mlir-tensor-attach-address-space)

;; The pattern we want to expand
(define-conversion-pattern cast-pattern
  :match 
  (%r = "onnx.Cast" (%ctx %input) ((to = !to_type)) : (!input-type) -> !output-type)
  :rewrite 
  (begin
    (%0 = "hipsr.placeholder" (%ctx %input)
          ((placeholder_type = "#hipsr.placeholder_type<normal>")) 
          : types -> result)
    (%1 = "hipsr.cast" (%ctx %input %0) 
          ((result_type = (mlir-tensor-attach-address-space !output-type)))
          : types -> result)
    %1))

;; Test it works
(display "Pattern procedure: ")
(display (procedure? cast-pattern))
(newline)

;; Call it
(display "Result: ")
(display (cast-pattern 2 'operands-ref 'rewriter 'type-converter))
(newline)
