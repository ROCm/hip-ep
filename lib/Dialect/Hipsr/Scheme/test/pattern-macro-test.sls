#!r6rs

(library (test pattern-macro-test)
  (export run-tests)
  (import (rnrs (6))
          (test test-framework)
          (prefix (test mock-ffi) ffi:)
          (mlir pattern-macro))
  
  (define mlir-operation-name ffi:mlir-operation-name)
  (define mlir-operation-get-operand (lambda (op idx) idx))
  (define mlir-operation-get-result (lambda (op idx) 999))
  (define mlir-value-get-type (lambda (val) 'f32))
  (define mlir-operation-get-attr (lambda (op name) 'f16))
  (define mlir-create-operation (lambda (name operands attrs) (list 'op name operands attrs)))
  (define mlir-tensor-attach-address-space ffi:mlir-tensor-attach-address-space)
  
  (define-conversion-pattern cast-pattern
    :match 
    (%r = "onnx.Cast" (%ctx %input) ((to = !to_type)) : (!input-type) -> !output-type)
    :rewrite 
    (%0 = "hipsr.placeholder" (%ctx %input)
          ((placeholder_type = "#hipsr.placeholder_type<normal>")) 
          : types -> result)
    (%1 = "hipsr.cast" (%ctx %input %0) 
          ((result_type = (mlir-tensor-attach-address-space !output-type)))
          : types -> result)
    %1)
  
  (define (run-tests)
    (test-begin "define-conversion-pattern")
    
    (test-assert "cast-pattern is a procedure"
      (procedure? cast-pattern))
    
    (test-assert "cast-pattern matches onnx.Cast"
      (let ([result (cast-pattern 2 'operands-ref 'rewriter 'type-converter)])
        (and (list? result)
             (eq? (car result) 'op)
             (equal? (cadr result) "hipsr.cast"))))
    
    (test-end)))

