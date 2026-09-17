#!r6rs

(library (test pattern-macro-test-simple)
  (export run-tests)
  (import (rnrs (6))
          (test test-framework)
          (prefix (test mock-ffi) ffi:)
          (mlir pattern-macro))
  
  (define mlir-operation-name ffi:mlir-operation-name)
  (define mlir-operation-get-operand (lambda (op idx) idx))
  (define mlir-operation-get-result (lambda (op idx) 999))
  (define mlir-value-get-type (lambda (val) 'mock-type))
  (define mlir-operation-get-attr (lambda (op name) 'mock-attr))
  
  (define-conversion-pattern cast-pattern
    :match 
    (%r = "onnx.Cast" (%ctx %input) ((to = !to_type)) : (!input-type) -> !output-type)
    :rewrite 
    (begin
      (%0 = (list 'placeholder %ctx %input !input-type !output-type))
      (%1 = (list 'cast %ctx %input %0 !to_type))
      %1))
  
  (define (run-tests)
    (test-begin "define-conversion-pattern-simple")
    
    (test-assert "cast-pattern is a procedure"
      (procedure? cast-pattern))
    
    (test-assert "cast-pattern matches onnx.Cast"
      (let ([result (cast-pattern 2 'operands-ref 'rewriter 'type-converter)])
        (and (list? result)
             (eq? (car result) 'cast))))
    
    (test-end)))

