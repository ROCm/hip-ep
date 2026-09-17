#!r6rs

(library (mlir pattern-macro)
  (export define-conversion-pattern)
  (import (rnrs (6)))

  (define-syntax define-conversion-pattern
    (lambda (stx)
      (syntax-case stx (:match :rewrite :with = : ->)
        
        ;; Pattern: single operand, compute %ctx, then two operations
        [(_ pattern-name
            :match (%result = op-name-str (%operand1) ((attr-name = !attr-val)) : (!type1) -> !output-type)
            :rewrite %root :with
            (%v0 = expr0)
            (%v1 = "hipsr.placeholder" (operands1 ...) ((attrs1 ...)) rest1 ...)
            (%v2 = "hipsr.cast" (operands2 ...) ((attrs2 ...)) rest2 ...))
         
         (with-syntax ([mlir-op-name (datum->syntax #'pattern-name 'mlir-operation-name)]
                       [mlir-get-result (datum->syntax #'pattern-name 'mlir-operation-get-result)]
                       [mlir-get-type (datum->syntax #'pattern-name 'mlir-value-get-type)]
                       [mlir-get-attr (datum->syntax #'pattern-name 'mlir-operation-get-attr)]
                       [mlir-create-placeholder-op (datum->syntax #'pattern-name 'mlir-create-placeholder-op)]
                       [mlir-create-cast-op (datum->syntax #'pattern-name 'mlir-create-cast-op)]
                       [value-array-ref-at (datum->syntax #'pattern-name 'value-array-ref-at)])
           
           #'(define pattern-name
               (lambda (op operands-ref rewriter type-converter)
                 (if (not (string=? (mlir-op-name op) op-name-str))
                     #f
                     (let* ([%result (mlir-get-result op 0)]
                            [%operand1 (value-array-ref-at operands-ref 0)]
                            [!type1 (mlir-get-type %operand1)]
                            [!output-type (mlir-get-type %result)]
                            [!attr-val (mlir-get-attr op (symbol->string 'attr-name))])
                       (let ([%root %result])
                         (let ([%v0 expr0])
                           ;; "hipsr.placeholder" -> mlir-create-placeholder-op
                           ;; Signature: (ctx input output-type placeholder-type-int)
                           (let ([%v1 (mlir-create-placeholder-op operands1 ... 0)])
                             ;; "hipsr.cast" -> mlir-create-cast-op
                             ;; Signature: (ctx input placeholder output-type)
                             (let ([%v2 (mlir-create-cast-op operands2 ...)])
                               %v2)))))))))]
        
        ;; Pattern with two operands, two operations (for tests)
        [(_ pattern-name
            :match (%result = op-name-str (%ctx %input) ((attr-name = !attr-val)) : (!type1) -> !output-type)
            :rewrite %root :with
            (%v0 = op0-name (operands0 ...) ((a0-name = a0-val)) rest0 ...)
            (%v1 = op1-name (operands1 ...) ((a1-name = a1-val)) rest1 ...))
         
         (with-syntax ([mlir-op-name (datum->syntax #'pattern-name 'mlir-operation-name)]
                       [mlir-get-operand (datum->syntax #'pattern-name 'mlir-operation-get-operand)]
                       [mlir-get-result (datum->syntax #'pattern-name 'mlir-operation-get-result)]
                       [mlir-get-type (datum->syntax #'pattern-name 'mlir-value-get-type)]
                       [mlir-get-attr (datum->syntax #'pattern-name 'mlir-operation-get-attr)]
                       [mlir-create-operation (datum->syntax #'pattern-name 'mlir-create-operation)])
           
           #'(define pattern-name
               (lambda (op operands-ref rewriter type-converter)
                 (if (not (string=? (mlir-op-name op) op-name-str))
                     #f
                     (let* ([%result (mlir-get-result op 0)]
                            [%ctx (mlir-get-operand op 0)]
                            [%input (mlir-get-operand op 1)]
                            [!type1 (mlir-get-type %input)]
                            [!output-type (mlir-get-type %result)]
                            [!attr-val (mlir-get-attr op (symbol->string 'attr-name))])
                       (let ([%root %result])
                         (let ([%v0 (mlir-create-operation op0-name 
                                                           (list operands0 ...)
                                                           (list (cons 'a0-name a0-val)))])
                           (let ([%v1 (mlir-create-operation op1-name 
                                                             (list operands1 ...)
                                                             (list (cons 'a1-name a1-val)))])
                             %v1))))))))]
        
        [(_ . rest)
         (syntax-violation 'define-conversion-pattern
           "Invalid pattern syntax" stx)])))

) ;; end library
