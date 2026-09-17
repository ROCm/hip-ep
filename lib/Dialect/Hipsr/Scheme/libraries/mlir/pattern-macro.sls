#!r6rs

(library (mlir pattern-macro)
  (export define-conversion-pattern)
  (import (rnrs (6)))

  (define-syntax define-conversion-pattern
    (lambda (stx)
      (syntax-case stx (:match :rewrite = : ->)
        
        ;; Pattern with operation syntax
        ;;   attr-spec can be: (name = value) or just value
        ;; We'll match two specific patterns for now
        [(_ pattern-name
            :match (%result = op-name-str (%ctx %input) ((attr-name = !attr-val)) : (!type1) -> !output-type)
            :rewrite 
            (begin
              (%v0 = op0-name (operands0 ...) ((a0-name = a0-val)) rest0 ...)
              (%v1 = op1-name (operands1 ...) ((a1-name = a1-val)) rest1 ...)
              final-expr))
         
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
                       (let ([%v0 (mlir-create-operation op0-name 
                                                         (list operands0 ...)
                                                         (list (cons 'a0-name a0-val)))])
                         (let ([%v1 (mlir-create-operation op1-name 
                                                           (list operands1 ...)
                                                           (list (cons 'a1-name a1-val)))])
                           final-expr)))))))
        ]
        
        [(_ . rest)
         (syntax-violation 'define-conversion-pattern
           "Invalid pattern syntax" stx)])))

) ;; end library
