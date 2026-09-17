#!r6rs
;;===----------------------------------------------------------------------===;;
;; Pattern DSL Macro - define-conversion-pattern
;;===----------------------------------------------------------------------===;;

(library (mlir pattern-macro)
  (export define-conversion-pattern)
  (import (rnrs (6)))

  (define-syntax define-conversion-pattern
    (lambda (stx)
      (syntax-case stx (:match :rewrite)
        [(_ pattern-name :match op-name-str :rewrite rewrite-proc)
         (with-syntax ([mlir-op-name (datum->syntax #'pattern-name 'mlir-operation-name)])
           #'(define pattern-name
               (lambda (op operands-ref rewriter type-converter)
                 (if (string=? (mlir-op-name op) op-name-str)
                     (rewrite-proc op operands-ref rewriter type-converter)
                     #f))))]
        [(_ . rest)
         (syntax-violation 'define-conversion-pattern
           "Expected: (define-conversion-pattern name :match \"op\" :rewrite proc)"
           stx)])))

) ;; end library
