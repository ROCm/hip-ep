#!r6rs
;;===----------------------------------------------------------------------===;;
;; Pattern DSL Macro - define-conversion-pattern
;;
;; Generates pattern matching code from declarative DSL syntax
;;===----------------------------------------------------------------------===;;

(library (mlir pattern-macro)
  (export define-conversion-pattern)
  (import (rnrs (6)))

  ;;===--------------------------------------------------------------------===;;
  ;; Main Macro
  ;;===--------------------------------------------------------------------===;;

  ;; Syntax: (define-conversion-pattern pattern-name "op-name"
  ;;           :rewrite (lambda (op operands-ref rewriter type-converter) ...))
  ;;
  ;; Generates a function with signature:
  ;;   (lambda (op operands-ref rewriter type-converter) ...)
  ;;
  (define-syntax define-conversion-pattern
    (lambda (stx)
      (syntax-case stx (:match :rewrite)

        ;; Simple form: just match op-name and execute rewrite
        [(_ pattern-name :match op-name-str :rewrite rewrite-proc)
         #'(define pattern-name
             (lambda (op operands-ref rewriter type-converter)
               ;; Note: mlir-operation-name is from (mlir ffi)
               ;; The generated code assumes it's available at runtime
               (if (string=? (mlir-operation-name op) op-name-str)
                   (rewrite-proc op operands-ref rewriter type-converter)
                   #f)))]

        [(_ . rest)
         (syntax-violation 'define-conversion-pattern
           "Expected: (define-conversion-pattern name :match \"op\" :rewrite proc)"
           stx)])))

) ;; end library
