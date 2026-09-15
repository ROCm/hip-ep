#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Pattern DSL - define-conversion-pattern macro
;;
;; Implements the pattern matching DSL for OpConversionPattern.
;; See tech/design/2026-09-14-mlir-hipsr-pattern-dsl.md for syntax.
;;
;;===----------------------------------------------------------------------===;;

(library (pattern-dsl)
  (export define-conversion-pattern %unbound bind-or-test)
  (import (rnrs (6))
          (only (chezscheme) gensym format))

  ;; Sentinel value for unbound capture variables
  (define %unbound (gensym "unbound"))

  ;; Bind-or-test semantics for pattern matching
  ;; Returns #t on successful bind or match, #f on mismatch
  (define-syntax bind-or-test
    (syntax-rules ()
      [(_ var actual equal-pred)
       (cond
         [(eq? var %unbound)
          (set! var actual)
          #t]
         [else
          (equal-pred var actual)])]))

  ;; Helper: extract capture variables from match clause
  ;; Returns list of variable identifiers
  (define (extract-captures stx)
    (syntax-case stx (= :type ->)
      ;; Single result: %r = "op" ...
      [(result = op-name . rest)
       (identifier? #'result)
       (cons #'result (extract-captures-from-rest #'rest))]

      ;; Multiple results: (%r1 %r2) = "op" ...
      [((result ...) = op-name . rest)
       (append (syntax->list #'(result ...))
               (extract-captures-from-rest #'rest))]

      [_ '()]))

  ;; Helper: extract captures from operands, attributes, types
  (define (extract-captures-from-rest stx)
    (syntax-case stx (:type ->)
      ;; Operands: (%op1 %op2 ...)
      [((operand ...) . rest)
       (append (filter identifier? (syntax->list #'(operand ...)))
               (extract-captures-from-rest #'rest))]

      ;; Attributes: (:attr $val) ...
      [((:attr-name attr-val) . rest)
       (cons #'attr-val (extract-captures-from-rest #'rest))]

      ;; Type signature: :type (input-types ...) -> output-type
      [(:type (input-type ...) -> output-type . rest)
       (append (filter identifier? (syntax->list #'(input-type ...)))
               (if (identifier? #'output-type)
                   (list #'output-type)
                   '())
               (extract-captures-from-rest #'rest))]

      ;; Type signature: :type (input-types ...) -> (output-type ...)
      [(:type (input-type ...) -> (output-type ...) . rest)
       (append (filter identifier? (syntax->list #'(input-type ...)))
               (filter identifier? (syntax->list #'(output-type ...)))
               (extract-captures-from-rest #'rest))]

      [_ '()]))

  ;; define-conversion-pattern macro
  ;;
  ;; Syntax:
  ;;   (define-conversion-pattern pattern-name
  ;;     :if-match
  ;;       %result = "dialect.op" (%operand ...) (:attr-name $attr-value) ...
  ;;                 :type (type ...) -> type
  ;;       ...
  ;;     :rewrite %root
  ;;       body-expr ...)
  ;;
  ;; Expands to a function that takes (op rewriter) and returns #t/#f for match.
  ;; On successful match, executes rewrite body and replaces operation.
  (define-syntax define-conversion-pattern
    (lambda (x)
      (syntax-case x (:if-match :rewrite :type)
        [(_ pattern-name
            :if-match match-clause ...
            :rewrite root-binding
            rewrite-body ...)

         ;; For now: simplified implementation for single operation patterns
         ;; TODO: Handle multi-operation patterns (def-use chain walking)

         (syntax-case #'(match-clause ...) (= :type ->)
           ;; Single operation pattern
           [(single-clause)
            (syntax-case #'single-clause (= :type ->)
              ;; Parse: %result = "op-name" (%operands ...) (:attrs ...) :type ...
              [(result = op-name (operand ...) attr-specs ... :type type-spec ...)

               #'(define (pattern-name op rewriter)
                   ;; Initialize all capture variables to %unbound
                   (let ([result %unbound]
                         [operand %unbound] ...
                         ;; TODO: Extract attribute and type variables
                         )

                     ;; Match operation name
                     (and (string=? (mlir-operation-name op) op-name)

                          ;; TODO: Match operand count
                          ;; TODO: Bind operands with bind-or-test
                          ;; TODO: Match/bind attributes
                          ;; TODO: Match/bind types

                          ;; Execute rewrite body on successful match
                          (begin
                            (set! result op) ;; Bind result to matched operation
                            rewrite-body ...
                            #t))))])]

           ;; Multi-operation pattern
           [_
            #'(define (pattern-name op rewriter)
                (error 'pattern-name
                       "Multi-operation patterns not yet implemented"))])])))

) ;; end library (pattern-dsl)
