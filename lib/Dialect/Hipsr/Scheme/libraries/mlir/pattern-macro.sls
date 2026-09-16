#!r6rs
;;===----------------------------------------------------------------------===;;
;; Pattern DSL Macro - define-conversion-pattern
;;
;; Architecture: Multi-pass compiler
;;   Pass 1: Parse DSL syntax → AST (pattern-info records)
;;   Pass 2: Validate and analyze AST
;;   Pass 3: Generate Scheme code from AST
;;===----------------------------------------------------------------------===;;

(library (mlir pattern-macro)
  (export define-conversion-pattern)
  (import (rnrs (6))
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; AST Definition
  ;;===--------------------------------------------------------------------===;;

  ;; Single operation pattern
  (define-record-type op-pattern
    (fields result-var       ; %result or (%r1 %r2) for multiple results
            op-name          ; "dialect.op"
            operands         ; (%op1 %op2 ...)
            input-types      ; (!t1 !t2 ...)
            output-type))    ; !out or (!out1 !out2)

  ;; Complete pattern AST
  (define-record-type pattern-ast
    (fields name             ; pattern-name
            operations       ; list of op-pattern
            root-var         ; %root
            body))           ; (body ...)

  ;;===--------------------------------------------------------------------===;;
  ;; Pass 1: Parse DSL Syntax → AST
  ;;===--------------------------------------------------------------------===;;

  ;; Parse :if-match clause into list of op-pattern records
  (define (parse-operations stx)
    (syntax-case stx ()
      [() '()]
      ;; Single result operation
      [((%result = op-name (%operands ...) :type (!input-types ...) -> !output-type) . rest)
       (cons (make-op-pattern #'%result #'op-name
                              #'(%operands ...)
                              #'(!input-types ...)
                              #'!output-type)
             (parse-operations #'rest))]
      ;; Multiple results operation (future)
      ;; TODO: Add support for ((%r1 %r2) = "op" ...)
      [else
       (syntax-violation 'parse-operations
         "Invalid operation pattern" stx)]))

  ;;===--------------------------------------------------------------------===;;
  ;; Pass 2: Analyze AST
  ;;===--------------------------------------------------------------------===;;

  ;; Extract all capture variables from AST
  (define (extract-all-vars ast)
    ;; TODO: Walk AST and collect %vars, !types, $attrs
    '())

  ;;===--------------------------------------------------------------------===;;
  ;; Pass 3: Code Generation
  ;;===--------------------------------------------------------------------===;;

  ;; Generate pattern function from AST
  (define (codegen-pattern ast)
    (let ([ops (pattern-ast-operations ast)]
          [root (pattern-ast-root-var ast)]
          [body (pattern-ast-body ast)])

      ;; For now: single operation only
      (if (not (= (length ops) 1))
          (error 'codegen-pattern "Multi-op patterns not yet implemented")

          (let ([op-info (car ops)])
            (codegen-single-op-pattern
              (pattern-ast-name ast)
              op-info
              root
              body)))))

  ;; Generate code for single operation pattern
  (define (codegen-single-op-pattern name-stx op-info root-stx body-stx)
    (let ([result-var (op-pattern-result-var op-info)]
          [op-name (op-pattern-op-name op-info)]
          [operands (op-pattern-operands op-info)]
          [input-types (op-pattern-input-types op-info)]
          [output-type (op-pattern-output-type op-info)])

      (with-syntax ([pattern-name name-stx]
                    [%result result-var]
                    [op-name-str op-name]
                    [(%operands ...) operands]
                    [(!input-types ...) input-types]
                    [!output-type output-type]
                    [%root root-stx]
                    [(body-form ...) body-stx]
                    [num-operands (length (syntax->datum operands))])

        #'(define (pattern-name ctx rewriter op operands)

            ;; Inner: MATCH
            (define (match op operands)
              (and (string=? (mlir-operation-name op) op-name-str)
                   (= (mlir-operation-num-operands op) num-operands)
                   (= (mlir-operation-num-results op) 1)))

            ;; Inner: REWRITE
            (define (rewrite ctx rewriter op operands)
              ;; Extract SSA values
              (let* ([%result (mlir-operation-get-result-value op 0)]
                     . (codegen-operand-bindings #'(%operands ...) 0))
                ;; Extract types
                (let ([!output-type (mlir-value-get-type %result)]
                      . (codegen-type-bindings #'(!input-types ...) #'(%operands ...)))
                  ;; Execute user body
                  body-form ...)))

            ;; Main
            (if (match op operands)
                (begin (rewrite ctx rewriter op operands) #t)
                #f)))))

  ;; Helper: Generate operand bindings
  (define-syntax codegen-operand-bindings
    (lambda (x)
      (syntax-case x ()
        [(_ () idx) #'()]
        [(_ (var) idx)
         #'((var (mlir-operation-get-operand-value op idx)))]
        [(_ (var rest ...) idx)
         #'((var (mlir-operation-get-operand-value op idx))
            . (codegen-operand-bindings (rest ...) (+ idx 1)))])))

  ;; Helper: Generate type bindings
  (define-syntax codegen-type-bindings
    (lambda (x)
      (syntax-case x ()
        [(_ () ()) #'()]
        [(_ (tvar) (ovar))
         #'((tvar (mlir-value-get-type ovar)))]
        [(_ (tvar trest ...) (ovar orest ...))
         #'((tvar (mlir-value-get-type ovar))
            . (codegen-type-bindings (trest ...) (orest ...)))])))

  ;;===--------------------------------------------------------------------===;;
  ;; Main Macro
  ;;===--------------------------------------------------------------------===;;

  (define-syntax define-conversion-pattern
    (lambda (stx)
      (syntax-case stx (:if-match :rewrite)

        [(_ pattern-name
            :if-match
              op-patterns ...
            :rewrite root-var
              body ...)

         ;; Pass 1: Parse to AST
         (let* ([ops (parse-operations #'(op-patterns ...))]
                [ast (make-pattern-ast #'pattern-name ops #'root-var #'(body ...))])

           ;; Pass 2: Analyze (TODO)
           ;; (analyze-ast ast)

           ;; Pass 3: CodeGen
           (codegen-pattern ast))]

        [(_ . rest)
         (syntax-violation 'define-conversion-pattern
           "Invalid syntax" stx)])))

) ;; end library
