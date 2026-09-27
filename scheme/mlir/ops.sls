#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir ops) — expression-level MLIR operation builder
;;
;; Provides two forms:
;;
;;   (with-current-mlir-builder (rw loc) body ...)
;;     Installs rw (RewriterBase*) and loc (Operation* for location/IP)
;;     as dynamic context for the duration of body.
;;
;;   (with-mlir-ops op-form ...)
;;     Builds a sequence of MLIR operations using the current builder.
;;     Returns the last result (Scheme convention).
;;
;; op-form syntax:
;;   (%var = "op.name" (operands...) -> result-type)      single-result
;;   ((%a %b) = "op.name" (operands...) -> (t1 t2 ...))  multi-result
;;   (%var = scheme-expr)                                  Scheme escape
;;
;; Operands prefixed with ! are types — they are filtered from the value
;; operand list and must appear only in the -> clause as result types.
;; (The same convention as the define-conversion-pattern DSL.)
;;
;;===----------------------------------------------------------------------===;;

(library (mlir ops)
  ;; with-current-mlir-builder, with-current-block-builder and
  ;; current-mlir-build-fn are now in (mlir ffi); re-exported here.
  (export current-mlir-build-fn
          with-current-mlir-builder
          with-current-block-builder
          with-mlir-ops)

  (import (rnrs (6))
          (only (chezscheme) syntax->list syntax->datum datum->syntax)
          (rename (rime loop) (:with :rime-with))
          (for (rename (rime loop) (:with :rime-with)) expand)
          (mlir ffi))

  ;;===--------------------------------------------------------------------===;;
  ;; with-mlir-ops
  ;;===--------------------------------------------------------------------===;;

  (define-syntax with-mlir-ops
    (lambda (stx)

      ;; True when identifier starts with ! (type convention, not a value)
      (define (type-id? x)
        (let ([d (syntax->datum x)])
          (and (symbol? d)
               (let ([s (symbol->string d)])
                 (and (> (string-length s) 0)
                      (char=? #\! (string-ref s 0)))))))

      ;; Return only the value operands from a mixed operand+type list
      (define (value-operands ops-stx)
        (filter (lambda (x) (not (type-id? x)))
                (syntax->list ops-stx)))

      ;; Combine var and expr into a let* binding form #'(var expr).
      ;; Named function so with-syntax is compiled in isolation.
      (define (make-binding var expr)
        (with-syntax ([v var] [e expr]) #'(v e)))

      ;; Process one op-form. Returns a flat list of (var . expr) pairs.
      ;; idx is used to generate a unique tmp name for multi-result ops.
      ;;
      ;; Uses datum comparison for = and -> rather than free-identifier=?
      ;; so the macro works regardless of the using context's imports.
      (define (arrow? x)   (eq? (syntax->datum x) '->))
      (define (eq-sym? x)  (eq? (syntax->datum x) '=))
      (define (attrs-kw? x)(eq? (syntax->datum x) ':attrs))
      (define (index-kw? x)(eq? (syntax->datum x) ':index))

      ;; Generate attr-setter code for one attr form.
      ;; (name val)       → (mlir-operation-set-attr       new-op "name" val)
      ;; (name val :index)→ (mlir-operation-set-index-attr new-op "name" val)
      (define (make-attr-setter attr-stx)
        (let ([af (syntax->list attr-stx)])
          (let* ([name-raw (syntax->datum (list-ref af 0))]
                 [name-str (if (string? name-raw) name-raw (symbol->string name-raw))]
                 [val      (list-ref af 1)])
            (if (and (= (length af) 3) (index-kw? (list-ref af 2)))
                (with-syntax ([n name-str] [v val])
                  #'(mlir-operation-set-index-attr new-op n v))
                (with-syntax ([n name-str] [v val])
                  #'(mlir-operation-set-attr new-op n v))))))

      ;; Emit let*/get-result code, inserting attr-setters between build and get-result.
      (define (emit-single-result %var op-name vals result-type attr-setters)
        (let ([zero-result? (null? (syntax->datum result-type))])
          (if zero-result?
              (with-syntax ([var %var] [(v ...) vals] [n op-name]
                            [(setter ...) attr-setters])
                (list (cons #'var
                            #'(let ([new-op ((current-mlir-build-fn) n (list v ...) '())])
                                setter ... new-op))))
              (with-syntax ([var %var] [(v ...) vals] [n op-name] [rt result-type]
                            [(setter ...) attr-setters])
                (list (cons #'var
                            #'(let ([new-op ((current-mlir-build-fn) n (list v ...) (list rt))])
                                setter ...
                                (mlir-operation-get-result new-op 0))))))))

      (define (process-op op-stx idx)
        (let ([form (syntax->list op-stx)])
          (cond
            ;; Single-result MLIR op with :attrs:
            ;;   (%var = "op" (ops) :attrs ((name val ...) ...) -> result-type)  length=8
            [(and form
                  (= (length form) 8)
                  (eq-sym?  (list-ref form 1))
                  (string?  (syntax->datum (list-ref form 2)))
                  (attrs-kw? (list-ref form 4))
                  (arrow?   (list-ref form 6)))
             (emit-single-result
               (list-ref form 0)
               (syntax->datum (list-ref form 2))
               (value-operands (list-ref form 3))
               (list-ref form 7)
               (map make-attr-setter (syntax->list (list-ref form 5))))]

            ;; Single-result MLIR op (no attrs):
            ;;   (%var = "op.name" (operands...) -> result-type)   length=6
            [(and form
                  (= (length form) 6)
                  (eq-sym? (list-ref form 1))
                  (string? (syntax->datum (list-ref form 2)))
                  (arrow? (list-ref form 4)))
             (emit-single-result
               (list-ref form 0)
               (syntax->datum (list-ref form 2))
               (value-operands (list-ref form 3))
               (list-ref form 5)
               '())]

            ;; Multi-result MLIR op:
            ;;   ((%a %b) = "op.name" (operands...) -> (t1 t2 ...))   length=6,
            ;;   but first element is itself a list
            [(and form
                  (= (length form) 6)
                  (eq-sym? (list-ref form 1))
                  (string? (syntax->datum (list-ref form 2)))
                  (arrow? (list-ref form 4))
                  (list? (syntax->list (list-ref form 0))))
             (let* ([rvars    (syntax->list (list-ref form 0))]
                    [op-name  (syntax->datum (list-ref form 2))]
                    [vals     (value-operands (list-ref form 3))]
                    [rtypes   (syntax->list (list-ref form 5))]
                    [tmp      (datum->syntax stx
                                (string->symbol
                                  (string-append "%op-tmp-" (number->string idx))))])
               (with-syntax ([(v ...) vals] [n op-name]
                             [(rt ...) rtypes] [t tmp])
                 (cons (cons #'t
                             #'((current-mlir-build-fn) n (list v ...) (list rt ...)))
                       (let loop ([vs rvars] [i 0] [acc '()])
                         (if (null? vs)
                             (reverse acc)
                             (loop (cdr vs) (+ i 1)
                                   (cons (cons (car vs)
                                               #`(mlir-operation-get-result t #,i))
                                         acc)))))))]

            ;; Scheme escape: (%var = expr)   length=3
            [(and form
                  (= (length form) 3)
                  (eq-sym? (list-ref form 1)))
             (list (cons (list-ref form 0) (list-ref form 2)))]

            [else
             (syntax-violation 'with-mlir-ops "invalid op form" op-stx)])))

      ;; Expand: collect all bindings, emit a flat let*
      (syntax-case stx ()
        [(_ op ...)
         (let* ([ops      (syntax->list #'(op ...))]
                [pairs    (apply append
                                 (loop :for op-stx :in ops
                                       :for i :from 0
                                       :collect (process-op op-stx i)))]
                [last-var (and (pair? pairs)
                               (car (car (reverse pairs))))])
           (if (null? pairs)
               #'(if #f #f)
               (with-syntax ([(binding ...) (map make-binding
                                                  (map car pairs)
                                                  (map cdr pairs))]
                             [result last-var])
                 #'(let* (binding ...) result))))])))

) ;; end library (mlir ops)
