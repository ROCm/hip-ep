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
          (only (chezscheme) syntax->list syntax->datum datum->syntax parameterize)
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

      ;; Accept both 'hipsr.placeholder and "hipsr.placeholder" as op names
      (define (op-name? x)
        (let ([d (syntax->datum x)])
          (or (string? d) (symbol? d))))
      (define (op-name->str x)
        (let ([d (syntax->datum x)])
          (if (string? d) d (symbol->string d))))

      ;; Generate attr-setter code for one attr form.
      ;; (name val)           → mlir-operation-set-attr
      ;; (name val :index)    → mlir-operation-set-index-attr
      ;; (name val :i32-array)→ mlir-operation-set-dense-i32-array
      (define (make-attr-setter attr-stx)
        (let* ([af      (syntax->list attr-stx)]
               [len     (length af)]
               [name-d  (syntax->datum (list-ref af 0))]
               [name-s  (if (string? name-d) name-d (symbol->string name-d))]
               [val     (list-ref af 1)]
               [qual    (and (= len 3) (syntax->datum (list-ref af 2)))])
          (with-syntax ([n name-s] [v val])
            (cond
              [(eq? qual ':index)
               #'(mlir-operation-set-index-attr new-op n v)]
              [(eq? qual ':i32-array)
               #'(mlir-operation-set-dense-i32-array new-op n v)]
              [else
               #'(mlir-operation-set-attr new-op n v)]))))

      ;;-------------------------------------------------------------------
      ;; Region helpers
      ;;-------------------------------------------------------------------

      ;; Parse one block form: (^bb0 ((arg : !type) ...) body-op ...)
      ;; Returns (arg-vars arg-types body-ops) as three lists of syntax.
      (define (parse-block-form blk-stx)
        (let* ([items    (syntax->list blk-stx)]
               [arg-list (syntax->list (list-ref items 1))]  ; ((a : !t) ...)
               [body-ops (cddr items)])                       ; body-op ...
          (let-values ([(arg-vars arg-types)
                        (let loop ([al arg-list] [vs '()] [ts '()])
                          (if (null? al)
                              (values (reverse vs) (reverse ts))
                              (let* ([entry (syntax->list (car al))]
                                     ;; entry = (var : !type)
                                     [v (list-ref entry 0)]
                                     [t (list-ref entry 2)])
                                (loop (cdr al) (cons v vs) (cons t ts)))))])
            (list arg-vars arg-types body-ops))))

      ;; Emit the code that fills one pre-allocated region of new-op.
      ;; region-idx — integer index.
      ;; blk-stx    — the (^bb0 (...) body ...) form.
      ;; Returns a syntax expression (begin ...) suitable as a statement.
      (define (emit-one-region region-idx blk-stx idx)
        (let* ([parsed    (parse-block-form blk-stx)]
               [arg-vars  (list-ref parsed 0)]
               [arg-types (list-ref parsed 1)]
               [body-ops  (list-ref parsed 2)]
               ;; body-ops is a plain Scheme list; cons it into a syntax list for with-syntax.
               [body-stx  (with-syntax ([(body ...) body-ops])
                             #'(with-mlir-ops body ...))]
               [n-args    (length arg-vars)]
               [arg-bind-pairs
                (let loop ([vs arg-vars] [i 0] [acc '()])
                  (if (null? vs)
                      (reverse acc)
                      (loop (cdr vs) (+ i 1)
                            (cons (cons (car vs)
                                        #`(mlir-block-get-argument block #,i))
                                  acc))))])
          (with-syntax ([ri   region-idx]
                        [(at ...) arg-types]
                        [(arg-binding ...) (map make-binding
                                                (map car arg-bind-pairs)
                                                (map cdr arg-bind-pairs))]
                        [body body-stx])
            ;; new-op is in scope from the enclosing let in emit-op.
            ;; Use new-op as the location source for mlir-build-op-in-block.
            #'(let* ([region (mlir-op-get-region new-op ri)]
                     [block  (mlir-new-block region (list at ...))]
                     arg-binding ...)
                (let ([b (mlir-builder-at-block-end block)])
                  (parameterize ([current-builder
                                   (lambda (name ops types . rest)
                                     (mlir-create-op b new-op name ops types
                                                     (if (pair? rest) (car rest) 0)))])
                    body)
                  (mlir-destroy-builder b))))))

      ;; Emit region-fill code for all blocks in a :regions clause.
      ;; regions-stx — the list form after :regions: ((^bb0 ...) ...)
      ;; Returns (nregions . region-stmts-list).
      (define (emit-regions regions-stx idx)
        (let* ([blocks (syntax->list regions-stx)]
               [n      (length blocks)]
               [stmts  (let loop ([bs blocks] [i 0] [acc '()])
                         (if (null? bs)
                             (reverse acc)
                             (loop (cdr bs) (+ i 1)
                                   (cons (emit-one-region i (car bs) idx) acc))))])
          (cons n stmts)))

      ;;-------------------------------------------------------------------
      ;; Emit helpers
      ;;-------------------------------------------------------------------

      ;; Emit (var . expr) pair(s) for a single-identifier result variable.
      ;; result-type #f or '() → zero-result op (returns new-op itself).
      (define (emit-single %var op-name operands result-type attr-setters region-info)
        (let* ([zero?        (or (not result-type)
                                 (null? (syntax->datum result-type)))]
               [nregions     (if region-info (car region-info) 0)]
               [region-stmts (if region-info (cdr region-info) '())])
          (with-syntax ([var %var]
                        [(v ...) operands]
                        [n op-name]
                        [(setter ...) attr-setters]
                        [(region-stmt ...) region-stmts])
            (if zero?
                (with-syntax ([nr nregions])
                  (list (cons #'var
                              #'(let ([new-op ((current-builder) n (list v ...) '() nr)])
                                  setter ...
                                  region-stmt ...
                                  new-op))))
                (with-syntax ([nr nregions] [rt result-type])
                  (list (cons #'var
                              #'(let ([new-op ((current-builder) n (list v ...) (list rt) nr)])
                                  setter ...
                                  region-stmt ...
                                  (mlir-operation-get-result new-op 0)))))))))

      ;; Emit (var . expr) pairs for a multi-result list variable like (%a %b).
      (define (emit-multi multi-vars op-name operands rtypes attr-setters idx)
        (let ([tmp (datum->syntax #'with-mlir-ops
                     (string->symbol (string-append "%op-tmp-" (number->string idx))))])
          (with-syntax ([(v ...) operands] [n op-name]
                        [(rt ...) rtypes] [t tmp]
                        [(setter ...) attr-setters])
            (cons (cons #'t #'((current-builder) n (list v ...) (list rt ...) 0))
                  (let loop ([vs multi-vars] [i 0] [acc '()])
                    (if (null? vs)
                        (reverse acc)
                        (loop (cdr vs) (+ i 1)
                              (cons (cons (car vs)
                                          #`(mlir-operation-get-result t #,i))
                                    acc))))))))

      ;; process-op: parse one op-form and return a flat list of (var . expr) pairs.
      ;;
      ;; Uses datum comparison for = and -> (not syntax-case literals) so the
      ;; macro works regardless of the call site's import context (hygiene-safe).
      (define (eq-datum? stx sym) (eq? (syntax->datum stx) sym))
      (define (multi? var) (list? (syntax->datum var)))

      (define (scan-modifiers form)
        ;; Scan form list for :attrs idx, :regions idx, -> idx.
        (let loop ([i 4] [attrs #f] [regions #f] [arrow #f])
          (if (>= i (length form))
              (list attrs regions arrow)
              (let ([d (syntax->datum (list-ref form i))])
                (cond
                  [(eq? d ':attrs)   (loop (+ i 1) i regions arrow)]
                  [(eq? d ':regions) (loop (+ i 1) attrs i arrow)]
                  [(eq? d '->)       (loop (+ i 1) attrs regions i)]
                  [else              (loop (+ i 1) attrs regions arrow)])))))

      (define (process-op op-stx idx)
        (let ([form (syntax->list op-stx)])
          (cond
            ;; Scheme escape: (var = expr) — length 3, expr is not an op-name
            [(and form
                  (= (length form) 3)
                  (eq-datum? (list-ref form 1) '=)
                  (not (op-name? (list-ref form 2))))
             (list (cons (list-ref form 0) (list-ref form 2)))]

            ;; MLIR op: var = op-name (ops) [modifiers] -> type
            [(and form
                  (>= (length form) 6)
                  (eq-datum? (list-ref form 1) '=)
                  (op-name? (list-ref form 2)))
             (let* ([%var      (list-ref form 0)]
                    [op-name   (op-name->str (list-ref form 2))]
                    [operands  (value-operands (list-ref form 3))]
                    [scan      (scan-modifiers form)]
                    [attrs-pos (list-ref scan 0)]
                    [regs-pos  (list-ref scan 1)]
                    [arrow-pos (list-ref scan 2)]
                    [result-type (and arrow-pos (list-ref form (+ arrow-pos 1)))]
                    [setters   (if attrs-pos
                                   (map make-attr-setter
                                        (syntax->list (list-ref form (+ attrs-pos 1))))
                                   '())]
                    [reg-info  (if regs-pos
                                   (emit-regions (list-ref form (+ regs-pos 1)) idx)
                                   #f)])
               (if (multi? %var)
                   (emit-multi (syntax->list %var) op-name operands
                               (syntax->list result-type) setters idx)
                   (emit-single %var op-name operands result-type setters reg-info)))]

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
