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
      ;; Datum comparison for keywords: robust across library boundaries since
      ;; with-mlir-ops users don't need to import (mlir pattern-keywords).
      (define (arrow? x)        (eq? (syntax->datum x) '->))
      (define (eq-sym? x)       (eq? (syntax->datum x) '=))
      (define (attrs-kw? x)     (eq? (syntax->datum x) ':attrs))
      (define (regions-kw? x)   (eq? (syntax->datum x) ':regions))
      (define (index-kw? x)     (eq? (syntax->datum x) ':index))
      (define (i32-array-kw? x) (eq? (syntax->datum x) ':i32-array))
      (define (colon-kw? x)     (eq? (syntax->datum x) ':))

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
        (let ([af (syntax->list attr-stx)])
          (let* ([name-raw (syntax->datum (list-ref af 0))]
                 [name-str (if (string? name-raw) name-raw (symbol->string name-raw))]
                 [val      (list-ref af 1)])
            (cond
              [(and (= (length af) 3) (index-kw? (list-ref af 2)))
               (with-syntax ([n name-str] [v val])
                 #'(mlir-operation-set-index-attr new-op n v))]
              [(and (= (length af) 3) (i32-array-kw? (list-ref af 2)))
               (with-syntax ([n name-str] [v val])
                 #'(mlir-operation-set-dense-i32-array new-op n v))]
              [else
               (with-syntax ([n name-str] [v val])
                 #'(mlir-operation-set-attr new-op n v))]))))

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
                  (with-current-block-builder (b new-op)
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
      ;; Op form scanner: find :attrs and :regions positions
      ;;-------------------------------------------------------------------

      ;; Scan a form list for :attrs and :regions keyword positions.
      ;; Returns an alist: ((attrs . idx-or-#f) (regions . idx-or-#f) (arrow . idx) (result . idx))
      (define (scan-form form)
        (let loop ([i 0] [attrs-idx #f] [regions-idx #f] [arrow-idx #f])
          (if (>= i (length form))
              `((attrs . ,attrs-idx) (regions . ,regions-idx) (arrow . ,arrow-idx))
              (let ([x (list-ref form i)])
                (cond
                  [(attrs-kw? x)   (loop (+ i 1) i       regions-idx arrow-idx)]
                  [(regions-kw? x) (loop (+ i 1) attrs-idx i          arrow-idx)]
                  [(arrow? x)      (loop (+ i 1) attrs-idx regions-idx i)]
                  [else            (loop (+ i 1) attrs-idx regions-idx arrow-idx)])))))

      ;;-------------------------------------------------------------------
      ;; Emit helpers: build op with optional attrs and regions
      ;;-------------------------------------------------------------------

      ;; Core emitter: given all parts, emit the (var binding) pair(s).
      ;; result-type — syntax for the type, or #f for zero-result.
      ;; attr-setters — list of setter syntax forms.
      ;; region-info  — #f or (nregions . stmts-list).
      (define (emit-op %var op-name vals result-type attr-setters region-info)
        (let* ([zero-result? (or (not result-type)
                                 (null? (syntax->datum result-type)))]
               [nregions     (if region-info (car region-info) 0)]
               [region-stmts (if region-info (cdr region-info) '())])
          (with-syntax ([var %var]
                        [(v ...) vals]
                        [n op-name]
                        [(setter ...) attr-setters]
                        [(region-stmt ...) region-stmts])
            (if zero-result?
                ;; zero-result: bind var to the op itself
                (if (> nregions 0)
                    (with-syntax ([nr nregions])
                      (list (cons #'var
                                  #'(let ([new-op ((current-mlir-build-with-regions-fn) n (list v ...) '() nr)])
                                      setter ...
                                      region-stmt ...
                                      new-op))))
                    (list (cons #'var
                                #'(let ([new-op ((current-mlir-build-fn) n (list v ...) '())])
                                    setter ... new-op))))
                ;; single result: bind var to result 0
                (if (> nregions 0)
                    (with-syntax ([nr nregions] [rt result-type])
                      (list (cons #'var
                                  #'(let ([new-op ((current-mlir-build-with-regions-fn) n (list v ...) (list rt) nr)])
                                      setter ...
                                      region-stmt ...
                                      (mlir-operation-get-result new-op 0)))))
                    (with-syntax ([rt result-type])
                      (list (cons #'var
                                  #'(let ([new-op ((current-mlir-build-fn) n (list v ...) (list rt))])
                                      setter ...
                                      (mlir-operation-get-result new-op 0))))))))))

      ;; process-op: parse one form and return a flat list of (var . expr) pairs.
      ;;
      ;; Recognized forms (var may be a list for multi-result):
      ;;   var = op-name (ops) -> type
      ;;   var = op-name (ops) :attrs ((n v q?) ...) -> type
      ;;   var = op-name (ops) :regions ((^bb0 ...) ...) -> type
      ;;   var = op-name (ops) :attrs (...) :regions (...) -> type
      ;;   var = expr                          (scheme escape)
      (define (process-op op-stx idx)
        (let ([form (syntax->list op-stx)])
          (cond
            ;; Scheme escape: (var = expr) — length 3, no op-name at position 2
            [(and form
                  (= (length form) 3)
                  (eq-sym? (list-ref form 1)))
             (list (cons (list-ref form 0) (list-ref form 2)))]

            ;; MLIR op: var = op-name (ops) [modifiers] -> type
            [(and form
                  (>= (length form) 6)
                  (eq-sym? (list-ref form 1))
                  (op-name? (list-ref form 2)))
             (let* ([%var      (list-ref form 0)]
                    [op-name   (op-name->str (list-ref form 2))]
                    [operands  (value-operands (list-ref form 3))]
                    [scan      (scan-form form)]
                    [attrs-pos (cdr (assq 'attrs   scan))]
                    [regs-pos  (cdr (assq 'regions scan))]
                    [arrow-pos (cdr (assq 'arrow   scan))]
                    [result-type (list-ref form (+ arrow-pos 1))]
                    [attr-setters
                     (if attrs-pos
                         (map make-attr-setter (syntax->list (list-ref form (+ attrs-pos 1))))
                         '())]
                    [region-info
                     (if regs-pos
                         (emit-regions (list-ref form (+ regs-pos 1)) idx)
                         #f)]
                    ;; multi-result: %var is a list form like (%a %b); use datum to check safely
                    [multi-vars (and (list? (syntax->datum %var))
                                     (syntax->list %var))])
               (if multi-vars
                   ;; Multi-result: tmp binding + N result extractions
                   (let* ([rtypes (syntax->list result-type)]
                          [tmp    (datum->syntax #'with-mlir-ops
                                    (string->symbol (string-append "%op-tmp-" (number->string idx))))])
                     (with-syntax ([(v ...) operands] [n op-name]
                                   [(rt ...) rtypes] [t tmp]
                                   [(setter ...) attr-setters])
                       (cons (cons #'t
                                   #'((current-mlir-build-fn) n (list v ...) (list rt ...)))
                             (let loop ([vs multi-vars] [i 0] [acc '()])
                               (if (null? vs)
                                   (reverse acc)
                                   (loop (cdr vs) (+ i 1)
                                         (cons (cons (car vs)
                                                     #`(mlir-operation-get-result t #,i))
                                               acc)))))))
                   ;; Single-result (or zero-result)
                   (emit-op %var op-name operands result-type attr-setters region-info)))]

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
