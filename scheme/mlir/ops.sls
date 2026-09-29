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
;;     Installs rw (RewriterBase*) and loc (Operation* for location/IP)
;;     as dynamic context for the duration of body.
;;
;;   (with-mlir-ops op-form ...)
;;     Builds a sequence of MLIR operations using the current builder.
;;     Returns the last result (Scheme convention).
;;
;; op-form syntax:
;;   (%var = "op.name" (operands...) modifiers... -> result-type)   single-result
;;   ((%a %b) = "op.name" (operands...) modifiers... -> (t1 t2))   multi-result
;;   (%var = scheme-expr)                                            Scheme escape
;;   (op-name (operands...) modifiers...)                           statement (no result)
;;
;; Modifiers (any order, between operands and ->):
;;   (name = val)            — set string attr
;;   (name = val :index)     — set index attr
;;   (name = val :i32-array) — set dense i32 array attr
;;   (^label ((arg : !type) ...) body ...) — region block
;;
;; Operands prefixed with ! are types filtered from the value operand list.
;; ,@list splices a dynamic list into the operand position:
;;   (%r = tensor.from_elements (,@%dim-vals) -> !type)
;;   (%r = some.op (v1 ,@%extra v2) -> !type)
;;
;;===----------------------------------------------------------------------===;;

(library (mlir ops)
  (export with-mlir-ops)

  (import (except (rnrs (6)) =)
          (only (chezscheme) syntax->list syntax->datum datum->syntax parameterize)
          (rename (rime loop) (:with :rime-with))
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (only (mlir pattern-keywords) = : -> :index :region) expand)
          (mlir ir)
          (for (only (mlir ir) mlir-build-operation) expand))

  ;;===--------------------------------------------------------------------===;;
  ;; with-mlir-ops
  ;;===--------------------------------------------------------------------===;;

  ;; NOTE: the = symbol in op-forms must be the DSL = from (mlir pattern-keywords),
  ;; not the R6RS numeric =. Libraries using (except (rnrs) =) satisfy this;
  ;; others must import (only (mlir pattern-keywords) =) explicitly.
  (define-syntax with-mlir-ops
    (lambda (stx)

      ;;-------------------------------------------------------------------
      ;; Main entry
      ;;-------------------------------------------------------------------

      ;; Collect all (var . expr) pairs from every op form and emit a flat
      ;; let* returning the last bound variable.
      (define (main)
        (syntax-case stx ()
          [(_ op ...)
           (let ([pairs (loop :for op-stx :in (syntax->list #'(op ...))
                              :for index :from 0
                              :append (process-op op-stx index))])
             (if (null? pairs)
                 #'(if #f #f)
                 (with-syntax ([(binding ...) (loop :for pair :in pairs
                                                    :collect (make-binding pair))]
                               [result        (car (car (reverse pairs)))])
                   #'(let* (binding ...) result))))]))

      ;;-------------------------------------------------------------------
      ;; Op form parser
      ;;-------------------------------------------------------------------

      ;; Parse one op-form; return a flat list of (var . expr) pairs.
      (define (process-op op-stx index)
        (syntax-case op-stx (= ->)
          ;; Scheme escape
          [(var = expr)
           (and (identifier? #'var) (not (op-name? #'expr)))
           (list (cons #'var #'expr))]
          ;; Single-result: normalize (var) to ((var)) and result-type to (result-type)
          ;; then recurse. identifier? guard prevents re-matching after normalization.
          [(var = op (operands ...) modifiers ... -> result-type)
           (and (identifier? #'var) (op-name? #'op))
           (process-op #'((var) = op (operands ...) modifiers ... -> (result-type)) index)]
          ;; Multi-result (and normalized single): result vars and result types are lists
          [((var ...) = op (operands ...) modifiers ... -> (result-type ...))
           (and (op-name? #'op)
                (for-all identifier? (syntax->list #'(var ...)))
                (eqv? (length (syntax->list #'(var ...)))
                      (length (syntax->list #'(result-type ...)))))
           (let-values ([(attr-setter-fns region-fill-fns) (parse-modifiers #'(modifiers ...))])
             (emit-multi (syntax->list #'(var ...)) (op-name->str #'op)
                         (value-operands #'(operands ...))
                         (syntax->list #'(result-type ...)) attr-setter-fns
                         region-fill-fns index))]
          ;; Statement: no var, no ->. Normalize to (() = op ...) then recurse.
          [(op (operands ...) modifiers ...)
           (op-name? #'op)
           (process-op #'(() = op (operands ...) modifiers ... -> ()) index)]
          [_ (syntax-violation 'with-mlir-ops "invalid op form" op-stx)]))

      ;;-------------------------------------------------------------------
      ;; Modifier parser
      ;;-------------------------------------------------------------------

      ;; Parse the optional modifier clause between the operand list and ->.
      ;; modifiers-stx is the captured (modifiers ...) syntax list.
      ;;
      ;; Returns (values attr-setter-fns region-fill-fns) where:
      ;;   attr-setter-fns  — Scheme list of closures (lambda (new-op-stx) → setter-syntax)
      ;;   region-fill-fns  — Scheme list of closures (lambda (new-op-stx) → fill-stmt-syntax)
      ;;
      ;; Blocks are parsed immediately; closures carry the pre-parsed structure.
      ;; No re-parsing occurs later.
      ;;
      ;; Attr entries:
      ;;   (name = val)         — set string attr
      ;;   (name = val :index)  — set index attr
      ;;
      ;; Region forms — two syntaxes:
      ;;   (^label ((arg : type) ...) body ...)
      ;;       Shorthand: one region containing exactly one block.
      ;;       Normalised to (:region (^label ...)) and re-parsed.
      ;;
      ;;   (:region (^label ((arg : type) ...) body ...) ...)
      ;;       Full form: one region containing one or more blocks.
      ;;
      ;; Examples:
      ;;   %r = "arith.constant" () ("value" = 42 :index) -> (i32)
      ;;   %r = "scf.if" (%c) (^bb0 () then-body...) -> (i1)          ;; shorthand
      ;;   %r = "scf.for" (%lo %hi %step)
      ;;          (:region (^bb0 ((i : index)) body...) (^bb1 () exit...))
      ;;          -> (i32)
      (define (parse-modifiers modifiers-stx)
        (let loop ([rest modifiers-stx] [attr-setter-fns '()] [region-fill-fns '()])
          (syntax-case rest (= : :region)
            [()
             (values attr-setter-fns region-fill-fns)]
            ;; Attr entry
            [((name = val . qualifier) . remaining)
             (loop #'remaining
                   (append attr-setter-fns (list (make-attr-setter #'(name = val . qualifier))))
                   region-fill-fns)]
            ;; Shorthand: bare block — extract components from pattern vars directly
            [((label ((arg : type) ...) . body) . remaining)
             (block-label? #'label)
             (let* ([region-index (length region-fill-fns)]
                    [fill-fn      (make-block-fill-fn
                                    (syntax->list #'(arg ...))
                                    (syntax->list #'(type ...))
                                    (syntax->list #'body))])
               (loop #'remaining attr-setter-fns
                     (append region-fill-fns (list (make-region-fill-fn region-index (list fill-fn))))))]
            ;; Full form: pattern matches all block structures directly via nested ellipsis
            [((:region (label ((arg : type) ...) . body) ...) . remaining)
             (and (for-all block-label? (syntax->list #'(label ...)))
                  (for-all identifier?
                           (apply append (map syntax->list (syntax->list #'((arg ...) ...))))))
             (let* ([region-index  (length region-fill-fns)]
                    [block-fill-fns (map make-block-fill-fn
                                        (map syntax->list (syntax->list #'((arg ...) ...)))
                                        (map syntax->list (syntax->list #'((type ...) ...)))
                                        (map syntax->list (syntax->list #'(body ...))))])
               (loop #'remaining attr-setter-fns
                     (append region-fill-fns (list (make-region-fill-fn region-index block-fill-fns)))))]
            [_ (syntax-violation 'with-mlir-ops "invalid modifier entry" rest)])))

      ;; True when x is a block label identifier starting with ^.
      (define (block-label? x)
        (let ([datum (syntax->datum x)])
          (and (symbol? datum)
               (char=? #\^ (string-ref (symbol->string datum) 0)))))

      ;; Returns a closure (lambda (new-op-stx) → setter-syntax) for one attr form.
      ;; Calling the closure with the Operation* syntax produces the setter expression.
      ;; (name = val)            → mlir-operation-set-attr
      ;; (name = val :index)     → mlir-operation-set-index-attr
      ;; (name = val :i32-array) → mlir-operation-set-dense-i32-array
      (define (make-attr-setter attr-stx)
        (define (name->str x)
          (let ([datum (syntax->datum x)])
            (if (string? datum) datum (symbol->string datum))))
        (syntax-case attr-stx (= :index :i32-array)
          [(name = val :index)
           (let ([name-str (name->str #'name)] [val-stx #'val])
             (lambda (new-op-stx)
               (with-syntax ([new-op new-op-stx] [n name-str] [v val-stx])
                 #'(mlir-operation-set-index-attr new-op n v))))]
          [(name = val :i32-array)
           (let ([name-str (name->str #'name)] [val-stx #'val])
             (lambda (new-op-stx)
               (with-syntax ([new-op new-op-stx] [n name-str] [v val-stx])
                 #'(mlir-operation-set-dense-i32-array new-op n v))))]
          [(name = val)
           (let ([name-str (name->str #'name)] [val-stx #'val])
             (lambda (new-op-stx)
               (with-syntax ([new-op new-op-stx] [n name-str] [v val-stx])
                 #'(mlir-operation-set-attr new-op n v))))]))

      ;;-------------------------------------------------------------------
      ;; Code emitters
      ;;-------------------------------------------------------------------

      ;; Core code generator. Produces the (var . expr) pairs that become let*
      ;; bindings in the final expansion.
      ;;
      ;; Always generates a tmp binding for the Operation* itself:
      ;;   (%op-tmp-N . (let ([new-op (builder name operands result-types nregions)])
      ;;                  setter ...           ; apply attributes
      ;;                  region-fill-stmt ... ; fill each region
      ;;                  new-op))
      ;;
      ;; Then appends one binding per result variable:
      ;;   — non-empty result-types: (%var . (mlir-operation-get-result %op-tmp-N i))
      ;;   — empty result-types:     (%var . %op-tmp-N)  (var bound to op itself)
      ;;
      ;; Parameters:
      ;;   result-vars   — Scheme list of result variable syntax objects
      ;;   op-name       — string op name, e.g. "hipsr.placeholder"
      ;;   operands      — Scheme list of value operand syntax objects
      ;;   result-types  — Scheme list of result-type syntax objects (may be empty)
      ;;   attr-setter-fns  — Scheme list of closures (lambda (new-op-stx) → setter-syntax)
      ;;   region-fill-fns  — Scheme list of closures (lambda (new-op-stx) → fill-stmt-syntax)
      ;;   index            — integer op index, used to generate a unique %op-tmp-N name
      (define (emit-multi result-vars op-name operands result-types attr-setter-fns region-fill-fns index)
        (let* ([nregions  (length region-fill-fns)]
               [new-op-id (datum->syntax #'with-mlir-ops 'new-op)]
               [tmp       (datum->syntax #'with-mlir-ops
                            (string->symbol (string-append "%op-tmp-" (number->string index))))])
          (with-syntax ([operands-expr operands] [name op-name]
                        [(result-type ...) result-types] [tmp-var tmp]
                        [new-op new-op-id]
                        [(setter ...) (map (lambda (fn) (fn new-op-id)) attr-setter-fns)]
                        [(region-fill-stmt ...) (map (lambda (fn) (fn new-op-id)) region-fill-fns)]
                        [nregions nregions])
            (cons (cons #'tmp-var
                        #'(let ([new-op (mlir-build-operation name
                                         operands-expr (list result-type ...) nregions)])
                            setter ...
                            region-fill-stmt ...
                            new-op))
                  (if (null? result-types)
                      ;; zero-result: bind each var directly to new-op
                      (map (lambda (var) (cons var #'tmp-var)) result-vars)
                      ;; extract each result by index
                      (let loop ([vars result-vars] [i 0] [acc '()])
                        (if (null? vars)
                            (reverse acc)
                            (loop (cdr vars) (+ i 1)
                                  (cons (cons (car vars)
                                              #`(mlir-operation-get-result tmp-var #,i))
                                        acc)))))))))

      ;; Returns a closure (lambda (new-op-stx) → fill-stmt-syntax) for one region.
      ;; region-index    — 0-based index of this region within the op.
      ;; block-fill-fns  — Scheme list of block-fill closures from make-block-fill-fn.
      (define (make-region-fill-fn region-index block-fill-fns)
        (lambda (new-op-stx)
          (let ([region-id (datum->syntax #'with-mlir-ops 'region)])
            (with-syntax ([new-op      new-op-stx]
                          [region      region-id]
                          [region-idx  region-index]
                          [(block-fill-stmt ...)
                           (map (lambda (fn) (fn new-op-stx region-id)) block-fill-fns)])
              #'(let ([region (mlir-op-get-region new-op region-idx)])
                  block-fill-stmt ...)))))

      ;; Returns a closure (lambda (new-op-stx region-stx) → block-fill-syntax).
      ;; Takes pre-extracted components — no re-parsing of syntax.
      ;;   arg-vars  — Scheme list of arg variable syntax objects
      ;;   arg-types — Scheme list of arg type syntax objects
      ;;   body-ops  — Scheme list of body op syntax objects
      (define (make-block-fill-fn arg-vars arg-types body-ops)
        (let* ([body-stx       (with-syntax ([(body ...) body-ops])
                                 #'(with-mlir-ops body ...))]
               [arg-bind-pairs (loop :for var :in arg-vars
                                    :for i :from 0
                                    :collect (cons var #`(mlir-block-get-argument block #,i)))])
          (lambda (new-op-stx region-stx)
            (with-syntax ([(arg-type ...) arg-types]
                          [(arg-binding ...) (loop :for pair :in arg-bind-pairs
                                                   :collect (make-binding pair))]
                          [body    body-stx]
                          [new-op  new-op-stx]
                          [region  region-stx])
              #'(let* ([block (mlir-new-block region (list arg-type ...))]
                       arg-binding ...)
                  (with-block-builder (builder block new-op)
                    body))))))


      ;;-------------------------------------------------------------------
      ;; Leaf helpers
      ;;-------------------------------------------------------------------

      ;; Lift a (var-stx . expr-stx) cons cell into a syntax binding (var expr).
      (define (make-binding pair)
        (with-syntax ([var (car pair)] [expr (cdr pair)]) #'(var expr)))

      ;; Accept both 'hipsr.placeholder and "hipsr.placeholder" as op names.
      (define (op-name? x)
        (let ([datum (syntax->datum x)])
          (or (string? datum) (symbol? datum))))
      (define (op-name->str x)
        (let ([datum (syntax->datum x)])
          (if (string? datum) datum (symbol->string datum))))

      ;; Build a runtime expression for the operand list from (operands ...).
      ;; Operands prefixed with ! are filtered (they are types, not values).
      ;; ,@list splices a dynamic list: (v1 ,@%more v2) → (append (list v1) %more (list v2))
      ;; All-static result: #'(list v1 v2 ...)
      ;; Any splice present: #'(append (list v1) %splice (list v2) ...)
      (define (value-operands operands-stx)
        (define (splice? x)
          (let ([d (syntax->datum x)])
            (and (pair? d) (eq? (car d) 'unquote-splicing))))
        (let ([items (filter (lambda (x) (not (type-id? x)))
                             (syntax->list operands-stx))])
          (if (for-all (lambda (x) (not (splice? x))) items)
              ;; All static — simple (list ...)
              (with-syntax ([(v ...) items]) #'(list v ...))
              ;; Mixed — build with append, grouping static runs
              (let loop ([rest items] [static-run '()] [chunks '()])
                (cond
                  [(null? rest)
                   (let ([final-chunks
                          (if (null? static-run)
                              (reverse chunks)
                              (reverse (cons (with-syntax ([(v ...) (reverse static-run)])
                                              #'(list v ...))
                                            chunks)))])
                     (with-syntax ([(chunk ...) final-chunks])
                       #'(append chunk ...)))]
                  [(splice? (car rest))
                   (let* ([splice-expr (cadr (syntax->list (car rest)))]
                          [chunks+     (if (null? static-run)
                                          (cons splice-expr chunks)
                                          (cons splice-expr
                                                (cons (with-syntax ([(v ...) (reverse static-run)])
                                                        #'(list v ...))
                                                      chunks)))])
                     (loop (cdr rest) '() chunks+))]
                  [else
                   (loop (cdr rest) (cons (car rest) static-run) chunks)])))))

      ;; True when identifier starts with ! (type convention, not a value).
      (define (type-id? x)
        (let ([datum (syntax->datum x)])
          (and (symbol? datum)
               (let ([str (symbol->string datum)])
                 (and (> (string-length str) 0)
                      (char=? #\! (string-ref str 0)))))))

      (main)))


) ;; end library (mlir ops)
