#!r6rs
(library (mlir pattern-codegen)
  (export generate-debug-ast
          generate-pattern-matchAndRewrite
          generate-debug-codegen
          make-unbound-value)
  (import (rnrs)
          (only (chezscheme) syntax->list syntax->datum syntax-object->datum record-rtd record-type-field-names record-accessor identifier? parameterize)
          (rename (rime loop) (:with :rime-with))
          (for (only (chezscheme) syntax->list syntax->datum record-rtd record-type-field-names record-accessor identifier? parameterize) expand)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (mlir pattern-ast) expand)
          (for (mlir pattern-analyze) expand)
          (for (mlir ffi) expand)
          (for (only (mlir pattern-keywords) :index) expand))

  ;;=======================================================================
  ;; Call graph
  ;;=======================================================================
  ;;
  ;; generate-pattern-matchAndRewrite
  ;; ├── generate-root-result-setters  (set! %varN (mlir-operation-get-result op N)) per root result
  ;; │   └── find-root-op
  ;; ├── generate-rewrite-bindings (loc-op)  one (var . binding) per :rewrite op
  ;; │   └── generate-one-rewrite-binding (loc-op)
  ;; │       └── generate-region-code (loc-op)  for each :regions clause
  ;; │           └── generate-one-region (loc-op)
  ;; │               ├── generate-rewrite-bindings  ◄─ recursive (block-ops)
  ;; │               └── generate-rewrite-code 'region  ◄─ recursive
  ;; │                   (with-current-block-builder installs current-mlir-build-fn)
  ;; ├── collect-all-variables
  ;; ├── generate-check-code           (and check₀ check₁ …) for :match
  ;; │   └── action->check-code
  ;; ├── generate-rewrite-code 'conversion/'rewrite  final let* + replaceOp/return
  ;; │   (wraps in with-current-mlir-builder → current-mlir-build-fn = mlir-build-op rw loc)
  ;; └── generate-then-let-bindings    ((var expr) …) for :then-let
  ;;
  ;; generate-debug-ast / generate-debug-codegen  (debug path, not on hot path)
  ;; └── record->alist
  ;;
  ;;=======================================================================
  ;; Entry points (called from pattern-macro.sls waterfall)
  ;;=======================================================================

  (define (generate-debug-ast ast-rec)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)])
      (let ([alist-data (record->alist ast-rec)])
        (with-syntax ([ast-list (datum->syntax #'fname `',alist-data)])
          #'(define fname (lambda () ast-list))))))

  (define (generate-debug-codegen ast-rec generated-code)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)])
      (let ([code-datum (syntax-object->datum generated-code)])
        (with-syntax ([code-list (datum->syntax #'fname `',code-datum)])
          #'(define fname (lambda () code-list))))))

  (define (generate-pattern-matchAndRewrite ast-rec)
    ;; All four are syntax identifiers from the user's call site (guaranteed by validation).
    ;; They become the lambda parameters in the generated function, so references to them
    ;; in :then-let expressions share the same binding via hygiene.
    (let* ([op             (ast-pattern-expand-param-op             ast-rec)] ; syntax-identifier
           [operands-ref   (ast-pattern-expand-param-operands-ref   ast-rec)] ; syntax-identifier
           [rewriter       (ast-pattern-expand-param-rewriter       ast-rec)] ; syntax-identifier
           [type-converter (ast-pattern-expand-param-type-converter ast-rec)]) ; syntax-identifier

      ;; Independent reads from the AST — no ordering required.
      (let ([match-vec      (ast-pattern-expand-match          ast-rec)] ; vector of ast-match-expand
            [binding-mgr   (ast-pattern-expand-match-bindings  ast-rec)] ; binding-manager hashtable
            [actions        (ast-pattern-expand-match-actions   ast-rec)] ; list of action records
            [root-op-name   (ast-pattern-expand-root-op-name   ast-rec)] ; syntax-string e.g. #'"onnx.Cast"
            [pattern-type   (ast-pattern-expand-pattern-type   ast-rec)] ; symbol: 'conversion or 'rewrite
            [rewrite-ops    (ast-pattern-expand-rewrite         ast-rec)] ; list of ast-operation-expand
            [then-let-bindings (ast-pattern-expand-then-let           ast-rec)]) ; list of ast-then-let-binding-expand

        ;; Computed values — each may depend on earlier bindings in this block.
        (let* ([num-ops          (vector-length match-vec)]                        ; integer
               [root-op             (find-root-op match-vec root-op-name)]                      ; ast-match-expand
               [root-result-vars    (ast-match-expand-result-var root-op)]                     ; list of syntax-identifier
               [root-result-setters (generate-root-result-setters root-result-vars op)]        ; list of syntax: (set! %varN (mlir-operation-get-result op N))
               ;; Builder is dynamic — generate-rewrite-bindings no longer receives rw/loc.
               [op-bindings      (generate-rewrite-bindings rewrite-ops op)] ; list of (result-var . let-binding)
               [match-vars       (collect-all-variables binding-mgr)]              ; list of syntax-identifier
               [then-let-vars       (map ast-then-let-binding-expand-var then-let-bindings)] ; list of syntax-identifier
               [rewrite-vars     (map car op-bindings)]                            ; list of syntax-identifier
               [all-vars         (append match-vars then-let-vars rewrite-vars)])     ; list of syntax-identifier

          ;; Code generation — the two halves are independent of each other.
          (let ([check-code  (generate-check-code actions match-vec operands-ref)] ; syntax (and ...)
                [rewrite-code (generate-rewrite-code op-bindings pattern-type rewriter op)]) ; syntax

        (with-syntax ([fname    (ast-pattern-expand-function-name ast-rec)]
                      [(param ...) (list op operands-ref rewriter type-converter)]
                      [(var ...) all-vars]
                      [num-operations num-ops]
                      [(root-result-setter ...) root-result-setters]
                      [(then-let-binding ...) (generate-then-let-bindings then-let-bindings)]
                      [checks  check-code]
                      [rewrite rewrite-code]
                      [rw-param rewriter]
                      [loc-param op])
          ;; with-current-mlir-builder installs current-mlir-build-fn so all DSL
          ;; ops call (current-mlir-build-fn) → mlir-build-op rw loc.
          #'(define fname
              (lambda (param ...)
                (let ([var (make-unbound-value)] ...
                      [all-operations (make-vector num-operations (make-unbound-value))])
                  root-result-setter ...
                  (if checks
                      (let* (then-let-binding ...)
                        (with-current-mlir-builder (rw-param loc-param)
                          rewrite))
                      #f))))))))))

  ;;=======================================================================
  ;; Rewrite code (final let* + replaceOp)
  ;;=======================================================================
  ;;
  ;; generate-rewrite-code
  ;;
  ;; op-bindings — a flat list of (result-var . let-binding) pairs produced by
  ;;   generate-rewrite-bindings.  Single-result ops contribute one pair;
  ;;   multi-result ops contribute N+1 pairs (a tmp binding for the op itself,
  ;;   then one per result variable).
  ;;
  ;;   Single-result example — (%c = "hipsr.cast" (%x %y) -> !t):
  ;;     (%c . (let* ([new-op (mlir-build-op rw loc "hipsr.cast" ...)]) (mlir-operation-get-result new-op 0)))
  ;;
  ;;   Multi-result example — ((%a %b) = "hipsr.foo" (%x) -> (!t1 !t2)):
  ;;     (%multi-tmp-0 . (let* ([new-op (mlir-build-op rw loc "hipsr.foo" ...)]) new-op))
  ;;     (%a           . (mlir-operation-get-result %multi-tmp-0 0))
  ;;     (%b           . (mlir-operation-get-result %multi-tmp-0 1))
  ;;
  ;;   pair->binding-form converts each (var . expr) pair to a #'(var expr) syntax form
  ;;   for use as a let* binding. Defined as a named function (not an anonymous lambda)
  ;;   so its with-syntax is compiled in isolation without triggering ChezScheme's
  ;;   compile-time forward-reference check across the library body.
  ;;
  ;; last-var — the result-var of the final pair in op-bindings.  For a single-
  ;;   result rewrite that is the op's lone result; for a multi-result rewrite it
  ;;   is the last named result variable (e.g. %b above).  Used as the replacement
  ;;   value for 'conversion patterns.
  ;;
  ;; pattern-type — three cases:
  ;;   'conversion — set IP before op, build all ops in let*, replace matched op
  ;;                 with last-var via mlir-replace-op, return #t
  ;;   'rewrite    — set IP before op, build all ops in let*, return last-var directly
  ;;   'region     — IP already at block end (set by caller); build ops for side
  ;;                 effect only; return unspecified (if #f #f)
  ;;
  ;; Generated shape ('conversion):
  ;;   (let* ((%p (let* ([new-op (mlir-build-op rw loc "hipsr.placeholder" ...)]) (mlir-operation-get-result new-op 0)))
  ;;          (%c (let* ([new-op (mlir-build-op rw loc "hipsr.cast" ...)]) (mlir-operation-get-result new-op 0))))
  ;;     (mlir-replace-op rw op %c)
  ;;     #t)
  ;;   mlir-build-op sets its own IP to before loc each time it is called.
  ;;
  ;; Generated shape ('conversion, multi-result last op):
  ;;   (let* ((%multi-tmp-0 (let* ([new-op (mlir-build-op rw loc "hipsr.foo" ...)]) new-op))
  ;;          (%a           (mlir-operation-get-result %multi-tmp-0 0))
  ;;          (%b           (mlir-operation-get-result %multi-tmp-0 1)))
  ;;     (mlir-replace-op rw op %b)
  ;;     #t)
  ;;
  ;; Generated shape ('region):
  ;;   (let* ((var0 binding0) ...)
  ;;     (if #f #f))

  ;; Helper: combine a (var . expr) pair from op-bindings into a let* binding form.
  ;; Defined as a named function (not a lambda) so with-syntax is compiled in isolation,
  ;; avoiding ChezScheme's compile-time forward-reference issue.
  (define (pair->binding-form p)
    (with-syntax ([v (car p)] [e (cdr p)])
      #'(v e)))

  ;; generate-rewrite-code
  ;; op-bindings   — flat list of (var . expr) pairs
  ;; pattern-type  — 'conversion | 'rewrite | 'region
  ;; rw / op       — syntax identifiers needed only for 'conversion and 'rewrite;
  ;;                 omit (or pass #f) for 'region.
  (define (generate-rewrite-code op-bindings pattern-type . rw+op)
    (let ([rw (if (null? rw+op) #f (car  rw+op))]
          [op (if (null? rw+op) #f (cadr rw+op))])
      (if (null? op-bindings)
          #'#t
          (let* ([bindings (map pair->binding-form op-bindings)]
                 [last-var (car (car (reverse op-bindings)))])
            (case pattern-type
              [(conversion)
               (with-syntax ([(binding ...) bindings]
                             [result last-var])
                 #`(let* (binding ...)
                     (mlir-replace-op #,rw #,op result)
                     #t))]
              [(rewrite)
               (with-syntax ([(binding ...) bindings]
                             [result last-var])
                 #`(let* (binding ...)
                     result))]
              [(region)
               ;; Builder context set by with-current-block-builder in generate-one-region.
               (with-syntax ([(binding ...) bindings])
                 #'(let* (binding ...)
                     (if #f #f)))])))))

    ;;=======================================================================
  ;; :then-let bindings
  ;;=======================================================================

  (define (generate-then-let-bindings then-let-list)
    (loop :for binding-rec :in then-let-list
          :rime-with var  := (ast-then-let-binding-expand-var  binding-rec)
          :rime-with expr := (ast-then-let-binding-expand-expr binding-rec)
          :collect (list var expr)))

  ;;=======================================================================
  ;; Match-phase check code
  ;;=======================================================================

  (define (generate-check-code actions match-vec operands-ref)
    (if (null? actions)
        #'#t
        (let ([checks (map (lambda (act) (action->check-code act match-vec operands-ref)) actions)])
          #`(and #,@checks))))

  (define (action->check-code action match-vec operands-ref)
    (let ([tag (car action)])
      (case tag
        [(:set-current-op)
         (let* ([fields (cdr action)]
                [op-idx (cdr (assq 'op-idx fields))]
                [var    (cdr (assq 'var fields))])
           #`(let ([def-op (mlir-value-get-defining-op #,var)])
               (and def-op
                    (begin
                      (vector-set! all-operations #,op-idx def-op)
                      #t))))]

        [(:check-op)
         (let* ([fields      (cdr action)]
                [op-idx      (cdr (assq 'op-idx fields))]
                [match-op    (vector-ref match-vec op-idx)]
                [op-name     (ast-match-expand-op-name match-op)]
                [num-results (length (ast-match-expand-result-var match-op))])
           #`(and (string=? (mlir-operation-name (vector-ref all-operations #,op-idx))
                            #,(syntax->datum op-name))
                  (= (mlir-operation-num-results (vector-ref all-operations #,op-idx))
                     #,num-results)))]

        [(:bind-operand)
         (let* ([fields      (cdr action)]
                [op-idx      (cdr (assq 'op-idx fields))]
                [var         (cdr (assq 'var fields))]
                [operand-idx (cdr (assq 'operand-idx fields))])
           #`(begin
               (set! #,var (mlir-operation-get-operand-value
                            (vector-ref all-operations #,op-idx)
                            #,operand-idx))
               #t))]

        [(:bind-argument-operand)
         (let* ([fields      (cdr action)]
                [var         (cdr (assq 'var fields))]
                [operand-idx (cdr (assq 'operand-idx fields))])
           #`(if (< #,operand-idx (value-array-ref-size #,operands-ref))
                 (let ([val (value-array-ref-at #,operands-ref #,operand-idx)])
                   (and (not (zero? val))
                        (begin (set! #,var val) #t)))
                 #f))]

        [(:check-eq)
         (let* ([fields      (cdr action)]
                [op-idx      (cdr (assq 'op-idx fields))]
                [operand-idx (cdr (assq 'operand-idx fields))]
                [var         (cdr (assq 'var fields))])
           #`(value-equal? (get-operand (vector-ref all-operations #,op-idx)
                                        #,operand-idx)
                           #,var))]

        [else
         (error 'action->check-code "Unknown action type" tag)])))

  ;;=======================================================================
  ;; Rewrite bindings (let* chain for created ops)
  ;;=======================================================================
  ;;
  ;; Returns a FLAT list of (var . binding) pairs.
  ;; Each item in rewrite-ops is either:
  ;;   ast-scheme-binding-expand → one pair (var . scheme-expr)
  ;;   ast-operation-expand single-result → one pair
  ;;   ast-operation-expand multi-result → N+1 pairs (tmp-op + N result vars)

  ;; generate-rewrite-bindings: no longer takes rw/loc-op — the current
  ;; dynamic builder context (current-mlir-build-fn) is set by the caller
  ;; via with-current-mlir-builder or with-current-block-builder.
  (define (generate-rewrite-bindings rewrite-ops loc-op)
    (loop :for op-rec :in rewrite-ops
          :for idx :from 0
          :rime-with pairs := (if (ast-scheme-binding-expand? op-rec)
                                  (list (generate-scheme-binding op-rec idx))
                                  (generate-one-rewrite-binding op-rec idx loc-op))
          :append pairs))

  (define (generate-scheme-binding binding-rec idx)
    (let* ([var  (ast-scheme-binding-expand-var  binding-rec)]
           [expr (ast-scheme-binding-expand-expr binding-rec)]
           [result-var
            (if (and (identifier? var)
                     (let ([s (syntax->datum var)])
                       (or (eq? s '_) (eq? s 'void))))
                ;; discard — generate a fresh unused name
                (datum->syntax #'here
                  (string->symbol (string-append "%scheme-discard-" (number->string idx))))
                var)])
      (cons result-var expr)))

  ;; Returns a LIST of (var . binding) pairs.
  ;; Single-result → list of one pair.
  ;; Multi-result → list of (op-tmp . op-binding) + (var_i . result-i-binding) ...
  ;;
  ;; Builder dispatch: generated code calls (current-mlir-build-fn) and
  ;; (current-mlir-build-with-regions-fn) at runtime; the dynamic context is
  ;; set by with-current-mlir-builder (rewriter path) or
  ;; with-current-block-builder (region body path).
  (define (generate-one-rewrite-binding op-rec idx loc-op)
    (let* ([result-var-raw (ast-operation-expand-result-var op-rec)]
           [is-multi?   (pair? (syntax->datum result-var-raw))]
           [is-empty?   (and (not is-multi?)
                             (null? (if (identifier? result-var-raw)
                                        (list result-var-raw)
                                        (syntax->datum result-var-raw))))]
           [op-name     (syntax->datum (ast-operation-expand-op-name op-rec))]
           [all-operands (syntax->list (ast-operation-expand-operands op-rec))]
           [operands    (filter (lambda (s)
                                  (not (char=? (string-ref (symbol->string (syntax->datum s)) 0) #\!)))
                                all-operands)]
           [result-types-raw (ast-operation-expand-result-types op-rec)]
           [attrs       (syntax->list (ast-operation-expand-attributes op-rec))]
           [regions-raw (ast-operation-expand-regions op-rec)]
           [regions     (cond [(null? regions-raw) '()]
                              [(pair? regions-raw) regions-raw]
                              [else '()])]
           [region-code (generate-region-code regions loc-op)])
      (if is-multi?
          ;; Multi-result: build op into a tmp, then extract each result
          (let* ([result-vars  (syntax->list result-var-raw)]
                 [op-tmp       (datum->syntax #'here
                                 (string->symbol (string-append "%multi-tmp-" (number->string idx))))]
                 [types-code   (if (null? (syntax->datum result-types-raw))
                                   #''()
                                   (with-syntax ([(t ...) (syntax->list result-types-raw)])
                                     #'(list t ...)))]
                 [op-binding
                  (with-syntax ([name op-name]
                                [(operand ...) operands]
                                [types types-code]
                                [regions-emit region-code])
                    (if (null? regions)
                        #'(let* ([new-op ((current-mlir-build-fn) name (list operand ...) types)])
                             regions-emit
                             new-op)
                        (with-syntax ([nregions (length regions)])
                          #'(let* ([new-op ((current-mlir-build-with-regions-fn) name (list operand ...) types nregions)])
                               regions-emit
                               new-op))))]
                 [result-pairs
                  (let loop ([vs result-vars] [i 0] [racc '()])
                    (if (null? vs)
                        (reverse racc)
                        (loop (cdr vs) (+ i 1)
                              (cons (cons (car vs)
                                          (with-syntax ([tmp op-tmp] [i-val i])
                                            #'(mlir-operation-get-result tmp i-val)))
                                    racc))))])
            (cons (cons op-tmp op-binding) result-pairs))
          ;; Single-result (existing path, wrapped in list)
          (let* ([result-var (if is-empty?
                                 (datum->syntax #'here
                                   (string->symbol (string-append "%rewrite-tmp-" (number->string idx))))
                                 result-var-raw)]
                 [result-types (if (null? (syntax->datum result-types-raw))
                                   #''()
                                   (with-syntax ([t result-types-raw]) #'(list t)))]
                 [binding
                  (if (null? attrs)
                      (with-syntax ([name op-name]
                                    [(operand ...) operands]
                                    [types result-types]
                                    [regions-emit region-code])
                        (if (null? regions)
                            #'(let* ([new-op ((current-mlir-build-fn) name (list operand ...) types)])
                                 regions-emit
                                 (mlir-operation-get-result new-op 0))
                            (with-syntax ([nregions (length regions)])
                              #'(let* ([new-op ((current-mlir-build-with-regions-fn) name (list operand ...) types nregions)])
                                   regions-emit
                                   (mlir-operation-get-result new-op 0)))))
                      (with-syntax ([name op-name]
                                    [(operand ...) operands]
                                    [types result-types]
                                    [(attr-setter ...) (map generate-attr-setter attrs)]
                                    [regions-emit region-code])
                        (if (null? regions)
                            #'(let* ([new-op ((current-mlir-build-fn) name (list operand ...) types)])
                                 attr-setter ...
                                 regions-emit
                                 (mlir-operation-get-result new-op 0))
                            (with-syntax ([nregions (length regions)])
                              #'(let* ([new-op ((current-mlir-build-with-regions-fn) name (list operand ...) types nregions)])
                                   attr-setter ...
                                   regions-emit
                                   (mlir-operation-get-result new-op 0))))))])
            (list (cons result-var binding))))))

  ;;=======================================================================
  ;; Region code
  ;;=======================================================================

  ;; generate-region-code: loc-op is a syntax identifier used for datum->syntax hygiene.
  ;; Generates a (begin stmt ...) block that fills each pre-allocated region of new-op.
  (define (generate-region-code regions loc-op)
    (if (null? regions)
        #'(if #f #f)
        (let ([region-stmts
               (let loop ([rs regions] [i 0] [acc '()])
                 (if (null? rs)
                     (reverse acc)
                     (loop (cdr rs) (+ i 1)
                           (cons (generate-one-region loc-op (car rs) i) acc))))])
          (with-syntax ([(stmt ...) region-stmts])
            #'(begin stmt ...)))))

  ;; generate-one-region: creates a block inside region i of new-op, builds a fresh
  ;; OpBuilder at block end, then recursively generates the block body ops.
  ;; with-current-block-builder installs the OpBuilder into current-mlir-build-fn
  ;; and current-block-builder so both the DSL and :scheme escapes resolve correctly.
  (define (generate-one-region loc-op region-rec region-idx)
    (let* ([blocks    (ast-region-expand-blocks region-rec)]
           [block-rec (car blocks)]
           [block-args (ast-block-expand-arguments block-rec)]
           [block-ops  (ast-block-expand-operations block-rec)]
           [arg-vars  (map car  block-args)]
           [arg-types (map cadr block-args)]
           [arg-bindings
            (let loop ([vars arg-vars] [i 0] [acc '()])
              (if (null? vars)
                  (reverse acc)
                  (loop (cdr vars) (+ i 1)
                        (cons (with-syntax ([v (car vars)] [idx i])
                                #'(v (mlir-block-get-argument block idx)))
                              acc))))]
           ;; Recurse inside with-current-block-builder which sets current-mlir-build-fn
           [block-op-bindings (generate-rewrite-bindings block-ops loc-op)]
           [nested-code       (generate-rewrite-code block-op-bindings 'region)])
      (with-syntax ([ri region-idx]
                    [loc-id loc-op]
                    [(arg-type ...) arg-types]
                    [(arg-binding ...) arg-bindings]
                    [nested nested-code])
        #'(let* ([region (mlir-op-get-region new-op ri)]
                 [block  (mlir-new-block region (list arg-type ...))]
                 arg-binding ...)
            ;; with-current-block-builder sets both current-mlir-build-fn and
            ;; current-block-builder so :scheme escapes in the body resolve correctly.
            (let ([b (mlir-builder-at-block-end block)])
              (with-current-block-builder (b loc-id)
                nested)
              (mlir-destroy-builder b))))))

  ;;=======================================================================
  ;; Attribute setter
  ;;=======================================================================

  (define (generate-attr-setter attr-stx)
    ;; Datum comparison for qualifiers avoids free-identifier=? hygiene issues.
    (syntax-case attr-stx ()
      [(attr-name value qualifier)
       (eq? (syntax->datum #'qualifier) ':index)
       (with-syntax ([name-str (symbol->string (syntax->datum #'attr-name))])
         #'(mlir-operation-set-index-attr new-op name-str value))]
      [(attr-name value qualifier)
       (eq? (syntax->datum #'qualifier) ':i32-array)
       (with-syntax ([name-str (symbol->string (syntax->datum #'attr-name))])
         #'(mlir-operation-set-dense-i32-array new-op name-str value))]
      [(attr-name value)
       (with-syntax ([name-str (symbol->string (syntax->datum #'attr-name))])
         #'(mlir-operation-set-attr new-op name-str value))]))

  ;;=======================================================================
  ;; Root op lookup and initialization
  ;;=======================================================================

  (define (find-root-op match-vec root-op-name-stx)
    (let ([root-op-name (syntax->datum root-op-name-stx)])
      (loop :initially := #f
            :for idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec idx)
            :rime-with op-name  := (syntax->datum (ast-match-expand-op-name match-op))
            :when (string=? op-name root-op-name)
            :break match-op)))

  (define (generate-root-result-setters root-result-vars op-param)
    (loop :for var :in root-result-vars
          :for idx :from 0
          :collect #`(set! #,var (mlir-operation-get-result #,op-param #,idx))))

  ;;=======================================================================
  ;; Variable collection
  ;;=======================================================================

  (define (collect-all-variables binding-mgr)
    (vector->list (hashtable-keys (binding-manager-bindings binding-mgr))))

  ;;=======================================================================
  ;; Leaf utilities
  ;;=======================================================================

  (define (make-unbound-value) (if #f #f))

  (define (record->alist obj)
    (let ([datum (syntax-object->datum obj)])
      (cond
        [(not (eq? datum obj)) datum]
        [(hashtable? obj)
         (let* ([keys (vector->list (hashtable-keys obj))]
                [sorted-keys (list-sort (lambda (a b)
                                          (string<? (if (identifier? a)
                                                        (symbol->string (syntax->datum a))
                                                        (symbol->string a))
                                                    (if (identifier? b)
                                                        (symbol->string (syntax->datum b))
                                                        (symbol->string b))))
                                        keys)])
           (loop :for key :in sorted-keys
                 :collect (cons (record->alist key)
                               (record->alist (hashtable-ref obj key #f)))))]
        [(record? obj)
         (let* ([rtd (record-rtd obj)]
                [field-names (vector->list (record-type-field-names rtd))])
           (loop :for name :in field-names
                 :for i :from 0
                 :collect (let* ([accessor (record-accessor rtd i)]
                                 [value (accessor obj)])
                            (cons name (record->alist value)))))]
        [(list? obj) (map record->alist obj)]
        [(vector? obj) (vector->list (vector-map record->alist obj))]
        [else obj])))

)
