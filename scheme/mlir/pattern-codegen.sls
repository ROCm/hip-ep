#!r6rs
(library (mlir pattern-codegen)
  (export generate-debug-ast
          generate-pattern-matchAndRewrite
          generate-debug-codegen
          make-unbound-value)
  (import (rnrs)
          (only (chezscheme) syntax->list syntax->datum syntax-object->datum record-rtd record-type-field-names record-accessor identifier?)
          (rename (rime loop) (:with :rime-with))
          (for (only (chezscheme) syntax->list syntax->datum record-rtd record-type-field-names record-accessor identifier?) expand)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (mlir pattern-ast) expand)
          (for (mlir pattern-analyze) expand)
          (for (mlir ffi) expand)
          (for (only (mlir ops) with-mlir-ops) expand))

  ;;=======================================================================
  ;; Call graph
  ;;=======================================================================
  ;;
  ;; generate-pattern-matchAndRewrite
  ;; ├── generate-root-result-setters  (set! %varN (mlir-operation-get-result op N)) per root result
  ;; │   └── find-root-op
  ;; ├── collect-all-variables
  ;; ├── generate-check-code           (and check₀ check₁ …) for :match
  ;; │   └── action->check-code
  ;; ├── generate-rewrite-code (raw-body)  wraps with-current-mlir-builder + with-mlir-ops
  ;; │   :rewrite :with body forwarded verbatim to with-mlir-ops; no AST round-trip
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
            [raw-rewrite    (ast-pattern-expand-rewrite         ast-rec)] ; list of raw syntax objects
            [then-let-bindings (ast-pattern-expand-then-let    ast-rec)]) ; list of ast-then-let-binding-expand

        ;; Computed values — each may depend on earlier bindings in this block.
        (let* ([num-ops          (vector-length match-vec)]
               [root-op             (find-root-op match-vec root-op-name)]
               [root-result-vars    (ast-match-expand-result-var root-op)]
               [root-result-setters (generate-root-result-setters root-result-vars op)]
               ;; :rewrite :with body kept as raw syntax — forwarded to with-mlir-ops.
               ;; Rewrite vars are NOT pre-declared in the outer let; with-mlir-ops
               ;; declares them in its own let*.
               [match-vars       (collect-all-variables binding-mgr)]
               [then-let-vars    (map ast-then-let-binding-expand-var then-let-bindings)]
               [all-vars         (append match-vars then-let-vars)])

          ;; Code generation — the two halves are independent of each other.
          (let ([check-code  (generate-check-code actions match-vec operands-ref)]
                [rewrite-code (generate-rewrite-code raw-rewrite pattern-type rewriter op)])

        (with-syntax ([fname    (ast-pattern-expand-function-name ast-rec)]
                      [(param ...) (list op operands-ref rewriter type-converter)]
                      [(var ...) all-vars]
                      [num-operations num-ops]
                      [(root-result-setter ...) root-result-setters]
                      [(then-let-binding ...) (generate-then-let-bindings then-let-bindings)]
                      [checks  check-code]
                      [rewrite rewrite-code])
          #'(define fname
              (lambda (param ...)
                (let ([var (make-unbound-value)] ...
                      [all-operations (make-vector num-operations (make-unbound-value))])
                  root-result-setter ...
                  (if checks
                      (let* (then-let-binding ...)
                        rewrite)
                      #f))))))))))

  ;;=======================================================================
  ;; Rewrite code — thin wrapper delegating to with-mlir-ops
  ;;=======================================================================
  ;;
  ;; generate-rewrite-code
  ;;
  ;; raw-body     — list of raw syntax objects (the :rewrite :with op-forms)
  ;; pattern-type — 'conversion | 'rewrite
  ;; rw / op      — syntax identifiers for the rewriter and matched operation
  ;;
  ;; Generated shape ('conversion):
  ;;   (with-current-mlir-builder (rw op)
  ;;     (let ([result (with-mlir-ops form ...)])
  ;;       (mlir-replace-op rw op result)
  ;;       #t))
  ;;
  ;; with-mlir-ops handles op-forms, :attrs, :regions, and :scheme escapes.
  ;; with-current-mlir-builder installs current-mlir-build-fn so all ops
  ;; dispatch through mlir-build-op rw loc.
  (define (generate-rewrite-code raw-body pattern-type rw op)
    (if (null? raw-body)
        #'#t
        (with-syntax ([(form ...) raw-body])
          (case pattern-type
            [(conversion)
             #`(with-current-mlir-builder (#,rw #,op)
                 (let ([result (with-mlir-ops form ...)])
                   (mlir-replace-op #,rw #,op result)
                   #t))]
            [(rewrite)
             #`(with-current-mlir-builder (#,rw #,op)
                 (let ([result (with-mlir-ops form ...)])
                   (mlir-replace-op #,rw #,op result)
                   #t))]))))

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
