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
          (for (mlir ffi) expand))

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
               [op-bindings      (generate-rewrite-bindings rewrite-ops rewriter op)] ; list of (result-var . let-binding)
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
  ;; Rewrite code (final let* + replaceOp)
  ;;=======================================================================
  ;;
  ;; generate-rewrite-code
  ;;
  ;; op-bindings — one entry per operation in the :rewrite ... :with clause.
  ;;   Each entry is (result-var . let-binding) where result-var is the %name
  ;;   before = and let-binding is the generated code that calls mlir-build-op
  ;;   and evaluates to its result 0.
  ;;   NOTE: only the first result of each op is captured; multi-result ops
  ;;   are not yet supported.
  ;;
  ;; pattern-type — 'conversion: last result replaces the matched op via
  ;;                mlir-replace-op, returns #t.
  ;;                'rewrite: last result is returned directly.
  ;;
  ;; Generated shape ('conversion):
  ;;   (let* ((var0 binding0) (var1 binding1) ...)
  ;;     (mlir-replace-op rw op var-last)
  ;;     #t)

  (define (generate-rewrite-code op-bindings pattern-type rw op)
    (if (null? op-bindings)
        #'#t
        (let* ([bindings (map cdr op-bindings)]
               [last-var (car (car (reverse op-bindings)))])
          (if (eq? pattern-type 'conversion)
              (with-syntax ([(binding ...) bindings]
                            [result last-var])
                #`(let* (binding ...)
                    (mlir-replace-op #,rw #,op result)
                    #t))
              (with-syntax ([(binding ...) bindings]
                            [result last-var])
                #'(let* (binding ...)
                    result))))))

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

  (define (generate-rewrite-bindings rewrite-ops rw loc-op)
    (loop :for op-rec :in rewrite-ops
          :for idx :from 0
          :rime-with pair := (generate-one-rewrite-binding op-rec idx rw loc-op)
          :collect pair))

  (define (generate-one-rewrite-binding op-rec idx rw loc-op)
    (let* ([result-var-raw (ast-operation-expand-result-var op-rec)]
           [is-empty? (null? (if (identifier? result-var-raw)
                                 (list result-var-raw)
                                 (syntax->datum result-var-raw)))]
           [result-var (if is-empty?
                           (datum->syntax #'here
                             (string->symbol (string-append "%rewrite-tmp-" (number->string idx))))
                           result-var-raw)]
           [op-name     (syntax->datum (ast-operation-expand-op-name op-rec))]
           [all-operands (syntax->list (ast-operation-expand-operands op-rec))]
           [operands    (filter (lambda (s)
                                  (not (char=? (string-ref (symbol->string (syntax->datum s)) 0) #\!)))
                                all-operands)]
           [result-types (ast-operation-expand-result-types op-rec)]
           [attrs       (syntax->list (ast-operation-expand-attributes op-rec))]
           [regions-raw (ast-operation-expand-regions op-rec)]
           [regions     (cond [(null? regions-raw) '()]
                              [(pair? regions-raw) regions-raw]
                              [else '()])]
           [region-code (generate-region-code regions rw loc-op)])
      (cons result-var
            (if (null? attrs)
                (with-syntax ([var result-var]
                              [name op-name]
                              [(operand ...) operands]
                              [types result-types]
                              [rw-id rw]
                              [loc-id loc-op]
                              [regions-emit region-code])
                  #'(var (let* ([_ (mlir-set-insertion-point-before rw-id loc-id)]
                                [new-op (mlir-build-op rw-id loc-id name (list operand ...) (list types))])
                           regions-emit
                           (mlir-operation-get-result new-op 0))))
                (with-syntax ([var result-var]
                              [name op-name]
                              [(operand ...) operands]
                              [types result-types]
                              [rw-id rw]
                              [loc-id loc-op]
                              [(attr-setter ...) (map generate-attr-setter attrs)]
                              [regions-emit region-code])
                  #'(var (let* ([_ (mlir-set-insertion-point-before rw-id loc-id)]
                                [new-op (mlir-build-op rw-id loc-id name (list operand ...) (list types))])
                           attr-setter ...
                           regions-emit
                           (mlir-operation-get-result new-op 0))))))))

  ;;=======================================================================
  ;; Region code
  ;;=======================================================================

  (define (generate-region-code regions rw loc-op)
    (if (null? regions)
        #'(begin)
        (let ([region-stmts
               (let loop ([rs regions] [i 0] [acc '()])
                 (if (null? rs)
                     (reverse acc)
                     (loop (cdr rs) (+ i 1)
                           (cons (generate-one-region rw loc-op (car rs) i) acc))))])
          (with-syntax ([(stmt ...) region-stmts]
                        [rw-id rw]
                        [loc-id loc-op])
            #'(begin
                stmt ...
                (mlir-set-insertion-point-before rw-id loc-id))))))

  (define (generate-one-region rw loc-op region-rec region-idx)
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
           [nested-code
            (if (null? block-ops)
                #'(begin)
                (with-syntax ([(emit ...)
                               (map (lambda (nested-op)
                                      (generate-nested-op-emit rw loc-op nested-op))
                                    block-ops)])
                  #'(begin emit ...)))])
      (with-syntax ([ri region-idx]
                    [rw-id rw]
                    [loc-id loc-op]
                    [(arg-type ...) arg-types]
                    [(arg-binding ...) arg-bindings]
                    [nested nested-code])
        #'(let* ([region (mlir-op-get-region new-op ri)]
                 [block  (mlir-region-create-block rw-id region (list arg-type ...))]
                 arg-binding ...)
            nested))))

  (define (generate-nested-op-emit rw loc-op nested-op-rec)
    (let* ([op-name (syntax->datum (ast-operation-expand-op-name nested-op-rec))]
           [all-operands (syntax->list (ast-operation-expand-operands nested-op-rec))]
           [operands (filter (lambda (s)
                               (not (char=? (string-ref (symbol->string (syntax->datum s)) 0) #\!)))
                             all-operands)]
           [result-types (ast-operation-expand-result-types nested-op-rec)]
           [result-list-code (if (null? (syntax->datum result-types))
                                 #''()
                                 (with-syntax ([types result-types])
                                   #'(list types)))])
      (with-syntax ([name op-name]
                    [(operand ...) operands]
                    [rw-id rw]
                    [loc-id loc-op]
                    [result-list result-list-code])
        #'(mlir-build-op rw-id loc-id name (list operand ...) result-list))))

  ;;=======================================================================
  ;; Attribute setter
  ;;=======================================================================

  (define (generate-attr-setter attr-stx)
    (syntax-case attr-stx ()
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
