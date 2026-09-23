#!r6rs
(library (mlir pattern-codegen)
  (export generate-debug-ast
          generate-pattern-matchAndRewrite
          generate-debug-codegen)
  (import (rnrs)
          (only (chezscheme) syntax->list syntax->datum syntax-object->datum record-rtd record-type-field-names record-accessor identifier? hashtable-keys)
          (rename (rime loop) (:with :rime-with))
          (for (only (chezscheme) syntax->list syntax->datum record-rtd record-type-field-names record-accessor) expand)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (mlir pattern-ast) expand)
          (for (mlir pattern-analyze) expand))  ; for binding-manager-bindings

  ;;=======================================================================
  ;; Phase 4: Code generation - generate lambda from analyzed AST
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; Generic record to alist conversion
  ;;-----------------------------------------------------------------------

  (define (record->alist obj)
    "Convert a record or any value to an alist. Recursively handles nested records.
     Uses syntax-object->datum to handle all syntax objects (identifiers and forms).
     Converts hashtables to alists."
    (let ([datum (syntax-object->datum obj)])
      (cond
        ;; If syntax-object->datum converted something, use the datum
        [(not (eq? datum obj))
         datum]
        ;; Hashtable: convert to alist with sorted keys
        [(hashtable? obj)
         (let* ([keys (vector->list (hashtable-keys obj))]
                [sorted-keys (list-sort (lambda (a b)
                                          (string<? (symbol->string a)
                                                   (symbol->string b)))
                                        keys)])
           (loop :for key :in sorted-keys
                 :collect (cons (record->alist key)
                               (record->alist (hashtable-ref obj key #f)))))]
        ;; Otherwise check if it's a record
        [(record? obj)
         (let* ([rtd (record-rtd obj)]
                [field-names (vector->list (record-type-field-names rtd))])
           (loop :for name :in field-names
                 :for i :from 0
                 :collect (let* ([accessor (record-accessor rtd i)]
                                 [value (accessor obj)])
                            (cons name (record->alist value)))))]
        ;; Lists and vectors
        [(list? obj)
         (map record->alist obj)]
        [(vector? obj)
         (vector->list (vector-map record->alist obj))]
        ;; Anything else
        [else obj])))

  ;;-----------------------------------------------------------------------
  ;; Main entry points (called from pattern-macro.sls waterfall)
  ;;-----------------------------------------------------------------------
  ;; generate-debug-ast: returns lambda that returns AST as alist
  ;; generate-pattern-matchAndRewrite: returns actual pattern matching code
  ;; generate-debug-codegen: returns lambda that returns generated code as datum

  ;;-----------------------------------------------------------------------
  ;; Debug mode: AST output (parse phase)
  ;;-----------------------------------------------------------------------

  (define (generate-debug-ast ast-rec)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)])
      (let ([alist-data (record->alist ast-rec)])
        (with-syntax ([ast-list (datum->syntax #'fname `',alist-data)])
          #'(define fname (lambda () ast-list))))))

  ;;-----------------------------------------------------------------------
  ;; Debug mode: Actions output (analyze phase)
  ;;-----------------------------------------------------------------------

  (define (generate-debug-actions ast-rec)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)])
      (let ([alist-data (record->alist ast-rec)])
        (with-syntax ([ast-list (datum->syntax #'fname `',alist-data)])
          #'(define fname (lambda () ast-list))))))

  ;;-----------------------------------------------------------------------
  ;; Debug mode: Codegen output (codegen phase)
  ;;-----------------------------------------------------------------------

  (define (generate-debug-codegen ast-rec generated-code)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)])
      (let ([code-datum (syntax->datum generated-code)])
        #`(define fname (lambda () '#,code-datum)))))

  ;;-----------------------------------------------------------------------
  ;; Pattern matcher generation
  ;;-----------------------------------------------------------------------

  (define (generate-pattern-matchAndRewrite ast-rec)
    (let* ([match-vec (ast-pattern-expand-match ast-rec)]
           [binding-mgr (ast-pattern-expand-match-bindings ast-rec)]
           [actions (ast-pattern-expand-match-actions ast-rec)]
           [root-var (ast-pattern-expand-root-var ast-rec)]
           [root-op-name (ast-pattern-expand-root-op-name ast-rec)]
           [num-ops (vector-length match-vec)]

           ;; Find root operation to get all its result variables
           [root-op (find-root-op match-vec root-op-name)]
           [root-result-vars (ast-match-expand-result-var root-op)]

           ;; Generate root initialization code (bind all result variables)
           [root-inits (generate-root-inits root-result-vars)]

           ;; Collect all variables from binding manager
           [all-vars (collect-all-variables binding-mgr)]

           ;; Generate check code from actions
           [check-code (generate-check-code actions match-vec)])

      (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                    [(var ...) all-vars]
                    [num-operations num-ops]
                    [(root-init ...) root-inits]
                    [checks check-code])
        #'(define fname
            (lambda (op operands-ref rewriter type-converter)
              (let ([var (make-unbound-value)] ...
                    [all-operations (make-vector num-operations (make-unbound-value))])
                ;; Bind all result variables of root operation
                root-init ...

                ;; Match pattern and rewrite if successful
                (if checks
                    (error 'todo "rewrite not implemented yet")
                    #f)))))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate root initialization statements
  ;;-----------------------------------------------------------------------
  ;;
  ;; For each root result variable, generate: (set! var (mlir-operation-get-result op idx))
  ;;
  (define (generate-root-inits root-result-vars)
    (loop :for var :in root-result-vars
          :for idx :from 0
          :collect #`(set! #,var (mlir-operation-get-result op #,idx))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Find root operation by name
  ;;-----------------------------------------------------------------------

  (define (find-root-op match-vec root-op-name-stx)
    (let ([root-op-name (syntax->datum root-op-name-stx)])
      (loop :initially := #f
            :for idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec idx)
            :rime-with op-name := (syntax->datum (ast-match-expand-op-name match-op))
            :when (string=? op-name root-op-name)
            :break match-op)))

  ;;-----------------------------------------------------------------------
  ;; Helper: Collect all variables from binding manager
  ;;-----------------------------------------------------------------------

  (define (collect-all-variables binding-mgr)
    (vector->list (hashtable-keys (binding-manager-bindings binding-mgr))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate check code from actions
  ;;-----------------------------------------------------------------------

  (define (generate-check-code actions match-vec)
    (if (null? actions)
        #'#t
        (let ([checks (map (lambda (act) (action->check-code act match-vec)) actions)])
          #`(and #,@checks))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Translate single action to check code
  ;;-----------------------------------------------------------------------

  (define (action->check-code action match-vec)
    (let ([tag (car action)])
      (case tag
        [(:set-current-op)
         (let* ([fields (cdr action)]
                [op-idx (cdr (assq 'op-idx fields))]
                [var (cdr (assq 'var fields))])
           ;; Navigate: get defining operation of the value and store in all-operations
           ;; Check for nullptr (block arguments have no defining op)
           #`(let ([def-op (mlir-value-get-defining-op #,var)])
               (and def-op
                    (begin
                      (vector-set! all-operations #,op-idx def-op)
                      #t))))]

        [(:check-op)
         (let* ([fields (cdr action)]
                [op-idx (cdr (assq 'op-idx fields))]
                [match-op (vector-ref match-vec op-idx)]
                [op-name (ast-match-expand-op-name match-op)]
                [num-results (length (ast-match-expand-result-var match-op))])
           #`(and (string=? (mlir-operation-name (vector-ref all-operations #,op-idx))
                            #,(syntax->datum op-name))
                  (= (mlir-operation-num-results (vector-ref all-operations #,op-idx))
                     #,num-results)))]

        [(:bind-operand)
         (let* ([fields (cdr action)]
                [op-idx (cdr (assq 'op-idx fields))]
                [var (cdr (assq 'var fields))]
                [operand-idx (cdr (assq 'operand-idx fields))])
           #`(begin
               (set! #,var (mlir-operation-get-operand-value
                            (vector-ref all-operations #,op-idx)
                            #,operand-idx))
               #t))]

        [(:check-eq)
         (let* ([fields (cdr action)]
                [op-idx (cdr (assq 'op-idx fields))]
                [operand-idx (cdr (assq 'operand-idx fields))]
                [var (cdr (assq 'var fields))])
           #`(value-equal? (get-operand (vector-ref all-operations #,op-idx)
                                        #,operand-idx)
                           #,var))]

        [else
         (error 'action->check-code "Unknown action type" tag)])))

  ;;-----------------------------------------------------------------------
  ;; AST to datum conversion (for debug modes)
  ;;-----------------------------------------------------------------------

  (define (match-expand->datum match-exp)
    (list 'match
          (syntax->datum (ast-match-expand-result-var match-exp))
          (syntax->datum (ast-match-expand-op-name match-exp))
          (syntax->datum (ast-match-expand-operands match-exp))
          (syntax->datum (ast-match-expand-where-expr match-exp))))

  (define (operation-expand->datum op-exp)
    (list 'rewrite
          (syntax->datum (ast-operation-expand-result-var op-exp))
          (syntax->datum (ast-operation-expand-op-name op-exp))
          (syntax->datum (ast-operation-expand-operands op-exp))
          (syntax->datum (ast-operation-expand-regions op-exp))
          (syntax->datum (ast-operation-expand-attributes op-exp))
          (syntax->datum (ast-operation-expand-result-types op-exp))))

  (define (where-binding-expand->datum where-exp)
    (list (syntax->datum (ast-where-binding-expand-var where-exp))
          (syntax->datum (ast-where-binding-expand-expr where-exp))))

  (define (action->datum action)
    ;; Convert action list to datum, handling syntax objects in labeled fields
    ;; Action format: (:tag (field-name . value) ...)
    (cons (car action)  ; Keep tag as-is
          (map (lambda (field)
                 ;; field is (field-name . value)
                 (let ([field-name (car field)]
                       [field-value (cdr field)])
                   (cons field-name
                         (if (identifier? field-value)
                             (syntax->datum field-value)
                             field-value))))
               (cdr action)))))
