#!r6rs
(library (mlir pattern-codegen)
  (export generate-debug-ast
          generate-pattern-matchAndRewrite
          generate-debug-codegen
          ;; Dummy parameter names for hygiene
          op rewriter type-converter operands-ref make-unbound-value)
  (import (rnrs)
          (only (chezscheme) syntax->list syntax->datum syntax-object->datum record-rtd record-type-field-names record-accessor identifier?)
          (rename (rime loop) (:with :rime-with))
          (for (only (chezscheme) syntax->list syntax->datum record-rtd record-type-field-names record-accessor identifier?) expand)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (mlir pattern-ast) expand)
          (for (mlir pattern-analyze) expand)  ; for binding-manager-bindings
          (for (mlir ffi) expand))  ; FFI identifiers needed in generated syntax

  ;; Dummy bindings so hygiene system knows these identifiers exist
  ;; They're never actually used - just needed for quasiquote references
  (define op (if #f #f))  ; Use if to avoid constant folding issues
  (define rewriter (if #f #f))
  (define type-converter (if #f #f))
  (define operands-ref (if #f #f))
  (define (make-unbound-value) (if #f #f))  ; Sentinel for uninitialized variables

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
        ;; Keys can be symbols or syntax identifiers - handle both
        [(hashtable? obj)
         (let* ([keys (vector->list (hashtable-keys obj))]
                [sorted-keys (list-sort (lambda (a b)
                                          (let ([a-str (if (identifier? a)
                                                          (symbol->string (syntax->datum a))
                                                          (symbol->string a))]
                                                [b-str (if (identifier? b)
                                                          (symbol->string (syntax->datum b))
                                                          (symbol->string b))])
                                            (string<? a-str b-str)))
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
      ;; Use syntax-object->datum which properly handles free identifiers
      ;; (doesn't try to evaluate them at expansion time)
      (let ([code-datum (syntax-object->datum generated-code)])
        (with-syntax ([code-list (datum->syntax #'fname `',code-datum)])
          #'(define fname (lambda () code-list))))))

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
           [pattern-type (ast-pattern-expand-pattern-type ast-rec)]
           [rewrite-ops (ast-pattern-expand-rewrite ast-rec)]
           [where-bindings (ast-pattern-expand-where ast-rec)]

           ;; Find root operation to get all its result variables
           [root-op (find-root-op match-vec root-op-name)]
           [root-result-vars (ast-match-expand-result-var root-op)]

           ;; Generate root initialization code (bind all result variables)
           [root-inits (generate-root-inits root-result-vars)]

           ;; Generate rewrite bindings (returns list of (var . binding) pairs)
           [rewrite-pairs (generate-rewrite-bindings rewrite-ops)]

           ;; Collect all variables that need initialization
           [match-vars (collect-all-variables binding-mgr)]
           [where-vars (map ast-where-binding-expand-var where-bindings)]
           [rewrite-vars (map car rewrite-pairs)]  ; Extract vars from pairs
           [all-vars (append match-vars where-vars rewrite-vars)]

           ;; Generate check code from actions
           [check-code (generate-check-code actions match-vec)]

           ;; Generate rewrite code using pre-generated pairs
           [rewrite-code (generate-rewrite-code-from-pairs rewrite-pairs pattern-type)])

      (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                    [(var ...) all-vars]
                    [num-operations num-ops]
                    [(root-init ...) root-inits]
                    [(where-binding ...) (generate-where-let-bindings where-bindings #'fname)]
                    [checks check-code]
                    [rewrite rewrite-code])
        #'(define fname
            (lambda (op operands-ref rewriter type-converter)
              (let ([var (make-unbound-value)] ...
                    [all-operations (make-vector num-operations (make-unbound-value))])
                ;; Bind all result variables of root operation
                root-init ...

                ;; Match pattern and rewrite if successful
                (if checks
                    ;; Wrap rewrite with where bindings from :then-let
                    (let* (where-binding ...)
                      rewrite)
                    #f)))))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate rewrite code
  ;;-----------------------------------------------------------------------
  ;;
  ;; For conversion patterns: generate let* bindings, then replaceOp
  ;; For rewrite patterns: generate let* bindings, then return last result
  ;;
  (define (generate-rewrite-code-from-pairs pairs pattern-type)
    (if (null? pairs)
        #'#t  ; No rewrite ops, just return #t for success
        (let* ([bindings (map cdr pairs)]  ; Extract bindings from (var . binding) pairs
               [last-var (car (car (reverse pairs)))])  ; Last var from last pair
          (if (eq? pattern-type 'conversion)
              ;; Conversion pattern: replaceOp with last result
              (with-syntax ([(binding ...) bindings]
                            [result last-var])
                #'(let* (binding ...)
                    (mlir-replace-op op result)
                    #t))
              ;; Rewrite pattern: return last result
              (with-syntax ([(binding ...) bindings]
                            [result last-var])
                #'(let* (binding ...)
                    result))))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate let* bindings for rewrite operations
  ;;-----------------------------------------------------------------------
  ;;
  ;; Each operation becomes: (result-var (mlir-create-generic-op "op.name" operands-list types-list))
  ;;
  (define (generate-rewrite-bindings rewrite-ops)
    (loop :for op-rec :in rewrite-ops
          :for idx :from 0
          :rime-with pair := (generate-one-rewrite-binding op-rec idx)
          :collect pair))

  (define (generate-one-rewrite-binding op-rec idx)
    (let* ([result-var-raw (ast-operation-expand-result-var op-rec)]
           ;; Check if empty by converting to datum and checking null
           [is-empty? (let ([datum (if (identifier? result-var-raw)
                                       result-var-raw  ; Keep identifiers as-is
                                       (syntax->datum result-var-raw))])
                        (null? datum))]
           [result-var (if is-empty?
                           (datum->syntax #'here (string->symbol (string-append "%rewrite-tmp-" (number->string idx))))
                           result-var-raw)]
           [op-name (syntax->datum (ast-operation-expand-op-name op-rec))]
           [operands (ast-operation-expand-operands op-rec)]
           [result-types (ast-operation-expand-result-types op-rec)]
           [attrs (syntax->list (ast-operation-expand-attributes op-rec))])
      (cons result-var  ; Return the var so caller knows what was generated
            (if (null? attrs)
                ;; No attributes - simple case
                (with-syntax ([var result-var]
                              [name op-name]
                              [(operand ...) operands]
                              [types result-types])
                  #'(var (let ([new-op (mlir-create-generic-op name (list operand ...) (list types))])
                           (mlir-operation-get-result new-op 0))))
                ;; Has attributes - wrap with let to set them
                (with-syntax ([var result-var]
                              [name op-name]
                              [(operand ...) operands]
                              [types result-types]
                              [(attr-setter ...) (map generate-attr-setter attrs)])
                  #'(var (let ([new-op (mlir-create-generic-op name (list operand ...) (list types))])
                           attr-setter ...
                           (mlir-operation-get-result new-op 0))))))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate attribute setter call
  ;;-----------------------------------------------------------------------
  ;;
  ;; Attribute syntax: (attr-name value)
  ;; Generates: (mlir-operation-set-attr new-op "attr-name" value)
  ;;
  (define (generate-attr-setter attr-stx)
    (syntax-case attr-stx ()
      [(attr-name value)
       (with-syntax ([name-str (symbol->string (syntax->datum #'attr-name))])
         #'(mlir-operation-set-attr new-op name-str value))]))

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate where bindings as let* bindings (:then-let)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Each where binding becomes: (var expr) for let*
  ;;
  (define (generate-where-let-bindings where-list ctx-id)
    ;; DON'T recontextualize - preserve original syntax
    ;; The trick: wrap everything in (with-syntax ...) to inject op, rewriter, etc.
    (loop :for binding-rec :in where-list
          :rime-with var := (ast-where-binding-expand-var binding-rec)
          :rime-with expr := (ast-where-binding-expand-expr binding-rec)
          :collect (list var expr)))

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

  (define (collect-where-variables where-list)
    (loop :for binding-rec :in where-list
          :collect (ast-where-binding-expand-var binding-rec)))

  (define (collect-rewrite-variables rewrite-ops)
    (loop :for op-rec :in rewrite-ops
          :rime-with result-var := (ast-operation-expand-result-var op-rec)
          :when (not (null? result-var))  ; Skip void/anonymous operations
          :collect result-var))

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

        [(:bind-argument-operand)
         (let* ([fields (cdr action)]
                [var (cdr (assq 'var fields))]
                [operand-idx (cdr (assq 'operand-idx fields))])
           #`(begin
               ;; Check bounds: operand-idx < operands-ref size
               (if (< #,operand-idx (value-array-ref-size operands-ref))
                   (let ([val (value-array-ref-at operands-ref #,operand-idx)])
                     ;; Check for nullptr (uptr is 0)
                     (and (not (zero? val))
                          (begin
                            (set! #,var val)
                            #t)))
                   #f)))]

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
