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
          (for (mlir pattern-analyze) expand)  ; for binding-manager-bindings
          (for (mlir ffi) expand))  ; FFI identifiers needed in generated syntax


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
    ;; FIRST: Extract user parameters - these are needed for code generation templates
    (let* ([user-params (ast-pattern-expand-parameters ast-rec)]
           [params (if (null? user-params)
                       #'(op operands-ref rewriter type-converter)
                       user-params)]
           ;; Extract individual parameters - bind them early so templates can use them
           [op (car params)]
           [operands-ref (cadr params)]
           [rewriter (caddr params)]
           [type-converter (cadddr params)])

      ;; NOW: Generate code with parameters in scope
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
             [root-inits (generate-root-inits root-result-vars op)]

             ;; Generate rewrite bindings (returns list of (var . binding) pairs)
             [rewrite-pairs (generate-rewrite-bindings rewrite-ops rewriter op)]

             ;; Collect all variables that need initialization
             [match-vars (collect-all-variables binding-mgr)]
             [where-vars (map ast-where-binding-expand-var where-bindings)]
             [rewrite-vars (map car rewrite-pairs)]  ; Extract vars from pairs
             [all-vars (append match-vars where-vars rewrite-vars)]

             ;; Generate check code from actions (pass operands-ref parameter)
             [check-code (generate-check-code actions match-vec operands-ref)]

             ;; Generate rewrite code using pre-generated pairs
             [rewrite-code (generate-rewrite-code-from-pairs rewrite-pairs pattern-type rewriter op)])

      (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                    [(param ...) params]
                    [(var ...) all-vars]
                    [num-operations num-ops]
                    [(root-init ...) root-inits]
                    [(where-binding ...) (generate-where-let-bindings where-bindings)]
                    [checks check-code]
                    [rewrite rewrite-code])
        ;; Use user-provided parameter names - they share lexical scope with :then-let
        #'(define fname
            (lambda (param ...)
              (let ([var (make-unbound-value)] ...
                    [all-operations (make-vector num-operations (make-unbound-value))])
                ;; Bind all result variables of root operation
                root-init ...

                ;; Match pattern and rewrite if successful
                (if checks
                    ;; Wrap rewrite with where bindings from :then-let
                    (let* (where-binding ...)
                      rewrite)
                    #f))))))))  ;; Extra paren for outer let* that binds parameters

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate rewrite code
  ;;-----------------------------------------------------------------------
  ;;
  ;; For conversion patterns: generate let* bindings, then replaceOp
  ;; For rewrite patterns: generate let* bindings, then return last result
  ;;
  (define (generate-rewrite-code-from-pairs pairs pattern-type rw op)
    (if (null? pairs)
        #'#t  ; No rewrite ops, just return #t for success
        (let* ([bindings (map cdr pairs)]  ; Extract bindings from (var . binding) pairs
               [last-var (car (car (reverse pairs)))])  ; Last var from last pair
          (if (eq? pattern-type 'conversion)
              ;; Conversion pattern: replaceOp with last result
              (with-syntax ([(binding ...) bindings]
                            [result last-var])
                #`(let* (binding ...)
                    (mlir-replace-op #,rw #,op result)
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
  (define (generate-rewrite-bindings rewrite-ops rw loc-op)
    (loop :for op-rec :in rewrite-ops
          :for idx :from 0
          :rime-with pair := (generate-one-rewrite-binding op-rec idx rw loc-op)
          :collect pair))

  (define (generate-one-rewrite-binding op-rec idx rw loc-op)
    (let* ([result-var-raw (ast-operation-expand-result-var op-rec)]
           [is-empty? (let ([datum (if (identifier? result-var-raw)
                                       result-var-raw
                                       (syntax->datum result-var-raw))])
                        (null? datum))]
           [result-var (if is-empty?
                           (datum->syntax #'here (string->symbol (string-append "%rewrite-tmp-" (number->string idx))))
                           result-var-raw)]
           [op-name (syntax->datum (ast-operation-expand-op-name op-rec))]
           [all-operands (syntax->list (ast-operation-expand-operands op-rec))]
           [operands (filter (lambda (operand-stx)
                               (let ([name (symbol->string (syntax->datum operand-stx))])
                                 (not (char=? (string-ref name 0) #\!))))
                             all-operands)]
           [result-types (ast-operation-expand-result-types op-rec)]
           [attrs (syntax->list (ast-operation-expand-attributes op-rec))]
           [regions-raw (ast-operation-expand-regions op-rec)]
           ;; Normalise: regions field may be '() or a list of ast-region-expand records
           [regions (cond [(null? regions-raw) '()]
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

  ;;-----------------------------------------------------------------------
  ;; Helper: Generate region emission code
  ;;-----------------------------------------------------------------------
  ;;
  ;; For each region: create block, bind args, emit nested ops.
  ;; After all regions, restores insertion point before loc-op.
  ;;
  (define (generate-region-code regions rw loc-op)
    (if (null? regions)
        #'(begin) ; nothing to emit
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
    (let* ([blocks (ast-region-expand-blocks region-rec)]
           [block-rec (car blocks)]          ; assume single block
           [block-args (ast-block-expand-arguments block-rec)]   ; list of (var type) stx pairs
           [block-ops  (ast-block-expand-operations block-rec)]  ; list of ast-operation-expand
           [arg-vars  (map car block-args)]
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
           ;; () in the -> clause means zero results; otherwise a single type identifier
           [result-list-code (if (null? (syntax->datum result-types))
                                 #''()
                                 (with-syntax ([types result-types])
                                   #'(list types)))])
      (with-syntax ([name op-name]
                    [(operand ...) operands]
                    [rw-id rw]
                    [loc-id loc-op]
                    [result-list result-list-code])
        ;; Emit at current insertion point (end of block, set by mlir-region-create-block)
        #'(mlir-build-op rw-id loc-id name (list operand ...) result-list))))

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
  (define (generate-where-let-bindings where-list)
    ;; Keep original syntax context - do NOT recontextualize
    ;; The with-syntax wrapping will handle parameter injection
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
  (define (generate-root-inits root-result-vars op-param)
    (loop :for var :in root-result-vars
          :for idx :from 0
          :collect #`(set! #,var (mlir-operation-get-result #,op-param #,idx))))

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

  (define (generate-check-code actions match-vec operands-ref)
    (if (null? actions)
        #'#t
        (let ([checks (map (lambda (act) (action->check-code act match-vec operands-ref)) actions)])
          #`(and #,@checks))))

  ;;-----------------------------------------------------------------------
  ;; Helper: Translate single action to check code
  ;;-----------------------------------------------------------------------

  (define (action->check-code action match-vec operands-ref)
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
               (if (< #,operand-idx (value-array-ref-size #,operands-ref))
                   (let ([val (value-array-ref-at #,operands-ref #,operand-idx)])
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
