#!r6rs
(library (mlir pattern-analyze)
  (export analyze-ast)
  (import (rnrs)
          (only (chezscheme) syntax->list format printf)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (mlir pattern-ast))

  ;;=======================================================================
  ;; Phase 3: Analysis - build bindings and actions for pattern matching
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; Main entry point
  ;;-----------------------------------------------------------------------

  (define (analyze-ast ast-rec)
    (analyze-match-operations ast-rec)
    ast-rec)

  ;;-----------------------------------------------------------------------
  ;; Match operations analysis
  ;;-----------------------------------------------------------------------

  (define (analyze-match-operations ast-rec)
    ;; Match field is already a vector (normalized earlier)
    (let* ([match-vec (ast-pattern-expand-match ast-rec)]
           [root-var (ast-pattern-expand-root-var ast-rec)]
           [root-idx (find-root-operation match-vec root-var)])

      (unless root-idx
        (syntax-violation 'analyze-match-operations
          "Root variable not found in any match operation result"
          root-var))

      ;; Extract root operation name
      (ast-pattern-expand-root-op-name-set! ast-rec
        (ast-match-expand-op-name (vector-ref match-vec root-idx)))

      ;; Collect all identifiers (binding-manager)
      (let* ([binding-mgr (collect-all-identifiers match-vec)]
             [visited (make-vector (vector-length match-vec) #f)]
             ;; Find which result index the root-var is
             [root-op (vector-ref match-vec root-idx)]
             [root-result-idx (find-result-index root-op root-var)])

        (unless root-result-idx
          (syntax-violation 'analyze-match-operations
            "Root variable not found in operation results" root-var))

        ;; Build match actions (includes binding for all results and operands)
        (let ([actions (build-match-actions match-vec root-idx visited binding-mgr)])

          ;; Check all result variables are bound
          (check-all-results-bound! binding-mgr)

          (ast-pattern-expand-match-bindings-set! ast-rec binding-mgr)
          (ast-pattern-expand-match-actions-set! ast-rec actions)

          ;; Warn about unvisited operations
          (warn-unvisited-operations match-vec visited)))))

  ;;-----------------------------------------------------------------------
  ;; DAG traversal and action generation
  ;;-----------------------------------------------------------------------

  (define (build-match-actions match-vec op-idx visited binding-mgr)
    (if (vector-ref visited op-idx)
        '()  ; Already visited - return early
        (begin
          ;; Mark as visited
          (vector-set! visited op-idx #t)

          (let* ([match-op (vector-ref match-vec op-idx)]
                 [actions '()])

            ;; Generate check actions for this operation
            (set! actions (cons (list ':set-current-op op-idx) actions))
            (set! actions (cons (list ':check-op-name op-idx) actions))
            (set! actions (cons (list ':check-num-results op-idx) actions))
            (set! actions (cons (list ':check-num-operands op-idx) actions))
            (set! actions (cons (list ':check-input-types op-idx) actions))
            (set! actions (cons (list ':check-output-types op-idx) actions))

            ;; Bind all result variables for this operation
            (loop :for result-var :in (ast-match-expand-result-var match-op)
                  :do (set! actions
                         (cons (action-bind-result! binding-mgr result-var)
                               actions)))

            ;; Process operands: bind or check equality
            (let ([operands (syntax->list (ast-match-expand-operands match-op))])
              (loop :for operand :in operands
                    :for operand-idx :from 0
                    :do (if (is-binding-bound? binding-mgr operand)
                            ;; Already bound - check equality
                            (set! actions
                              (cons (action-check-operand-equal binding-mgr operand op-idx operand-idx)
                                    actions))
                            ;; Not bound yet - bind it
                            (set! actions
                              (cons (action-bind-operand! binding-mgr operand op-idx operand-idx)
                                    actions))))

              ;; Recursively traverse operands (DAG edges)
              (loop :for operand :in operands
                    :rime-with operand-op-idx := (find-operation-by-result match-vec operand)
                    :when operand-op-idx
                    :do (set! actions
                           (append (build-match-actions match-vec operand-op-idx
                                                       visited binding-mgr)
                                   actions))))

            (reverse actions)))))

  ;;-----------------------------------------------------------------------
  ;; Identifier collection and binding manager construction
  ;;-----------------------------------------------------------------------

  (define (collect-all-identifiers match-vec)
    ;; Two-pass collection:
    ;; Pass 1: collect operand occurrences (last write wins)
    ;; Pass 2: results overwrite everything
    ;; Note: duplicate result variables are validated earlier, guaranteed not to happen here
    (let ([ht (make-hashtable identifier-hash bound-identifier=?)])

      ;; Pass 1: Collect operand occurrences (last write wins)
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (loop :for operand-var :in (syntax->list (ast-match-expand-operands match-op))
                      :for operand-idx :from 0
                      :do (hashtable-set! ht operand-var
                            (make-binding-entry operand-var #f #f #f op-idx operand-idx #f))))

      ;; Pass 2: Results overwrite (duplicate results validated earlier)
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (loop :for result-var :in (ast-match-expand-result-var match-op)
                      :for result-idx :from 0
                      :do (hashtable-set! ht result-var
                            (make-binding-entry result-var #t op-idx result-idx #f #f #f))))

      ;; Return binding manager
      (make-binding-manager ht)))

  ;;-----------------------------------------------------------------------
  ;; Binding records and manager
  ;;-----------------------------------------------------------------------

  ;; Record: binding-entry
  ;; Represents a single identifier binding (result or operand)
  (define-record-type binding-entry
    (fields id                    ; identifier (syntax object)
            is-result?            ; #t if result variable, #f if only operand
            result-op-idx         ; if is-result?, the operation index (else #f)
            result-idx            ; if is-result?, the result index (else #f)
            operand-op-idx        ; if operand, one operand occurrence op-idx (else #f)
            operand-idx           ; if operand, one operand occurrence index (else #f)
            (mutable bound?)))    ; #t when bound, initially #f

  ;; Record: binding-manager
  ;; Manages all identifier bindings for pattern matching
  (define-record-type binding-manager
    (fields bindings-table))      ; hashtable: identifier → binding-entry

  (define (binding-manager-bindings mgr)
    (binding-manager-bindings-table mgr))

  (define (find-binding-entry mgr id)
    (hashtable-ref (binding-manager-bindings mgr) id #f))

  (define (is-binding-bound? mgr id)
    (let ([entry (find-binding-entry mgr id)])
      (and entry (binding-entry-bound? entry))))

  ;;-----------------------------------------------------------------------
  ;; Binding actions
  ;;-----------------------------------------------------------------------

  (define (action-bind-result! mgr id)
    ;; Mark a result variable as bound - returns :bind-result action
    ;; Note: result-op-idx and result-idx are already set during collect-all-identifiers
    (let ([entry (find-binding-entry mgr id)])
      (unless entry
        (syntax-violation 'action-bind-result!
          "Identifier not found in bindings" id))
      (when (binding-entry-bound? entry)
        (syntax-violation 'action-bind-result!
          "Identifier already bound" id))
      (unless (binding-entry-is-result? entry)
        (syntax-violation 'action-bind-result!
          "Identifier is not a result variable" id))
      ;; Mark as bound
      (binding-entry-bound?-set! entry #t)
      ;; Return :bind-result action for codegen
      (list ':bind-result id
            (binding-entry-result-op-idx entry)
            (binding-entry-result-idx entry))))

  (define (action-bind-operand! mgr id op-idx operand-idx)
    ;; Bind an operand variable - updates entry in place, returns :bind-operand action
    ;; Note: operand location is already recorded during collect-all-identifiers
    (let ([entry (find-binding-entry mgr id)])
      (unless entry
        (syntax-violation 'action-bind-operand!
          "Identifier not found in bindings" id))
      (when (binding-entry-bound? entry)
        (syntax-violation 'action-bind-operand!
          "Identifier already bound" id))
      ;; Mark as bound
      (binding-entry-bound?-set! entry #t)
      ;; Return :bind-operand action using recorded location
      (list ':bind-operand id
            (binding-entry-operand-op-idx entry)
            (binding-entry-operand-idx entry))))

  (define (action-check-operand-equal mgr id op-idx operand-idx)
    ;; Check operand equality - returns :check-operand-equal action
    (let ([entry (find-binding-entry mgr id)])
      (unless entry
        (syntax-violation 'action-check-operand-equal
          "Identifier not found in bindings" id))
      (unless (binding-entry-bound? entry)
        (syntax-violation 'action-check-operand-equal
          "Identifier not bound yet" id))
      ;; Return action: check that operand at (op-idx, operand-idx) equals bound value
      (list ':check-operand-equal id op-idx operand-idx)))

  (define (check-all-results-bound! mgr)
    ;; Verify all result variables are bound (call at end of analysis)
    (let-values ([(keys values) (hashtable-entries (binding-manager-bindings mgr))])
      (vector-for-each
        (lambda (bentry)
          (when (and (binding-entry-is-result? bentry)
                     (not (binding-entry-bound? bentry)))
            (syntax-violation 'check-all-results-bound!
              "Result variable not bound during pattern matching"
              (binding-entry-id bentry))))
        values)))

  ;;-----------------------------------------------------------------------
  ;; Helper functions
  ;;-----------------------------------------------------------------------

  (define (identifier-hash id)
    (symbol-hash (syntax->datum id)))

  (define (warn-unvisited-operations match-vec visited)
    (loop :for op-idx :from 0 :below (vector-length match-vec)
          :rime-with match-op := (vector-ref match-vec op-idx)
          :when (not (vector-ref visited op-idx))
          :do (format #t "WARNING: Operation ~a at index ~a is not reachable from root~%"
                      (syntax->datum (ast-match-expand-op-name match-op)) op-idx)))

  (define (find-root-operation match-vec root-var)
    (or (loop :for op-idx :from 0 :below (vector-length match-vec)
              :rime-with match-op := (vector-ref match-vec op-idx)
              :rime-with result-vars := (ast-match-expand-result-var match-op)
              :break op-idx :if (loop :for var :in result-vars
                                      :break #t :if (bound-identifier=? var root-var)))
        #f))

  (define (find-result-index match-op target-var)
    (let ([result-vars (ast-match-expand-result-var match-op)])
      (let loop-inner ([vars result-vars] [idx 0])
        (cond
          [(null? vars) #f]
          [(bound-identifier=? (car vars) target-var) idx]
          [else (loop-inner (cdr vars) (+ idx 1))]))))

  (define (find-operation-by-result match-vec result-var)
    (or (loop :for op-idx :from 0 :below (vector-length match-vec)
              :rime-with match-op := (vector-ref match-vec op-idx)
              :rime-with result-vars := (ast-match-expand-result-var match-op)
              :break op-idx :if (loop :for var :in result-vars
                                      :break #t :if (bound-identifier=? var result-var)))
        #f)))
