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

  ;;-----------------------------------------------------------------------
  ;; Binding manager operations
  ;;-----------------------------------------------------------------------

  ;; Get bindings hashtable from manager
  (define (binding-manager-bindings mgr)
    (binding-manager-bindings-table mgr))

  ;; Find binding entry for identifier
  (define (find-binding-entry mgr id)
    (hashtable-ref (binding-manager-bindings mgr) id #f))

  ;; Check if identifier is already bound
  (define (is-binding-bound? mgr id)
    (let ([entry (find-binding-entry mgr id)])
      (and entry (binding-entry-bound? entry))))

  ;; Mark a result variable as bound - returns :bind-result action
  ;; Note: result-op-idx and result-idx are already set during collect-all-identifiers
  (define (action-bind-result! mgr id)
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

  ;; Bind an operand variable - updates entry in place, returns :bind-operand action
  ;; Note: operand location is already recorded during collect-all-identifiers
  (define (action-bind-operand! mgr id op-idx operand-idx)
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

  ;; Check operand equality - returns :check-operand-equal action
  (define (action-check-operand-equal mgr id op-idx operand-idx)
    (let ([entry (find-binding-entry mgr id)])
      (unless entry
        (syntax-violation 'action-check-operand-equal
          "Identifier not found in bindings" id))
      (unless (binding-entry-bound? entry)
        (syntax-violation 'action-check-operand-equal
          "Identifier not bound yet" id))
      ;; Return action: check that operand at (op-idx, operand-idx) equals bound value
      (list ':check-operand-equal id op-idx operand-idx)))

  ;; Verify all result variables are bound (call at end of analysis)
  (define (check-all-results-bound! mgr)
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
  ;; Helper functions for pattern analysis
  ;;-----------------------------------------------------------------------

  ;; Binding Manager Concept:
  ;;
  ;; A binding manager is a data structure that tracks all identifiers (variables)
  ;; appearing in pattern match operations. It is a hashtable mapping identifiers
  ;; to binding-entry records.
  ;;
  ;; Each binding-entry record contains:
  ;;   - id:              The identifier (syntax object), e.g., %0, %r
  ;;   - is-result?:      Boolean - does this identifier appear as a result variable?
  ;;                      #t if appears in result-var (left of =), #f if only in operands
  ;;   - result-op-idx:   If is-result?, which operation index defines it (else #f)
  ;;   - result-idx:      If is-result?, which result index within that operation (else #f)
  ;;   - operand-op-idx:  If used as operand, one occurrence's operation index (else #f)
  ;;   - operand-idx:     If used as operand, one occurrence's operand index (else #f)
  ;;   - bound?:          Boolean - has this identifier been bound during DAG traversal?
  ;;                      Initially #f, set to #t when we bind it (mutable)
  ;;
  ;; Construction (collect-all-identifiers):
  ;;   Pass 1: Collect operand occurrences (last write wins for operand location)
  ;;   Pass 2: Results overwrite (set is-result?=#t, result location)
  ;;           Duplicate results are validated earlier, guaranteed not to happen
  ;;
  ;; DAG traversal (build-match-actions):
  ;;   - For each result: bind it (mark bound?=#t)
  ;;   - For each operand:
  ;;     • If bound?=#f: bind it (mark bound?=#t)
  ;;     • If bound?=#t: generate check-equality action
  ;;
  ;; Validation (check-all-results-bound!):
  ;;   - Verify all entries with is-result?=#t also have bound?=#t
  ;;
  ;; Example: Pattern matching (%2r = "arith.add" (%r %r) ...)
  ;;
  ;;   After collect-all-identifiers:
  ;;     %2r: is-result?=#t, result-op-idx=0, result-idx=0, operand-op-idx=#f, operand-idx=#f, bound?=#f
  ;;     %r:  is-result?=#f, result-op-idx=#f, result-idx=#f, operand-op-idx=0, operand-idx=1, bound?=#f
  ;;          (last operand occurrence wins: operand[1], not operand[0])
  ;;
  ;;   During DAG traversal:
  ;;     Step 1: Bind result %2r → bound?=#t, action: (:bind-result %2r 0 0)
  ;;     Step 2: Bind operand[0] %r → bound?=#t, action: (:bind-operand %r 0 1) [uses recorded location]
  ;;     Step 3: Operand[1] %r already bound → action: (:check-operand-equal %r 0 1)
  ;;             Recognizes "double operand" pattern!

  ;; Helper: Warn about unvisited operations
  (define (warn-unvisited-operations match-vec visited)
    (loop :for op-idx :from 0 :below (vector-length match-vec)
          :rime-with match-op := (vector-ref match-vec op-idx)
          :when (not (vector-ref visited op-idx))
          :do (format #t "WARNING: Operation ~a at index ~a is not reachable from root~%"
                      (syntax->datum (ast-match-expand-op-name match-op)) op-idx)))

  ;; Find root operation index by matching root-var
  (define (find-root-operation match-vec root-var)
    (or (loop :for op-idx :from 0 :below (vector-length match-vec)
              :rime-with match-op := (vector-ref match-vec op-idx)
              :rime-with result-vars := (ast-match-expand-result-var match-op)
              :break op-idx :if (loop :for var :in result-vars
                                      :break #t :if (bound-identifier=? var root-var)))
        #f))

  ;; Find which result index a variable corresponds to in an operation
  (define (find-result-index match-op target-var)
    (let ([result-vars (ast-match-expand-result-var match-op)])
      (let loop-inner ([vars result-vars] [idx 0])
        (cond
          [(null? vars) #f]
          [(bound-identifier=? (car vars) target-var) idx]
          [else (loop-inner (cdr vars) (+ idx 1))]))))

  ;; Find operation index that produces a given result variable
  (define (find-operation-by-result match-vec result-var)
    (or (loop :for op-idx :from 0 :below (vector-length match-vec)
              :rime-with match-op := (vector-ref match-vec op-idx)
              :rime-with result-vars := (ast-match-expand-result-var match-op)
              :break op-idx :if (loop :for var :in result-vars
                                      :break #t :if (bound-identifier=? var result-var)))
        #f))

  ;; Collect all identifiers and create binding manager (two-pass)
  ;; Pass 1: collect operand occurrences (last write wins)
  ;; Pass 2: results overwrite everything
  ;; Note: duplicate result variables are validated earlier, guaranteed not to happen here
  ;; Hash function for identifiers using symbol-hash
  (define (identifier-hash id)
    (symbol-hash (syntax->datum id)))

  (define (collect-all-identifiers match-vec)
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

  ;; Build match-actions by traversing DAG from root operation
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
  ;; Main analysis entry point
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

  ;; Main entry point for analysis phase
  (define (analyze-ast ast-rec)
    (analyze-match-operations ast-rec)
    ast-rec))
