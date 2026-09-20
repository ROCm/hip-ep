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
    ;; Phase 2 validation guarantees: root variable exists as a result in some match operation
    (let* ([match-vec (ast-pattern-expand-match ast-rec)]
           [root-var (ast-pattern-expand-root-var ast-rec)]
           [root-idx (find-root-operation match-vec root-var)]
           [root-op (vector-ref match-vec root-idx)]
           [root-result-idx (find-result-index root-op root-var)])

      ;; Extract root operation name
      (ast-pattern-expand-root-op-name-set! ast-rec
        (ast-match-expand-op-name root-op))

      ;; Collect all identifiers (binding-manager)
      (let* ([binding-mgr (collect-all-identifiers match-vec)]
             [visited (make-vector (vector-length match-vec) #f)])

        ;; Create initial action: bind root variable to root operation result
        (let ([initial-action (list ':bind-root root-var root-idx root-result-idx)])

          ;; Build match actions (includes binding for all results and operands)
          (let ([actions (cons initial-action
                              (build-match-actions match-vec root-idx visited binding-mgr))])

            (ast-pattern-expand-match-bindings-set! ast-rec binding-mgr)
            (ast-pattern-expand-match-actions-set! ast-rec actions)

            ;; Warn about unvisited operations
            (warn-unvisited-operations match-vec visited))))))

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
    ;; Phase 2 validation guarantees: identifier exists, starts with %, is a result variable
    (let ([entry (find-binding-entry mgr id)])
      (binding-entry-bound?-set! entry #t)
      (list ':bind-result id
            (binding-entry-result-op-idx entry)
            (binding-entry-result-idx entry))))

  (define (action-bind-operand! mgr id op-idx operand-idx)
    ;; Bind an operand variable - updates entry in place, returns :bind-operand action
    ;; Note: operand location is already recorded during collect-all-identifiers
    ;; Phase 2 validation guarantees: identifier exists and starts with %
    (let ([entry (find-binding-entry mgr id)])
      (binding-entry-bound?-set! entry #t)
      (list ':bind-operand id
            (binding-entry-operand-op-idx entry)
            (binding-entry-operand-idx entry))))

  (define (action-check-operand-equal mgr id op-idx operand-idx)
    ;; Check operand equality - returns :check-operand-equal action
    ;; Phase 2 validation guarantees: identifier exists and starts with %
    ;; DAG traversal guarantees: identifier is already bound
    (list ':check-operand-equal id op-idx operand-idx))


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
    ;; Find which result index target-var occupies in match-op
    (let ([result-vars (ast-match-expand-result-var match-op)])
      (loop :for var :in result-vars
            :for idx :from 0
            :when (bound-identifier=? var target-var)
            :break idx
            :finally #f)))

  (define (find-operation-by-result match-vec result-var)
    (or (loop :for op-idx :from 0 :below (vector-length match-vec)
              :rime-with match-op := (vector-ref match-vec op-idx)
              :rime-with result-vars := (ast-match-expand-result-var match-op)
              :break op-idx :if (loop :for var :in result-vars
                                      :break #t :if (bound-identifier=? var result-var)))
        #f)))
