#!r6rs
(library (mlir pattern-validate)
  (export validate-ast)
  (import (rnrs)
          (for (only (chezscheme) syntax->list) expand)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (mlir pattern-ast) expand))

  ;;=======================================================================
  ;; Phase 2: Validation - validate syntax and semantics, normalize fields
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; Main entry point
  ;;-----------------------------------------------------------------------

  (define (validate-ast ast-rec)
    ;; Phase 2 validates and normalizes the AST from Phase 1 (parse).
    ;; Order matters: normalization must happen before validation that depends on it.

    ;; Rule: Function name must be an identifier
    (validate-ast-match-function-name ast-rec)

    ;; Rule: Root variable must be an identifier (checked before normalization)
    (validate-root-var-is-identifier ast-rec)

    ;; Normalization: match field from list to vector (enables indexed access in Phase 3)
    (normalize-ast-match-to-vector ast-rec)

    ;; Normalization: result-var from single identifier to list (uniform representation)
    (normalize-match-result-vars ast-rec)

    ;; Normalization: op-name from symbol to string (canonical form)
    (normalize-match-op-names ast-rec)

    ;; Rule: All identifiers in match operations must start with %
    ;; Rule: Result variables must be unique across all operations (FATAL if violated)
    ;; Rule: :where guards must be valid syntax objects (not validated for correctness)
    (validate-match-operations ast-rec)

    ;; Rule: Root variable must appear in results (after normalization)
    ;; Cache: root-op-index, root-result-idx, root-op-name for codegen
    (validate-and-cache-root-var ast-rec)

    ;; TODO: Document specific validation rules for rewrite operations
    (for-each validate-operation (ast-pattern-expand-rewrite ast-rec))

    ;; Rule: Where binding variables must be identifiers
    (validate-where-bindings ast-rec)

    ast-rec)

  ;;-----------------------------------------------------------------------
  ;; Top-level AST field validation
  ;;-----------------------------------------------------------------------

  (define (validate-ast-match-function-name ast-rec)
    (unless (identifier? (ast-pattern-expand-function-name ast-rec))
      (syntax-violation 'validate-ast "Function name must be an identifier"
                       (ast-pattern-expand-function-name ast-rec))))

  (define (validate-root-var-is-identifier ast-rec)
    (unless (identifier? (ast-pattern-expand-root-var ast-rec))
      (syntax-violation 'validate-ast "Root variable must be an identifier"
                       (ast-pattern-expand-root-var ast-rec))))

  ;;-----------------------------------------------------------------------
  ;; Normalization
  ;;-----------------------------------------------------------------------

  (define (normalize-ast-match-to-vector ast-rec)
    (ast-pattern-expand-match-set! ast-rec
      (list->vector (ast-pattern-expand-match ast-rec))))

  (define (normalize-match-result-vars ast-rec)
    (let ([match-vec (ast-pattern-expand-match ast-rec)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :rime-with result-var := (ast-match-expand-result-var match-op)
            :do (cond
                  ;; Single identifier: %r → (#'%r)
                  [(identifier? result-var)
                   (ast-match-expand-result-var-set! match-op (list result-var))]

                  ;; List syntax: #'(%a %b) → (#'%a #'%b)
                  ;; Also handles future variadic: #'(%a ...) and dotted: #'(%a . %rest)
                  [else
                   (ast-match-expand-result-var-set! match-op (syntax->list result-var))]))))

  (define (normalize-match-op-names ast-rec)
    (let ([match-vec (ast-pattern-expand-match ast-rec)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (normalize-op-name match-op))))

  ;;-----------------------------------------------------------------------
  ;; Match operations validation
  ;;-----------------------------------------------------------------------

  (define (validate-match-operations ast-rec)
    (validate-match-identifiers-start-with-% ast-rec)
    (validate-no-duplicate-result-variables ast-rec)
    (validate-where-guards ast-rec))

  (define (validate-match-identifiers-start-with-% ast-rec)
    (let ([match-vec (ast-pattern-expand-match ast-rec)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (begin
                  ;; Validate result variables start with %
                  (loop :for var :in (ast-match-expand-result-var match-op)
                        :do (validate-%-identifier var "Result"))
                  ;; Validate operand variables start with %
                  ;; Operands are ast-operand records (parsed and flattened in phase 1)
                  (loop :for operand :in (ast-match-expand-operands match-op)
                        :do (validate-%-identifier (ast-operand-var operand) "Operand"))))))

  (define (validate-no-duplicate-result-variables ast-rec)
    (let ([match-vec (ast-pattern-expand-match ast-rec)]
          [seen-results (make-hashtable identifier-hash bound-identifier=?)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (loop :for var :in (ast-match-expand-result-var match-op)
                      :do (begin
                            (when (hashtable-contains? seen-results var)
                              (syntax-violation 'validate-no-duplicate-result-variables
                                "Duplicate result variable" var))
                            (hashtable-set! seen-results var #t))))))

  (define (validate-where-guards ast-rec)
    ;; Validate :where guards in match operations
    ;; Guards are arbitrary Scheme expressions stored as syntax objects
    ;; Actual correctness validation happens at Scheme expansion time
    ;; We just check the field is properly set (syntax object or #f)
    (let ([match-vec (ast-pattern-expand-match ast-rec)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (if #f #f))))  ;; No validation needed - parser ensures correct type

  (define (validate-and-cache-root-var ast-rec)
    ;; Rule: Root variable must appear as a result in at least one match operation
    ;; Cache: Store root-op-index, root-result-idx, root-op-name for codegen phase
    ;; Must be called AFTER normalization (needs match vector and normalized result-vars)
    (let ([match-vec (ast-pattern-expand-match ast-rec)]
          [root-var (ast-pattern-expand-root-var ast-rec)])
      (let ([found (loop :initially := #f
                         :for op-idx :from 0 :below (vector-length match-vec)
                         :rime-with match-op := (vector-ref match-vec op-idx)
                         :rime-with result-vars := (ast-match-expand-result-var match-op)
                         :rime-with result-idx := (loop :initially := #f
                                                         :for var :in result-vars
                                                         :for idx :from 0
                                                         :when (bound-identifier=? var root-var)
                                                         :break idx)
                         :when result-idx
                         :break (cons op-idx result-idx))])
        (if found
            (begin
              ;; Cache indices for codegen
              (ast-pattern-expand-root-op-index-set! ast-rec (car found))
              (ast-pattern-expand-root-result-idx-set! ast-rec (cdr found))
              ;; Also cache root-op-name for convenience
              (let ([root-op (vector-ref match-vec (car found))])
                (ast-pattern-expand-root-op-name-set! ast-rec
                  (ast-match-expand-op-name root-op))))
            (syntax-violation 'validate-and-cache-root-var
              "Root variable not found in any match operation result" root-var)))))

  ;;-----------------------------------------------------------------------
  ;; Rewrite operation validation
  ;;-----------------------------------------------------------------------

  (define (validate-operation op)
    (validate-operation-result-var op)
    (validate-and-normalize-operation-name op)
    (validate-operation-regions op))

  (define (validate-operation-result-var op)
    (let ([result (ast-operation-expand-result-var op)])
      (unless (or (identifier? result)
                  (null? (syntax->datum result)))
        (syntax-violation 'validate-operation "Operation result must be identifier or ()"
                         result))))

  (define (validate-and-normalize-operation-name op)
    (let* ([op-name-stx (ast-operation-expand-op-name op)]
           [op-name-datum (syntax->datum op-name-stx)])
      (unless (or (string? op-name-datum) (symbol? op-name-datum))
        (syntax-violation 'validate-operation "Operation name must be string or symbol"
                         op-name-stx))
      ;; Normalize: convert symbol to string in-place
      (unless (string? op-name-datum)
        (ast-operation-expand-op-name-set! op
          (datum->syntax op-name-stx (symbol->string op-name-datum))))))

  (define (validate-operation-regions op)
    (let ([regions (ast-operation-expand-regions op)])
      (when (pair? regions)
        (for-each validate-region regions))))

  (define (validate-region region)
    (for-each validate-block (ast-region-expand-blocks region)))

  (define (validate-block block)
    (for-each validate-operation (ast-block-expand-operations block)))

  ;;-----------------------------------------------------------------------
  ;; Where binding validation
  ;;-----------------------------------------------------------------------

  (define (validate-where-bindings ast-rec)
    (for-each (lambda (where-binding)
                (unless (identifier? (ast-where-binding-expand-var where-binding))
                  (syntax-violation 'validate-ast "Where binding variable must be an identifier"
                                   (ast-where-binding-expand-var where-binding))))
              (ast-pattern-expand-where ast-rec)))

  ;;-----------------------------------------------------------------------
  ;; Utilities
  ;;-----------------------------------------------------------------------

  (define (identifier-hash id)
    (symbol-hash (syntax->datum id)))

  (define (validate-%-identifier var context-msg)
    (unless (identifier? var)
      (syntax-violation 'validate-match-operations
        (string-append context-msg " must be identifier") var))
    (let ([var-name (symbol->string (syntax->datum var))])
      (unless (char=? (string-ref var-name 0) #\%)
        (syntax-violation 'validate-match-operations
          (string-append context-msg " must start with %") var))))

  (define (normalize-op-name match-op)
    (let* ([op-name-stx (ast-match-expand-op-name match-op)]
           [op-name-datum (syntax->datum op-name-stx)])
      (unless (or (string? op-name-datum) (symbol? op-name-datum))
        (syntax-violation 'validate-match-operations
          "Operation name must be string or symbol" op-name-stx))
      (when (symbol? op-name-datum)
        (ast-match-expand-op-name-set! match-op
          (datum->syntax op-name-stx (symbol->string op-name-datum)))))))
