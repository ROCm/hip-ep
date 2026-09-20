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

  ;; Helper: validate function name is identifier
  (define (validate-ast-match-function-name ast-rec)
    (unless (identifier? (ast-pattern-expand-function-name ast-rec))
      (syntax-violation 'validate-ast "Function name must be an identifier"
                       (ast-pattern-expand-function-name ast-rec))))

  ;; Helper: validate root-var is identifier
  (define (validate-ast-match-root-var ast-rec)
    (unless (identifier? (ast-pattern-expand-root-var ast-rec))
      (syntax-violation 'validate-ast "Root variable must be an identifier"
                       (ast-pattern-expand-root-var ast-rec))))

  ;; Helper: normalize match field from list to vector
  (define (normalize-ast-match-to-vector ast-rec)
    (ast-pattern-expand-match-set! ast-rec
      (list->vector (ast-pattern-expand-match ast-rec))))

  ;; Helper: normalize result-var in match operations to list
  (define (normalize-match-result-vars ast-rec)
    (let ([match-vec (ast-pattern-expand-match ast-rec)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :rime-with result-var := (ast-match-expand-result-var match-op)
            :when (identifier? result-var)
            :do (ast-match-expand-result-var-set! match-op (list result-var)))))

  ;;=======================================================================
  ;; Rewrite operation validation (defined before validate-ast)
  ;;=======================================================================

  ;; Helper: validate result-var is identifier (or empty for void operations)
  (define (validate-operation-result-var op)
    (let ([result (ast-operation-expand-result-var op)])
      (unless (or (identifier? result)
                  (null? (syntax->datum result)))
        (syntax-violation 'validate-operation "Operation result must be identifier or ()"
                         result))))

  ;; Helper: validate and normalize operation op-name (string or symbol -> string)
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

  ;; Helper: validate operation regions (walk structure)
  (define (validate-operation-regions op)
    (let ([regions (ast-operation-expand-regions op)])
      (when (pair? regions)  ;; Only walk if non-empty
        (for-each validate-region regions))))

  ;; Validate a single operation and walk its structure
  (define (validate-operation op)
    (validate-operation-result-var op)
    (validate-and-normalize-operation-name op)
    (validate-operation-regions op))

  ;; Validate a region and walk its blocks
  (define (validate-region region)
    (for-each validate-block (ast-region-expand-blocks region)))

  ;; Validate a block and walk its operations
  (define (validate-block block)
    ;; Walk operations in this block
    (for-each validate-operation (ast-block-expand-operations block)))

  ;;=======================================================================
  ;; Main validation entry point
  ;;=======================================================================

  (define (validate-ast ast-rec)
    (validate-ast-match-function-name ast-rec)
    (normalize-ast-match-to-vector ast-rec)
    (normalize-match-result-vars ast-rec)
    (validate-ast-match-root-var ast-rec)

    ;; Validate match operations
    (validate-match-operations ast-rec)

    ;; Validate each rewrite operation
    (for-each validate-operation (ast-pattern-expand-rewrite ast-rec))

    ;; Validate where bindings
    (for-each (lambda (where-binding)
               (unless (identifier? (ast-where-binding-expand-var where-binding))
                 (syntax-violation 'validate-ast "Where binding variable must be an identifier"
                                  (ast-where-binding-expand-var where-binding))))
             (ast-pattern-expand-where ast-rec))

    ast-rec)

  ;; Helper: validate identifier starts with %
  (define (validate-%-identifier var context-msg)
    (unless (identifier? var)
      (syntax-violation 'validate-match-operations
        (string-append context-msg " must be identifier") var))
    (let ([var-name (symbol->string (syntax->datum var))])
      (unless (char=? (string-ref var-name 0) #\%)
        (syntax-violation 'validate-match-operations
          (string-append context-msg " must start with %") var))))

  ;; Helper: normalize op-name (string or symbol -> string)
  (define (normalize-op-name match-op)
    (let* ([op-name-stx (ast-match-expand-op-name match-op)]
           [op-name-datum (syntax->datum op-name-stx)])
      (unless (or (string? op-name-datum) (symbol? op-name-datum))
        (syntax-violation 'validate-match-operations
          "Operation name must be string or symbol" op-name-stx))
      (when (symbol? op-name-datum)
        (ast-match-expand-op-name-set! match-op
          (datum->syntax op-name-stx (symbol->string op-name-datum))))))

  ;; Hash function for identifiers using symbol-hash
  (define (identifier-hash id)
    (symbol-hash (syntax->datum id)))

  ;; Validate match operations (match field and result-vars already normalized)
  (define (validate-match-operations ast-rec)
    (let ([match-vec (ast-pattern-expand-match ast-rec)]
          [seen-results (make-hashtable identifier-hash bound-identifier=?)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (begin
                  (normalize-op-name match-op)
                  (loop :for var :in (ast-match-expand-result-var match-op)
                        :do (begin
                              (validate-%-identifier var "Result")
                              ;; Check for duplicate result variables
                              (when (hashtable-contains? seen-results var)
                                (syntax-violation 'validate-match-operations
                                  "Duplicate result variable" var))
                              (hashtable-set! seen-results var #t)))
                  (loop :for var :in (syntax->list (ast-match-expand-operands match-op))
                        :do (validate-%-identifier var "Operand"))))))

)
