#!r6rs
(library (mlir pattern-validate)
  (export validate-ast)
  (import (rnrs)
          (for (only (chezscheme) syntax->list) expand)
          (for (rename (rime loop) (:with :rime-with)) expand)
          (for (mlir pattern-ast) expand))

  ;;=======================================================================
  ;; Phase 2: Validate AST record and normalize fields
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

  (define (validate-ast ast-rec)
    (validate-ast-match-function-name ast-rec)
    (normalize-ast-match-to-vector ast-rec)
    (validate-ast-match-root-var ast-rec)

    ;; Validate and normalize match operations
    (validate-and-analyze-pattern-matching ast-rec)

    ;; Validate and normalize each rewrite operation
    (for-each validate-operation (ast-pattern-expand-rewrite ast-rec))

    ;; Validate where bindings
    (for-each (lambda (where-binding)
               (unless (identifier? (ast-where-binding-expand-var where-binding))
                 (syntax-violation 'validate-ast "Where binding variable must be an identifier"
                                  (ast-where-binding-expand-var where-binding))))
             (ast-pattern-expand-where ast-rec))

    ast-rec)

  ;; Validate and analyze match operations (coordinator)
  (define (validate-and-analyze-pattern-matching ast-rec)
    (validate-match-operations ast-rec)
    (analyze-match-operations ast-rec))

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

  ;; Validate match operations (match field is already a vector)
  (define (validate-match-operations ast-rec)
    (let ([match-vec (ast-pattern-expand-match ast-rec)])
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :do (normalize-op-name match-op)
            (:loop :for var :in (let ([rv (ast-match-expand-result-var match-op)])
                                  (if (identifier? rv) (list rv) (syntax->list rv)))
                   :do (validate-%-identifier var "Result"))
            (:loop :for var :in (syntax->list (ast-match-expand-operands match-op))
                   :do (validate-%-identifier var "Operand")))))

  ;; Analyze match operations - build match-bindings and match-actions
  (define (analyze-match-operations ast-rec)
    ;; Match field is already a vector (normalized earlier)
    (let* ([match-vec (ast-pattern-expand-match ast-rec)]
           [bindings (make-eq-hashtable)]
           [actions '()])

      ;; Walk through each match operation
      (loop :for op-idx :from 0 :below (vector-length match-vec)
            :rime-with match-op := (vector-ref match-vec op-idx)
            :rime-with op-name := (syntax->datum (ast-match-expand-op-name match-op))
            :rime-with results := (let ([rv (ast-match-expand-result-var match-op)])
                                    (if (identifier? rv) (list rv) (syntax->list rv)))
            :rime-with operands := (syntax->list (ast-match-expand-operands match-op))

            ;; Create :match-operation action
            :do (set! actions (cons (list ':match-operation op-name op-idx) actions))

            ;; Process results with nested loop
            (:loop :for result-idx :from 0
                   :for var :in results
                   :do (let ([matcher (list ':match-result result-idx op-idx)])
                         (hashtable-set! bindings var matcher)
                         (set! actions (cons matcher actions))))

            ;; Process operands with nested loop
            (:loop :for operand-idx :from 0
                   :for var :in operands
                   :do (if (hashtable-ref bindings var #f)
                           ;; Reuse: create :match-exact-value
                           (set! actions (cons (list ':match-exact-value var operand-idx op-idx) actions))
                           ;; First occurrence: create :match-operand
                           (let ([matcher (list ':match-operand operand-idx op-idx)])
                             (hashtable-set! bindings var matcher)
                             (set! actions (cons matcher actions))))))

      ;; Store results (reverse actions to maintain order)
      (ast-pattern-expand-match-bindings-set! ast-rec bindings)
      (ast-pattern-expand-match-actions-set! ast-rec (reverse actions))

      ;; Find root operation and extract its name
      (let ([root-var (ast-pattern-expand-root-var ast-rec)])
        (loop :for op-idx :from 0 :below (vector-length match-vec)
              :rime-with match-op := (vector-ref match-vec op-idx)
              :rime-with result-var := (ast-match-expand-result-var match-op)
              :rime-with is-root? := (if (identifier? result-var)
                                        (free-identifier=? result-var root-var)
                                        (loop :for var :in (syntax->list result-var)
                                              :break #t :if (free-identifier=? var root-var)
                                              :finally #f))
              :do (when is-root?
                    (ast-pattern-expand-root-op-name-set! ast-rec
                      (ast-match-expand-op-name match-op)))))))

  ;; Validate a single operation and walk its structure
  (define (validate-operation op)
    ;; Validate result-var is identifier (or empty for void operations)
    (let ([result (ast-operation-expand-result-var op)])
      (unless (or (identifier? result)
                  (null? (syntax->datum result)))
        (syntax-violation 'validate-operation "Operation result must be identifier or ()"
                         result)))
    ;; Validate and normalize op-name
    (let* ([op-name-stx (ast-operation-expand-op-name op)]
           [op-name-datum (syntax->datum op-name-stx)])
      (unless (or (string? op-name-datum) (symbol? op-name-datum))
        (syntax-violation 'validate-operation "Operation name must be string or symbol"
                         op-name-stx))
      ;; Normalize: convert symbol to string in-place
      (unless (string? op-name-datum)
        (ast-operation-expand-op-name-set! op
          (datum->syntax op-name-stx (symbol->string op-name-datum)))))
    ;; Walk regions - regions is a list of ast-region-expand records
    (let ([regions (ast-operation-expand-regions op)])
      (when (pair? regions)  ;; Only walk if non-empty
        (for-each validate-region regions))))

  ;; Validate a region and walk its blocks
  (define (validate-region region)
    (for-each validate-block (ast-region-expand-blocks region)))

  ;; Validate a block and walk its operations
  (define (validate-block block)
    ;; Walk operations in this block
    (for-each validate-operation (ast-block-expand-operations block)))
)
