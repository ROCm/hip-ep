#!r6rs
(library (mlir pattern-macro)
  (export define-conversion-pattern
          :match :rewrite :with :where :debug-ast :debug-matching = : ->
          ast-pattern?
          ast-pattern-function-name
          ast-pattern-root-op-name
          ast-pattern-match
          ast-pattern-rewrite
          ast-pattern-where
          ast-pattern-debug-ast?
          ast-pattern-debug-matching?)
  (import (except (rnrs (6)) =)
          (for (only (chezscheme) syntax->list) expand))  ;; Import syntax->list for expansion time


  ;; Define keywords as syntax (for cross-library hygiene)
  (define-syntax :match (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :rewrite (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :with (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :where (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-ast (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-matching (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax = (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax : (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))

  ;; Runtime AST record (created when :debug-ast is used)
  ;; Contains datums (symbols, strings, lists) for inspection
  (define-record-type ast-pattern
    (fields function-name      ;; symbol - name of the pattern function
            root-op-name       ;; string - operation name for OpConversionPattern (e.g., "onnx.Cast")
            match              ;; list - match operations (at least one, raw s-expressions)
            rewrite            ;; list - rewrite operations (at least one, raw s-expressions)
            where              ;; list - where bindings (zero or more, raw s-expressions)
            debug-ast?         ;; boolean - return AST instead of compiled pattern
            debug-matching?))  ;; boolean - add debug prints during matching

  (define-syntax define-conversion-pattern
    (lambda (stx)
      
      ;; Expansion-time AST record for the whole pattern
      ;; Contains syntax objects to preserve lexical context for code generation
      (define-record-type (ast-pattern-expand make-ast-pattern-expand ast-pattern-expand?)
        (fields (mutable function-name)    ;; syntax - identifier for pattern function
                (mutable root-var)         ;; syntax - identifier for root result var (e.g., #'%out)
                (mutable root-op-name)     ;; syntax - string literal for op name (e.g., #'"onnx.Cast")
                (mutable match)            ;; list of ast-match-expand - parsed match operations
                (mutable rewrite)          ;; list of ast-operation-expand - parsed rewrite operations
                (mutable where)            ;; list of ast-where-binding-expand - parsed where bindings
                (mutable debug-ast?)       ;; boolean - whether :debug-ast flag is present
                (mutable debug-matching?)))  ;; boolean - whether :debug-matching flag is present

      ;; Expansion-time AST record for match operations
      ;; Contains syntax objects for code generation
      (define-record-type (ast-match-expand make-ast-match-expand ast-match-expand?)
        (fields result-var     ;; syntax - identifier (e.g., #'%out)
                op-name        ;; syntax - string literal (e.g., #'"onnx.Cast")
                operands       ;; syntax - operand list (e.g., #'(%in))
                attributes     ;; syntax - attribute list (e.g., #'())
                input-types    ;; syntax - input types (e.g., #'(!in-type))
                output-type))  ;; syntax - output type (e.g., #'!out-type)

      ;; Expansion-time AST record for rewrite operations
      ;; Contains syntax objects for code generation
      (define-record-type (ast-operation-expand make-ast-operation-expand ast-operation-expand?)
        (fields result-var     ;; syntax - identifier (e.g., #'%new)
                op-name        ;; syntax - string literal (e.g., #'"hipsr.cast")
                operands       ;; syntax - operand expressions
                attributes     ;; syntax - attribute expressions
                rest))         ;; syntax - remainder (types, etc.)

      ;; Expansion-time AST record for where bindings
      ;; Contains syntax objects for code generation
      (define-record-type (ast-where-binding-expand make-ast-where-binding-expand ast-where-binding-expand?)
        (fields var            ;; syntax - identifier (e.g., #'%ctx)
                expr))         ;; syntax - expression to compute

      ;;=======================================================================
      ;; Phase 1: Parse whole syntax to AST record - pure pattern matching
      ;;=======================================================================
      
      (define (parse-to-ast whole-stx)
        (syntax-case whole-stx ()
          [(_ . rest)
           (parse-rest #'rest (make-ast-pattern-expand #f #f #f '() '() '() #f #f))]))
      
      ;; Parse flags and clauses
      (define (parse-rest rest ast)
        (syntax-case rest (:debug-ast :debug-matching :match :rewrite :with :where)
          ;; Debug flags
          [(:debug-ast . more)
           (begin
             (ast-pattern-expand-debug-ast?-set! ast #t)
             (parse-rest #'more ast))]
          
          [(:debug-matching . more)
           (begin
             (ast-pattern-expand-debug-matching?-set! ast #t)
             (parse-rest #'more ast))]
          
          ;; Main pattern structure with guards
          [(fname 
            :match (match-op ...)
            :rewrite root :with (rewrite-op ...)
            . where-rest)
           (and (identifier? #'fname)
                (identifier? #'root)
                (not (null? (syntax->list #'(match-op ...))))
                (not (null? (syntax->list #'(rewrite-op ...)))))
           (begin
             ;; Store syntax objects, not datums
             (ast-pattern-expand-function-name-set! ast #'fname)
             (ast-pattern-expand-root-var-set! ast #'root)
             ;; Parse each match operation into ast-match-expand using map
             (ast-pattern-expand-match-set! ast 
               (map parse-match-operation (syntax->list #'(match-op ...))))
             ;; Parse each rewrite operation into ast-operation-expand using map
             (ast-pattern-expand-rewrite-set! ast
               (map parse-rewrite-operation (syntax->list #'(rewrite-op ...))))
             (parse-where #'where-rest ast)
             ast)]
          
          [_ (syntax-violation 'define-conversion-pattern "Invalid pattern syntax (expected fname :match (...) :rewrite root :with (...))" rest)]))
      
      ;;=======================================================================
      ;; Parse individual match operation
      ;; Input: (%out = "onnx.Cast" (%in) () : (!in-type) -> !out-type)
      ;; Output: ast-match-expand record with syntax objects
      ;;=======================================================================
      (define (parse-match-operation op-stx)
        (syntax-case op-stx ()
          ;; Pattern: (result = "op.name" (operands ...) (attrs ...) : (input-types ...) -> output-type)
          ;; WHY datum matching: Literal keywords use free-identifier=? which fails when syntax objects
          ;; cross function boundaries - the = in user's input has different lexical context than
          ;; the = in this helper's literals list, even when properly exported/imported.
          ;; Datum matching with eq? checks symbol names only, bypassing the binding context issue.
          ;; PARSE PHASE: Only check pattern structure (= : -> positions), no validation
          [(result eq op-name operands attrs colon input-types arrow output-type)
           (and (eq? (syntax->datum #'eq) '=)
                (eq? (syntax->datum #'colon) ':)
                (eq? (syntax->datum #'arrow) '->))
           (make-ast-match-expand
             #'result          ;; Store syntax object as-is
             #'op-name         ;; Store syntax object as-is (validation will check/convert)
             #'operands        ;; Store syntax object (TODO: parse structure)
             #'attrs           ;; Store syntax object (TODO: parse structure)
             #'input-types     ;; Store syntax object (TODO: parse structure)
             #'output-type)]   ;; Store syntax object

          [_ (syntax-violation 'parse-match-operation
               "Invalid match operation syntax (expected: result = \"op.name\" operands attrs : types -> type)"
               op-stx)]))
      
      ;;=======================================================================
      ;; Parse individual rewrite operation
      ;; Input: (%new = "hipsr.cast" (%ctx %in) ((attr val)) : (!t1 !t2) -> !t3)
      ;; Output: ast-operation-expand record with syntax objects
      ;;=======================================================================
      (define (parse-rewrite-operation op-stx)
        (syntax-case op-stx ()
          ;; Pattern: (result = "op.name" (operands ...) (attrs ...) rest ...)
          ;; WHY datum matching: Same reason as parse-match-operation - literal keywords fail
          ;; when syntax objects are passed between functions due to lexical context mismatch.
          ;; PARSE PHASE: Only check pattern structure (= position), no validation
          [(result eq op-name operands attrs rest ...)
           (eq? (syntax->datum #'eq) '=)
           (make-ast-operation-expand
             #'result          ;; Store syntax object as-is
             #'op-name         ;; Store syntax object as-is (validation will check/convert)
             #'operands        ;; Store syntax object (TODO: parse structure)
             #'attrs           ;; Store syntax object (TODO: parse structure)
             #'(rest ...))]    ;; Store syntax object
          
          ;; TODO: Add more patterns for variations
          
          [_ (syntax-violation 'parse-rewrite-operation 
               "Invalid rewrite operation syntax (expected: result = \"op.name\" operands attrs ...)"
               op-stx)]))
      
      ;; Helper: parse optional :where clause
      (define (parse-where rest-stx ast)
        (syntax-case rest-stx (:where)
          [(:where ((var expr) ...))
           (ast-pattern-expand-where-set! ast 
             (map (lambda (binding)
                    (syntax-case binding ()
                      [(v e)
                       (identifier? #'v)
                       (make-ast-where-binding-expand #'v #'e)]
                      [_ (syntax-violation 'parse-where "Invalid where binding (expected: (var expr))" binding)]))
                  (syntax->list #'((var expr) ...))))]
          [()
           (if #f #f)]
          [_ (syntax-violation 'define-conversion-pattern "Expected :where ((var expr) ...) or end" rest-stx)]))

      ;;=======================================================================
      ;; Phase 2: Validate AST record and normalize fields
      ;;=======================================================================
      (define (validate-ast ast-rec)
        ;; Validate function name
        (unless (identifier? (ast-pattern-expand-function-name ast-rec))
          (syntax-violation 'validate-ast "Function name must be an identifier"
                           (ast-pattern-expand-function-name ast-rec)))

        ;; Validate root-var
        (unless (identifier? (ast-pattern-expand-root-var ast-rec))
          (syntax-violation 'validate-ast "Root variable must be an identifier"
                           (ast-pattern-expand-root-var ast-rec)))

        ;; Validate and normalize each match operation
        (ast-pattern-expand-match-set! ast-rec
          (map (lambda (match-op)
                 ;; Validate result-var is identifier
                 (unless (identifier? (ast-match-expand-result-var match-op))
                   (syntax-violation 'validate-ast "Match result must be an identifier"
                                    (ast-match-expand-result-var match-op)))
                 ;; Validate and normalize op-name (string or symbol -> string)
                 (let* ([op-name-stx (ast-match-expand-op-name match-op)]
                        [op-name-datum (syntax->datum op-name-stx)])
                   (unless (or (string? op-name-datum) (symbol? op-name-datum))
                     (syntax-violation 'validate-ast "Operation name must be string or symbol"
                                      op-name-stx))
                   (let ([normalized-name (if (string? op-name-datum)
                                             op-name-stx
                                             (datum->syntax op-name-stx (symbol->string op-name-datum)))])
                     (make-ast-match-expand
                       (ast-match-expand-result-var match-op)
                       normalized-name  ;; Now guaranteed to be string syntax
                       (ast-match-expand-operands match-op)
                       (ast-match-expand-attributes match-op)
                       (ast-match-expand-input-types match-op)
                       (ast-match-expand-output-type match-op)))))
               (ast-pattern-expand-match ast-rec)))

        ;; Validate and normalize each rewrite operation
        (ast-pattern-expand-rewrite-set! ast-rec
          (map (lambda (rewrite-op)
                 ;; Validate result-var is identifier
                 (unless (identifier? (ast-operation-expand-result-var rewrite-op))
                   (syntax-violation 'validate-ast "Rewrite result must be an identifier"
                                    (ast-operation-expand-result-var rewrite-op)))
                 ;; Validate and normalize op-name
                 (let* ([op-name-stx (ast-operation-expand-op-name rewrite-op)]
                        [op-name-datum (syntax->datum op-name-stx)])
                   (unless (or (string? op-name-datum) (symbol? op-name-datum))
                     (syntax-violation 'validate-ast "Operation name must be string or symbol"
                                      op-name-stx))
                   (let ([normalized-name (if (string? op-name-datum)
                                             op-name-stx
                                             (datum->syntax op-name-stx (symbol->string op-name-datum)))])
                     (make-ast-operation-expand
                       (ast-operation-expand-result-var rewrite-op)
                       normalized-name  ;; Now guaranteed to be string syntax
                       (ast-operation-expand-operands rewrite-op)
                       (ast-operation-expand-attributes rewrite-op)
                       (ast-operation-expand-rest rewrite-op)))))
               (ast-pattern-expand-rewrite ast-rec)))

        ;; Validate where bindings
        (for-each (lambda (where-binding)
                   (unless (identifier? (ast-where-binding-expand-var where-binding))
                     (syntax-violation 'validate-ast "Where binding variable must be an identifier"
                                      (ast-where-binding-expand-var where-binding))))
                 (ast-pattern-expand-where ast-rec))

        ;; Find the match operation that produces root-var and extract its op-name
        (let ([root-var-stx (ast-pattern-expand-root-var ast-rec)]
              [match-ops (ast-pattern-expand-match ast-rec)])
          (let find-root ([ops match-ops])
            (when (pair? ops)
              (let ([match-op (car ops)])
                (if (free-identifier=? (ast-match-expand-result-var match-op) root-var-stx)
                    (ast-pattern-expand-root-op-name-set! ast-rec
                      (ast-match-expand-op-name match-op))
                    (find-root (cdr ops))))))
          ast-rec))

      ;;=======================================================================
      ;; Phase 3: Generate code from validated AST record
      ;;=======================================================================
      (define (generate-code whole-stx ast-rec)
        (syntax-case whole-stx ()
          [(macro-name . _)
           (let ([fname-stx (ast-pattern-expand-function-name ast-rec)]
                 [root-op-name-stx (ast-pattern-expand-root-op-name ast-rec)]
                 [match-ops (ast-pattern-expand-match ast-rec)]
                 [rewrite-ops (ast-pattern-expand-rewrite ast-rec)]
                 [where-bindings (ast-pattern-expand-where ast-rec)]
                 [debug-ast? (ast-pattern-expand-debug-ast? ast-rec)]
                 [debug-matching? (ast-pattern-expand-debug-matching? ast-rec)])
             (if debug-ast?
                 ;; For :debug-ast, create runtime ast-pattern record with datums
                 (with-syntax ([fname-id fname-stx]
                               [fname-datum (syntax->datum fname-stx)]
                               [root-op-name-datum (syntax->datum root-op-name-stx)]
                               [match-ops-datum (syntax->datum #'match-ops)]  ;; TODO: convert properly
                               [rewrite-ops-datum (syntax->datum #'rewrite-ops)]  ;; TODO: convert properly
                               [where-bindings-datum (syntax->datum #'where-bindings)]  ;; TODO: convert properly
                               [debug-ast-val debug-ast?]
                               [debug-matching-val debug-matching?])
                   #'(define fname-id 
                       (make-ast-pattern 'fname-datum 
                                       'root-op-name-datum 
                                       'match-ops-datum 
                                       'rewrite-ops-datum 
                                       'where-bindings-datum 
                                       'debug-ast-val 
                                       'debug-matching-val)))
                 ;; For normal mode, generate lambda using syntax objects directly
                 #`(define #,fname-stx 
                     (lambda (op operands-ref rewriter type-converter) 
                       #f))))]))

      (let* ([ast-rec (parse-to-ast stx)]
             [validated (validate-ast ast-rec)])
        (generate-code stx validated))))

) ;; end library
