#!r6rs
(library (mlir pattern-macro)
  (export define-conversion-pattern
          :match :rewrite :with :where :debug-ast :debug-matching
          = : -> :region :regions :attrs
          ast-pattern?
          ast-pattern-function-name
          ast-pattern-root-op-name
          ast-pattern-match
          ast-pattern-rewrite
          ast-pattern-where
          ast-pattern-debug-ast?
          ast-pattern-debug-matching?
          make-ast-pattern)  ;; Re-export for generated code
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
  (define-syntax :region (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :regions (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :attrs (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))

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
      ;; Mutable fields allow validation to normalize in-place
      (define-record-type (ast-match-expand make-ast-match-expand ast-match-expand?)
        (fields (mutable result-var)     ;; syntax - identifier (e.g., #'%out)
                (mutable op-name)        ;; syntax - string literal (e.g., #'"onnx.Cast")
                (mutable operands)       ;; syntax - operand list (e.g., #'(%in))
                (mutable attributes)     ;; syntax - attribute list (e.g., #'())
                (mutable input-types)    ;; syntax - input types (e.g., #'(!in-type))
                (mutable output-type)))  ;; syntax - output type (e.g., #'!out-type)

      ;; Expansion-time AST record for rewrite operations
      ;; Contains syntax objects for code generation
      ;; Field order matches MLIR generic syntax: operands, regions, attributes, types
      ;; Note: No input-types field because input types are implicit in operands
      ;;       (each operand is a Value with .getType()). Only result types are needed.
      ;; Mutable fields allow validation to normalize in-place
      (define-record-type (ast-operation-expand make-ast-operation-expand ast-operation-expand?)
        (fields (mutable result-var)     ;; syntax - identifier (e.g., #'%new)
                (mutable op-name)        ;; syntax - string literal (e.g., #'"hipsr.cast")
                (mutable operands)       ;; syntax - operand expressions
                (mutable regions)        ;; syntax - regions (before attributes in MLIR syntax)
                (mutable attributes)     ;; syntax - attribute expressions
                (mutable result-types))) ;; syntax - result type(s) for operation output

      ;; Expansion-time AST record for where bindings
      ;; Contains syntax objects for code generation
      ;; Mutable fields allow validation to normalize in-place
      (define-record-type (ast-where-binding-expand make-ast-where-binding-expand ast-where-binding-expand?)
        (fields (mutable var)            ;; syntax - identifier (e.g., #'%ctx)
                (mutable expr)))         ;; syntax - expression to compute

      ;; Expansion-time AST record for regions (mimics MLIR Region)
      ;; A region contains a list of blocks
      ;; Mutable fields allow validation to normalize in-place
      (define-record-type (ast-region-expand make-ast-region-expand ast-region-expand?)
        (fields (mutable blocks)))       ;; list of ast-block-expand

      ;; Expansion-time AST record for blocks (mimics MLIR Block)
      ;; A block has a label, arguments, and operations
      ;; Mutable fields allow validation to normalize in-place
      (define-record-type (ast-block-expand make-ast-block-expand ast-block-expand?)
        (fields (mutable label)          ;; syntax - block label (e.g., #'^bb0)
                (mutable arguments)      ;; list of (var type) pairs - block arguments
                (mutable operations)))   ;; list of ast-operation-expand

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
        (syntax-case op-stx (= : ->)
          ;; Full pattern: (result = "op.name" (operands ...) (attrs ...) : (input-types ...) -> output-type)
          [(result = op-name operands attrs : input-types -> output-type)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'attrs
             #'input-types
             #'output-type)]

          ;; No types: (result = "op.name" (operands ...) (attrs ...))
          [(result = op-name operands attrs)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'attrs
             #'()              ;; Empty input-types
             #'())]            ;; Empty output-type

          ;; No attrs, with types: (result = "op.name" (operands ...) : (input-types ...) -> output-type)
          [(result = op-name operands : input-types -> output-type)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'()              ;; Empty attrs
             #'input-types
             #'output-type)]

          ;; Minimal: (result = "op.name" (operands ...))
          [(result = op-name operands)
           (make-ast-match-expand
             #'result
             #'op-name
             #'operands
             #'()              ;; Empty attrs
             #'()              ;; Empty input-types
             #'())]            ;; Empty output-type

          [_ (syntax-violation 'parse-match-operation
               "Invalid match operation syntax (expected: result = \"op.name\" operands [attrs] [: types -> type])"
               op-stx)]))
      
      ;;=======================================================================
      ;; Parse individual rewrite operation
      ;; Keywords introduce sections that accumulate elements until next keyword
      ;;   (result-list =)? "op.name" (operands) [:regions region1 region2 ...]? [:attrs attr1 attr2 ...]? -> result-types
      ;;
      ;; Examples:
      ;;   (%r = "op" (%a %b) -> i32)                        - with operands
      ;;   (%r = "op" (%a) :attrs [a1 v1] [a2 v2] -> i32)    - with attrs (flat!)
      ;;   (%r = "op" (%a) :regions [region1] -> i32)        - with one region (flat!)
      ;;   (%r = "op" (%a) :regions [r1] [r2] -> i32)        - with two regions (flat!)
      ;;   (%r = "op" (%a) :regions [r1] :attrs [a1 v1] -> i32)  - full
      ;;   ((%r1 %r2) = "op" (%a) -> (i32 i32))              - multiple results
      ;;   ("op" (%a) -> ())                                 - no results (void)
      ;;
      ;; Keywords as section markers (not wrappers):
      ;;   :regions [r1] [r2] :attrs [a1] [a2] -> types
      ;;   ^^^^^^^^ ^^^^^^^^^ ^^^^^^ ^^^^^^^^^
      ;;   Section  Elements  Section Elements
      ;;=======================================================================
      (define (parse-rewrite-operation op-stx)
        ;; Step 1: Extract result(s), =, op-name, and rest
        (syntax-case op-stx (=)
          ;; Pattern: (result-part = op-name . rest) or ((results ...) = op-name . rest)
          [(result-part = op-name . rest)
           (parse-rewrite-rest #'result-part #'op-name #'rest)]

          ;; Pattern: (op-name . rest) - no result
          ;; Validation will check if op-name is valid string/symbol
          [(op-name . rest)
           (parse-rewrite-rest #'() #'op-name #'rest)]

          [_ (syntax-violation 'parse-rewrite-operation
               "Invalid operation syntax (expected: [result =] \"op.name\" (operands) ...)"
               op-stx)]))

      ;; Step 2: Parse the rest: (operands) [:regions (...)]? [:attrs [...]]? [-> result-types]?
      ;; Create record with operands, then mutate as we find optional parts
      (define (parse-rewrite-rest result-stx op-name-stx rest-stx)
        (syntax-case rest-stx ()
          [(operands . more)
           (let ([rec (make-ast-operation-expand
                        result-stx
                        op-name-stx
                        #'operands
                        #'()  ;; Empty regions (may be updated)
                        #'()  ;; Empty attrs (may be updated)
                        #'())]) ;; Empty result-types (may be updated)
             (parse-rewrite-optional rec #'more)
             rec)]))

      ;; Helper: parse sections :regions, :attrs, then REQUIRED -> result-types
      ;; Sections accumulate elements until next keyword
      (define (parse-rewrite-optional rec rest-stx)
        (syntax-case rest-stx (:regions :attrs ->)
          ;; :regions starts region section - accumulate until :attrs, ->, or end
          [(:regions . more)
           (parse-regions-section rec #'more '())]

          ;; :attrs starts attrs section - accumulate until -> or end
          [(:attrs . more)
           (parse-attrs-section rec #'more '())]

          ;; -> result-types (REQUIRED - must be present)
          [(-> result-types)
           (ast-operation-expand-result-types-set! rec #'result-types)]

          ;; Error - -> result-types is mandatory
          [_ (syntax-violation 'parse-rewrite-optional
               "Missing -> result-types (required for operations with results)"
               rest-stx)]))

      ;; Accumulate regions until hitting :attrs, ->, or end
      (define (parse-regions-section rec rest-stx regions-acc)
        (syntax-case rest-stx (:attrs ->)
          ;; Hit :attrs - done with regions, start attrs section
          [(:attrs . more)
           (begin
             (ast-operation-expand-regions-set! rec (reverse regions-acc))
             (parse-attrs-section rec #'more '()))]

          ;; Hit -> - done with regions, parse result types
          [(-> result-types)
           (begin
             (ast-operation-expand-regions-set! rec (reverse regions-acc))
             (ast-operation-expand-result-types-set! rec #'result-types))]

          ;; Another region - accumulate it
          [(region . more)
           (parse-regions-section rec #'more (cons #'region regions-acc))]

          ;; Error - must have -> result-types
          [_ (syntax-violation 'parse-regions-section
               "Expected region, :attrs, or -> result-types"
               rest-stx)]))

      ;; Accumulate attrs until hitting -> or end
      (define (parse-attrs-section rec rest-stx attrs-acc)
        (syntax-case rest-stx (->)
          ;; Hit -> - done with attrs, parse result types
          [(-> result-types)
           (begin
             (ast-operation-expand-attributes-set! rec (reverse attrs-acc))
             (ast-operation-expand-result-types-set! rec #'result-types))]

          ;; Another attr - accumulate it
          [(attr . more)
           (parse-attrs-section rec #'more (cons #'attr attrs-acc))]

          ;; Error - must have -> result-types
          [_ (syntax-violation 'parse-attrs-section
               "Expected attr or -> result-types"
               rest-stx)]))
      
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

        ;; Validate and normalize each rewrite operation (in-place with mutation)
        (for-each (lambda (rewrite-op)
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
                      ;; Normalize: convert symbol to string in-place
                      (unless (string? op-name-datum)
                        (ast-operation-expand-op-name-set! rewrite-op
                          (datum->syntax op-name-stx (symbol->string op-name-datum))))))
                  (ast-pattern-expand-rewrite ast-rec))

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
      ;; Helper: Convert expansion-time AST to runtime data (datums)
      ;;=======================================================================
      ;; Convert ast-match-expand to datum for runtime AST record
      (define (match-expand->datum match-exp)
        (list 'match
              (syntax->datum (ast-match-expand-result-var match-exp))
              (syntax->datum (ast-match-expand-op-name match-exp))
              (syntax->datum (ast-match-expand-operands match-exp))
              (syntax->datum (ast-match-expand-attributes match-exp))
              (syntax->datum (ast-match-expand-input-types match-exp))
              (syntax->datum (ast-match-expand-output-type match-exp))))

      ;; Convert ast-operation-expand to datum for runtime AST record
      (define (operation-expand->datum op-exp)
        (list 'rewrite
              (syntax->datum (ast-operation-expand-result-var op-exp))
              (syntax->datum (ast-operation-expand-op-name op-exp))
              (syntax->datum (ast-operation-expand-operands op-exp))
              (syntax->datum (ast-operation-expand-regions op-exp))
              (syntax->datum (ast-operation-expand-attributes op-exp))
              (syntax->datum (ast-operation-expand-result-types op-exp))))

      ;; Convert ast-where-binding-expand to datum for runtime AST record
      (define (where-binding-expand->datum where-exp)
        (list (syntax->datum (ast-where-binding-expand-var where-exp))
              (syntax->datum (ast-where-binding-expand-expr where-exp))))

      ;;=======================================================================
      ;; Phase 3: Generate code from validated AST record
      ;;=======================================================================
      (define (generate-code whole-stx ast-rec)
        (syntax-case whole-stx ()
          [(macro-name . _)
           (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                        [root-op-name (ast-pattern-expand-root-op-name ast-rec)])
             (let ([match-ops (ast-pattern-expand-match ast-rec)]
                   [rewrite-ops (ast-pattern-expand-rewrite ast-rec)]
                   [where-bindings (ast-pattern-expand-where ast-rec)]
                   [debug-ast? (ast-pattern-expand-debug-ast? ast-rec)]
                   [debug-matching? (ast-pattern-expand-debug-matching? ast-rec)])
               (if debug-ast?
                   ;; For :debug-ast mode, generate record constructor call with actual data
                   ;; Convert syntax objects to datums for runtime inspection
                   (with-syntax ([fname-sym #'(quote fname)]
                                [root-op-str (syntax->datum #'root-op-name)]
                                [match-data (datum->syntax #'macro-name
                                              (list 'quote (map match-expand->datum match-ops)))]
                                [rewrite-data (datum->syntax #'macro-name
                                                (list 'quote (map operation-expand->datum rewrite-ops)))]
                                [where-data (datum->syntax #'macro-name
                                              (list 'quote (map where-binding-expand->datum where-bindings)))]
                                [debug-ast-flag (datum->syntax #'macro-name debug-ast?)]
                                [debug-match-flag (datum->syntax #'macro-name debug-matching?)])
                     #'(define fname
                         (make-ast-pattern fname-sym
                                           root-op-str
                                           match-data
                                           rewrite-data
                                           where-data
                                           debug-ast-flag
                                           debug-match-flag)))
                   ;; For normal mode, generate lambda using syntax objects directly
                   #'(define fname
                       (lambda (op operands-ref rewriter type-converter)
                         #f)))))]))

      (let* ([ast-rec (parse-to-ast stx)]
             [validated (validate-ast ast-rec)])
        (generate-code stx validated))))

) ;; end library
