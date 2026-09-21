#!r6rs
;;=======================================================================
;; Pattern Parser - Phase 1
;;=======================================================================
;;
;; Parses pattern DSL syntax into AST records (defined in pattern-ast.sls).
;; This is phase 1 of the 4-phase macro expansion pipeline.
;;
;; INPUT:  Raw syntax from define-conversion-pattern macro
;; OUTPUT: ast-pattern-expand record with parsed structure
;;
;; Key responsibilities:
;; - Pattern match syntax structure and extract components
;; - Create AST records with syntax objects (NOT datums)
;; - Handle optional clauses (attributes, types, regions, where)
;; - Parse debug flags (:debug-parse, :debug-analyze, etc.)
;; - Validate basic syntax structure (guards in syntax-case)
;;
;; Does NOT:
;; - Validate semantics (that's phase 2: pattern-validate.sls)
;; - Build bindings or actions (that's phase 3: pattern-analyze.sls)
;; - Generate code (that's phase 4: pattern-codegen.sls)
;;
;; Strategy:
;; - Tail-recursive parsing with accumulator AST record
;; - Mutable AST fields allow incremental construction
;; - Keywords imported at expansion time (for syntax-case matching)
;;
;;=======================================================================

(library (mlir pattern-parse)
  (export parse-to-ast)
  (import (except (rnrs) =)
          (for (only (chezscheme) syntax->list) expand)
          (for (mlir pattern-keywords) expand)  ;; KEY: Import keywords at expansion time!
          (for (mlir pattern-ast) expand))

  ;;=======================================================================
  ;; Main entry point
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; parse-to-ast - Entry point for parsing
  ;;-----------------------------------------------------------------------
  ;;
  ;; INPUT:  whole-stx - entire syntax from (define-conversion-pattern ...)
  ;; OUTPUT: ast-pattern-expand record
  ;;
  ;; Strips the macro name and delegates to parse-rest for actual parsing.
  ;;
  (define (parse-to-ast whole-stx)
    (syntax-case whole-stx ()
      [(_ . rest)
       ;; Create empty AST record and parse incrementally
       (parse-rest #'rest (make-ast-pattern-expand #f #f #f '() #f #f '() '() #f #f #f #f))]))

  ;;-----------------------------------------------------------------------
  ;; parse-rest - Tail-recursive parser with accumulator AST
  ;;-----------------------------------------------------------------------
  ;;
  ;; Parses fname and debug flags (in any order), then the main pattern.
  ;; Uses tail recursion to handle optional debug flags.
  ;;
  ;; Pattern syntax:
  ;;   function-name [:debug-flags]* OR [:debug-flags]* function-name
  ;;   :match (match-operations...)
  ;;   :rewrite root-var :with (rewrite-operations...)
  ;;   [:where ((var expr)...)]?
  ;;
  ;; Strategy: Parse fname and debug flags until we hit :match, then parse main structure.
  ;;
  (define (parse-rest rest ast)
    (syntax-case rest (:debug-parse :debug-analyze :debug-codegen :debug-matching :match :rewrite :with :where)
      ;; Debug flags - continue parsing
      [(:debug-parse . more)
       (begin
         (ast-pattern-expand-debug-parse?-set! ast #t)
         (parse-rest #'more ast))]

      [(:debug-analyze . more)
       (begin
         (ast-pattern-expand-debug-analyze?-set! ast #t)
         (parse-rest #'more ast))]

      [(:debug-codegen . more)
       (begin
         (ast-pattern-expand-debug-codegen?-set! ast #t)
         (parse-rest #'more ast))]

      [(:debug-matching . more)
       (begin
         (ast-pattern-expand-debug-matching?-set! ast #t)
         (parse-rest #'more ast))]

      ;; Function name (if not already set) - continue parsing
      [(fname . more)
       (and (identifier? #'fname)
            (not (ast-pattern-expand-function-name ast)))
       (begin
         (ast-pattern-expand-function-name-set! ast #'fname)
         (parse-rest #'more ast))]

      ;; Main pattern structure - fname must be set by now
      [(:match (match-op ...)
        :rewrite root :with (rewrite-op ...)
        . where-rest)
       (and (ast-pattern-expand-function-name ast)  ; fname already parsed
            (identifier? #'root)
            (not (null? (syntax->list #'(match-op ...))))
            (not (null? (syntax->list #'(rewrite-op ...)))))
       (begin
         ;; Store syntax objects, not datums
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

  ;;-----------------------------------------------------------------------
  ;; parse-match-operation - Parse one match operation
  ;;-----------------------------------------------------------------------
  ;;
  ;; Naming convention (enforced by validation, not parser):
  ;;   - Result variables: identifiers starting with % (e.g., %a, %out, %result)
  ;;   - Operand variables: identifiers starting with % (e.g., %x, %input)
  ;;   - Type variables: identifiers starting with ! (e.g., !t1, !f32)
  ;;   - Operation names: string literals OR symbols
  ;;       Parser accepts: "onnx.Add" or onnx.Add
  ;;       Validation normalizes symbols to strings: 'onnx.Add → "onnx.Add"
  ;;
  ;; Result syntax (all supported):
  ;;   - Single result:     %r = "op" (...)  or  %r = op (...)
  ;;   - Multiple results:  (%a %b) = "op" (...)
  ;;   - Variadic (future): (%a ...) = "op" (...)      [not validated yet]
  ;;   - Dotted (future):   (%a %b . %rest) = "op" (...) [not validated yet]
  ;;
  ;; Operand syntax:
  ;;   - Variables: (%x %y) - identifiers starting with %
  ;;   - Can be result variables from other operations or free variables
  ;;
  ;; Attributes and types are optional (4 variants for each result form):
  ;;   1. Full:     result = "op" (operands) (attrs) : (types) -> type
  ;;   2. No attrs: result = "op" (operands) : (types) -> type
  ;;   3. No types: result = "op" (operands) (attrs)
  ;;   4. Minimal:  result = "op" (operands)
  ;;
  ;; Parser accepts result/operands as-is, validation checks naming conventions.
  ;;
  ;; Returns ast-match-expand record with syntax objects.
  ;;
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

  ;;-----------------------------------------------------------------------
  ;; parse-rewrite-operation - Parse one rewrite operation
  ;;-----------------------------------------------------------------------
  ;;
  ;; Two-step parsing strategy:
  ;;   Step 1: Extract result(s), op-name (this function)
  ;;   Step 2: Parse rest: operands, optional sections (parse-rewrite-rest)
  ;;
  ;; Handles operations with/without results:
  ;;   With result:  (%out = "op" (operands) -> !type)
  ;;   Without:      ("op" (operands) -> ())
  ;;
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

  ;;-----------------------------------------------------------------------
  ;; parse-rewrite-rest - Parse operation body after result and op-name
  ;;-----------------------------------------------------------------------
  ;;
  ;; Required: (operands)
  ;; Optional: :regions (...) :attrs [...] -> result-types
  ;;
  ;; The -> result-types is REQUIRED for operations with results.
  ;; Delegates to parse-rewrite-optional to handle optional sections.
  ;;
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

  ;;-----------------------------------------------------------------------
  ;; parse-rewrite-optional - Parse optional sections after operands
  ;;-----------------------------------------------------------------------
  ;;
  ;; Handles three optional sections in this order:
  ;;   :regions (region...)  - one or more regions
  ;;   :attrs [attr...]      - attribute list
  ;;   -> result-types       - REQUIRED if operation has results
  ;;
  ;; Uses accumulator pattern for :regions and :attrs sections
  ;; (parse-regions-section and parse-attrs-section).
  ;;
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

  ;;-----------------------------------------------------------------------
  ;; parse-regions-section - Accumulate regions
  ;;-----------------------------------------------------------------------
  ;;
  ;; Tail-recursive accumulator pattern. Collects regions until hitting:
  ;;   - :attrs keyword (start attrs section)
  ;;   - -> keyword (parse result types, end)
  ;;
  ;; Regions are accumulated in reverse order (cons), then reversed when done.
  ;;
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

      ;; Another region - parse it and accumulate
      [(region . more)
       (let ([region-rec (parse-region #'region)])
         (parse-regions-section rec #'more (cons region-rec regions-acc)))]

      ;; Error - must have -> result-types
      [_ (syntax-violation 'parse-regions-section
           "Expected region, :attrs, or -> result-types"
           rest-stx)]))

  ;;-----------------------------------------------------------------------
  ;; parse-region - Parse a single region (list of blocks)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Regions are wrapped in parens:
  ;;   :regions ((block1 block2) (block3))
  ;;            ^^^^^^^^^^^^^^^^  ^^^^^^^^
  ;;             region 1        region 2
  ;;
  (define (parse-region region-stx)
    (syntax-case region-stx ()
      [(block ...)
       (let ([blocks (map parse-block (syntax->list #'(block ...)))])
         (make-ast-region-expand blocks))]
      [_ (syntax-violation 'parse-region
           "Invalid region syntax (expected: list of blocks)"
           region-stx)]))

  ;;-----------------------------------------------------------------------
  ;; parse-block - Parse a single block
  ;;-----------------------------------------------------------------------
  ;;
  ;; Block syntax:
  ;;   (^label ((%arg : type)...) operations...)
  ;;
  ;; Example:
  ;;   (^bb0 ((%x : !t1) (%y : !t2))
  ;;     (%sum = "arith.addi" (%x %y) -> !t1)
  ;;     ("scf.yield" (%sum) -> ()))
  ;;
  (define (parse-block block-stx)
    (syntax-case block-stx (:)
      ;; ((%var : type) ...) matches zero or more arguments
      [(label ((%var : type) ...) operation ...)
       (let ([args (syntax->list #'((%var type) ...))]
             [ops (map parse-rewrite-operation (syntax->list #'(operation ...)))])
         (make-ast-block-expand #'label args ops))]

      [_ (syntax-violation 'parse-block
           "Invalid block syntax (expected: (^label ((%var : type) ...) operations...))"
           block-stx)]))

  ;;-----------------------------------------------------------------------
  ;; parse-attrs-section - Accumulate attributes
  ;;-----------------------------------------------------------------------
  ;;
  ;; Tail-recursive accumulator pattern. Collects attributes until hitting:
  ;;   - -> keyword (parse result types, end)
  ;;
  ;; Attributes are accumulated in reverse order (cons), then reversed when done.
  ;;
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

  ;;-----------------------------------------------------------------------
  ;; parse-where - Parse optional :where clause
  ;;-----------------------------------------------------------------------
  ;;
  ;; Where clause syntax:
  ;;   :where ((var1 expr1) (var2 expr2) ...)
  ;;
  ;; Each binding is (var expr) where expr is arbitrary Scheme code.
  ;; Used to compute additional values for the rewrite.
  ;;
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
)
