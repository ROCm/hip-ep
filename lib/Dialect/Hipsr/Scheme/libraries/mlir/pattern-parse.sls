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

;;=======================================================================
;; Syntax Reference: Match Operations
;;=======================================================================
;;
;; Variables: All identifiers must start with %
;;   - Results: %a, %out, %result
;;   - Operands: %x, %input
;;
;; Operation name: string or symbol
;;   - "onnx.Add" or onnx.Add
;;
;; Operand groups:
;;   (%x %y)                           - all required
;;   (%x (&optional %y %z))            - x required, y,z optional
;;   (%x (&optional %y) %z)            - x required, y optional, z required
;;   (%x (&variadic %rest))            - x required, rest variadic
;;   (%x (&optional %y) (&variadic %w)) - combined
;;
;; Note: &optional/&variadic require AttrSizedOperandSegments trait
;;
;; Guards (:where clause):
;;   :where <scheme-expr>
;;   - Returns truthy to match, falsy to fail
;;   - Early return (executes immediately after matching operation)
;;   - Can access result variables, call FFI functions
;;   Example: :where (let ([$ks (mlir-operation-get-attribute %a "kernel_shape")])
;;                     (and $ks (is-1x1-kernel? $ks)))
;;
;; Complete syntax:
;;   result = "op" (operands) [:where <expr>]
;;
;; Type matching: NOT SUPPORTED
;;   - MLIR's DRR/PDLL also make types optional
;;   - Type verification happens in operation verifiers
;;
;;=======================================================================

(library (mlir pattern-parse)
  (export parse-to-ast)
  (import (except (rnrs) =)
          (for (only (chezscheme) syntax->list) expand)
          (for (mlir pattern-keywords) expand)
          (for (mlir pattern-ast) expand))

  ;;=======================================================================
  ;; SECTION 1: Entry Points
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; parse-to-ast - Entry point
  ;;-----------------------------------------------------------------------
  ;; Strips macro name, delegates to parse-rest.
  ;;
  (define (parse-to-ast whole-stx)
    (syntax-case whole-stx ()
      [(_ . rest)
       ;; Create empty AST record and parse incrementally
       (parse-rest #'rest (make-ast-pattern-expand #f #f #f #f #f '() #f #f '() '() #f #f #f #f))]))

  ;;-----------------------------------------------------------------------
  ;; parse-rest - Parse function name, debug flags, then dispatch to :match
  ;;-----------------------------------------------------------------------
  ;; Tail-recursive: accumulates debug flags, then calls parse-match-ops-recursive.
  ;;
  (define (parse-rest rest ast)
    (syntax-case rest (:debug-parse :debug-analyze :debug-codegen :debug-matching :match :then-let :rewrite :with)
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
      [(:match . match-rest)
       (ast-pattern-expand-function-name ast)  ; fname already parsed
       ;; Parse match operations recursively
       (parse-match-ops-recursive #'match-rest '() ast)]

  ;;=======================================================================
  ;; SECTION 2: Recursive Collectors (High-Level Parsing)
  ;;=======================================================================
  ;;
  ;; These functions orchestrate the parsing by recursively collecting
  ;; operations and dispatching to detail parsers.
  ;;
  ;; Call graph:
  ;;   parse-match-ops-recursive  → parse-after-match → parse-rewrite-ops-recursive
  ;;        ↓ creates AST directly         ↓ parses :then-let         ↓ calls detail parser
  ;;   make-ast-match-expand          parse-rewrite-ops...    parse-rewrite-operation
  ;;
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; parse-match-ops-recursive - Collect match operations
  ;;-----------------------------------------------------------------------
  ;; Stops at :then-let or :rewrite. Creates AST records directly.
  ;;
  (define (parse-match-ops-recursive rest-stx acc-ops ast)
    (syntax-case rest-stx (:then-let :rewrite :where =)
      ;; Stop: :then-let or :rewrite
      [(:then-let . _)
       (begin
         (ast-pattern-expand-match-set! ast (reverse acc-ops))
         (parse-after-match rest-stx ast))]

      [(:rewrite . _)
       (begin
         (ast-pattern-expand-match-set! ast (reverse acc-ops))
         (parse-after-match rest-stx ast))]

      ;; Match operation WITH :where guard
      [(result = op-name (operand ...) :where guard-expr . rest)
       (identifier? #'result)
       (let ([match-op (make-ast-match-expand #'result #'op-name #'(operand ...) #'guard-expr)])
         (parse-match-ops-recursive #'rest (cons match-op acc-ops) ast))]

      ;; Match operation WITHOUT :where guard
      [(result = op-name (operand ...) . rest)
       (identifier? #'result)
       (let ([match-op (make-ast-match-expand #'result #'op-name #'(operand ...) #f)])
         (parse-match-ops-recursive #'rest (cons match-op acc-ops) ast))]

      [_ (syntax-violation 'parse-match-ops-recursive
           "Invalid match operation (expected: result = \"op\" (...) [:where expr])" rest-stx)]))

  ;;-----------------------------------------------------------------------
  ;; parse-after-match - Parse optional :then-let, then :rewrite
  ;;-----------------------------------------------------------------------
  (define (parse-after-match rest-stx ast)
    (syntax-case rest-stx (:then-let :rewrite :with)
      ;; Pattern 1: :then-let followed by :rewrite
      [(:then-let ((var expr) ...) :rewrite root :with . rewrite-rest)
       (identifier? #'root)
       (begin
         (ast-pattern-expand-root-var-set! ast #'root)
         (ast-pattern-expand-where-set! ast
           (map (lambda (binding)
                  (syntax-case binding ()
                    [(v e)
                     (identifier? #'v)
                     (make-ast-where-binding-expand #'v #'e)]
                    [_ (syntax-violation 'parse-after-match "Invalid :then-let binding (expected: (var expr))" binding)]))
                (syntax->list #'((var expr) ...))))
         ;; Parse rewrite operations recursively
         (parse-rewrite-ops-recursive #'rewrite-rest '() ast))]

      ;; Pattern 2: :rewrite without :then-let
      [(:rewrite root :with . rewrite-rest)
       (identifier? #'root)
       (begin
         (ast-pattern-expand-root-var-set! ast #'root)
         ;; Parse rewrite operations recursively
         (parse-rewrite-ops-recursive #'rewrite-rest '() ast))]

      [_ (syntax-violation 'define-conversion-pattern
           "Expected [:then-let (...)] :rewrite root :with rewrite-ops..." rest-stx)]))

  ;;-----------------------------------------------------------------------
  ;; parse-rewrite-ops-recursive - Collect rewrite operations
  ;;-----------------------------------------------------------------------
  ;; Requires parentheses around each operation (complex syntax).
  ;; Delegates to parse-rewrite-operation for details.
  ;;
  (define (parse-rewrite-ops-recursive rest-stx acc-ops ast)
    (syntax-case rest-stx (=)
      ;; End of operations
      [()
       (ast-pattern-expand-rewrite-set! ast (reverse acc-ops))
       ast]

      ;; TODO: This is complex - rewrite operations can span multiple lines
      ;; with :regions, :attrs, etc. For now, require explicit parentheses
      ;; around each rewrite operation (similar to old syntax)
      ;; Future: implement proper operation boundary detection
      [(op-syntax . rest)
       (let ([rewrite-op (parse-rewrite-operation #'op-syntax)])
         (parse-rewrite-ops-recursive #'rest (cons rewrite-op acc-ops) ast))]

      [_ (syntax-violation 'parse-rewrite-ops-recursive
           "Invalid rewrite operation syntax" rest-stx)]))
)
      [_ (syntax-violation 'define-conversion-pattern "Invalid pattern syntax (expected fname :match match-ops... :rewrite root :with rewrite-ops...)" rest)]))

  ;;=======================================================================
  ;; SECTION 3: Detail Parsers (Low-Level Parsing)
  ;;=======================================================================
  ;;
  ;; These functions parse individual rewrite operations and components.
  ;; Match operations are parsed directly in Section 2 (simple syntax).
  ;; Rewrite operations are complex: operands + :regions + :attrs + -> types
  ;;
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; parse-rewrite-operation - Parse one rewrite operation
  ;;-----------------------------------------------------------------------
  ;; Extracts result, op-name; delegates rest to parse-rewrite-rest.
  ;;
  (define (parse-rewrite-operation op-stx)
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
  ;; parse-rewrite-rest - Parse operands and optional sections
  ;;-----------------------------------------------------------------------
  ;; Syntax: (operands) [:regions ...] [:attrs ...] [-> types]
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
  ;; parse-match-ops-recursive - Recursively parse match operations
  ;;-----------------------------------------------------------------------
  ;;
  ;; Collects match operations until hitting :then-let or :rewrite
  ;;
  ;; Pattern matching style (not list processing):
  ;;   1. Match one operation: result = "op" (...) [:where expr]
  ;;   2. Recurse on rest
  ;;   3. Stop when hitting :then-let or :rewrite
  ;;
