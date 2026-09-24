#!r6rs
;;=======================================================================
;; Pattern AST Record Definitions
;;=======================================================================
;;
;; This library defines the AST record types used throughout the 4-phase
;; pattern DSL macro expansion pipeline:
;;
;;   Phase 1 (parse)    - Parse syntax to AST records (this file defines structure)
;;   Phase 2 (validate) - Normalize and validate AST in-place (mutable fields)
;;   Phase 3 (analyze)  - Build bindings and actions, store in AST
;;   Phase 4 (codegen)  - Generate final lambda from AST
;;
;; All fields store syntax objects (not datums) to preserve lexical context
;; for code generation. Mutable fields allow later phases to normalize/enrich
;; the AST in-place without creating new records.
;;
;;=======================================================================

(library (mlir pattern-ast)
  (export ast-pattern-expand make-ast-pattern-expand make-default-ast-pattern-expand ast-pattern-expand?
          ast-pattern-expand-pattern-type ast-pattern-expand-pattern-type-set!
          ast-pattern-expand-function-name ast-pattern-expand-function-name-set!
          ast-pattern-expand-parameters ast-pattern-expand-parameters-set!
          ast-pattern-expand-root-var ast-pattern-expand-root-var-set!
          ast-pattern-expand-root-op-name ast-pattern-expand-root-op-name-set!
          ast-pattern-expand-root-op-index ast-pattern-expand-root-op-index-set!
          ast-pattern-expand-root-result-idx ast-pattern-expand-root-result-idx-set!
          ast-pattern-expand-match ast-pattern-expand-match-set!
          ast-pattern-expand-match-bindings ast-pattern-expand-match-bindings-set!
          ast-pattern-expand-match-actions ast-pattern-expand-match-actions-set!
          ast-pattern-expand-rewrite ast-pattern-expand-rewrite-set!
          ast-pattern-expand-where ast-pattern-expand-where-set!
          ast-pattern-expand-debug-parse? ast-pattern-expand-debug-parse?-set!
          ast-pattern-expand-debug-validate? ast-pattern-expand-debug-validate?-set!
          ast-pattern-expand-debug-analyze? ast-pattern-expand-debug-analyze?-set!
          ast-pattern-expand-debug-codegen? ast-pattern-expand-debug-codegen?-set!
          ast-pattern-expand-debug-matching? ast-pattern-expand-debug-matching?-set!

          ast-match-expand make-ast-match-expand ast-match-expand?
          ast-match-expand-result-var ast-match-expand-result-var-set!
          ast-match-expand-op-name ast-match-expand-op-name-set!
          ast-match-expand-operands ast-match-expand-operands-set!
          ast-match-expand-where-expr ast-match-expand-where-expr-set!

          ast-operation-expand make-ast-operation-expand ast-operation-expand?
          ast-operation-expand-result-var ast-operation-expand-result-var-set!
          ast-operation-expand-op-name ast-operation-expand-op-name-set!
          ast-operation-expand-operands ast-operation-expand-operands-set!
          ast-operation-expand-regions ast-operation-expand-regions-set!
          ast-operation-expand-attributes ast-operation-expand-attributes-set!
          ast-operation-expand-result-types ast-operation-expand-result-types-set!

          ast-where-binding-expand make-ast-where-binding-expand ast-where-binding-expand?
          ast-where-binding-expand-var ast-where-binding-expand-var-set!
          ast-where-binding-expand-expr ast-where-binding-expand-expr-set!

          ast-region-expand make-ast-region-expand ast-region-expand?
          ast-region-expand-blocks ast-region-expand-blocks-set!

          ast-block-expand make-ast-block-expand ast-block-expand?
          ast-block-expand-label ast-block-expand-label-set!
          ast-block-expand-arguments ast-block-expand-arguments-set!
          ast-block-expand-operations ast-block-expand-operations-set!

          ast-operand make-ast-operand ast-operand?
          ast-operand-kind ast-operand-kind-set!
          ast-operand-var ast-operand-var-set!)
  (import (rnrs))

  ;;=======================================================================
  ;; AST RECORD HIERARCHY (top-down tree order: root → leaves)
  ;;=======================================================================
  ;;
  ;; Tree structure of pattern AST (top-down presentation):
  ;;
  ;; ast-pattern-expand (ROOT)
  ;; ├── match: list of ast-match-expand
  ;; │   └── operands: list of ast-operand
  ;; ├── where: list of ast-where-binding-expand
  ;; └── rewrite: list of ast-operation-expand
  ;;     └── regions: list of ast-region-expand
  ;;         └── blocks: list of ast-block-expand
  ;;             └── operations: list of ast-operation-expand (recursive)
  ;;
  ;; Reading flow: Start at root (pattern definition), drill down into
  ;; match operations, operand details, rewrite operations, region/block structure.
  ;;
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; ROOT RECORD: ast-pattern-expand
  ;;-----------------------------------------------------------------------
  ;;
  ;; Represents a complete pattern definition across all phases.
  ;; Created by parse phase, enriched by validate/analyze phases, consumed by codegen.
  ;;
  ;; Contains:
  ;;   - match: list of ast-match-expand (operations to match)
  ;;   - where: list of ast-where-binding-expand (computed bindings)
  ;;   - rewrite: list of ast-operation-expand (operations to construct)
  ;;
  (define-record-type (ast-pattern-expand make-ast-pattern-expand ast-pattern-expand?)
    (fields
      (mutable pattern-type)     ;; Phase 1 (parse): symbol - 'conversion or 'rewrite
                                 ;; Determines operand binding behavior in codegen:
                                 ;; - 'conversion: root operation uses operands-ref parameter
                                 ;; - 'rewrite: all operations use mlir-operation-get-operand-value

      (mutable function-name)    ;; Phase 1 (parse): syntax identifier - name of generated pattern function
                                 ;; Example: #'my-pattern

      (mutable parameters)       ;; Phase 1 (parse): list of syntax identifiers - lambda parameters
                                 ;; Example: (#'op #'operands-ref #'rewriter #'type-converter)
                                 ;; User-provided parameters that will be visible in :then-let scope

      (mutable root-var)         ;; Phase 1 (parse): syntax identifier - result variable of the root operation
                                 ;; Example: #'%out
                                 ;; The root operation is the one matched against the input op
                                 ;; Phase 2 (validate): validated and confirmed to be a result var

      (mutable root-op-name)     ;; Phase 1 (parse): not set (initialized to #f)
                                 ;; Phase 2 (validate): syntax string - operation name of root operation
                                 ;; Example: #'"onnx.Cast"
                                 ;; Set after finding which match operation defines root-var

      (mutable root-op-index)    ;; Phase 1 (parse): not set (initialized to #f)
                                 ;; Phase 2 (validate): integer - index of root operation in match vector
                                 ;; Example: 2 (if root is 3rd operation in :match clause)
                                 ;; Cached for efficiency, used by codegen to initialize root variable

      (mutable root-result-idx)  ;; Phase 1 (parse): not set (initialized to #f)
                                 ;; Phase 2 (validate): integer - index of root var in root operation's result list
                                 ;; Example: 0 (if root var is first result)
                                 ;;          1 (if root var is second result in (%a %b) = "op"(...))
                                 ;; Cached for efficiency, used by codegen with mlir-operation-get-result

      (mutable match)            ;; Phase 1 (parse): list of ast-match-expand (parsed in order)
                                 ;; Phase 2 (validate): vector of ast-match-expand (normalized for indexing)
                                 ;; Indexing: vector index = operation index in DAG
                                 ;; Phase 3 (analyze): used to generate match-actions
                                 ;; Contains: list/vector of ast-match-expand records

      (mutable match-bindings)   ;; Phase 1 (parse): not set (initialized to #f)
                                 ;; Phase 2 (validate): not set (still #f)
                                 ;; Phase 3 (analyze): binding-manager record
                                 ;; Maps all identifiers (results and operands) to binding-entry records
                                 ;; Used by codegen to collect all variables

      (mutable match-actions)    ;; Phase 1 (parse): not set (initialized to #f)
                                 ;; Phase 2 (validate): not set (still #f)
                                 ;; Phase 3 (analyze): list of actions
                                 ;; Ordered sequence of runtime matching actions
                                 ;; Actions: :set-current-op, :check-op, :bind-operand, :check-eq
                                 ;; Used by codegen to generate pattern matching code

      (mutable rewrite)          ;; Phase 1 (parse): list of ast-operation-expand - rewrite operations
                                 ;; Operations to construct when pattern matches
                                 ;; Contains: list of ast-operation-expand records
                                 ;; Currently unused (rewrite not implemented yet)

      (mutable where)            ;; Phase 1 (parse): list of ast-where-binding-expand - constraint bindings
                                 ;; Additional computed bindings for rewrite
                                 ;; Contains: list of ast-where-binding-expand records
                                 ;; Currently unused (rewrite not implemented yet)

      (mutable debug-parse?)     ;; Phase 1 (parse): boolean - :debug-parse flag
                                 ;; When true, codegen outputs parsed AST as datum

      (mutable debug-validate?)  ;; Phase 1 (parse): boolean - :debug-validate flag
                                 ;; When true, codegen outputs validated AST as datum

      (mutable debug-analyze?)   ;; Phase 1 (parse): boolean - :debug-analyze flag
                                 ;; When true, codegen outputs match-actions as datum

      (mutable debug-codegen?)   ;; Phase 1 (parse): boolean - :debug-codegen flag
                                 ;; When true, codegen outputs generated code as quoted datum

      (mutable debug-matching?)))  ;; Phase 1 (parse): boolean - :debug-matching flag
                                   ;; When true, generated code prints trace during matching
                                   ;; Currently unused (not implemented yet)

  ;;=======================================================================
  ;; Default Constructor for ast-pattern-expand
  ;;=======================================================================
  ;;
  ;; Creates a new pattern AST with all fields initialized to default values.
  ;; Only pattern-type needs to be specified ('conversion or 'rewrite).
  ;;
  (define (make-default-ast-pattern-expand pattern-type)
    (make-ast-pattern-expand
      pattern-type  ;; pattern-type: 'conversion or 'rewrite
      #f            ;; function-name: set by parse-rest
      '()           ;; parameters: set by parse-rest (list of parameter identifiers)
      #f            ;; root-var: set by parse-rest
      #f            ;; root-op-name: set by validate phase
      #f            ;; root-op-index: set by validate phase
      #f            ;; root-result-idx: set by validate phase
      '()           ;; match: accumulated during parse
      #f            ;; match-bindings: set by analyze phase
      #f            ;; match-actions: set by analyze phase
      '()           ;; rewrite: accumulated during parse
      '()           ;; where: accumulated during parse
      #f            ;; debug-parse?: set by parse-rest if :debug-parse present
      #f            ;; debug-validate?: set by parse-rest if :debug-validate present
      #f            ;; debug-analyze?: set by parse-rest if :debug-analyze present
      #f            ;; debug-codegen?: set by parse-rest if :debug-codegen present
      #f))          ;; debug-matching?: set by parse-rest if :debug-matching present

  ;;-----------------------------------------------------------------------
  ;; MATCH-LEVEL RECORD: ast-match-expand (child of ast-pattern-expand)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Used by: ast-pattern-expand (match field)
  ;;
  ;; Represents a single operation to match in the pattern.
  ;; Corresponds to one line in the :match clause.
  ;;
  ;; Contains:
  ;;   - operands: list of ast-operand records
  ;;
  ;; Example DSL input (single result):
  ;;   (%b = "op2" (%a) () : (!t2) -> !t3)
  ;;
  ;; Example DSL input (multiple results):
  ;;   ((%a %b) = "op2" (%x) () : (!t) -> (!t1 !t2))
  ;;
  (define-record-type (ast-match-expand make-ast-match-expand ast-match-expand?)
    (fields
      (mutable result-var)     ;; Phase 1 (parse): syntax identifier OR syntax list
                               ;;          Single: #'%b
                               ;;          Multiple: #'(%a %b)
                               ;;          Variadic (future): #'(%a ...)
                               ;;          Dotted (future): #'(%a %b . %rest)
                               ;; Phase 2 (validate): list of syntax identifiers (normalized)
                               ;;          Single becomes: (#'%b)
                               ;;          Multiple becomes: (#'%a #'%b)
                               ;; Normalization: syntax->list converts all forms to uniform list

      (mutable op-name)        ;; Phase 1 (parse): syntax string OR syntax symbol
                               ;;          String: #'"op2"
                               ;;          Symbol: #'op2
                               ;; Phase 2 (validate): syntax string (normalized)
                               ;;          Symbol converted: #'op2 → #'"op2"

      (mutable operands)       ;; Phase 1 (parse): list of ast-operand records (flattened)
                               ;; Phase 2 (validate): unchanged (records already validated during parse)
                               ;; Phase 3 (analyze): used to generate operand binding actions
                               ;; Contains: list of ast-operand records
                               ;;
                               ;; Input syntax: (%x (&optional %y %z) %w (&variadic %rest))
                               ;; Parsed to flat list:
                               ;;   [ast-operand('required, #'%x),
                               ;;    ast-operand('optional, #'%y),
                               ;;    ast-operand('optional, #'%z),
                               ;;    ast-operand('required, #'%w),
                               ;;    ast-operand('variadic, #'%rest)]
                               ;;
                               ;; Each operand is tagged with kind: 'required, 'optional, or 'variadic
                               ;; Operand vars can be result variables (from other ops) or free variables
                               ;;
                               ;; Note: Optional/variadic require AttrSizedOperandSegments trait
                               ;; and runtime operandSegmentSizes attribute to calculate positions

      (mutable where-expr)))   ;; Phase 1 (parse): syntax object - pure Scheme guard expression
                               ;; Executes after matching this operation (early return on failure)
                               ;; Default: #'#f (no guard, always succeeds)
                               ;; Example: #'(let ([$ks (mlir-operation-get-attribute %a "kernel_shape")])
                               ;;             (and $ks (is-1x1-kernel? $ks)))
                               ;; Can access: matched result-var, operands, any previously bound vars

  ;;-----------------------------------------------------------------------
  ;; OPERAND-LEVEL RECORD: ast-operand (child of ast-match-expand)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Used by: ast-match-expand (operands field)
  ;;
  ;; Represents a single operand in a match operation with its kind tag.
  ;; The (&optional ...) and (&variadic ...) groups in the DSL syntax are
  ;; flattened during parsing - each variable gets its own record.
  ;;
  ;; Example DSL input: (%x (&optional %y %z) %w (&variadic %rest))
  ;; Phase 1 (parse): Flattened to list of ast-operand records:
  ;;   [ast-operand('required, #'%x),
  ;;    ast-operand('optional, #'%y),
  ;;    ast-operand('optional, #'%z),
  ;;    ast-operand('required, #'%w),
  ;;    ast-operand('variadic, #'%rest)]
  ;;
  ;; Note: Optional/variadic require AttrSizedOperandSegments trait and
  ;; runtime operandSegmentSizes attribute to calculate access positions.
  ;;
  (define-record-type (ast-operand make-ast-operand ast-operand?)
    (fields
      (mutable kind)           ;; Phase 1 (parse): symbol - 'required, 'optional, or 'variadic
                               ;; Phase 2 (validate): unchanged
                               ;; Phase 3 (analyze): used to compute operand segment positions

      (mutable var)))          ;; Phase 1 (parse): syntax identifier (e.g., #'%x)
                               ;; Phase 2 (validate): validated to start with %
                               ;; Phase 3 (analyze): used to bind/access this operand

  ;;-----------------------------------------------------------------------
  ;; WHERE-BINDING-LEVEL RECORD: ast-where-binding-expand (child of ast-pattern-expand)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Used by: ast-pattern-expand (where field)
  ;;
  ;; Represents a computed binding in the :where clause.
  ;; Used to compute additional values needed for rewrite.
  ;;
  ;; Example DSL input:
  ;;   :where ((!new-type (compute-type !old-type))
  ;;           (%ctx (get-context)))
  ;;
  (define-record-type (ast-where-binding-expand make-ast-where-binding-expand ast-where-binding-expand?)
    (fields
      (mutable var)            ;; Phase 1 (parse): syntax identifier - variable to bind
                               ;; Example: #'!new-type or #'%ctx

      (mutable expr)))         ;; Phase 1 (parse): syntax expression - Scheme expression to evaluate
                               ;; Example: #'(compute-type !old-type)
                               ;; Currently unused (rewrite not implemented)

  ;;-----------------------------------------------------------------------
  ;; REWRITE-OPERATION-LEVEL RECORD: ast-operation-expand (child of ast-pattern-expand)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Used by: ast-pattern-expand (rewrite field), ast-block-expand (operations field)
  ;;
  ;; Represents a single operation to construct in the rewrite.
  ;; Corresponds to one line in the :rewrite :with clause.
  ;;
  ;; Contains:
  ;;   - regions: list of ast-region-expand records
  ;;
  ;; Example DSL input:
  ;;   (%out = "hipsr.cast" (%x) :regions (...) :attrs [("to", !t3)] -> !t3)
  ;;
  ;; Field order matches MLIR generic operation syntax:
  ;;   result = op-name (operands) :regions (...) :attrs [...] -> result-types
  ;;
  ;; Note: No input-types field. Input types are implicit in operand Values
  ;;       (each Value has .getType()). Only result types need to be specified.
  ;;
  (define-record-type (ast-operation-expand make-ast-operation-expand ast-operation-expand?)
    (fields
      (mutable result-var)     ;; Phase 1 (parse): syntax identifier or list - result variable(s)
                               ;; Example: #'%out for single result
                               ;; Example: #'(%a %b) for multiple results
                               ;; Empty #'() for operations with no results

      (mutable op-name)        ;; Phase 1 (parse): syntax string - operation name to construct
                               ;; Example: #'"hipsr.cast"

      (mutable operands)       ;; Phase 1 (parse): syntax list - operand expressions
                               ;; Example: #'(%x) means pass bound variable %x
                               ;; Operands must be bound by match or where clauses

      (mutable regions)        ;; Phase 1 (parse): list of ast-region-expand - nested regions
                               ;; Example: control flow ops like scf.if have regions
                               ;; Contains: list of ast-region-expand records
                               ;; Currently unused (region construction not implemented)

      (mutable attributes)     ;; Phase 1 (parse): syntax list - attribute expressions
                               ;; Example: #'(("to" !t3)) for typed attribute
                               ;; Currently unused (attribute construction not implemented)

      (mutable result-types))) ;; Phase 1 (parse): syntax - result type expression(s)
                               ;; Example: #'!t3 for single result type
                               ;; Example: #'(!t1 !t2) for multiple result types
                               ;; Currently unused (rewrite not implemented)

  ;;-----------------------------------------------------------------------
  ;; REGION-LEVEL RECORD: ast-region-expand (child of ast-operation-expand)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Used by: ast-operation-expand (regions field)
  ;;
  ;; Represents a region in MLIR (a list of blocks).
  ;; Used by operations like scf.if, scf.while that contain nested code.
  ;;
  ;; Contains:
  ;;   - blocks: list of ast-block-expand records
  ;;
  (define-record-type (ast-region-expand make-ast-region-expand ast-region-expand?)
    (fields
      (mutable blocks)))       ;; Phase 1 (parse): list of ast-block-expand
                               ;; Each region contains one or more blocks
                               ;; Contains: list of ast-block-expand records

  ;;-----------------------------------------------------------------------
  ;; BLOCK-LEVEL RECORD: ast-block-expand (child of ast-region-expand)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Used by: ast-region-expand (blocks field)
  ;;
  ;; Represents a block in MLIR (a sequence of operations with a label).
  ;; Blocks can take arguments like function parameters.
  ;;
  ;; Contains:
  ;;   - operations: list of ast-operation-expand records (recursive)
  ;;
  ;; Example DSL input:
  ;;   (^bb0 ((%arg0 : !t1) (%arg1 : !t2))
  ;;     (%sum = "arith.addi" (%arg0 %arg1) -> !t1)
  ;;     ("scf.yield" (%sum) -> ()))
  ;;
  (define-record-type (ast-block-expand make-ast-block-expand ast-block-expand?)
    (fields
      (mutable label)          ;; Phase 1 (parse): syntax identifier - block label
                               ;; Example: #'^bb0 (^ prefix is convention)

      (mutable arguments)      ;; Phase 1 (parse): list of (var type) syntax pairs
                               ;; Example: ((#'%arg0 #'!t1) (#'%arg1 #'!t2))
                               ;; Block arguments are like function parameters

      (mutable operations)))   ;; Phase 1 (parse): list of ast-operation-expand
                               ;; Operations in this block (can be recursive - blocks in regions in ops)
                               ;; Contains: list of ast-operation-expand records (recursive)
)
