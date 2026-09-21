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
  (export ast-pattern-expand make-ast-pattern-expand ast-pattern-expand?
          ast-pattern-expand-function-name ast-pattern-expand-function-name-set!
          ast-pattern-expand-root-var ast-pattern-expand-root-var-set!
          ast-pattern-expand-root-op-name ast-pattern-expand-root-op-name-set!
          ast-pattern-expand-match ast-pattern-expand-match-set!
          ast-pattern-expand-match-bindings ast-pattern-expand-match-bindings-set!
          ast-pattern-expand-match-actions ast-pattern-expand-match-actions-set!
          ast-pattern-expand-rewrite ast-pattern-expand-rewrite-set!
          ast-pattern-expand-where ast-pattern-expand-where-set!
          ast-pattern-expand-debug-parse? ast-pattern-expand-debug-parse?-set!
          ast-pattern-expand-debug-analyze? ast-pattern-expand-debug-analyze?-set!
          ast-pattern-expand-debug-codegen? ast-pattern-expand-debug-codegen?-set!
          ast-pattern-expand-debug-matching? ast-pattern-expand-debug-matching?-set!

          ast-match-expand make-ast-match-expand ast-match-expand?
          ast-match-expand-result-var ast-match-expand-result-var-set!
          ast-match-expand-op-name ast-match-expand-op-name-set!
          ast-match-expand-operands ast-match-expand-operands-set!
          ast-match-expand-attributes ast-match-expand-attributes-set!
          ast-match-expand-input-types ast-match-expand-input-types-set!
          ast-match-expand-output-type ast-match-expand-output-type-set!

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
          ast-block-expand-operations ast-block-expand-operations-set!)
  (import (rnrs))

  ;;-----------------------------------------------------------------------
  ;; Top-level pattern record
  ;;-----------------------------------------------------------------------
  ;;
  ;; Represents a complete pattern definition across all phases.
  ;; Created by parse phase, enriched by validate/analyze phases, consumed by codegen.
  ;;
  (define-record-type (ast-pattern-expand make-ast-pattern-expand ast-pattern-expand?)
    (fields
      (mutable function-name)    ;; syntax identifier - name of generated pattern function
                                 ;; Example: #'my-pattern

      (mutable root-var)         ;; syntax identifier - result variable of the root operation
                                 ;; Example: #'%out
                                 ;; The root operation is the one matched against the input op

      (mutable root-op-name)     ;; syntax string - operation name of root operation
                                 ;; Example: #'"onnx.Cast"
                                 ;; Set by analyze phase after finding root operation

      (mutable match)            ;; Phase 1: list of ast-match-expand (parsed)
                                 ;; Phase 2: vector of ast-match-expand (normalized by validate)
                                 ;; Indexing: vector index = operation index in DAG

      (mutable match-bindings)   ;; Phase 3: binding-manager record (analyze phase)
                                 ;; Maps all identifiers (results and operands) to binding-entry records
                                 ;; Used by codegen to collect all variables

      (mutable match-actions)    ;; Phase 3: list of actions (analyze phase)
                                 ;; Ordered sequence of runtime matching actions
                                 ;; Actions: :set-current-op, :check-op, :bind-operand, :check-eq
                                 ;; Used by codegen to generate pattern matching code

      (mutable rewrite)          ;; list of ast-operation-expand - rewrite operations
                                 ;; Operations to construct when pattern matches
                                 ;; Currently unused (rewrite not implemented yet)

      (mutable where)            ;; list of ast-where-binding-expand - constraint bindings
                                 ;; Additional computed bindings for rewrite
                                 ;; Currently unused (rewrite not implemented yet)

      (mutable debug-parse?)     ;; boolean - :debug-parse flag
                                 ;; When true, codegen outputs parsed AST as datum

      (mutable debug-analyze?)   ;; boolean - :debug-analyze flag
                                 ;; When true, codegen outputs match-actions as datum

      (mutable debug-codegen?)   ;; boolean - :debug-codegen flag
                                 ;; When true, codegen outputs generated code as quoted datum

      (mutable debug-matching?)))  ;; boolean - :debug-matching flag
                                   ;; When true, generated code prints trace during matching
                                   ;; Currently unused (not implemented yet)

  ;;-----------------------------------------------------------------------
  ;; Match operation record
  ;;-----------------------------------------------------------------------
  ;;
  ;; Represents a single operation to match in the pattern.
  ;; Corresponds to one line in the :match clause.
  ;;
  ;; Example input syntax:
  ;;   (%b = "op2" (%a) () : (!t2) -> !t3)
  ;;
  (define-record-type (ast-match-expand make-ast-match-expand ast-match-expand?)
    (fields
      (mutable result-var)     ;; Phase 1: syntax identifier - single result variable
                               ;;          Example: #'%b
                               ;; Phase 2: list of syntax identifiers - normalized
                               ;;          Example: (#'%b) for single result
                               ;;          Example: (#'%a #'%b) for multiple results
                               ;; Normalization allows uniform handling of single/multiple results

      (mutable op-name)        ;; syntax string - operation name to match
                               ;; Example: #'"op2"

      (mutable operands)       ;; syntax list - operand variables
                               ;; Example: #'(%a) means operation takes %a as input
                               ;; Operands can be result variables (from other ops) or free variables

      (mutable attributes)     ;; syntax list - attribute constraints
                               ;; Example: #'() for no attributes
                               ;; Currently unused (attribute matching not implemented)

      (mutable input-types)    ;; syntax list - input type variables
                               ;; Example: #'(!t2) means operand has type !t2
                               ;; Currently unused (type matching not implemented)

      (mutable output-type)))  ;; syntax type - output type variable
                               ;; Example: #'!t3 means result has type !t3
                               ;; Currently unused (type matching not implemented)

  ;;-----------------------------------------------------------------------
  ;; Rewrite operation record
  ;;-----------------------------------------------------------------------
  ;;
  ;; Represents a single operation to construct in the rewrite.
  ;; Corresponds to one line in the :rewrite :with clause.
  ;;
  ;; Example input syntax:
  ;;   (%out = "hipsr.cast" (%x) :attrs [("to", !t3)] -> !t3)
  ;;
  ;; Field order matches MLIR generic operation syntax:
  ;;   result = op-name (operands) :regions (...) :attrs [...] -> result-types
  ;;
  ;; Note: No input-types field. Input types are implicit in operand Values
  ;;       (each Value has .getType()). Only result types need to be specified.
  ;;
  (define-record-type (ast-operation-expand make-ast-operation-expand ast-operation-expand?)
    (fields
      (mutable result-var)     ;; syntax identifier or list - result variable(s)
                               ;; Example: #'%out for single result
                               ;; Example: (#'%a #'%b) for multiple results
                               ;; Empty #'() for operations with no results

      (mutable op-name)        ;; syntax string - operation name to construct
                               ;; Example: #'"hipsr.cast"

      (mutable operands)       ;; syntax list - operand expressions
                               ;; Example: #'(%x) means pass bound variable %x
                               ;; Operands must be bound by match or where clauses

      (mutable regions)        ;; list of ast-region-expand - nested regions
                               ;; Example: control flow ops like scf.if have regions
                               ;; Currently unused (region construction not implemented)

      (mutable attributes)     ;; syntax list - attribute expressions
                               ;; Example: #'(("to" !t3)) for typed attribute
                               ;; Currently unused (attribute construction not implemented)

      (mutable result-types))) ;; syntax - result type expression(s)
                               ;; Example: #'!t3 for single result type
                               ;; Example: #'(!t1 !t2) for multiple result types
                               ;; Currently unused (rewrite not implemented)

  ;;-----------------------------------------------------------------------
  ;; Where binding record
  ;;-----------------------------------------------------------------------
  ;;
  ;; Represents a computed binding in the :where clause.
  ;; Used to compute additional values needed for rewrite.
  ;;
  ;; Example input syntax:
  ;;   :where ((!new-type (compute-type !old-type))
  ;;           (%ctx (get-context)))
  ;;
  (define-record-type (ast-where-binding-expand make-ast-where-binding-expand ast-where-binding-expand?)
    (fields
      (mutable var)            ;; syntax identifier - variable to bind
                               ;; Example: #'!new-type or #'%ctx

      (mutable expr)))         ;; syntax expression - Scheme expression to evaluate
                               ;; Example: #'(compute-type !old-type)
                               ;; Currently unused (rewrite not implemented)

  ;;-----------------------------------------------------------------------
  ;; Region record (for control flow operations)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Represents a region in MLIR (a list of blocks).
  ;; Used by operations like scf.if, scf.while that contain nested code.
  ;;
  (define-record-type (ast-region-expand make-ast-region-expand ast-region-expand?)
    (fields
      (mutable blocks)))       ;; list of ast-block-expand
                               ;; Each region contains one or more blocks

  ;;-----------------------------------------------------------------------
  ;; Block record (for control flow)
  ;;-----------------------------------------------------------------------
  ;;
  ;; Represents a block in MLIR (a sequence of operations with a label).
  ;; Blocks can take arguments like function parameters.
  ;;
  ;; Example input syntax:
  ;;   (^bb0 ((%arg0 : !t1) (%arg1 : !t2))
  ;;     (%sum = "arith.addi" (%arg0 %arg1) -> !t1)
  ;;     ("scf.yield" (%sum) -> ()))
  ;;
  (define-record-type (ast-block-expand make-ast-block-expand ast-block-expand?)
    (fields
      (mutable label)          ;; syntax identifier - block label
                               ;; Example: #'^bb0
                               ;; Labels start with ^ by convention

      (mutable arguments)      ;; list of (var type) syntax pairs
                               ;; Example: ((#'%arg0 #'!t1) (#'%arg1 #'!t2))
                               ;; Block arguments are like function parameters

      (mutable operations)))   ;; list of ast-operation-expand
                               ;; Operations in this block
)
