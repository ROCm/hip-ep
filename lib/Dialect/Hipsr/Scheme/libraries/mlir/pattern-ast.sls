#!r6rs
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
          ast-pattern-expand-debug-ast? ast-pattern-expand-debug-ast?-set!
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

  ;; Expansion-time AST record for the whole pattern
  ;; Contains syntax objects to preserve lexical context for code generation
  (define-record-type (ast-pattern-expand make-ast-pattern-expand ast-pattern-expand?)
    (fields (mutable function-name)    ;; syntax - identifier for pattern function
            (mutable root-var)         ;; syntax - identifier for root result var (e.g., #'%out)
            (mutable root-op-name)     ;; syntax - string literal for op name (e.g., #'"onnx.Cast")
            (mutable match)            ;; list of ast-match-expand - parsed match operations
            (mutable match-bindings)   ;; hashtable: identifier -> matcher (first occurrence)
            (mutable match-actions)    ;; list of matchers (ordered, for codegen)
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
)
