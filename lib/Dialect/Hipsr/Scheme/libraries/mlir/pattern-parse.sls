#!r6rs
(library (mlir pattern-parse)
  (export parse-to-ast)
  (import (except (rnrs) =)
          (for (only (chezscheme) syntax->list) expand)
          (for (mlir pattern-keywords) expand)  ;; KEY: Import keywords at expansion time!
          (for (mlir pattern-ast) expand))

  ;;=======================================================================
  ;; Phase 1: Parse whole syntax to AST record - pure pattern matching
  ;;=======================================================================

  (define (parse-to-ast whole-stx)
    (syntax-case whole-stx ()
      [(_ . rest)
       (parse-rest #'rest (make-ast-pattern-expand #f #f #f '() #f #f '() '() #f #f #f #f))]))

  ;; Parse flags and clauses
  (define (parse-rest rest ast)
    (syntax-case rest (:debug-parse :debug-analyze :debug-codegen :debug-matching :match :rewrite :with :where)
      ;; Debug flags
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

  ;; Parse individual match operation
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

  ;; Parse individual rewrite operation
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

      ;; Another region - parse it and accumulate
      [(region . more)
       (let ([region-rec (parse-region #'region)])
         (parse-regions-section rec #'more (cons region-rec regions-acc)))]

      ;; Error - must have -> result-types
      [_ (syntax-violation 'parse-regions-section
           "Expected region, :attrs, or -> result-types"
           rest-stx)]))

  ;; Parse a region: a list of blocks wrapped in parens
  (define (parse-region region-stx)
    (syntax-case region-stx ()
      [(block ...)
       (let ([blocks (map parse-block (syntax->list #'(block ...)))])
         (make-ast-region-expand blocks))]
      [_ (syntax-violation 'parse-region
           "Invalid region syntax (expected: list of blocks)"
           region-stx)]))

  ;; Parse a block: (^label (args...) operations...)
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
)
