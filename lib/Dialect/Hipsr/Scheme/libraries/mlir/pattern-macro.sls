#!r6rs
(library (mlir pattern-macro)
  (export define-conversion-pattern
          ast-pattern?
          ast-pattern-function-name
          ast-pattern-root-op-name
          ast-pattern-match
          ast-pattern-rewrite
          ast-pattern-where
          ast-pattern-debug-ast?
          ast-pattern-debug-matching?)
  (import (rnrs (6))
          (for (rime loop) expand)      ;; Import loop macro for expansion time
          (for (only (chezscheme) syntax->list) expand))    ;; Import syntax->list for expansion time

  ;; Runtime AST record (created when :debug-ast is used)
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
      (define-record-type (ast-pattern-expand make-ast-pattern-expand ast-pattern-expand?)
        (fields (mutable function-name)    ;; symbol - name of the pattern function
                (mutable root-var)         ;; symbol - which match result is the root (e.g., %out)
                (mutable root-op-name)     ;; string - operation name extracted from match (e.g., "onnx.Cast")
                (mutable match)            ;; list - match operations (at least one, will be list of ast-match-expand)
                (mutable rewrite)          ;; list - rewrite operations (at least one, will be list of ast-operation-expand)
                (mutable where)            ;; list - where bindings (zero or more, will be list of ast-where-binding-expand)
                (mutable debug-ast?)       ;; boolean - whether :debug-ast flag is present
                (mutable debug-matching?)))  ;; boolean - whether :debug-matching flag is present

      ;; Expansion-time AST record for match operations
      (define-record-type (ast-match-expand make-ast-match-expand ast-match-expand?)
        (fields result-var     ;; symbol - result variable (e.g., %out)
                op-name        ;; string - operation name (e.g., "onnx.Cast")
                operands       ;; list - operand variables (zero or more)
                attributes     ;; list - attribute bindings (zero or more)
                input-types    ;; list - input type expressions (zero or more)
                output-type))  ;; expression - output type expression

      ;; Expansion-time AST record for rewrite operations
      (define-record-type (ast-operation-expand make-ast-operation-expand ast-operation-expand?)
        (fields result-var     ;; symbol - result variable (e.g., %new)
                op-name        ;; string - operation name (e.g., "hipsr.cast")
                operands       ;; list - operand expressions (zero or more)
                attributes     ;; list - attribute expressions (zero or more)
                rest))         ;; list - remainder (types, etc.)

      ;; Expansion-time AST record for where bindings
      (define-record-type (ast-where-binding-expand make-ast-where-binding-expand ast-where-binding-expand?)
        (fields var            ;; symbol - binding variable (e.g., %ctx)
                expr))         ;; expression - Scheme expression to compute

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
                (not (null? #'(match-op ...)))
                (not (null? #'(rewrite-op ...))))
           (begin
             (ast-pattern-expand-function-name-set! ast (syntax->datum #'fname))
             (ast-pattern-expand-root-var-set! ast (syntax->datum #'root))
             ;; Parse each match operation into ast-match-expand using loop
             (ast-pattern-expand-match-set! ast 
               (loop :for op-stx :in (syntax->list #'(match-op ...))
                     :collect (parse-match-operation op-stx)))
             ;; Parse each rewrite operation into ast-operation-expand using loop
             (ast-pattern-expand-rewrite-set! ast
               (loop :for op-stx :in (syntax->list #'(rewrite-op ...))
                     :collect (parse-rewrite-operation op-stx)))
             (parse-where #'where-rest ast)
             ast)]
          
          ;; Error: function name must be identifier
          [(fname :match . _)
           (not (identifier? #'fname))
           (syntax-violation 'define-conversion-pattern "Function name must be an identifier" #'fname)]
          
          ;; Error: root var must be identifier
          [(_ :match _ :rewrite root . _)
           (not (identifier? #'root))
           (syntax-violation 'define-conversion-pattern "Root variable must be an identifier" #'root)]
          
          ;; Error: match operations cannot be empty
          [(_ :match () . _)
           (syntax-violation 'define-conversion-pattern "At least one match operation required" #'())]
          
          ;; Error: rewrite operations cannot be empty
          [(_ :match _ :rewrite _ :with () . _)
           (syntax-violation 'define-conversion-pattern "At least one rewrite operation required" #'())]
          
          [_ (syntax-violation 'define-conversion-pattern "Invalid pattern syntax (expected fname :match (...) :rewrite root :with (...))" rest)]))
      
      ;;=======================================================================
      ;; Parse individual match operation
      ;; Input: (%out = "onnx.Cast" (%in) () : (!in-type) -> !out-type)
      ;; Output: ast-match-expand record
      ;;=======================================================================
      (define (parse-match-operation op-stx)
        (syntax-case op-stx (= :)
          ;; Pattern: (result-var = "op.name" (operands ...) (attrs ...) : (input-types ...) -> output-type)
          [(result = op-name operands attrs : input-types -> output-type)
           (and (identifier? #'result)
                (string? (syntax->datum #'op-name)))
           (make-ast-match-expand
             (syntax->datum #'result)          ;; result-var: symbol
             (syntax->datum #'op-name)         ;; op-name: string
             (syntax->datum #'operands)        ;; operands: list (TODO: parse structure)
             (syntax->datum #'attrs)           ;; attributes: list (TODO: parse structure)
             (syntax->datum #'input-types)     ;; input-types: list (TODO: parse structure)
             (syntax->datum #'output-type))]   ;; output-type: expression
          
          ;; TODO: Add more patterns for variations
          
          [_ (syntax-violation 'parse-match-operation 
               "Invalid match operation syntax (expected: result = \"op.name\" operands attrs : types -> type)"
               op-stx)]))
      
      ;;=======================================================================
      ;; Parse individual rewrite operation
      ;; Input: (%new = "hipsr.cast" (%ctx %in) ((attr val)) : (!t1 !t2) -> !t3)
      ;; Output: ast-operation-expand record
      ;;=======================================================================
      (define (parse-rewrite-operation op-stx)
        (syntax-case op-stx (= :)
          ;; Pattern: (result = "op.name" (operands ...) (attrs ...) : types ... -> output-type)
          [(result = op-name operands attrs rest ...)
           (and (identifier? #'result)
                (string? (syntax->datum #'op-name)))
           (make-ast-operation-expand
             (syntax->datum #'result)          ;; result-var: symbol
             (syntax->datum #'op-name)         ;; op-name: string
             (syntax->datum #'operands)        ;; operands: list (TODO: parse structure)
             (syntax->datum #'attrs)           ;; attributes: list (TODO: parse structure)
             (syntax->datum #'(rest ...)))]    ;; rest: types, etc.
          
          ;; TODO: Add more patterns for variations
          
          [_ (syntax-violation 'parse-rewrite-operation 
               "Invalid rewrite operation syntax (expected: result = \"op.name\" operands attrs ...)"
               op-stx)]))
      
      ;; Helper: parse optional :where clause
      (define (parse-where rest-stx ast)
        (syntax-case rest-stx (:where)
          [(:where ((var expr) ...))
           (ast-pattern-expand-where-set! ast 
             (loop :for binding :in (syntax->list #'((var expr) ...))
                   :collect (syntax-case binding ()
                              [(v e) 
                               (identifier? #'v)
                               (make-ast-where-binding-expand
                                 (syntax->datum #'v)
                                 (syntax->datum #'e))]
                              [_ (syntax-violation 'parse-where "Invalid where binding (expected: (var expr))" binding)])))]
          [()
           (if #f #f)]
          [_ (syntax-violation 'define-conversion-pattern "Expected :where ((var expr) ...) or end" rest-stx)]))

      ;;=======================================================================
      ;; Phase 2: Validate AST record
      ;;=======================================================================
      (define (validate-ast ast-rec)
        ;; Find the match operation that produces root-var and extract its op-name
        (let ([root-var (ast-pattern-expand-root-var ast-rec)]
              [match-ops (ast-pattern-expand-match ast-rec)])
          (let find-root ([ops match-ops])
            (when (pair? ops)
              (let ([match-op (car ops)])
                (if (eq? (ast-match-expand-result-var match-op) root-var)
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
           (let ([fname (ast-pattern-expand-function-name ast-rec)]
                 [root-op-name (ast-pattern-expand-root-op-name ast-rec)]
                 [match-ops (ast-pattern-expand-match ast-rec)]
                 [rewrite-ops (ast-pattern-expand-rewrite ast-rec)]
                 [where-bindings (ast-pattern-expand-where ast-rec)]
                 [debug-ast? (ast-pattern-expand-debug-ast? ast-rec)]
                 [debug-matching? (ast-pattern-expand-debug-matching? ast-rec)])
             (with-syntax ([fname-id (datum->syntax #'macro-name fname)]
                           [fname-q (datum->syntax #'macro-name fname)]
                           [root-op-name-q (datum->syntax #'macro-name root-op-name)]
                           [match-ops-q (datum->syntax #'macro-name match-ops)]
                           [rewrite-ops-q (datum->syntax #'macro-name rewrite-ops)]
                           [where-bindings-q (datum->syntax #'macro-name where-bindings)]
                           [debug-ast-q (datum->syntax #'macro-name debug-ast?)]
                           [debug-matching-q (datum->syntax #'macro-name debug-matching?)])
               (if debug-ast?
                   #'(define fname-id (make-ast-pattern 'fname-q 'root-op-name-q 'match-ops-q 'rewrite-ops-q 'where-bindings-q 'debug-ast-q 'debug-matching-q))
                   #'(define fname-id (lambda (op operands-ref rewriter type-converter) #f)))))]))

      (let* ([ast-rec (parse-to-ast stx)]
             [validated (validate-ast ast-rec)])
        (generate-code stx validated))))

) ;; end library
