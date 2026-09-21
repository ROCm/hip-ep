#!r6rs
(library (mlir pattern-codegen)
  (export generate-code)
  (import (rnrs)
          (for (only (chezscheme) syntax->list) expand)
          (for (mlir pattern-ast) expand))

  ;;=======================================================================
  ;; Phase 4: Code generation - generate lambda from analyzed AST
  ;;=======================================================================

  ;;-----------------------------------------------------------------------
  ;; Main entry point
  ;;-----------------------------------------------------------------------

  (define (generate-code whole-stx ast-rec)
    (syntax-case whole-stx ()
      [(macro-name . _)
       (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                    [root-op-name (ast-pattern-expand-root-op-name ast-rec)])
         (let ([match-ops (ast-pattern-expand-match ast-rec)]
               [rewrite-ops (ast-pattern-expand-rewrite ast-rec)]
               [where-bindings (ast-pattern-expand-where ast-rec)]
               [match-actions (ast-pattern-expand-match-actions ast-rec)]
               [debug-parse? (ast-pattern-expand-debug-parse? ast-rec)]
               [debug-analyze? (ast-pattern-expand-debug-analyze? ast-rec)]
               [debug-codegen? (ast-pattern-expand-debug-codegen? ast-rec)]
               [debug-matching? (ast-pattern-expand-debug-matching? ast-rec)])
           (if (or debug-parse? debug-analyze? debug-codegen?)
               ;; For debug modes, return list with requested data
               (let ([fname-sym (syntax->datum #'fname)]
                     [root-op-str (syntax->datum #'root-op-name)]
                     [match-data (if debug-parse?
                                     (map match-expand->datum (vector->list match-ops))
                                     #f)]
                     [rewrite-data (if debug-parse?
                                       (map operation-expand->datum rewrite-ops)
                                       #f)]
                     [where-data (if debug-parse?
                                     (map where-binding-expand->datum where-bindings)
                                     #f)]
                     [actions-data (if debug-analyze?
                                       (map action->datum match-actions)
                                       #f)])
                 (with-syntax ([ast-list (datum->syntax #'macro-name
                                           `(list 'function-name ',fname-sym
                                                  'root-op-name ,root-op-str
                                                  ,@(if match-data `('match ',match-data) '())
                                                  ,@(if rewrite-data `('rewrite ',rewrite-data) '())
                                                  ,@(if where-data `('where ',where-data) '())
                                                  ,@(if actions-data `('match-actions ',actions-data) '())
                                                  'debug-parse? ,debug-parse?
                                                  'debug-analyze? ,debug-analyze?
                                                  'debug-codegen? ,debug-codegen?
                                                  'debug-matching? ,debug-matching?))])
                   #'(define fname ast-list)))
               ;; For normal mode, generate lambda
               #'(define fname
                   (lambda (op operands-ref rewriter type-converter)
                     #f)))))]))

  ;;-----------------------------------------------------------------------
  ;; AST to datum conversion (for debug modes)
  ;;-----------------------------------------------------------------------

  (define (match-expand->datum match-exp)
    (list 'match
          (syntax->datum (ast-match-expand-result-var match-exp))
          (syntax->datum (ast-match-expand-op-name match-exp))
          (syntax->datum (ast-match-expand-operands match-exp))
          (syntax->datum (ast-match-expand-attributes match-exp))
          (syntax->datum (ast-match-expand-input-types match-exp))
          (syntax->datum (ast-match-expand-output-type match-exp))))

  (define (operation-expand->datum op-exp)
    (list 'rewrite
          (syntax->datum (ast-operation-expand-result-var op-exp))
          (syntax->datum (ast-operation-expand-op-name op-exp))
          (syntax->datum (ast-operation-expand-operands op-exp))
          (syntax->datum (ast-operation-expand-regions op-exp))
          (syntax->datum (ast-operation-expand-attributes op-exp))
          (syntax->datum (ast-operation-expand-result-types op-exp))))

  (define (where-binding-expand->datum where-exp)
    (list (syntax->datum (ast-where-binding-expand-var where-exp))
          (syntax->datum (ast-where-binding-expand-expr where-exp))))

  (define (action->datum action)
    ;; Convert action list to datum, handling syntax objects in action elements
    (map (lambda (elem)
           (if (identifier? elem)
               (syntax->datum elem)
               elem))
         action)))
