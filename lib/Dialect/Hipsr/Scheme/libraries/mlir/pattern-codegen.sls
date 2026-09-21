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

  (define (generate-code ast-rec)
    (cond
      [(ast-pattern-expand-debug-parse? ast-rec)
       (generate-debug-ast ast-rec)]

      [(ast-pattern-expand-debug-analyze? ast-rec)
       (generate-debug-actions ast-rec)]

      [(ast-pattern-expand-debug-codegen? ast-rec)
       (generate-debug-codegen ast-rec)]

      [else
       (generate-pattern-matcher ast-rec)]))

  ;;-----------------------------------------------------------------------
  ;; Debug mode: AST output (parse phase)
  ;;-----------------------------------------------------------------------

  (define (generate-debug-ast ast-rec)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                  [root-op-name (ast-pattern-expand-root-op-name ast-rec)])
      (let* ([fname-sym (syntax->datum #'fname)]
             [root-op-str (syntax->datum #'root-op-name)]
             [match-data (map match-expand->datum
                              (vector->list (ast-pattern-expand-match ast-rec)))]
             [rewrite-data (map operation-expand->datum
                                (ast-pattern-expand-rewrite ast-rec))]
             [where-data (map where-binding-expand->datum
                              (ast-pattern-expand-where ast-rec))])
        (with-syntax ([ast-list (datum->syntax #'fname
                                  `(list 'function-name ',fname-sym
                                         'root-op-name ,root-op-str
                                         'match ',match-data
                                         'rewrite ',rewrite-data
                                         'where ',where-data
                                         'debug-parse? #t
                                         'debug-analyze? #f
                                         'debug-codegen? #f
                                         'debug-matching? #f))])
          #'(define fname ast-list)))))

  ;;-----------------------------------------------------------------------
  ;; Debug mode: Actions output (analyze phase)
  ;;-----------------------------------------------------------------------

  (define (generate-debug-actions ast-rec)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)]
                  [root-op-name (ast-pattern-expand-root-op-name ast-rec)])
      (let* ([fname-sym (syntax->datum #'fname)]
             [root-op-str (syntax->datum #'root-op-name)]
             [actions-data (map action->datum
                                (ast-pattern-expand-match-actions ast-rec))])
        (with-syntax ([ast-list (datum->syntax #'fname
                                  `(list 'function-name ',fname-sym
                                         'root-op-name ,root-op-str
                                         'match-actions ',actions-data
                                         'debug-parse? #f
                                         'debug-analyze? #t
                                         'debug-codegen? #f
                                         'debug-matching? #f))])
          #'(define fname ast-list)))))

  ;;-----------------------------------------------------------------------
  ;; Debug mode: Codegen output (codegen phase)
  ;;-----------------------------------------------------------------------

  (define (generate-debug-codegen ast-rec)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)])
      (let ([fname-sym (syntax->datum #'fname)])
        (with-syntax ([ast-list (datum->syntax #'fname
                                  `(list 'function-name ',fname-sym
                                         'debug-parse? #f
                                         'debug-analyze? #f
                                         'debug-codegen? #t
                                         'debug-matching? #f))])
          #'(define fname ast-list)))))

  ;;-----------------------------------------------------------------------
  ;; Pattern matcher generation
  ;;-----------------------------------------------------------------------

  (define (generate-pattern-matcher ast-rec)
    (with-syntax ([fname (ast-pattern-expand-function-name ast-rec)])
      #'(define fname
          (lambda (op operands-ref rewriter type-converter)
            #f))))

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
    ;; Convert action list to datum, handling syntax objects in labeled fields
    ;; Action format: (:tag (field-name . value) ...)
    (cons (car action)  ; Keep tag as-is
          (map (lambda (field)
                 ;; field is (field-name . value)
                 (let ([field-name (car field)]
                       [field-value (cdr field)])
                   (cons field-name
                         (if (identifier? field-value)
                             (syntax->datum field-value)
                             field-value))))
               (cdr action)))))
