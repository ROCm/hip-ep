#!r6rs
(library (mlir pattern-macro)
  (export define-conversion-pattern
          :match :then-let :rewrite :with :where
          :debug-parse :debug-validate :debug-analyze :debug-codegen :debug-matching
          = : -> :region :regions)
  (import (except (rnrs) =)
          (mlir pattern-keywords)  ;; Import keywords at run time for re-export
          (for (mlir pattern-keywords) expand)  ;; Also at expand time
          (for (mlir pattern-ast) expand)  ;; For AST predicates
          (for (mlir pattern-parse) expand)
          (for (mlir pattern-validate) expand)
          (for (mlir pattern-analyze) expand)
          (for (mlir pattern-codegen) expand))

  ;; Main macro: orchestrate 4 phases
  ;; Phase 1: Parse -> AST
  ;; Phase 2: Validate -> checked AST (skipped if :debug-parse)
  ;; Phase 3: Analyze -> AST with bindings and actions (skipped if :debug-parse or :debug-analyze)
  ;; Phase 4: Codegen -> final code
  (define-syntax define-conversion-pattern
    (lambda (stx)
      (let* ([ast-rec (parse-to-ast stx)]
             ;; Skip validation if :debug-parse is set
             [validated (if (ast-pattern-expand-debug-parse? ast-rec)
                            ast-rec
                            (validate-ast ast-rec))]
             ;; Skip analysis if :debug-parse is set (validation already skipped)
             ;; or if :debug-analyze? is set (only skip analysis, validation ran)
             [analyzed (if (or (ast-pattern-expand-debug-parse? validated)
                              (ast-pattern-expand-debug-validate? validated))
                           validated
                           (analyze-ast validated))])
        (generate-code analyzed))))
)
