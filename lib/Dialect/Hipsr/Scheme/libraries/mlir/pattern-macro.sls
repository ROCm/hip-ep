#!r6rs
(library (mlir pattern-macro)
  (export define-conversion-pattern
          :match :rewrite :with :where
          :debug-parse :debug-analyze :debug-codegen :debug-matching
          = : -> :region :regions :attrs)
  (import (except (rnrs) =)
          (mlir pattern-keywords)  ;; Import keywords at run time for re-export
          (for (mlir pattern-keywords) expand)  ;; Also at expand time
          (for (mlir pattern-parse) expand)
          (for (mlir pattern-validate) expand)
          (for (mlir pattern-analyze) expand)
          (for (mlir pattern-codegen) expand))

  ;; Main macro: orchestrate 4 phases
  ;; Phase 1: Parse -> AST
  ;; Phase 2: Validate -> checked AST
  ;; Phase 3: Analyze -> AST with bindings and actions
  ;; Phase 4: Codegen -> final code
  (define-syntax define-conversion-pattern
    (lambda (stx)
      (let* ([ast-rec (parse-to-ast stx)]
             [validated (validate-ast ast-rec)]
             [analyzed (analyze-ast validated)])
        (generate-code analyzed))))
)
