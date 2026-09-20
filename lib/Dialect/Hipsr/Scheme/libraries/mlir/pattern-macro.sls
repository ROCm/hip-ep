#!r6rs
(library (mlir pattern-macro)
  (export define-conversion-pattern
          :match :rewrite :with :where :debug-ast :debug-matching
          = : -> :region :regions :attrs)
  (import (except (rnrs) =)
          (mlir pattern-keywords)  ;; Import keywords at run time for re-export
          (for (mlir pattern-keywords) expand)  ;; Also at expand time
          (for (mlir pattern-parse) expand)
          (for (mlir pattern-validate) expand)
          (for (mlir pattern-codegen) expand))

  ;; Main macro: orchestrate parse -> validate -> codegen
  (define-syntax define-conversion-pattern
    (lambda (stx)
      (let* ([ast-rec (parse-to-ast stx)]
             [validated (validate-ast ast-rec)])
        (generate-code stx validated))))
)
