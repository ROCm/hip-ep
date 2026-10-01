#!r6rs
(library (mlir ddr)
  (export define-conversion-pattern
          :if-match :then-let :rewrite :with :where
          :debug-parse :debug-validate :debug-analyze :debug-codegen :debug-matching
          = : -> :region :regions
          :index
          make-unbound-value)
  (import (except (rnrs) =)
          (mlir ddr keywords)  ;; Import keywords at run time for re-export
          (mlir core ir)               ;; with-rewrite-builder etc. used in generated code
          (mlir ddr rewrite)               ;; with-mlir-ops used in generated :rewrite bodies
          (for (mlir ddr keywords) expand)  ;; Also at expand time
          (for (mlir ddr ast) expand)  ;; For AST predicates
          (for (mlir ddr parse) expand)
          (for (mlir ddr validate) expand)
          (for (mlir ddr analyze) expand)
          (for (mlir ddr codegen) expand)
          (for (only (mlir ddr codegen) make-unbound-value) expand))

  ;; Main macro: orchestrate 4 phases (waterfall style)
  ;; Phase 1: Parse -> AST
  ;; Phase 2: Validate -> checked AST (skipped if :debug-parse)
  ;; Phase 3: Analyze -> AST with bindings and actions (skipped if :debug-parse or :debug-validate)
  ;; Phase 4: Codegen -> final code (or debug output)
  (define-syntax define-conversion-pattern
    (lambda (stx)
      (let ([ast-rec (parse-to-ast stx)])
        (if (ast-pattern-expand-debug-parse? ast-rec)
            (generate-debug-ast ast-rec)
            (let ([validated (validate-ast ast-rec)])
              (if (ast-pattern-expand-debug-validate? validated)
                  (generate-debug-ast validated)
                  (let ([analyzed (analyze-ast validated)])
                    (if (ast-pattern-expand-debug-analyze? analyzed)
                        (generate-debug-ast analyzed)
                        (let ([real-code (generate-pattern-matchAndRewrite analyzed)])
                          (if (ast-pattern-expand-debug-codegen? analyzed)
                              (generate-debug-codegen analyzed real-code)
                              real-code))))))))))
)
