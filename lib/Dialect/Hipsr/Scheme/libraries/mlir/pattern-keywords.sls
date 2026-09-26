#!r6rs
(library (mlir pattern-keywords)
  (export :match :then-let :rewrite :with :where
          :debug-parse :debug-validate :debug-analyze :debug-codegen :debug-matching
          = : -> :region :regions)
  (import (except (rnrs) =))

  ;; Define keywords as syntax (for cross-library hygiene)
  (define-syntax :match (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :then-let (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :rewrite (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :with (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :where (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-parse (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-validate (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-analyze (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-codegen (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :debug-matching (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax = (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax : (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax -> (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :region (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
  (define-syntax :regions (lambda (x) (syntax-violation 'pattern-keyword "misplaced aux keyword" x)))
)
