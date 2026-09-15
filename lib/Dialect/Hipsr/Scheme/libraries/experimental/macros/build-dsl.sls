#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; Build DSL - mlir-build macro
;;
;; General-purpose MLIR IR construction DSL.
;; See tech/design/2026-09-14-mlir-build-dsl.md for syntax.
;;
;;===----------------------------------------------------------------------===;;

(library (build-dsl)
  (export mlir-build)
  (import (rnrs (6))
          (only (chezscheme) format))

  ;; mlir-build macro
  ;;
  ;; Syntax:
  ;;   (mlir-build
  ;;     %result = "dialect.op" (%operand ...)
  ;;               [:regions (region ...)]
  ;;               [(:attr-name ,value)]
  ;;               :type (type ...) -> type
  ;;     ...)
  ;;
  ;; Creates MLIR operations with automatic location handling.
  (define-syntax mlir-build
    (lambda (x)
      (syntax-case x (= :type :regions)
        [(_ op-spec ...)

         ;; TODO: Parse operation specifications
         ;; TODO: Generate mlir-create-* calls
         ;; TODO: Handle regions and attributes with quasiquote/unquote

         #'(begin
             (error 'mlir-build
                    "Macro implementation not yet complete"))])))

) ;; end library (build-dsl)
