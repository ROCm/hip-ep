#!r6rs
;;===----------------------------------------------------------------------===;;
;; Cast Pattern - Using Macro
;;===----------------------------------------------------------------------===;;

(library (patterns cast-macro-test)
  (export onnx-cast->hipsr-macro)
  (import (rnrs (6))
          (mlir ffi)
          (mlir pattern-macro))

  ;; Use the macro to define Cast pattern
  (define-conversion-pattern onnx-cast->hipsr-macro
    :if-match
      %cast = "onnx.Cast" (%input) :type (!input-type) -> !output-type
    :rewrite %cast
      ;; Get context
      (let ([ctx (mlir-get-hipsr-context-arg op)])
        ;; Create replacement operations
        (let* ([%placeholder (mlir-create-placeholder-op ctx %input !output-type 0)]
               [%new-cast (mlir-create-cast-op ctx %input %placeholder !output-type)])
          ;; Replace old with new
          (mlir-replace-op op %new-cast))))

) ;; end library
