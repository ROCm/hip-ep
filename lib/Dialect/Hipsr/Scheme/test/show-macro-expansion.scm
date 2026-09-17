#!/usr/bin/env scheme-script

;; Load the macro library directly
(import (chezscheme)
        (mlir pattern-macro))

;; Expand the macro
(define expanded
  (expand 
    '(define-conversion-pattern cast-pattern
       :match 
       (%r = "onnx.Cast" (%ctx %input) ((to = !to_type)) : (!input-type) -> !output-type)
       :rewrite 
       (begin
         (%0 = "hipsr.placeholder" (%ctx %input)
               ((placeholder_type = "#hipsr.placeholder_type<normal>")) 
               : types -> result)
         (%1 = "hipsr.cast" (%ctx %input %0) 
               ((result_type = (mlir-tensor-attach-address-space !output-type)))
               : types -> result)
         %1))))

(pretty-print (syntax->datum expanded))
