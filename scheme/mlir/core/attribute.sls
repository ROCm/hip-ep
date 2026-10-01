#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; (mlir core attribute) — MLIR attribute construction.
;;
;; Mirrors mlir/IR/Attribute.h. Attributes are first-class opaque uptr values
;; (Attribute::getAsOpaquePointer / getFromOpaquePointer).
;;
;;   (make-mlir-attribute ctx type value)
;;     ctx   : MLIRContext* uptr
;;     type  : a keyword symbol, e.g. :i64, :index, :i32-array, :i64-array,
;;             :dense-resource, or any future :foo registered as
;;             mlir_make_attr_foo in C++.
;;     value : Scheme value whose shape matches the C++ expectation for that type:
;;               :i64        — Scheme integer
;;               :index      — Scheme integer
;;               :i32-array  — Scheme list of integers
;;               :i64-array  — Scheme list of integers
;;               :dense-resource — Scheme list (result-type-uptr key-string
;;                                              data-addr-integer data-size-integer)
;;
;; C++ convention: every mlir_make_attr_<type> function has the uniform
;; signature (uptr ctx, ptr value) → uptr.  New attribute types are
;; discoverable automatically via foreign-entry? — no Scheme change needed.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core attribute)
  (export make-mlir-attribute
          ;; Type inspection
          mlir-type-element-type
          mlir-type-integer-width
          mlir-type-is-unsigned
          ;; Attribute inspection
          mlir-op-get-float-attr
          mlir-attr-is-splat
          mlir-attr-splat-float-value
          mlir-op-get-operand-segment-sizes)

  (import (rnrs)
          (only (chezscheme) foreign-procedure foreign-entry?
                make-eq-hashtable hashtable-ref hashtable-set!))

  ;; Derive the C symbol name from a type keyword.
  ;; :dense-resource → "mlir_make_attr_dense_resource"
  ;; :i64            → "mlir_make_attr_i64"
  (define (type->sym-name type)
    (let* ([s    (symbol->string type)]
           [s    (substring s 1 (string-length s))]   ; strip leading ":"
           [body (list->string
                   (map (lambda (c) (if (char=? c #\-) #\_ c))
                        (string->list s)))])
      (string-append "mlir_make_attr_" body)))

  ;; === Type inspection ===

  (define mlir-type-element-type
    (foreign-procedure "mlir_type_element_type" (uptr) uptr))

  (define mlir-type-integer-width
    (foreign-procedure "mlir_type_integer_width" (uptr) uptr))

  (define mlir-type-is-unsigned
    (foreign-procedure "mlir_type_is_unsigned" (uptr) boolean))

  ;; === Attribute inspection ===

  (define mlir-op-get-float-attr
    (foreign-procedure "mlir_op_get_float_attr" (uptr string) double))

  (define mlir-attr-is-splat
    (foreign-procedure "mlir_attr_is_splat" (uptr) boolean))

  (define mlir-attr-splat-float-value
    (foreign-procedure "mlir_attr_splat_float_value" (uptr) double))

  (define mlir-op-get-operand-segment-sizes
    (foreign-procedure "mlir_op_get_operand_segment_sizes" (uptr) scheme-object))

  ;; Per-type procedure cache: type keyword → foreign-procedure wrapper.
  ;; 'missing means the C symbol was not found via foreign-entry?.
  (define %cache (make-eq-hashtable))

  ;; Look up (or cache) the C procedure for a given type keyword.
  ;; Returns the procedure, or #f if the type is not registered.
  (define (lookup-proc type)
    (or (hashtable-ref %cache type #f)
        (let* ([sym  (type->sym-name type)]
               [proc (and (foreign-entry? sym)
                          (foreign-procedure sym (uptr scheme-object) uptr))])
          (hashtable-set! %cache type (or proc 'missing))
          proc)))

  ;; Construct an MLIR attribute by type keyword.
  ;; Dispatches dynamically to mlir_make_attr_<type> via foreign-entry?.
  ;; ctx:   MLIRContext* uptr — provides context for attribute construction
  ;; type:  keyword symbol like :i64, :index, :i32-array, :i64-array,
  ;;        :dense-resource, or any :foo for which mlir_make_attr_foo is registered
  ;; value: Scheme value appropriate for the type (see file header)
  ;; Returns: Attribute opaque uptr (Attribute::getAsOpaquePointer())
  ;; Raises:  error if type is unknown or C symbol not registered
  (define (make-mlir-attribute ctx type value)
    (let ([proc (lookup-proc type)])
      (if (and proc (not (eq? proc 'missing)))
          (proc ctx value)
          (error 'make-mlir-attribute
                 "unknown or unavailable attr type" type))))

) ;; end library (mlir core attribute)
