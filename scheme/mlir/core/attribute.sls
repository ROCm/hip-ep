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
;;     value : Scheme value whose shape matches the C++ expectation for that type.
;;
;; C++ convention: every mlir_make_attr_<type> function has the uniform
;; signature (uptr ctx, ptr value) → uptr.  New attribute types are
;; discoverable automatically via foreign-entry? — no Scheme change needed.
;;
;;===----------------------------------------------------------------------===;;

(library (mlir core attribute)
  (export make-mlir-attribute)

  (import (rnrs)
          (only (chezscheme) foreign-procedure foreign-entry?
                make-eq-hashtable hashtable-ref hashtable-set!
                symbol->string string-map string-append substring
                string-length))

  ;; Derive the C symbol name from a type keyword.
  ;; :dense-resource → "mlir_make_attr_dense_resource"
  (define (type->sym-name type)
    (let* ([s (symbol->string type)]
           [s (substring s 1 (string-length s))]   ; strip leading ":"
           [s (string-map (lambda (c) (if (char=? c #\-) #\_ c)) s)])
      (string-append "mlir_make_attr_" s)))

  ;; Cache: type keyword → foreign-procedure wrapper (or #f if unavailable).
  (define %cache (make-eq-hashtable))

  (define (lookup-proc type)
    (or (hashtable-ref %cache type #f)
        (let* ([sym  (type->sym-name type)]
               [proc (and (foreign-entry? sym)
                          (foreign-procedure sym (uptr scheme-object) uptr))])
          (hashtable-set! %cache type (or proc 'missing))
          proc)))

  (define (make-mlir-attribute ctx type value)
    (let ([proc (lookup-proc type)])
      (if (and proc (not (eq? proc 'missing)))
          (proc ctx value)
          (error 'make-mlir-attribute
                 "unknown or unavailable attr type" type))))

) ;; end library (mlir core attribute)
