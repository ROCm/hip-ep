#!r6rs
;; Minimal test to reproduce FFI crash

(library (test-ffi-crash)
  (export run-tests)
  (import (chezscheme))

  ;; Test 1: Simplest void function
  (define test-void-simple
    (foreign-procedure "test_void_simple" (unsigned-64) void))

  ;; Test 2: Void function like working log functions
  (define test-void-like-log
    (foreign-procedure "test_void_like_log" (string) void))

  ;; Test 3: No-op function
  (define test-void-noop
    (foreign-procedure "test_void_noop" (unsigned-64) void))

  ;; Test 4: With MLIR headers
  (define test-void-with-mlir-headers
    (foreign-procedure "test_void_with_mlir_headers" (unsigned-64) void))

  ;; Test 5: With pointer cast
  (define test-void-with-cast
    (foreign-procedure "test_void_with_cast" (unsigned-64) void))

  (define (run-tests)
    (display "Test 1: test-void-simple\n")
    (test-void-simple 12345)

    (display "Test 2: test-void-like-log\n")
    (test-void-like-log "hello from scheme")

    (display "Test 3: test-void-noop\n")
    (test-void-noop 0)

    (display "Test 4: test-void-with-mlir-headers\n")
    (test-void-with-mlir-headers 0)

    (display "Test 5: test-void-with-cast\n")
    (test-void-with-cast 99999)

    (display "All tests passed!\n"))
)
