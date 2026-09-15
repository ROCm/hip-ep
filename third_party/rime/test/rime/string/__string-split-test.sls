(library (test rime string __string-split-test)
  (export main)
  (import (rnrs (6))
          (rime string __string-split)
          (rime unit-test))
  (define-test test-string-split
    (CHECK equal?
           (string-split "::" "a::b")
           '("a" "b"))

    (CHECK equal?
           (string-split "::" "a::::b")
           '("a" "" "b"))

    (CHECK equal?
           (string-split "::" "::abc::::def")
           '("" "abc" "" "def"))

    (CHECK equal?
           (string-split "::" "::abc::::def::")
           '("" "abc" "" "def" "")))

  (define (main)
    (run-all-tests)))
