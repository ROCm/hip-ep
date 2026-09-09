#!r6rs
(library (rime unit-test)
  (export run-all-tests define-test CHECK)
  (import (rime unit-test __define-test)
          (rime unit-test __check)))
