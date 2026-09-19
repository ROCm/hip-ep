#!r6rs
(library (test proper-import)
  (export test1)
  (import (rnrs (6))
          (mlir pattern-macro))  ;; Import both macro AND keywords at runtime
  
  (define-conversion-pattern test1
    :match ((%out = "test.op" (%in) () : (!t) -> !t))
    :rewrite %out :with ((%new = "new.op" (%in) () : (!t) -> !t))))
