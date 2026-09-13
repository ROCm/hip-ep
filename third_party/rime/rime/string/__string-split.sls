(library (rime string __string-split)
  (export string-split)
  (import (rnrs (6))
          (rime loop)
          (rime logging)
          (rime string __string-start-with)
          (rime string __string-find-all))

  (define (string-split seperator string)
    (loop :trace-parser :trace-codegen
          :for offset :upfrom 0 :below (string-length string) :by step
          :with seperator-found? := (string-starts-with seperator offset string)
          :collect (substring string start offset) :if seperator-found?
          :with step := (if seperator-found? (string-length seperator) 1) :initially 1
          :with start := (if seperator-found? (fx+ offset step) start) :initially 0
          :finally :collect (substring string start (string-length string))
          )))
