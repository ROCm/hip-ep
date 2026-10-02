(library (rime string __string-find-all)
  (export string-find-all)
  (import (rnrs (6))
          (rime loop)
          (rime string __string-start-with))

  (define (string-find-all predicator string)
    (loop :for index :upfrom 0
          :for _c1 :in-string string
          :collect index :if (predicator string index)
          )))
