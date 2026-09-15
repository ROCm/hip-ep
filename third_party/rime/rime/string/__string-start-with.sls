(library (rime string __string-start-with)
  (export string-starts-with)
  (import (rnrs (6))
          (rime loop))
  (define (string-starts-with prefix offset string)
    (loop :for index :upfrom 0
          :for c1 :in-string prefix :offset 0
          :for c2 :in-string string :offset offset
          :break :unless (char=? c1 c2)
          :count
          :finally (fx=? :return-value (string-length prefix))
          )))
