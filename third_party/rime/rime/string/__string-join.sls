#!r6rs
(library (rime string __string-join)
  (export string-join)
  (import (rnrs (6))
          (rime loop))
  (define (string-join seperator list-of-strings)
    (loop :for s :in list-of-strings
          :join-string s :seperator seperator)))
