#!r6rs
(library (rime logging)
  (export logger
          object-to-string
          log-level
          :source-location
          :who
          :level
          :trace
          :info
          :debug
          :warning
          :error
          :critical)
  (import (rime logging __logger)))
