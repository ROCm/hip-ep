#!r6rs
;;===----------------------------------------------------------------------===;;
;;
;; Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
;; Licensed under the MIT License.
;;
;;===----------------------------------------------------------------------===;;
;;
;; PrintPass - Example MLIR pass written entirely in Scheme
;;
;; This pass demonstrates writing MLIR transformations in pure Scheme using
;; the FFI bindings to MLIR operations. It walks all operations in a module
;; and prints their structure with configurable detail levels.
;;
;;===----------------------------------------------------------------------===;;

(library (printpass)
  (export run-pass)
  (import (rnrs (6))
          (only (chezscheme) format)  ; format is Chez-specific
          (mlir ffi)
          (rime loop))

;;===----------------------------------------------------------------------===;;
;; Configuration
;;===----------------------------------------------------------------------===;;

;; Default configuration for the pass.
;; Returns a closure that looks up configuration keys.
(define (default-config)
  (lambda (key)
    (case key
      [(show-operands) #t]      ; Show operand details
      [(show-results) #t]       ; Show result details
      [(operation-filter) #f]   ; Filter operations (or #f for all)
      [else #f])))

;;===----------------------------------------------------------------------===;;
;; Private Helper Functions
;;===----------------------------------------------------------------------===;;

;; Format operands list as comma-separated string.
;; Returns empty string for operations with no operands, otherwise returns
;; a formatted string like " | Operands[4]: 123, 456, 789, 012".
(define (format-operands op num-operands)
  (loop :initially := ""
        :for i :from 0 :to (- num-operands 1)
        :with operand := (mlir-operation-get-operand op i)
        :join-string operand :seperator ", "
        :finally (if (zero? num-operands)
                     :return-value
                     (format " | Operands[~a]: ~a" num-operands :return-value))))

;; Format results list as comma-separated string.
;; Returns empty string for operations with no results, otherwise returns
;; a formatted string like " | Results[2]: 345, 678".
(define (format-results op num-results)
  (loop :initially := ""
        :for i :from 0 :to (- num-results 1)
        :with result := (mlir-operation-get-result op i)
        :join-string result :seperator ", "
        :finally (if (zero? num-results)
                     :return-value
                     (format " | Results[~a]: ~a" num-results :return-value))))

;; Format complete operation details according to configuration.
;; Returns a string like: Operation: "hip.add" | Operands[4]: ... | Results[1]: ...
(define (format-operation op config)
  (let ((name (mlir-operation-name op))
        (num-operands (mlir-operation-num-operands op))
        (num-results (mlir-operation-num-results op)))
    (format "Operation: ~s~a~a"
            name
            (if (config 'show-operands)
                (format-operands op num-operands)
                "")
            (if (config 'show-results)
                (format-results op num-results)
                ""))))

;; Check if operation should be processed based on filter configuration.
(define (should-process-operation? op-name config)
  (let ((filter (config 'operation-filter)))
    (or (not filter) (member op-name filter))))

;;===----------------------------------------------------------------------===;;
;; Public API
;;===----------------------------------------------------------------------===;;

;; Entry point called from C++.
;;
;; Parameters:
;;   module-op - The MLIR module operation to process
;;   args      - Optional configuration (closure accepting config keys)
;;
;; The pass walks all operations in the module and logs their structure
;; at trace level. Configuration controls which details are shown.
(define (run-pass module-op . args)
  (let ((config (if (null? args) (default-config) (car args))))
    (mlir-log-info "Starting Pure Scheme MLIR Pass")
    (mlir-log-debug (format "Module: ~a" (mlir-operation-name module-op)))

    ;; Walk all operations in the module
    (mlir-operation-walk module-op
      (lambda (op)
        (let ((op-name (mlir-operation-name op)))
          (when (should-process-operation? op-name config)
            (mlir-log-trace (format-operation op config))))))

    (mlir-log-info "Completed Pure Scheme MLIR Pass")))

) ;; end library (printpass)
