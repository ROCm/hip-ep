##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# E2E Integration Tests
#=============================================================================
# These tests used to compile MLIR -> per-model DLL and exec via
# hip-test-dll. The DLL path is gone (see ../../docs/design/alternatives/
# and the bitcode switchover plan): the compiler now emits LLVM bitcode
# that the EP JITs in-process. The bitcode-era equivalent harness lives
# at tools/hip-test-bc (loads via BitcodeJIT, same CLI surface as the
# retired hip-test-dll). This subdirectory will be wired back up to
# register MLIR -> .bc -> hip-test-bc invocations with ctest in a
# follow-up; the .mlir test inputs in ../lit/e2e/ are preserved.

message(STATUS "E2E integration tests disabled (pending hip-test-bc "
               "ctest wiring -- the harness exists, registration is the "
               "remaining work)")
