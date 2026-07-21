##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Aggregate hipsr transform-pass declarations/registration (upstream MLIR
# layout: one Passes.td / Passes.h.inc for the dialect's transforms).
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Hipsr)
add_public_tablegen_target(HipsrTransformsPassIncGen)
