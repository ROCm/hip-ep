##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipDialectIR STATIC
  HipDialect.cpp
)

# HipDialect.cpp aggregates every HIP op's TableGen-generated definitions; with
# the ops added for the Procyon model set it exceeds MSVC's default COFF
# section limit (fatal error C1128). /bigobj raises that limit.
if(MSVC)
  target_compile_options(HipDialectIR PRIVATE /bigobj)
endif()

add_dependencies(HipDialectIR
  HipDialectIncGen
  HipTypesIncGen
  HipOpsIncGen
)

target_link_libraries(HipDialectIR PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDestinationStyleOpInterface
  MLIRFuncDialect
)
