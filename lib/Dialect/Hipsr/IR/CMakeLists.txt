##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectIR STATIC
  HipsrDialect.cpp
  HipsrOps.cpp
  HipsrTypes.cpp
)

add_dependencies(HipsrDialectIR
  HipsrDialectIncGen
  HipsrEnumsIncGen
  HipsrAttrsIncGen
)

target_link_libraries(HipsrDialectIR PUBLIC
  MLIRIR
  MLIRSupport
)
