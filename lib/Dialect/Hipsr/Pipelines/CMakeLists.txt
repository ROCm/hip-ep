##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipsrDialectPipelines STATIC
  Pipelines.cpp
)

# The pipeline will add the hipsr transform passes, so link them now. Adding
# the first pass then only changes Pipelines.cpp.
add_dependencies(HipsrDialectPipelines
  HipsrTransformsPassIncGen
)

target_link_libraries(HipsrDialectPipelines PUBLIC
  HipsrDialectTransforms
  MLIRPass
  MLIRIR
  MLIRSupport
)
