##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipToTosa STATIC
	HipToTosa.cpp
)

add_dependencies(HipToTosa
	HipDialectIncGen
	HipTypesIncGen
	HipOpsIncGen
	HipPassesIncGen
)

target_link_libraries(HipToTosa PUBLIC
	HipDialectIR
	MLIRPass
	MLIRTensorDialect
	MLIRTosaDialect
)
