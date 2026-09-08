add_library(HipToTosa STATIC
	HipToTosa.cpp
)

add_dependencies(HipToTosa
	MLIRTosaOpsIncGen
	HipDialectIncGen
	HipTypesIncGen
	HipOpsIncGen
	HipPassesIncGen
)

target_link_libraries(HipToTosa PUBLIC
	HipDialectIR
	MLIRPass
	MLIRTosaDialect
)
