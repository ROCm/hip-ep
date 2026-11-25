##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# libhipep_backend without a device: descriptions of CPU-built ggml graphs,
# the sizes their operations make one, dynamic signatures and fits.
add_executable(hipep-backend-test test_hipep_backend.cpp)
target_link_libraries(hipep-backend-test PRIVATE hipep_backend_graph)
add_test(NAME HipepBackendUnitTest COMMAND hipep-backend-test)
