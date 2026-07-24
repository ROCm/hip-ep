##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_custom_target(morphizen-unit-test-GTest.hello.38
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GTest.hello
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GTest.hello.38 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GTest/hello"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GTest.hello
    )
target_sources(morphizen-unit-test-GTest.hello.38 PRIVATE
        # morphizen_unit_test_main.cpp # line 38 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConfigTest.Simple.26
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConfigTest.Simple
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConfigTest.Simple.26 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConfigTest/Simple"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConfigTest.Simple
    )
target_sources(morphizen-unit-test-ConfigTest.Simple.26 PRIVATE
        # morphizen/test_config.cpp # line 26 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConfigTest.EmptyProviderOption.34
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConfigTest.EmptyProviderOption
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConfigTest.EmptyProviderOption.34 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConfigTest/EmptyProviderOption"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConfigTest.EmptyProviderOption
    )
target_sources(morphizen-unit-test-ConfigTest.EmptyProviderOption.34 PRIVATE
        # morphizen/test_config.cpp # line 34 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConfigTest.ProviderOptionCacheDir.42
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConfigTest.ProviderOptionCacheDir
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConfigTest.ProviderOptionCacheDir.42 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConfigTest/ProviderOptionCacheDir"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConfigTest.ProviderOptionCacheDir
    )
target_sources(morphizen-unit-test-ConfigTest.ProviderOptionCacheDir.42 PRIVATE
        # morphizen/test_config.cpp # line 42 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConfigTest.SessionConfigs.54
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConfigTest.SessionConfigs
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConfigTest.SessionConfigs.54 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConfigTest/SessionConfigs"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConfigTest.SessionConfigs
    )
target_sources(morphizen-unit-test-ConfigTest.SessionConfigs.54 PRIVATE
        # morphizen/test_config.cpp # line 54 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.Load.16
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.Load
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.Load.16 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/Load"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.Load
    )
target_sources(morphizen-unit-test-ModelTest.Load.16 PRIVATE
        # morphizen/test_model.cpp # line 16 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.Clone.20
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.Clone
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.Clone.20 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/Clone"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.Clone
    )
target_sources(morphizen-unit-test-ModelTest.Clone.20 PRIVATE
        # morphizen/test_model.cpp # line 20 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.MainGraph.27
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.MainGraph
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.MainGraph.27 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/MainGraph"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.MainGraph
    )
target_sources(morphizen-unit-test-ModelTest.MainGraph.27 PRIVATE
        # morphizen/test_model.cpp # line 27 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.SetAndGetMetadata.32
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.SetAndGetMetadata
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.SetAndGetMetadata.32 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/SetAndGetMetadata"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.SetAndGetMetadata
    )
target_sources(morphizen-unit-test-ModelTest.SetAndGetMetadata.32 PRIVATE
        # morphizen/test_model.cpp # line 32 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.ImplicitConversion.49
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.ImplicitConversion
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.ImplicitConversion.49 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/ImplicitConversion"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.ImplicitConversion
    )
target_sources(morphizen-unit-test-ModelTest.ImplicitConversion.49 PRIVATE
        # morphizen/test_model.cpp # line 49 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.ModelCreationTest.66
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.ModelCreationTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.ModelCreationTest.66 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/ModelCreationTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.ModelCreationTest
    )
target_sources(morphizen-unit-test-ModelTest.ModelCreationTest.66 PRIVATE
        # morphizen/test_model.cpp # line 66 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.LoadAndSave.15
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.LoadAndSave
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.LoadAndSave.15 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/LoadAndSave"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.LoadAndSave
    )
target_sources(morphizen-unit-test-GraphTest.LoadAndSave.15 PRIVATE
        # morphizen/test_graph.cpp # line 15 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.FindNodeArgGraphInput.90
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.FindNodeArgGraphInput
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.FindNodeArgGraphInput.90 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/FindNodeArgGraphInput"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.FindNodeArgGraphInput
    )
target_sources(morphizen-unit-test-GraphTest.FindNodeArgGraphInput.90 PRIVATE
        # morphizen/test_graph.cpp # line 90 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.FindNodeArgGraphOutput.110
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.FindNodeArgGraphOutput
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.FindNodeArgGraphOutput.110 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/FindNodeArgGraphOutput"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.FindNodeArgGraphOutput
    )
target_sources(morphizen-unit-test-GraphTest.FindNodeArgGraphOutput.110 PRIVATE
        # morphizen/test_graph.cpp # line 110 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.NodesInTopologicalOrder.132
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.NodesInTopologicalOrder
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.NodesInTopologicalOrder.132 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/NodesInTopologicalOrder"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NodesInTopologicalOrder
    )
target_sources(morphizen-unit-test-GraphTest.NodesInTopologicalOrder.132 PRIVATE
        # morphizen/test_graph.cpp # line 132 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.NodeIndex.162
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.NodeIndex
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.NodeIndex.162 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/NodeIndex"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NodeIndex
    )
target_sources(morphizen-unit-test-GraphTest.NodeIndex.162 PRIVATE
        # morphizen/test_graph.cpp # line 162 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.FindConsumers.176
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.FindConsumers
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.FindConsumers.176 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/FindConsumers"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.FindConsumers
    )
target_sources(morphizen-unit-test-GraphTest.FindConsumers.176 PRIVATE
        # morphizen/test_graph.cpp # line 176 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.NodeArgFindProducer.190
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.NodeArgFindProducer
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.NodeArgFindProducer.190 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/NodeArgFindProducer"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NodeArgFindProducer
    )
target_sources(morphizen-unit-test-GraphTest.NodeArgFindProducer.190 PRIVATE
        # morphizen/test_graph.cpp # line 190 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.Fuse.202
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.Fuse
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.Fuse.202 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/Fuse"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.Fuse
    )
target_sources(morphizen-unit-test-GraphTest.Fuse.202 PRIVATE
        # morphizen/test_graph.cpp # line 202 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.TryFuse.248
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.TryFuse
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.TryFuse.248 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/TryFuse"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.TryFuse
    )
target_sources(morphizen-unit-test-GraphTest.TryFuse.248 PRIVATE
        # morphizen/test_graph.cpp # line 248 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.NewConstantInitializer.259
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.NewConstantInitializer
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.NewConstantInitializer.259 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/NewConstantInitializer"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NewConstantInitializer
    )
target_sources(morphizen-unit-test-GraphTest.NewConstantInitializer.259 PRIVATE
        # morphizen/test_graph.cpp # line 259 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.VirtualFuse.521
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.VirtualFuse
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.VirtualFuse.521 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/VirtualFuse"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.VirtualFuse
    )
target_sources(morphizen-unit-test-GraphTest.VirtualFuse.521 PRIVATE
        # morphizen/test_graph.cpp # line 521 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int8_scalar.42
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int8_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int8_scalar.42 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int8_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int8_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int8_scalar.42 PRIVATE
        # morphizen/test_const_data.cpp # line 42 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int8.53
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int8
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int8.53 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int8"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int8
    )
target_sources(morphizen-unit-test-ConstDataTest.int8.53 PRIVATE
        # morphizen/test_const_data.cpp # line 53 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint8_scalar.66
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint8_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint8_scalar.66 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint8_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint8_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint8_scalar.66 PRIVATE
        # morphizen/test_const_data.cpp # line 66 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint8.77
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint8
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint8.77 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint8"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint8
    )
target_sources(morphizen-unit-test-ConstDataTest.uint8.77 PRIVATE
        # morphizen/test_const_data.cpp # line 77 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int16_scalar.90
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int16_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int16_scalar.90 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int16_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int16_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int16_scalar.90 PRIVATE
        # morphizen/test_const_data.cpp # line 90 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int16.101
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int16
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int16.101 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int16"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int16
    )
target_sources(morphizen-unit-test-ConstDataTest.int16.101 PRIVATE
        # morphizen/test_const_data.cpp # line 101 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint16_scalar.114
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint16_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint16_scalar.114 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint16_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint16_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint16_scalar.114 PRIVATE
        # morphizen/test_const_data.cpp # line 114 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint16.125
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint16
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint16.125 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint16"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint16
    )
target_sources(morphizen-unit-test-ConstDataTest.uint16.125 PRIVATE
        # morphizen/test_const_data.cpp # line 125 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int32_scalar.138
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int32_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int32_scalar.138 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int32_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int32_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int32_scalar.138 PRIVATE
        # morphizen/test_const_data.cpp # line 138 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int32.149
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int32
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int32.149 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int32"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int32
    )
target_sources(morphizen-unit-test-ConstDataTest.int32.149 PRIVATE
        # morphizen/test_const_data.cpp # line 149 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint32_scalar.162
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint32_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint32_scalar.162 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint32_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint32_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint32_scalar.162 PRIVATE
        # morphizen/test_const_data.cpp # line 162 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint32.173
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint32
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint32.173 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint32"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint32
    )
target_sources(morphizen-unit-test-ConstDataTest.uint32.173 PRIVATE
        # morphizen/test_const_data.cpp # line 173 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int64_scalar.186
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int64_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int64_scalar.186 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int64_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int64_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int64_scalar.186 PRIVATE
        # morphizen/test_const_data.cpp # line 186 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int64.197
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int64
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int64.197 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int64"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int64
    )
target_sources(morphizen-unit-test-ConstDataTest.int64.197 PRIVATE
        # morphizen/test_const_data.cpp # line 197 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint64_scalar.210
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint64_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint64_scalar.210 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint64_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint64_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint64_scalar.210 PRIVATE
        # morphizen/test_const_data.cpp # line 210 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint64.221
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint64
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint64.221 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint64"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint64
    )
target_sources(morphizen-unit-test-ConstDataTest.uint64.221 PRIVATE
        # morphizen/test_const_data.cpp # line 221 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case0.92
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case0
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case0.92 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case0"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case0
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case0.92 PRIVATE
        # morphizen/test_anchor_point.cpp # line 92 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case1.93
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case1
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case1.93 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case1"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case1
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case1.93 PRIVATE
        # morphizen/test_anchor_point.cpp # line 93 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case2.94
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case2
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case2.94 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case2"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case2
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case2.94 PRIVATE
        # morphizen/test_anchor_point.cpp # line 94 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case3.95
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case3
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case3.95 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case3"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case3
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case3.95 PRIVATE
        # morphizen/test_anchor_point.cpp # line 95 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case4.96
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case4
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case4.96 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case4"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case4
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case4.96 PRIVATE
        # morphizen/test_anchor_point.cpp # line 96 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Append.97
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Append
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Append.97 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Append"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Append
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Append.97 PRIVATE
        # morphizen/test_anchor_point.cpp # line 97 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ImmutableMapTest.InsertSingleNode.22
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ImmutableMapTest.InsertSingleNode
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ImmutableMapTest.InsertSingleNode.22 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ImmutableMapTest/InsertSingleNode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ImmutableMapTest.InsertSingleNode
    )
target_sources(morphizen-unit-test-ImmutableMapTest.InsertSingleNode.22 PRIVATE
        # morphizen/test_immutable_map.cpp # line 22 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ImmutableMapTest.InsertMultipleNodes.29
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ImmutableMapTest.InsertMultipleNodes
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ImmutableMapTest.InsertMultipleNodes.29 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ImmutableMapTest/InsertMultipleNodes"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ImmutableMapTest.InsertMultipleNodes
    )
target_sources(morphizen-unit-test-ImmutableMapTest.InsertMultipleNodes.29 PRIVATE
        # morphizen/test_immutable_map.cpp # line 29 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PatternTest.CommutableNode.34
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PatternTest.CommutableNode
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PatternTest.CommutableNode.34 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PatternTest/CommutableNode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PatternTest.CommutableNode
    )
target_sources(morphizen-unit-test-PatternTest.CommutableNode.34 PRIVATE
        # morphizen/test_pattern.cpp # line 34 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PatternTest.LoadSaveBinary.89
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PatternTest.LoadSaveBinary
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PatternTest.LoadSaveBinary.89 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PatternTest/LoadSaveBinary"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PatternTest.LoadSaveBinary
    )
target_sources(morphizen-unit-test-PatternTest.LoadSaveBinary.89 PRIVATE
        # morphizen/test_pattern.cpp # line 89 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.ReadFileTest.46
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.ReadFileTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.ReadFileTest.46 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/ReadFileTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.ReadFileTest
    )
target_sources(morphizen-unit-test-PassContextTest.ReadFileTest.46 PRIVATE
        # morphizen/test_pass_context.cpp # line 46 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.UntarCacheTest.113
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.UntarCacheTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.UntarCacheTest.113 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/UntarCacheTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.UntarCacheTest
    )
target_sources(morphizen-unit-test-PassContextTest.UntarCacheTest.113 PRIVATE
        # morphizen/test_pass_context.cpp # line 113 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.TestEmptyFiles.128
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.TestEmptyFiles
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.TestEmptyFiles.128 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/TestEmptyFiles"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.TestEmptyFiles
    )
target_sources(morphizen-unit-test-PassContextTest.TestEmptyFiles.128 PRIVATE
        # morphizen/test_pass_context.cpp # line 128 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextConfigTest.Config.401
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextConfigTest.Config
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextConfigTest.Config.401 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextConfigTest/Config"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextConfigTest.Config
    )
target_sources(morphizen-unit-test-PassContextConfigTest.Config.401 PRIVATE
        # morphizen/test_pass_context.cpp # line 401 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextConfigTest.Target.415
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextConfigTest.Target
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextConfigTest.Target.415 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextConfigTest/Target"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextConfigTest.Target
    )
target_sources(morphizen-unit-test-PassContextConfigTest.Target.415 PRIVATE
        # morphizen/test_pass_context.cpp # line 415 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-NodeBuilderTest.SkipSimplifiedLayerNormalization.15
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=NodeBuilderTest.SkipSimplifiedLayerNormalization
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-NodeBuilderTest.SkipSimplifiedLayerNormalization.15 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/NodeBuilderTest/SkipSimplifiedLayerNormalization"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=NodeBuilderTest.SkipSimplifiedLayerNormalization
    )
target_sources(morphizen-unit-test-NodeBuilderTest.SkipSimplifiedLayerNormalization.15 PRIVATE
        # morphizen/test_node_builder.cpp # line 15 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarBallTest.TarTest.69
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarBallTest.TarTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarBallTest.TarTest.69 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarBallTest/TarTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarBallTest.TarTest
    )
target_sources(morphizen-unit-test-TarBallTest.TarTest.69 PRIVATE
        # morphizen/test_tarball.cpp # line 69 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarBallTest.Encrypt_Test.168
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarBallTest.Encrypt_Test
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarBallTest.Encrypt_Test.168 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarBallTest/Encrypt_Test"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarBallTest.Encrypt_Test
    )
target_sources(morphizen-unit-test-TarBallTest.Encrypt_Test.168 PRIVATE
        # morphizen/test_tarball.cpp # line 168 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PluginTest.StaticHelloPlugin.9
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PluginTest.StaticHelloPlugin
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PluginTest.StaticHelloPlugin.9 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PluginTest/StaticHelloPlugin"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PluginTest.StaticHelloPlugin
    )
target_sources(morphizen-unit-test-PluginTest.StaticHelloPlugin.9 PRIVATE
        # morphizen/test_plugin.cpp # line 9 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PluginTest.DynamicHelloPlugin.31
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PluginTest.DynamicHelloPlugin
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PluginTest.DynamicHelloPlugin.31 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PluginTest/DynamicHelloPlugin"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PluginTest.DynamicHelloPlugin
    )
target_sources(morphizen-unit-test-PluginTest.DynamicHelloPlugin.31 PRIVATE
        # morphizen/test_plugin.cpp # line 31 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarEntryTest.ReadFrom.11
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarEntryTest.ReadFrom
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarEntryTest.ReadFrom.11 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarEntryTest/ReadFrom"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarEntryTest.ReadFrom
    )
target_sources(morphizen-unit-test-TarEntryTest.ReadFrom.11 PRIVATE
        # morphizen/test_tar_entry.cpp # line 11 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.ReadFrom.15
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.ReadFrom
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.ReadFrom.15 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/ReadFrom"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.ReadFrom
    )
target_sources(morphizen-unit-test-TarFileTest.ReadFrom.15 PRIVATE
        # morphizen/test_tar_file.cpp # line 15 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.DoubleRead.30
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.DoubleRead
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.DoubleRead.30 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/DoubleRead"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.DoubleRead
    )
target_sources(morphizen-unit-test-TarFileTest.DoubleRead.30 PRIVATE
        # morphizen/test_tar_file.cpp # line 30 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.WriteOverride.177
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.WriteOverride
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.WriteOverride.177 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/WriteOverride"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.WriteOverride
    )
target_sources(morphizen-unit-test-TarFileTest.WriteOverride.177 PRIVATE
        # morphizen/test_tar_file.cpp # line 177 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.WriteTo.326
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.WriteTo
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.WriteTo.326 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/WriteTo"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.WriteTo
    )
target_sources(morphizen-unit-test-TarFileTest.WriteTo.326 PRIVATE
        # morphizen/test_tar_file.cpp # line 326 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.MemoryTar.382
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.MemoryTar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.MemoryTar.382 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/MemoryTar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.MemoryTar
    )
target_sources(morphizen-unit-test-TarFileTest.MemoryTar.382 PRIVATE
        # morphizen/test_tar_file.cpp # line 382 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-FileStreamTest.HelloWorld.11
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=FileStreamTest.HelloWorld
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-FileStreamTest.HelloWorld.11 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/FileStreamTest/HelloWorld"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=FileStreamTest.HelloWorld
    )
target_sources(morphizen-unit-test-FileStreamTest.HelloWorld.11 PRIVATE
        # morphizen/test_file_stream.cpp # line 11 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-MMapfileTest.create.9
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=MMapfileTest.create
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-MMapfileTest.create.9 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/MMapfileTest/create"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=MMapfileTest.create
    )
target_sources(morphizen-unit-test-MMapfileTest.create.9 PRIVATE
        # morphizen/test_mmap_file.cpp # line 9 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-MMapfileTest.CreateTar.19
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=MMapfileTest.CreateTar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-MMapfileTest.CreateTar.19 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/MMapfileTest/CreateTar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=MMapfileTest.CreateTar
    )
target_sources(morphizen-unit-test-MMapfileTest.CreateTar.19 PRIVATE
        # morphizen/test_mmap_file.cpp # line 19 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestOnnxRunner.RunOriginAndEpContextModel_embed_mode.157
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestOnnxRunner.RunOriginAndEpContextModel_embed_mode
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestOnnxRunner.RunOriginAndEpContextModel_embed_mode.157 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestOnnxRunner/RunOriginAndEpContextModel_embed_mode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestOnnxRunner.RunOriginAndEpContextModel_embed_mode
    )
target_sources(morphizen-unit-test-TestOnnxRunner.RunOriginAndEpContextModel_embed_mode.157 PRIVATE
        # test-onnx-runner/test-onnx-runner-main.cpp # line 157 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestOnnxRunner.RunOriginAndEpContextModel_non_embed_mode.160
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestOnnxRunner.RunOriginAndEpContextModel_non_embed_mode
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestOnnxRunner.RunOriginAndEpContextModel_non_embed_mode.160 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestOnnxRunner/RunOriginAndEpContextModel_non_embed_mode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestOnnxRunner.RunOriginAndEpContextModel_non_embed_mode
    )
target_sources(morphizen-unit-test-TestOnnxRunner.RunOriginAndEpContextModel_non_embed_mode.160 PRIVATE
        # test-onnx-runner/test-onnx-runner-main.cpp # line 160 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestOnnxRunner.SingleModelSingleSession.163
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestOnnxRunner.SingleModelSingleSession
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestOnnxRunner.SingleModelSingleSession.163 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestOnnxRunner/SingleModelSingleSession"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestOnnxRunner.SingleModelSingleSession
    )
target_sources(morphizen-unit-test-TestOnnxRunner.SingleModelSingleSession.163 PRIVATE
        # test-onnx-runner/test-onnx-runner-main.cpp # line 163 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestOnnxRunner.OfflineCompile.169
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestOnnxRunner.OfflineCompile
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestOnnxRunner.OfflineCompile.169 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestOnnxRunner/OfflineCompile"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestOnnxRunner.OfflineCompile
    )
target_sources(morphizen-unit-test-TestOnnxRunner.OfflineCompile.169 PRIVATE
        # test-onnx-runner/test-onnx-runner-main.cpp # line 169 don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end
