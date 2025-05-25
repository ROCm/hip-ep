##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_custom_target(morphizen-unit-test-GTest.hello.39
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GTest.hello
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GTest.hello.39 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GTest/hello"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GTest.hello
    )
target_sources(morphizen-unit-test-GTest.hello.39 PRIVATE
        morphizen_unit_test_main.cpp # line 39
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
        vaip/test_config.cpp # line 26
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
        vaip/test_config.cpp # line 34
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
        vaip/test_config.cpp # line 42
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
        vaip/test_config.cpp # line 54
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
        vaip/test_model.cpp # line 16
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.Clone.22
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.Clone
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.Clone.22 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/Clone"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.Clone
    )
target_sources(morphizen-unit-test-ModelTest.Clone.22 PRIVATE
        vaip/test_model.cpp # line 22
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.MainGraph.31
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.MainGraph
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.MainGraph.31 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/MainGraph"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.MainGraph
    )
target_sources(morphizen-unit-test-ModelTest.MainGraph.31 PRIVATE
        vaip/test_model.cpp # line 31
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.SetAndGetMetadata.38
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.SetAndGetMetadata
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.SetAndGetMetadata.38 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/SetAndGetMetadata"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.SetAndGetMetadata
    )
target_sources(morphizen-unit-test-ModelTest.SetAndGetMetadata.38 PRIVATE
        vaip/test_model.cpp # line 38
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.ImplicitConversion.56
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.ImplicitConversion
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.ImplicitConversion.56 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/ImplicitConversion"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.ImplicitConversion
    )
target_sources(morphizen-unit-test-ModelTest.ImplicitConversion.56 PRIVATE
        vaip/test_model.cpp # line 56
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ModelTest.ModelCreationTest.74
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ModelTest.ModelCreationTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest.ModelCreationTest.74 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ModelTest/ModelCreationTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.ModelCreationTest
    )
target_sources(morphizen-unit-test-ModelTest.ModelCreationTest.74 PRIVATE
        vaip/test_model.cpp # line 74
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-GraphTest.LoadAndSave.16
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=GraphTest.LoadAndSave
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest.LoadAndSave.16 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/GraphTest/LoadAndSave"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.LoadAndSave
    )
target_sources(morphizen-unit-test-GraphTest.LoadAndSave.16 PRIVATE
        vaip/test_graph.cpp # line 16
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
        vaip/test_graph.cpp # line 90
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
        vaip/test_graph.cpp # line 110
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
        vaip/test_graph.cpp # line 132
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
        vaip/test_graph.cpp # line 162
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
        vaip/test_graph.cpp # line 176
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
        vaip/test_graph.cpp # line 190
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
        vaip/test_graph.cpp # line 202
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
        vaip/test_graph.cpp # line 248
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
        vaip/test_graph.cpp # line 259
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
        vaip/test_graph.cpp # line 521
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int8_scalar.43
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int8_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int8_scalar.43 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int8_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int8_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int8_scalar.43 PRIVATE
        vaip/test_const_data.cpp # line 43
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int8.54
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int8
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int8.54 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int8"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int8
    )
target_sources(morphizen-unit-test-ConstDataTest.int8.54 PRIVATE
        vaip/test_const_data.cpp # line 54
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint8_scalar.67
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint8_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint8_scalar.67 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint8_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint8_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint8_scalar.67 PRIVATE
        vaip/test_const_data.cpp # line 67
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint8.78
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint8
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint8.78 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint8"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint8
    )
target_sources(morphizen-unit-test-ConstDataTest.uint8.78 PRIVATE
        vaip/test_const_data.cpp # line 78
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int16_scalar.91
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int16_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int16_scalar.91 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int16_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int16_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int16_scalar.91 PRIVATE
        vaip/test_const_data.cpp # line 91
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int16.102
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int16
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int16.102 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int16"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int16
    )
target_sources(morphizen-unit-test-ConstDataTest.int16.102 PRIVATE
        vaip/test_const_data.cpp # line 102
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint16_scalar.115
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint16_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint16_scalar.115 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint16_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint16_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint16_scalar.115 PRIVATE
        vaip/test_const_data.cpp # line 115
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint16.126
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint16
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint16.126 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint16"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint16
    )
target_sources(morphizen-unit-test-ConstDataTest.uint16.126 PRIVATE
        vaip/test_const_data.cpp # line 126
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int32_scalar.139
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int32_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int32_scalar.139 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int32_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int32_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int32_scalar.139 PRIVATE
        vaip/test_const_data.cpp # line 139
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int32.150
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int32
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int32.150 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int32"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int32
    )
target_sources(morphizen-unit-test-ConstDataTest.int32.150 PRIVATE
        vaip/test_const_data.cpp # line 150
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint32_scalar.163
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint32_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint32_scalar.163 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint32_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint32_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint32_scalar.163 PRIVATE
        vaip/test_const_data.cpp # line 163
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint32.174
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint32
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint32.174 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint32"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint32
    )
target_sources(morphizen-unit-test-ConstDataTest.uint32.174 PRIVATE
        vaip/test_const_data.cpp # line 174
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int64_scalar.187
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int64_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int64_scalar.187 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int64_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int64_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.int64_scalar.187 PRIVATE
        vaip/test_const_data.cpp # line 187
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.int64.198
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.int64
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.int64.198 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/int64"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int64
    )
target_sources(morphizen-unit-test-ConstDataTest.int64.198 PRIVATE
        vaip/test_const_data.cpp # line 198
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint64_scalar.211
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint64_scalar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint64_scalar.211 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint64_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint64_scalar
    )
target_sources(morphizen-unit-test-ConstDataTest.uint64_scalar.211 PRIVATE
        vaip/test_const_data.cpp # line 211
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ConstDataTest.uint64.222
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ConstDataTest.uint64
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest.uint64.222 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ConstDataTest/uint64"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint64
    )
target_sources(morphizen-unit-test-ConstDataTest.uint64.222 PRIVATE
        vaip/test_const_data.cpp # line 222
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case0.93
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case0
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case0.93 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case0"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case0
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case0.93 PRIVATE
        vaip/test_anchor_point.cpp # line 93
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case1.94
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case1
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case1.94 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case1"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case1
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case1.94 PRIVATE
        vaip/test_anchor_point.cpp # line 94
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case2.95
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case2
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case2.95 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case2"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case2
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case2.95 PRIVATE
        vaip/test_anchor_point.cpp # line 95
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case3.96
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case3
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case3.96 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case3"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case3
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case3.96 PRIVATE
        vaip/test_anchor_point.cpp # line 96
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Case4.97
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Case4
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Case4.97 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Case4"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case4
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Case4.97 PRIVATE
        vaip/test_anchor_point.cpp # line 97
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint.Append.98
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TestAnchorPoint.Append
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint.Append.98 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TestAnchorPoint/Append"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Append
    )
target_sources(morphizen-unit-test-TestAnchorPoint.Append.98 PRIVATE
        vaip/test_anchor_point.cpp # line 98
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ImmutableMapTest.InsertSingleNode.23
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ImmutableMapTest.InsertSingleNode
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ImmutableMapTest.InsertSingleNode.23 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ImmutableMapTest/InsertSingleNode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ImmutableMapTest.InsertSingleNode
    )
target_sources(morphizen-unit-test-ImmutableMapTest.InsertSingleNode.23 PRIVATE
        vaip/test_immutable_map.cpp # line 23
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-ImmutableMapTest.InsertMultipleNodes.31
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=ImmutableMapTest.InsertMultipleNodes
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ImmutableMapTest.InsertMultipleNodes.31 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/ImmutableMapTest/InsertMultipleNodes"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ImmutableMapTest.InsertMultipleNodes
    )
target_sources(morphizen-unit-test-ImmutableMapTest.InsertMultipleNodes.31 PRIVATE
        vaip/test_immutable_map.cpp # line 31
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PatternTest.CommutableNode.35
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PatternTest.CommutableNode
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PatternTest.CommutableNode.35 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PatternTest/CommutableNode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PatternTest.CommutableNode
    )
target_sources(morphizen-unit-test-PatternTest.CommutableNode.35 PRIVATE
        vaip/test_pattern.cpp # line 35
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PatternTest.LoadSaveBinary.90
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PatternTest.LoadSaveBinary
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PatternTest.LoadSaveBinary.90 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PatternTest/LoadSaveBinary"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PatternTest.LoadSaveBinary
    )
target_sources(morphizen-unit-test-PatternTest.LoadSaveBinary.90 PRIVATE
        vaip/test_pattern.cpp # line 90
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.ReadFileTest.41
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.ReadFileTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.ReadFileTest.41 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/ReadFileTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.ReadFileTest
    )
target_sources(morphizen-unit-test-PassContextTest.ReadFileTest.41 PRIVATE
        vaip/test_pass_context.cpp # line 41
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.UntarCacheTest.108
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.UntarCacheTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.UntarCacheTest.108 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/UntarCacheTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.UntarCacheTest
    )
target_sources(morphizen-unit-test-PassContextTest.UntarCacheTest.108 PRIVATE
        vaip/test_pass_context.cpp # line 108
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.TestEmptyFiles.123
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.TestEmptyFiles
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.TestEmptyFiles.123 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/TestEmptyFiles"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.TestEmptyFiles
    )
target_sources(morphizen-unit-test-PassContextTest.TestEmptyFiles.123 PRIVATE
        vaip/test_pass_context.cpp # line 123
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.TestCompress.243
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.TestCompress
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.TestCompress.243 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/TestCompress"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.TestCompress
    )
target_sources(morphizen-unit-test-PassContextTest.TestCompress.243 PRIVATE
        vaip/test_pass_context.cpp # line 243
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-PassContextTest.TestGzTar.258
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=PassContextTest.TestGzTar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest.TestGzTar.258 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/PassContextTest/TestGzTar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.TestGzTar
    )
target_sources(morphizen-unit-test-PassContextTest.TestGzTar.258 PRIVATE
        vaip/test_pass_context.cpp # line 258
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-NodeBuilderTest.SkipSimplifiedLayerNormalization.17
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=NodeBuilderTest.SkipSimplifiedLayerNormalization
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-NodeBuilderTest.SkipSimplifiedLayerNormalization.17 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/NodeBuilderTest/SkipSimplifiedLayerNormalization"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=NodeBuilderTest.SkipSimplifiedLayerNormalization
    )
target_sources(morphizen-unit-test-NodeBuilderTest.SkipSimplifiedLayerNormalization.17 PRIVATE
        vaip/test_node_builder.cpp # line 17
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarBallTest.TarTest.70
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarBallTest.TarTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarBallTest.TarTest.70 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarBallTest/TarTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarBallTest.TarTest
    )
target_sources(morphizen-unit-test-TarBallTest.TarTest.70 PRIVATE
        vaip/test_tarball.cpp # line 70
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarBallTest.CompressTest.129
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarBallTest.CompressTest
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarBallTest.CompressTest.129 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarBallTest/CompressTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarBallTest.CompressTest
    )
target_sources(morphizen-unit-test-TarBallTest.CompressTest.129 PRIVATE
        vaip/test_tarball.cpp # line 129
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarBallTest.Encrypt_Test.169
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarBallTest.Encrypt_Test
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarBallTest.Encrypt_Test.169 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarBallTest/Encrypt_Test"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarBallTest.Encrypt_Test
    )
target_sources(morphizen-unit-test-TarBallTest.Encrypt_Test.169 PRIVATE
        vaip/test_tarball.cpp # line 169
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
        vaip/test_plugin.cpp # line 9
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
        vaip/test_plugin.cpp # line 31
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarEntryTest.ReadFrom.12
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarEntryTest.ReadFrom
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarEntryTest.ReadFrom.12 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarEntryTest/ReadFrom"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarEntryTest.ReadFrom
    )
target_sources(morphizen-unit-test-TarEntryTest.ReadFrom.12 PRIVATE
        vaip/test_tar_entry.cpp # line 12
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.ReadFrom.14
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.ReadFrom
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.ReadFrom.14 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/ReadFrom"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.ReadFrom
    )
target_sources(morphizen-unit-test-TarFileTest.ReadFrom.14 PRIVATE
        vaip/test_tar_file.cpp # line 14
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.DoubleRead.29
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.DoubleRead
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.DoubleRead.29 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/DoubleRead"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.DoubleRead
    )
target_sources(morphizen-unit-test-TarFileTest.DoubleRead.29 PRIVATE
        vaip/test_tar_file.cpp # line 29
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.WriteOverride.176
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.WriteOverride
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.WriteOverride.176 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/WriteOverride"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.WriteOverride
    )
target_sources(morphizen-unit-test-TarFileTest.WriteOverride.176 PRIVATE
        vaip/test_tar_file.cpp # line 176
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-TarFileTest.WriteTo.293
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=TarFileTest.WriteTo
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest.WriteTo.293 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/TarFileTest/WriteTo"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.WriteTo
    )
target_sources(morphizen-unit-test-TarFileTest.WriteTo.293 PRIVATE
        vaip/test_tar_file.cpp # line 293
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
        vaip/test_file_stream.cpp # line 11
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-MMapfileTest.create.5
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=MMapfileTest.create
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-MMapfileTest.create.5 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/MMapfileTest/create"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=MMapfileTest.create
    )
target_sources(morphizen-unit-test-MMapfileTest.create.5 PRIVATE
        vaip/test_mmap_file.cpp # line 5
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end

add_custom_target(morphizen-unit-test-MMapfileTest.CreateTar.15
    COMMAND $<TARGET_FILE:${TEST_EXE_NAME}> --gtest_filter=MMapfileTest.CreateTar
    DEPENDS ${TEST_EXE_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-MMapfileTest.CreateTar.15 PROPERTIES
    FOLDER "morphizen/unit-tests/cases/MMapfileTest/CreateTar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=MMapfileTest.CreateTar
    )
target_sources(morphizen-unit-test-MMapfileTest.CreateTar.15 PRIVATE
        vaip/test_mmap_file.cpp # line 15
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end
