##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_custom_target(morphizen-unit-test-GTest-hello
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GTest.hello
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GTest-hello PROPERTIES
    FOLDER "morphizen/unit-tests/GTest/hello"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GTest.hello
    )
#end

add_custom_target(morphizen-unit-test-ConfigTest-Simple
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConfigTest.Simple
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConfigTest-Simple PROPERTIES
    FOLDER "morphizen/unit-tests/ConfigTest/Simple"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConfigTest.Simple
    )
#end

add_custom_target(morphizen-unit-test-ModelTest-Load
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ModelTest.Load
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest-Load PROPERTIES
    FOLDER "morphizen/unit-tests/ModelTest/Load"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.Load
    )
#end

add_custom_target(morphizen-unit-test-ModelTest-Clone
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ModelTest.Clone
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest-Clone PROPERTIES
    FOLDER "morphizen/unit-tests/ModelTest/Clone"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.Clone
    )
#end

add_custom_target(morphizen-unit-test-ModelTest-MainGraph
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ModelTest.MainGraph
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest-MainGraph PROPERTIES
    FOLDER "morphizen/unit-tests/ModelTest/MainGraph"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.MainGraph
    )
#end

add_custom_target(morphizen-unit-test-ModelTest-SetAndGetMetadata
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ModelTest.SetAndGetMetadata
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest-SetAndGetMetadata PROPERTIES
    FOLDER "morphizen/unit-tests/ModelTest/SetAndGetMetadata"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.SetAndGetMetadata
    )
#end

add_custom_target(morphizen-unit-test-ModelTest-ImplicitConversion
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ModelTest.ImplicitConversion
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest-ImplicitConversion PROPERTIES
    FOLDER "morphizen/unit-tests/ModelTest/ImplicitConversion"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.ImplicitConversion
    )
#end

add_custom_target(morphizen-unit-test-ModelTest-ModelCreationTest
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ModelTest.ModelCreationTest
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ModelTest-ModelCreationTest PROPERTIES
    FOLDER "morphizen/unit-tests/ModelTest/ModelCreationTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ModelTest.ModelCreationTest
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-LoadAndSave
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.LoadAndSave
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-LoadAndSave PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/LoadAndSave"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.LoadAndSave
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-FindNodeArgGraphInput
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.FindNodeArgGraphInput
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-FindNodeArgGraphInput PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/FindNodeArgGraphInput"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.FindNodeArgGraphInput
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-FindNodeArgGraphOutput
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.FindNodeArgGraphOutput
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-FindNodeArgGraphOutput PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/FindNodeArgGraphOutput"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.FindNodeArgGraphOutput
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-NodesInTopologicalOrder
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.NodesInTopologicalOrder
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-NodesInTopologicalOrder PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/NodesInTopologicalOrder"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NodesInTopologicalOrder
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-NodeIndex
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.NodeIndex
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-NodeIndex PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/NodeIndex"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NodeIndex
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-FindConsumers
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.FindConsumers
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-FindConsumers PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/FindConsumers"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.FindConsumers
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-NodeArgFindProducer
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.NodeArgFindProducer
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-NodeArgFindProducer PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/NodeArgFindProducer"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NodeArgFindProducer
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-Fuse
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.Fuse
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-Fuse PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/Fuse"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.Fuse
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-TryFuse
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.TryFuse
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-TryFuse PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/TryFuse"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.TryFuse
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-NewConstantInitializer
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.NewConstantInitializer
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-NewConstantInitializer PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/NewConstantInitializer"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.NewConstantInitializer
    )
#end

add_custom_target(morphizen-unit-test-GraphTest-VirtualFuse
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=GraphTest.VirtualFuse
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-GraphTest-VirtualFuse PROPERTIES
    FOLDER "morphizen/unit-tests/GraphTest/VirtualFuse"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=GraphTest.VirtualFuse
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int8_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int8_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int8_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int8_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int8_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int8
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int8
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int8 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int8"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int8
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint8_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint8_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint8_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint8_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint8_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint8
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint8
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint8 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint8"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint8
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int16_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int16_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int16_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int16_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int16_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int16
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int16
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int16 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int16"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int16
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint16_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint16_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint16_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint16_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint16_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint16
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint16
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint16 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint16"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint16
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int32_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int32_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int32_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int32_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int32_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int32
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int32
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int32 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int32"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int32
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint32_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint32_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint32_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint32_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint32_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint32
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint32
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint32 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint32"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint32
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int64_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int64_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int64_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int64_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int64_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-int64
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.int64
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-int64 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/int64"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.int64
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint64_scalar
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint64_scalar
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint64_scalar PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint64_scalar"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint64_scalar
    )
#end

add_custom_target(morphizen-unit-test-ConstDataTest-uint64
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ConstDataTest.uint64
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ConstDataTest-uint64 PROPERTIES
    FOLDER "morphizen/unit-tests/ConstDataTest/uint64"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ConstDataTest.uint64
    )
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint-Case0
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TestAnchorPoint.Case0
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint-Case0 PROPERTIES
    FOLDER "morphizen/unit-tests/TestAnchorPoint/Case0"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case0
    )
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint-Case1
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TestAnchorPoint.Case1
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint-Case1 PROPERTIES
    FOLDER "morphizen/unit-tests/TestAnchorPoint/Case1"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case1
    )
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint-Case2
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TestAnchorPoint.Case2
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint-Case2 PROPERTIES
    FOLDER "morphizen/unit-tests/TestAnchorPoint/Case2"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case2
    )
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint-Case3
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TestAnchorPoint.Case3
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint-Case3 PROPERTIES
    FOLDER "morphizen/unit-tests/TestAnchorPoint/Case3"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case3
    )
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint-Case4
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TestAnchorPoint.Case4
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint-Case4 PROPERTIES
    FOLDER "morphizen/unit-tests/TestAnchorPoint/Case4"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Case4
    )
#end

add_custom_target(morphizen-unit-test-TestAnchorPoint-Append
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TestAnchorPoint.Append
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TestAnchorPoint-Append PROPERTIES
    FOLDER "morphizen/unit-tests/TestAnchorPoint/Append"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TestAnchorPoint.Append
    )
#end

add_custom_target(morphizen-unit-test-ImmutableMapTest-InsertSingleNode
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ImmutableMapTest.InsertSingleNode
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ImmutableMapTest-InsertSingleNode PROPERTIES
    FOLDER "morphizen/unit-tests/ImmutableMapTest/InsertSingleNode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ImmutableMapTest.InsertSingleNode
    )
#end

add_custom_target(morphizen-unit-test-ImmutableMapTest-InsertMultipleNodes
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=ImmutableMapTest.InsertMultipleNodes
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-ImmutableMapTest-InsertMultipleNodes PROPERTIES
    FOLDER "morphizen/unit-tests/ImmutableMapTest/InsertMultipleNodes"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=ImmutableMapTest.InsertMultipleNodes
    )
#end

add_custom_target(morphizen-unit-test-PatternTest-CommutableNode
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=PatternTest.CommutableNode
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PatternTest-CommutableNode PROPERTIES
    FOLDER "morphizen/unit-tests/PatternTest/CommutableNode"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PatternTest.CommutableNode
    )
#end

add_custom_target(morphizen-unit-test-PatternTest-LoadSaveBinary
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=PatternTest.LoadSaveBinary
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PatternTest-LoadSaveBinary PROPERTIES
    FOLDER "morphizen/unit-tests/PatternTest/LoadSaveBinary"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PatternTest.LoadSaveBinary
    )
#end

add_custom_target(morphizen-unit-test-PassContextTest-ReadFileTest
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=PassContextTest.ReadFileTest
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest-ReadFileTest PROPERTIES
    FOLDER "morphizen/unit-tests/PassContextTest/ReadFileTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.ReadFileTest
    )
#end

add_custom_target(morphizen-unit-test-PassContextTest-UntarCacheTest
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=PassContextTest.UntarCacheTest
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest-UntarCacheTest PROPERTIES
    FOLDER "morphizen/unit-tests/PassContextTest/UntarCacheTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.UntarCacheTest
    )
#end

add_custom_target(morphizen-unit-test-PassContextTest-TestEmptyFiles
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=PassContextTest.TestEmptyFiles
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PassContextTest-TestEmptyFiles PROPERTIES
    FOLDER "morphizen/unit-tests/PassContextTest/TestEmptyFiles"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PassContextTest.TestEmptyFiles
    )
#end

add_custom_target(morphizen-unit-test-NodeBuilderTest-SkipSimplifiedLayerNormalization
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=NodeBuilderTest.SkipSimplifiedLayerNormalization
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-NodeBuilderTest-SkipSimplifiedLayerNormalization PROPERTIES
    FOLDER "morphizen/unit-tests/NodeBuilderTest/SkipSimplifiedLayerNormalization"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=NodeBuilderTest.SkipSimplifiedLayerNormalization
    )
#end

add_custom_target(morphizen-unit-test-TarBallTest-TarTest
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TarBallTest.TarTest
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarBallTest-TarTest PROPERTIES
    FOLDER "morphizen/unit-tests/TarBallTest/TarTest"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarBallTest.TarTest
    )
#end

add_custom_target(morphizen-unit-test-TarBallTest-Encrypt_Test
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TarBallTest.Encrypt_Test
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarBallTest-Encrypt_Test PROPERTIES
    FOLDER "morphizen/unit-tests/TarBallTest/Encrypt_Test"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarBallTest.Encrypt_Test
    )
#end

add_custom_target(morphizen-unit-test-PluginTest-StaticHelloPlugin
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=PluginTest.StaticHelloPlugin
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PluginTest-StaticHelloPlugin PROPERTIES
    FOLDER "morphizen/unit-tests/PluginTest/StaticHelloPlugin"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PluginTest.StaticHelloPlugin
    )
#end

add_custom_target(morphizen-unit-test-PluginTest-DynamicHelloPlugin
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=PluginTest.DynamicHelloPlugin
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-PluginTest-DynamicHelloPlugin PROPERTIES
    FOLDER "morphizen/unit-tests/PluginTest/DynamicHelloPlugin"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=PluginTest.DynamicHelloPlugin
    )
#end

add_custom_target(morphizen-unit-test-TarEntryTest-ReadFrom
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TarEntryTest.ReadFrom
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarEntryTest-ReadFrom PROPERTIES
    FOLDER "morphizen/unit-tests/TarEntryTest/ReadFrom"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarEntryTest.ReadFrom
    )
#end

add_custom_target(morphizen-unit-test-TarFileTest-ReadFrom
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TarFileTest.ReadFrom
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest-ReadFrom PROPERTIES
    FOLDER "morphizen/unit-tests/TarFileTest/ReadFrom"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.ReadFrom
    )
#end

add_custom_target(morphizen-unit-test-TarFileTest-DoubleRead
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TarFileTest.DoubleRead
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest-DoubleRead PROPERTIES
    FOLDER "morphizen/unit-tests/TarFileTest/DoubleRead"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.DoubleRead
    )
#end

add_custom_target(morphizen-unit-test-TarFileTest-WriteOverride
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TarFileTest.WriteOverride
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest-WriteOverride PROPERTIES
    FOLDER "morphizen/unit-tests/TarFileTest/WriteOverride"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.WriteOverride
    )
#end

add_custom_target(morphizen-unit-test-TarFileTest-WriteTo
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=TarFileTest.WriteTo
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-TarFileTest-WriteTo PROPERTIES
    FOLDER "morphizen/unit-tests/TarFileTest/WriteTo"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=TarFileTest.WriteTo1
    SOURCES morphizen/test_graph.cpp
  )

#end

add_custom_target(morphizen-unit-test-FileStreamTest-HelloWorld
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=FileStreamTest.HelloWorld
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-FileStreamTest-HelloWorld PROPERTIES
    FOLDER "morphizen/unit-tests/FileStreamTest/HelloWorld"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=FileStreamTest.HelloWorld
    )
#end

add_custom_target(morphizen-unit-test-OpDefTest-TestAddAndRemove
    COMMAND $<TARGET_FILE:morphizen-unit-test> --gtest_filter=OpDefTest.TestAddAndRemove
    DEPENDS morphizen-unit-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
set_target_properties(morphizen-unit-test-OpDefTest-TestAddAndRemove PROPERTIES
    FOLDER "morphizen/unit-tests/OpDefTest/TestAddAndRemove"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-unit-test>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter=OpDefTest.TestAddAndRemove
    )
#end
