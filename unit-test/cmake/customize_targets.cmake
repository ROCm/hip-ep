# add a target to run morphizen-unit-tests
add_custom_target(morphizen-run-unit-tests
    ${CMAKE_COMMAND} -E env CI=1
    $<TARGET_FILE:${TEST_EXE_NAME}>
    DEPENDS ${TEST_EXE_NAME}
)
set_target_properties(morphizen-run-unit-tests PROPERTIES
    FOLDER morphizen/unit-tests/targets
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${TEST_EXE_NAME}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
)

# add a target to run morphizen-pattern-gen
add_custom_target(morphizen-run-onnx-pattern-gen-resnet50
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND
        ${CMAKE_COMMAND} -E make_directory patterns
    COMMAND
    ${CMAKE_COMMAND} -E env
        $<TARGET_FILE:morphizen-pattern-gen> -f "${RESNET_50_ONNX_PATH}"
        -i 127 -o 128 -m ./patterns/relu_dq.mmd -j ./patterns/relu_dq.json
    DEPENDS morphizen-pattern-gen
)
set_target_properties(morphizen-run-onnx-pattern-gen-resnet50 PROPERTIES
    FOLDER morphizen/unit-tests/targets
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-pattern-gen>"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "-f ${RESNET_50_ONNX_PATH}  -i 127 -o 128 -m ./patterns/relu_dq.mmd -j ./patterns/relu_dq.json"
)

# add a target to run morphizen-onnx-grep
add_custom_target(morphizen-run-onnx-grep-resnet50
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND
    ${CMAKE_COMMAND} -E env
    $<TARGET_FILE:morphizen-onnx-grep>
      -f "${RESNET_50_ONNX_PATH}"
      -p ./patterns/relu_dq.json -n 128
    DEPENDS morphizen-onnx-grep morphizen-run-onnx-pattern-gen-resnet50
)
set_target_properties(morphizen-run-onnx-grep-resnet50 PROPERTIES
    FOLDER morphizen/unit-tests/targets
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:morphizen-pattern-gen>"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "-f ${RESNET_50_ONNX_PATH}  -p ./patterns/relu_dq.json -n 128"
)

# add a target to run all above targets
add_custom_target(morphizen-run-all-unit-tests
    COMMAND ${CMAKE_COMMAND} -E echo "ALL UNIT TESTS ARE OK"
    DEPENDS
    morphizen-run-unit-tests
    morphizen-run-onnx-pattern-gen-resnet50
    morphizen-run-onnx-grep-resnet50
)
set_target_properties(morphizen-run-all-unit-tests PROPERTIES
	FOLDER morphizen/unit-tests/targets
)