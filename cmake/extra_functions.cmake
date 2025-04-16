function(vaip_add_remote_target)
  set(options)
  set(oneValueArgs TARGET FILE URL EXPECTED_MD5)
  set(multiValueArgs PATCH_FILES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_FILE)
    message(FATAL_ERROR "vaip_add_remote_target: FILE not specified")
  endif()
  if(NOT ARG_URL)
    message(FATAL_ERROR "vaip_add_remote_target: URL not specified")
  endif()
  if(ARG_PATCH_FILES)
    find_package(Patch REQUIRED)
  endif()
  add_custom_command (
    OUTPUT ${ARG_FILE}
    COMMAND ${CMAKE_COMMAND}
    -DURL=${ARG_URL}
    -DFILE=${ARG_FILE}
    -DEXPECTED_MD5=${ARG_EXPECTED_MD5}
    -DPATCH_FILES=${ARG_PATCH_FILES}
    -DPatch_EXECUTABLE=${Patch_EXECUTABLE}
    -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/download.cmake
  )
  if(ARG_TARGET)
    add_custom_target(${ARG_TARGET} DEPENDS ${ARG_FILE})
  endif()
endfunction()

function(morphizen_add_python_target)
  set(options)
  set(oneValueArgs TARGET SCRIPT FOLDER)
  set(multiValueArgs ARGS DEPENDS OUTPUT)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT ARG_TARGET)
    message(FATAL_ERROR "morphizen_add_python_target: TARGET not specified")
  endif()
  if(NOT ARG_SCRIPT)
    message(FATAL_ERROR "morphizen_add_python_target: SCRIPT not specified")
  endif()
  add_custom_command (
    OUTPUT ${ARG_OUTPUT}
    COMMAND
      ${CMAKE_COMMAND} -E echo " -- Generating ${ARG_OUTPUT}"
    COMMAND
      ${CMAKE_COMMAND} -E echo " -- Running $<TARGET_FILE:Python3::Interpreter> -m  pip install --user numpy onnx"
    COMMAND
      ${CMAKE_COMMAND} -E echo " -- Running $<TARGET_FILE:Python3::Interpreter> ${ARG_SCRIPT} ${ARG_ARGS}"
    COMMAND
      ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
      $<TARGET_FILE:Python3::Interpreter> -m pip install --user numpy onnx
    COMMAND
      ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
      $<TARGET_FILE:Python3::Interpreter> ${ARG_SCRIPT} ${ARG_ARGS}
    DEPENDS ${ARG_SCRIPT} ${ARG_DEPENDS}
  )
  add_custom_target(${ARG_TARGET}
    DEPENDS ${ARG_OUTPUT}
    COMMENT "Generating ${ARG_OUTPUT} by runnning $<TARGET_FILE:Python3::Interpreter> ${ARG_SCRIPT} ${ARG_ARGS}"
  )
  if(ARG_FOLDER)
    set_target_properties(${ARG_TARGET} PROPERTIES FOLDER "${ARG_FOLDER}")
  endif()
endfunction(morphizen_add_python_target)


function(morphizen_add_library)
  set(options)
  set(oneValueArgs NAME VS_FOLDER INCLUDE_DIR SRC_DIR TEST_DIR SKIP_INSTALL OUTPUT_NAME)
  set(multiValueArgs SRCS DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # start check include dir
  if(NOT ARG_INCLUDE_DIR)
    set(ARG_INCLUDE_DIR "include")
  endif(NOT ARG_INCLUDE_DIR)
  # end check include dir

  # start to check src dir
  if(NOT ARG_SRC_DIR)
    set(ARG_SRC_DIR "src")
  endif(NOT ARG_SRC_DIR)
  # end check src dir

  # start to check test dir
  if(NOT ARG_TEST_DIR)
    set(ARG_TEST_DIR "test")
  endif(NOT ARG_TEST_DIR)
  # end check test dir

  # start to check IDE folder
  if(NOT ARG_VS_FOLDER)
    set(ARG_VS_FOLDER "vaip")
  endif(NOT ARG_VS_FOLDER)
  # end check IDE folder

  # start check target name
  if(NOT ARG_NAME)
    get_filename_component(ARG_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    set(COMPONENT_NAME
        ${ARG_NAME}
        PARENT_SCOPE)
  endif(NOT ARG_NAME)
  if(NOT ARG_OUTPUT_NAME)
    set (ARG_OUTPUT_NAME "vaip-${ARG_NAME}")
  endif()
  # end check target name

  # create the target
  message(STATUS "create target ${ARG_NAME} SHARED ${ARG_SRCS}")
  add_library(${ARG_NAME} ${ARG_SRCS})
  # create alias
  add_library(${PROJECT_NAME}::${ARG_NAME} ALIAS ${ARG_NAME})
  set_target_properties(${ARG_NAME} PROPERTIES FOLDER ${ARG_VS_FOLDER})

  target_link_libraries(${ARG_NAME} PUBLIC ${ARG_DEPENDS})
  # target_link_libraries(${ARG_NAME} PUBLIC -ltvm)
  target_include_directories(
    ${ARG_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>)
  target_include_directories(
    ${ARG_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

  # set all properties
  set_target_properties(${ARG_NAME} PROPERTIES OUTPUT_NAME
    ${ARG_OUTPUT_NAME})
  target_compile_definitions(
    ${ARG_NAME} PRIVATE -DOUTPUT_NAME="vaip-${ARG_NAME}")
  file(APPEND ${CMAKE_BINARY_DIR}/components.txt "${ARG_NAME}\n")
  if(TARGET morphizen-core-dynamic)
	  target_link_libraries(morphizen-core-dynamic PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ARG_NAME}>")
    message(STATUS "add WHOLE_ARCHIVE to morphizen-core-dynamic")
  endif(TARGET morphizen-core-dynamic)

  if(TARGET onnxruntime_vitisai_ep)
	  target_link_libraries(onnxruntime_vitisai_ep PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ARG_NAME}>")
    message(STATUS "add WHOLE_ARCHIVE to onnxruntime_vitisai_ep")
  endif(TARGET onnxruntime_vitisai_ep)

  install(
    TARGETS ${ARG_NAME}
    EXPORT ${ARG_NAME}-targets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
  install(
    EXPORT ${ARG_NAME}-targets
    NAMESPACE ${PROJECT_NAME}::
    COMPONENT base
    DESTINATION share/cmake/${PROJECT_NAME})

endfunction(morphizen_add_library)
