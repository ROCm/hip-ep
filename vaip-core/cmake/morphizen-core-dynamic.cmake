add_library(morphizen-core-dynamic SHARED src/main.cpp)
add_library (morphizen::morphizen-core-dynamic ALIAS morphizen-core-dynamic)
set_target_properties(morphizen-core-dynamic PROPERTIES FOLDER morphizen)
# set output name of morphizen-core-dynamic, it is required by VitisAI EP.
set_target_properties(morphizen-core-dynamic PROPERTIES OUTPUT_NAME ${morphizen_OUTPUT_NAME})

if(MSVC)
  # TODO: dumpbin is optional
  find_program(DUMPBIN_EXECUTABLE dumpbin REQUIRED)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def
    COMMAND "${DUMPBIN_EXECUTABLE}" "/symbols" "$<TARGET_FILE:morphizen-core-static>" > "${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt"
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/dump_symbol.py
      ${CMAKE_CURRENT_BINARY_DIR}/vaip_core_symbol.txt
      ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_vitisai_ep.def.in
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/dump_symbol.py
      ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_vitisai_ep.def.in
      $<TARGET_LINKER_LIBRARY_FILE:morphizen-core-static>
  )
# add_dependencies(morphizen-core-dynamic  morphizen-core-dynamic_def)
add_custom_target(morphizen-core-dynamic_def DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def)
target_sources(morphizen-core-dynamic PRIVATE onnxruntime_vitisai_ep.def)
set_target_properties(morphizen-core-dynamic_def PROPERTIES FOLDER morphizen)
endif(MSVC)

# NOTE: do not use $<LINK_LIBRARY:WHOLE_ARCHIVE, morphizen-core-static
#
# WHY, it seems that WHOLE_ARCHIVE can be only marked once. It means
# if we add the mark here, then, all targets dependes on
# morphizen-core-static directly or indirectly, must be marked with
# WHOLE_ARCHIVE. In stead, we manually maintain
# onnxruntime_vitisai_ep.def file. tools/parse_cl_link_error.py is
# used to parse the link error and update onnxruntime_vitisai_ep.def
# automatically, see tools/parse_cl_link_error.py for more details.
target_link_libraries(morphizen-core-dynamic
  PRIVATE
  morphizen-core-static)

target_include_directories(morphizen-core-dynamic
  PUBLIC
  $<BUILD_INTERFACE:$<TARGET_PROPERTY:morphizen-core-static,INTERFACE_INCLUDE_DIRECTORIES>>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(morphizen-core-dynamic PUBLIC cxx_std_17)
target_compile_definitions(morphizen-core-dynamic PUBLIC "-DONNX_NAMESPACE=onnx")
if(MSVC)
  target_compile_options(morphizen-core-dynamic PUBLIC "/Zc:__cplusplus")
endif(MSVC)

set_target_properties(morphizen-core-dynamic PROPERTIES
  VS_DEBUGGER_COMMAND "${CMAKE_INSTALL_PREFIX}\\bin\\test_onnx_runner.exe"
  VS_DEBUGGER_COMMAND_ARGUMENTS "${CMAKE_CURRENT_SOURCE_DIR}\\..\\..\\test_onnx_runner\\data\\pt_resnet50.onnx"
  VS_DEBUGGER_ENVIRONMENT "XLNX_ONNX_EP_VERBOSE=2
DEBUG_LOG_LEVEL=info
DEBUG_VAIP_PASS=1
MORPHIZEN_DEBUG_TAR_ENTRY=1
MORPHIZEN_DEBUG_TAR_FILE=1
DEBUG_TAR_CACHE=1
"
)
