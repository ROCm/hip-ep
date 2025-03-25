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
add_custom_target(morphizen-core-dynamic_def DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_vitisai_ep.def)
target_sources(morphizen-core-dynamic PRIVATE onnxruntime_vitisai_ep.def)
set_target_properties(morphizen-core-dynamic_def PROPERTIES FOLDER vaip)
add_dependencies(morphizen-core-dynamic  morphizen-core-dynamic_def)
endif(MSVC)
target_link_libraries(morphizen-core-dynamic PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,morphizen-core-static>")

target_include_directories(morphizen-core-dynamic PUBLIC
  $<BUILD_INTERFACE:$<TARGET_PROPERTY:morphizen-core-static,INTERFACE_INCLUDE_DIRECTORIES>>
)
