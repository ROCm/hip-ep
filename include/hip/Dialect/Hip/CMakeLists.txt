##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Keep existing plugin include paths in the build and install trees.
function(add_hip_dialect_header_alias header canonical)
  set(alias "${CMAKE_CURRENT_BINARY_DIR}/../${header}")
  file(GENERATE OUTPUT "${alias}" CONTENT "#include \"${canonical}\"\n")
  cmake_path(GET header PARENT_PATH component)
  install(FILES "${alias}" DESTINATION "include/hip/Dialect/${component}")
endfunction()

foreach(header IN ITEMS
    IR/HipBufferize.h
    IR/HipConstantData.h
    IR/HipDialect.h
    IR/HipShapeUtils.h
    IR/RuntimeAccessOpInterface.h
    Transforms/BufferUtils.h
    Transforms/Passes.h)
  add_hip_dialect_header_alias("${header}" "hip/Dialect/Hip/${header}")
endforeach()
add_hip_dialect_header_alias(Transforms/Pipelines.h hip/Compiler/Pipelines.h)
add_hip_dialect_header_alias(Hip/Transforms/Pipelines.h hip/Compiler/Pipelines.h)

# Some TableGen backends select definitions from the main input file only.
# Give those invocations the canonical file contents at the legacy path.
foreach(input IN ITEMS
    IR/HipDialect.td
    IR/HipDpsOpInterface.td
    IR/HipElementMoveInterface.td
    IR/HipOpStateInterface.td
    IR/HipOps.td
    IR/HipTypes.td
    IR/RuntimeAccessOpInterface.td
    Transforms/Passes.td)
  configure_file("${input}" "${CMAKE_CURRENT_BINARY_DIR}/../${input}" COPYONLY)
endforeach()

foreach(component IN ITEMS IR Transforms)
  add_subdirectory(${component})

  # A consumer can include a fragment more than once with different macros
  # (e.g. GET_OP_LIST and GET_OP_CLASSES). Forward without an include guard.
  get_directory_property(outputs DIRECTORY ${component} DEFINITION TABLEGEN_OUTPUT)
  list(REMOVE_DUPLICATES outputs)
  foreach(output IN LISTS outputs)
    cmake_path(GET output PARENT_PATH directory)
    if(directory STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/${component}")
      cmake_path(GET output FILENAME filename)
      file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../${component}/${filename}"
        CONTENT "#include \"hip/Dialect/Hip/${component}/${filename}\"\n")
    endif()
  endforeach()
endforeach()
