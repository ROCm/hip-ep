##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(OpenSSL QUIET)
if(TARGET OpenSSL::SSL)
  message(STATUS "found OpenSSL at ${OpenSSL_DIR} build morephizen-encryption with OpenSSL")
else()
  message(STATUS "cannot find_package(OpenSSL), build morephizen-encryption without OpenSSL")
endif()
set(LIB_NAME morphizen-encryption)
add_library(
  ${LIB_NAME}
  STATIC
  include/morphizen/encryption.hpp
  src/encryption.cpp
  )
add_library(morphizen::encryption ALIAS ${LIB_NAME})
## add an alias to keep it compatible with VAIP
add_library(vaip::encryption ALIAS ${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES FOLDER morphizen/encryption)
target_include_directories(${LIB_NAME}
  PRIVATE
    # compile time dependency on vaip_io.hpp
    # todo: use std::ostream and std::istream instead of vaip_io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../vaip-core/include
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(${LIB_NAME} PUBLIC vaip_io)

target_compile_options(${LIB_NAME} PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})
if(TARGET OpenSSL::SSL)
  target_compile_definitions(${LIB_NAME} PUBLIC -DWITH_OPENSSL)
  target_link_libraries(${LIB_NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto ${CMAKE_DL_LIBS} glog::glog)
  if (MSVC)
    # TODO: why we need these windows specific libraries
    target_link_libraries(${LIB_NAME} PRIVATE WS2_32.LIB GDI32.LIB ADVAPI32.LIB CRYPT32.LIB USER32.LIB)
  endif()
else()
  target_link_libraries(${LIB_NAME} PRIVATE glog::glog)
endif()

# install(
#   TARGETS ${LIB_NAME}
#   EXPORT ${LIB_NAME}-targets
#   RUNTIME DESTINATION bin
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib)
# install(
#   EXPORT ${LIB_NAME}-targets
#   NAMESPACE ${PROJECT_NAME}::
#   COMPONENT base
#   DESTINATION share/cmake/${PROJECT_NAME})
