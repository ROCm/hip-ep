cmake_minimum_required(VERSION 3.5)
project(morphizen-graph-opt VERSION 1.0.0 LANGUAGES C CXX)
find_package(glog REQUIRED)

add_executable(morphizen-graph-opt graph-opt.cpp)
target_include_directories(morphizen-graph-opt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
  )
target_link_libraries(morphizen-graph-opt PRIVATE vaip_core_dynamic glog::glog)
set_target_properties(morphizen-graph-opt PROPERTIES FOLDER vaip-tools)
install(TARGETS morphizen-graph-opt DESTINATION bin)