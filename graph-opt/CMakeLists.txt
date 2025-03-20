cmake_minimum_required(VERSION 3.5)
project(graph-opt VERSION 1.0.0 LANGUAGES C CXX)
find_package(glog REQUIRED)

add_executable(graph-opt graph-opt.cpp)
target_include_directories(graph-opt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
  )
target_link_libraries(graph-opt PRIVATE vaip_core_dynamic glog::glog)
set_target_properties(graph-opt PROPERTIES FOLDER vaip-tools)