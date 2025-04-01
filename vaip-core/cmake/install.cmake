set(TMP_INSTALL_TARGETS morphizen-core-static morphizen-core-dynamic zlibstatic GSL nlohmann_json)
#temp comments install for skip error
# -- Configuring done (32.8s)
#CMake Error: install(EXPORT "morphizen-core-targets" ...) includes target "morphizen-core-static" which requires tar#get "zlibstatic" that is not in any export set.
#CMake Error: install(EXPORT "morphizen-core-targets" ...) includes target "morphizen-core-static" which requires tar#get "nlohmann_json" that is not in any export set.
#CMake Error: install(EXPORT "morphizen-core-targets" ...) includes target "morphizen-core-static" which requires tar#get "GSL" that is not in any export set.
#-- Generating done (32.8s)
#
if(TARGET morphizen-core-dynamic)
install(
 TARGETS morphizen-core-dynamic
#  EXPORT morphizen-core-targets
 RUNTIME DESTINATION bin
 ARCHIVE DESTINATION lib
 LIBRARY DESTINATION lib)
# install(
#  EXPORT morphizen-core-targets
#  NAMESPACE ${PROJECT_NAME}::
#  COMPONENT base
#  DESTINATION share/cmake/${PROJECT_NAME}
#  EXCLUDE_FROM_ALL)
endif()

install(
  FILES ${PROTO_HDRS}
  DESTINATION include/morphizen)
install(
  DIRECTORY include/morphizen DESTINATION include)

install(
  DIRECTORY
  ${ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR}/vaip
  DESTINATION include)
