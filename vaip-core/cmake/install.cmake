set(TMP_INSTALL_TARGETS morphizen-core-static morphizen-core-dynamic)
install(
  TARGETS morphizen-core-dynamic morphizen-core-static
  EXPORT morphizen-core-targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  EXPORT morphizen-core-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})

install(
  FILES ${PROTO_HDRS}
  DESTINATION include/morphizen)
install(
  DIRECTORY include/morphizen DESTINATION include)

install(
  DIRECTORY
  ${ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR}/vaip
  DESTINATION include)
