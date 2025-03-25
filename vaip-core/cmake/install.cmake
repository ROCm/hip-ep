set(TMP_INSTALL_TARGETS morphizen-core-static morphizen-core-dynamic)
install(
  TARGETS ${TMP_INSTALL_TARGETS}
  EXPORT vaip-core-targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(
  EXPORT vaip-core-targets
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT base
  DESTINATION share/cmake/${PROJECT_NAME})

# install(
#   FILES ${CMAKE_CURRENT_BINARY_DIR}/config.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/anchor_point.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/capability.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/pass_context.pb.h
#         ${CMAKE_CURRENT_BINARY_DIR}/version.pb.h
#   DESTINATION include/vaip)


# foreach(CONFIG_FILE IN LISTS CONFIG_FILE_LIST)
#   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE} DESTINATION bin)
# endforeach()
# #
