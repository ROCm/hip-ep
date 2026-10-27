##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Checks that the linked runtime bitcode defines every accessor declared in
# hip/Runtime/PluginAPI.h exactly once, whichever runtime (real or mock) the
# build made.
#   cmake -DNM=<llvm-nm> -DBC=<runtime.bc> -DHEADER=<PluginAPI.h> -P <this file>

file(STRINGS "${HEADER}" decls REGEX "hipdnn_ep_state_[a-z0-9_]+\\(")
set(names "")
foreach(line IN LISTS decls)
  string(REGEX MATCH "hipdnn_ep_state_[a-z0-9_]+" name "${line}")
  list(APPEND names ${name})
endforeach()
list(REMOVE_DUPLICATES names)
list(LENGTH names count)
if(count EQUAL 0)
  message(FATAL_ERROR "no accessors found in ${HEADER}")
endif()

execute_process(COMMAND "${NM}" --defined-only "${BC}"
                OUTPUT_VARIABLE symbols RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "${NM} failed on ${BC}")
endif()

set(failed "")
foreach(name IN LISTS names)
  string(REGEX MATCHALL "[ \t]${name}\n" found "${symbols}")
  list(LENGTH found n)
  if(NOT n EQUAL 1)
    list(APPEND failed "${name} (${n} definitions)")
  endif()
endforeach()
if(failed)
  message(FATAL_ERROR "runtime bitcode ${BC}: ${failed}")
endif()
message(STATUS "${count} plugin API accessors defined once in ${BC}")
