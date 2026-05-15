##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip-inspect-bc reads metadata out of a per-model .bc artifact emitted by
# hip-compiler. It deliberately uses the lightweight `parseBitcodeFile` +
# global-walk path rather than going through BitcodeJIT -- inspection
# must work on dev machines without a ROCm install.
add_executable(hip-inspect-bc
  hip-inspect-bc.cpp
)

# LLVM components: Core / BitReader for IR parsing, Support for
# MemoryBuffer + JSON. No ORC / ExecutionEngine / CodeGen needed.
llvm_map_components_to_libnames(llvm_libs
  Core
  BitReader
  Support
)

target_link_libraries(hip-inspect-bc
  PRIVATE
  HipSupport
  ${llvm_libs}
)

install(TARGETS hip-inspect-bc
  RUNTIME DESTINATION bin
)
