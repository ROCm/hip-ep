/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_SUPPORT_GENERICTARGET_H
#define HIP_SUPPORT_GENERICTARGET_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <string>

namespace hip {

/// The generic ISA whose code objects run on `arch` ("gfx1151" ->
/// "gfx11-generic"), or "" when there is none. A custom-kernels library is
/// looked up as custom_kernels_<arch>, then custom_kernels_<generic>: the
/// runtime loads it that way (LlvmIrJit), and the compiler reads the row-pad
/// table from the same library.
inline std::string genericTargetFor(llvm::StringRef arch) {
  return llvm::StringSwitch<std::string>(arch)
      .Cases({"gfx900", "gfx902", "gfx904", "gfx906", "gfx909", "gfx90c"},
             "gfx9-generic")
      .Cases({"gfx942", "gfx950"}, "gfx9-4-generic")
      .Cases({"gfx1010", "gfx1011", "gfx1012", "gfx1013"}, "gfx10-1-generic")
      .Cases({"gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035",
              "gfx1036"},
             "gfx10-3-generic")
      .Cases({"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
              "gfx1152", "gfx1153"},
             "gfx11-generic")
      .Cases({"gfx1200", "gfx1201"}, "gfx12-generic")
      .Default("");
}

} // namespace hip

#endif // HIP_SUPPORT_GENERICTARGET_H
