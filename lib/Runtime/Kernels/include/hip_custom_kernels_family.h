/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CUSTOM_KERNELS_FAMILY_H
#define HIP_CUSTOM_KERNELS_FAMILY_H

/*
 * Custom-kernel shared-library naming for the family-common + per-arch-accel
 * split. Host and JIT loaders include this header to stay in sync with
 * lib/Runtime/Kernels/CMakeLists.txt.
 *
 * Layout:
 *   custom_kernels_<family>.{dll,so}     common ops (one device slice per family)
 *   custom_kernels_<arch>.{dll,so}       accel ops (WMMA/GQA/MatMulNBits/...)
 *
 * RDNA 3.5 APU family (gfx1150–gfx1153) shares custom_kernels_gfx115x, device
 * compiled with --offload-arch=gfx1151. Other arches use
 * custom_kernels_<arch>_common until a family entry is added.
 */

#include <string>

namespace hipdnn_ep {
namespace custom_kernels {

inline bool isGfx115xFamilyArch(const std::string &gcnArch) {
  return gcnArch == "gfx1150" || gcnArch == "gfx1151" ||
         gcnArch == "gfx1152" || gcnArch == "gfx1153";
}

// Basename without lib prefix or .dll/.so (e.g. custom_kernels_gfx115x).
inline std::string familyCommonBaseName(const std::string &gcnArch) {
  if (isGfx115xFamilyArch(gcnArch))
    return "custom_kernels_gfx115x";
  return "custom_kernels_" + gcnArch + "_common";
}

// Accel library basename (always per exact gcnArchName).
inline std::string accelBaseName(const std::string &gcnArch) {
  return "custom_kernels_" + gcnArch;
}

#ifdef _WIN32
inline std::string sharedLibraryFileName(const std::string &baseName) {
  return baseName + ".dll";
}
#else
inline std::string sharedLibraryFileName(const std::string &baseName) {
  return "lib" + baseName + ".so";
}
#endif

inline std::string familyCommonFileName(const std::string &gcnArch) {
  return sharedLibraryFileName(familyCommonBaseName(gcnArch));
}

inline std::string accelFileName(const std::string &gcnArch) {
  return sharedLibraryFileName(accelBaseName(gcnArch));
}

// Representative --offload-arch for a family common target (CMake uses the same).
inline std::string gfx115xFamilyOffloadArch() { return "gfx1151"; }

} // namespace custom_kernels
} // namespace hipdnn_ep

#endif // HIP_CUSTOM_KERNELS_FAMILY_H
