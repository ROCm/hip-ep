/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#ifndef VAIP_EXPORT_DLL
#  define VAIP_EXPORT_DLL 0
#endif
#ifndef VAIP_USER
#  define VAIP_USER 4
#endif

#if VAIP_EXPORT_DLL == 1
// ok to include by internal cpp files.
#else
#  if VAIP_USER == 1 || VAIP_USER == 2 || VAIP_USER == 3
#  else
#    error "please include morphizen/morphizen.hpp first"
#  endif
#endif
