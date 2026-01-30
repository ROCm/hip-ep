/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#ifndef MORPHIZEN_EXPORT_DLL
#  define MORPHIZEN_EXPORT_DLL 0
#endif
#ifndef MORPHIZEN_USER
#  define MORPHIZEN_USER 4
#endif

#if MORPHIZEN_EXPORT_DLL == 1
// ok to include by internal cpp files.
#else
#  if MORPHIZEN_USER == 1 || MORPHIZEN_USER == 2 || MORPHIZEN_USER == 3
#  else
#    error "please include morphizen/morphizen.hpp first"
#  endif
#endif
