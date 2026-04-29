/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_SUPPORT_CRASHHANDLER_H
#define HIP_SUPPORT_CRASHHANDLER_H

namespace hip {
void install_crash_handlers(const char *component_name);
} // namespace hip

#endif // HIP_SUPPORT_CRASHHANDLER_H
