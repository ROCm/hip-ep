/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
extern "C"
#if _WIN32
    __declspec(dllexport)
#endif
        const char* say_hello() {
  return "hello, world!";
}
