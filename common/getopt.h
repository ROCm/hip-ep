/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef GETOPT_H
#define GETOPT_H
extern int opterr, optind, optopt, optreset;
extern char* optarg;
int getopt(int nargc, char* const nargv[], const char* ostr) throw();
#endif
