/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_TAR_H
#define MORPHIZEN_TAR_H

/* Cross-platform alignment macro */
#if defined(_WIN32)
#  define PACKED(x) __declspec(align(x))
#else
#  define PACKED(x) __attribute__((packed, aligned(x)))
#endif

/* POSIX ustar constants */
#define TMAGIC "ustar" /* magic string */
#define TMAGLEN 6      /* length of magic string (includes null) */
#define TVERSION "00"  /* version string (no null) */
#define TVERSLEN 2     /* length of version string */

/* POSIX ustar header structure (512 bytes) */
typedef struct PACKED(1) {
  char name[100];     /* file name */
  char mode[8];       /* file mode */
  char uid[8];        /* owner user ID */
  char gid[8];        /* owner group ID */
  char size[12];      /* file size in bytes (octal) */
  char mtime[12];     /* modification time (octal) */
  char chksum[8];     /* header checksum */
  char typeflag;      /* link indicator */
  char linkname[100]; /* name of linked file */
  char magic[6];      /* "ustar" */
  char version[2];    /* "00" */
  char uname[32];     /* owner user name */
  char gname[32];     /* owner group name */
  char devmajor[8];   /* device major number */
  char devminor[8];   /* device minor number */
  char prefix[155];   /* filename prefix */
} HD_USTAR;

#endif /* MORPHIZEN_TAR_H */
