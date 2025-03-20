/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *Copyright (C) 2023 – 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include "getopt.h"

#include <stdio.h>
#include <string.h>
int opterr = 1, /* if error message should be printed */
    optind = 1, /* index into parent argv vector */
    optopt,     /* character checked for validity */
    optreset;   /* reset getopt */
char* optarg;   /* argument associated with option */
#define BADCH (int)'?'
#define BADARG (int)':'
#define EMSG ""

/*
 * getopt --
 *      Parse argc/argv argument vector.
 */
int getopt(int nargc, char* const nargv[], const char* ostr) throw() {
  static const char* place = EMSG; /* option letter processing */
  const char* oli;                 /* option letter list index */

  if (optreset || !*place) {       /* update scanning pointer */
    optreset = 0;
    if (optind >= nargc || *(place = nargv[optind]) != '-') {
      place = EMSG;
      return (-1);
    }
    if (place[1] && *++place == '-') { /* found "--" */
      ++optind;
      place = EMSG;
      return (-1);
    }
  } /* option letter okay? */
  if ((optopt = (int)*place++) == (int)':' || !(oli = strchr(ostr, optopt))) {
    /*
     * if the user didn't specify '-' as an option,
     * assume it means -1.
     */
    if (optopt == (int)'-')
      return (-1);
    if (!*place)
      ++optind;
    if (opterr && *ostr != ':')
      (void)printf("illegal option -- %c\n", optopt);
    return (BADCH);
  }
  if (*++oli != ':') { /* don't need argument */
    optarg = NULL;
    if (!*place)
      ++optind;
  } else {                        /* need an argument */
    if (*place)                   /* no white space */
      optarg = (char*)place;
    else if (nargc <= ++optind) { /* no arg */
      place = EMSG;
      if (*ostr == ':')
        return (BADARG);
      if (opterr)
        (void)printf("option requires an argument -- %c\n", optopt);
      return (BADCH);
    } else /* white space */
      optarg = nargv[optind];
    place = EMSG;
    ++optind;
  }
  return (optopt); /* dump back option letter */
}
