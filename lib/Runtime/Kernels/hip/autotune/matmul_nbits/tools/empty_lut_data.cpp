/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Empty LUT payload, for builds that must not consult a table.
 *
 * The sweep driver links this instead of the generated table on purpose: if it
 * loaded the real one it would get LUT hits for shapes it is supposed to be
 * measuring, and the next table would be a copy of the previous one rather than
 * a fresh measurement.
 *
 * The same stub is what CMake uses for an arch with no measured table, so this
 * is also the "no table" path exercised in ordinary builds.
 */
#include <cstddef>

extern "C" const unsigned char kMatmulNbitsLutData[1] = {0};
extern "C" const size_t kMatmulNbitsLutData_size = 0;
