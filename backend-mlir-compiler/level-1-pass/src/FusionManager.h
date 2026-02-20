/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef FUSION_MANAGER_H
#define FUSION_MANAGER_H

#include <string>

namespace morphizen {
class IPass;
}

namespace morphizen_cxx {
class GraphRef;
}

namespace hipdnn {
namespace level1pass {

class FusionManager {
public:
  // Fuse compiled artifact into ONNX Runtime graph
  // Returns true on success, false on failure
  static bool fuseGraph(morphizen::IPass &self, morphizen_cxx::GraphRef &graph,
                        const std::string &metadata,
                        const std::string &unique_id);
};

} // namespace level1pass
} // namespace hipdnn

#endif
