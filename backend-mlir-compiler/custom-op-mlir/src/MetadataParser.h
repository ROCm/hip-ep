/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef METADATA_PARSER_H
#define METADATA_PARSER_H

#include "mlir_compilation.pb.h"
#include <memory>
#include <optional>

namespace morphizen {
class MetaDefProto;
class PassContext;
} // namespace morphizen

namespace mlir_compilation {

namespace customop {

class MetadataParser {
public:
  // Parse MlirCompilationProto from JSON metadata string
  static std::optional<MlirCompilationProto>
  parse(const std::shared_ptr<const morphizen::PassContext> &context,
        const std::shared_ptr<morphizen::MetaDefProto> &meta_def);
};

} // namespace customop
} // namespace mlir_compilation

#endif
