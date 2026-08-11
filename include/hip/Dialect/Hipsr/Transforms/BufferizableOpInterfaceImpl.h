/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define HIPSR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace hipsr {

// Attaches every hipsr BufferizableOpInterface model. This is the only way to
// register them, so no caller can bring up a partial set.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
