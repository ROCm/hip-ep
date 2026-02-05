#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

// Include custom HIP dialect headers
#include "HipDialect.h"      // Defines the "hip" dialect
#include "HipPasses.h"       // Defines the "convert-hip-to-llvm" pass

int main(int argc, char **argv) {
  // A. Create a Dialect Registry
  mlir::DialectRegistry registry;
  
  // B. Register Standard Dialects (so you can use func, scf, etc.)
  mlir::registerAllDialects(registry);

  // C. Register HIP Custom Dialect
  registry.insert<mlir::hip::HipDialect>();

  // D. Register Standard Passes
  mlir::registerAllPasses();

  // E. Register HIP Custom Passes
  // This allows you to run: --convert-hip-to-llvm
  mlir::hip::registerHipPasses();

  // F. Run the tool (Just like mlir-opt)
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "hip-opt: custom compiler driver\n", registry));
}
