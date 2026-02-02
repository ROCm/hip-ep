#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

// // 1. Include YOUR custom headers
// #include "HipDnnDialect.h"      // Defines the "hipdnn" dialect
// #include "HipDnnPasses.h"       // Defines the "convert-hipdnn-to-llvm" pass

int main(int argc, char **argv) {
  // A. Create a Dialect Registry
  mlir::DialectRegistry registry;
  
  // B. Register Standard Dialects (so you can use func, scf, etc.)
  //registerAllDialects(registry);

  // C. Register YOUR Custom Dialect
  //registry.insert<hipdnn::HipDnnDialect>();

  // D. Register Standard Passes
  //registerAllPasses();

  // E. Register YOUR Custom Pass (The "script" we discussed)
  // This allows you to run: --convert-hipdnn-to-llvm
  //hipdnn::registerHipDnnPasses();

  // F. Run the tool (Just like mlir-opt)
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "hip-opt: custom compiler driver\n", registry));
}
