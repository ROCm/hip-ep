$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
. "$SCRIPT_DIR/run-external-command.ps1"
Set-Location "$Env:VAI_RT_WORKSPACE/onnxruntime"
Run python tools/ci_build/build.py --use_vitisai --config Debug --build_shared_lib --parallel --compile_no_warning_as_error `
    --skip_submodule_sync --build_dir "$Env:VAI_RT_BUILD_DIR/onnxruntime"   `
    --skip_tests `
    --enable_msvc_static_runtime `
    --cmake_extra_defines `
    "CMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    onnxruntime_BUILD_UNIT_TESTS=OFF `
    "CMAKE_CXX_FLAGS=/bigobj"
Run cmake --build "$Env:VAI_RT_BUILD_DIR/onnxruntime/Debug" --target install
Run python ../vai-rt/main.py --dev-mode --release_file=../vai-rt/release_file/latest_stx.txt --project zlib gsl gtest protobuf glog
