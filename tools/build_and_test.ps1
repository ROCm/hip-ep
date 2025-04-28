$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
. "$SCRIPT_DIR/run-external-command.ps1"

Run cmake -DCMAKE_FFIND_USE_PACKAGE_REGISTRY=OFF `
    -DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON `
    -DBUILD_SHARED_LIBS=OFF `
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" `
    -S "${Env:VAI_RT_WORKSPACE}/morphizen-demo" -B "$Env:VAI_RT_BUILD_DIR/morphizen-demo" `
    "-DCMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    "-DFETCHCONTENT_BASE_DIR=$Env:VAI_RT_PREFIX/morphizen_deps" `
    --fresh

$jobs = [Environment]::ProcessorCount

Run cmake  --build  "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -j $jobs --target install

Run ctest -j $jobs --test-dir "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -C Debug "-E" "example"   --output-on-failure
