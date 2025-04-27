$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
. "$SCRIPT_DIR/run-external-command.ps1"
Get-ChildItem env:
$env:CMAKE_PREFIX_PATH = "<prefix>;$env:CMAKE_PREFIX_PATH"
# e.g., $env:CMAKE_PREFIX_PATH = "C:/mylibs;$env:CMAKE_PREFIX_PATH"
Run cmake --version

Run cmake -DCMAKE_FFIND_USE_PACKAGE_REGISTRY=OFF `
    -DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON `
    -DBUILD_SHARED_LIBS=OFF `
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" `
    -S "${Env:VAI_RT_WORKSPACE}/morphizen-demo" -B "$Env:VAI_RT_BUILD_DIR/morphizen-demo" `
    --debug-find-pkg=glog `
    "-DCMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    "-DFETCHCONTENT_BASE_DIR=$Env:VAI_RT_PREFIX/morphizen_deps" `
    --fresh

$jobs = [Environment]::ProcessorCount

Run cmake  --build  "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -j $jobs --target install

Run cmake  --build  "$Env:VAI_RT_BUILD_DIR/morphizen-demo" --target morphizen-run-all-unit-tests

Run cmake  --build  "$Env:VAI_RT_BUILD_DIR/morphizen-demo" --target generate_ep_context_model
