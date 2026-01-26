# hipDNN Installation Guide for Windows

This guide provides step-by-step instructions for installing hipDNN on Windows to enable full graph validation in onnx-hipdnn-ep.

## Overview

To build onnx-hipdnn-ep with complete hipDNN graph validation support, you need:
1. TheRock ROCm SDK (nightly tarball for Windows)
2. Clang 20.x toolchain
3. Build hipDNN from source
4. Configure onnx-hipdnn-ep to use the installed hipDNN

## Prerequisites

### System Requirements
- Windows 10 or Windows 11 (Windows 11 recommended)
- AMD GPU with ROCm support
- Administrator privileges for system configuration

### Required Software
- Visual Studio 2022 with C++ workload
- CMake 3.25.2+
- Ninja build system
- Python 3
- Git (with Unix tools on PATH)

## Step-by-Step Installation

### 1. System Configuration

#### Enable Long Paths
Run this command in **Administrative PowerShell**:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### Enable Symlinks
Enable Developer Mode in Windows:
- **Windows 11**: Settings → System → For Developers → Developer Mode → toggle **On**
- **Windows 10**: Settings → Update & Security → For Developers → Developer Mode → toggle **On**

**Restart your computer** for these changes to take effect.

#### Configure Git
```bash
git config --global core.symlinks true
git config --global core.longpaths true
```

### 2. Install Build Tools (Using Chocolatey)

If you don't have Chocolatey, install it from https://chocolatey.org/install

Run these commands in **Administrative PowerShell**:

```powershell
# Install Visual Studio 2022 Build Tools
choco install visualstudio2022buildtools -y --params "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.ATL --add Microsoft.VisualStudio.Component.Windows11SDK.22621"

# Install other tools
choco install git.install -y --params "'/GitAndUnixToolsOnPath'"
choco install cmake --version=3.31.0 -y
choco install ninja -y
choco install python -y
```

### 3. Determine Your GPU Architecture

Download and extract Clang 20.x first (we'll use it to check GPU arch):

1. Download from: https://github.com/llvm/llvm-project/releases
   - Look for `LLVM-20.*.tar.xz` or `LLVM-20.*-win64.exe`
   
2. Extract to `C:\dist\clang` (no spaces in path)

3. Run `amdgpu-arch.exe` to find your GPU:
   ```cmd
   C:\dist\clang\bin\amdgpu-arch.exe
   ```
   
   This will output something like `gfx1103` - **record this value**.

4. Determine your GFX Family from this table:

   | GPU Architecture | GFX Family |
   |------------------|------------|
   | gfx900, gfx906, gfx908, gfx90a, gfx90c | gfx90X-all |
   | gfx940, gfx941, gfx942 | gfx94X-all |
   | gfx1030, gfx1031, gfx1032, gfx1034, gfx1035, gfx1036 | gfx103X-all |
   | gfx1100, gfx1101, gfx1102, gfx1103 | gfx110X-all |
   | gfx1150, gfx1151, gfx1152 | gfx115X-all |
   | gfx1200, gfx1201 | gfx120X-all |

### 4. Install TheRock ROCm SDK

1. **Download TheRock nightly tarball** for your GFX Family:
   - Go to: https://therock-nightly-tarball.s3.amazonaws.com/index.html
   - Find the latest `therock-dist-windows-gfx###-all-*.tar.gz` matching your GFX family
   - Example: `therock-dist-windows-gfx110X-all-7.10.0a20251103.tar.gz`

2. **Extract** to `C:\dist\therock` (no spaces in path)
   - After extraction, `C:\dist\therock\bin` should exist

3. **Set Environment Variables** (in **System Environment Variables**, not just session):
   ```cmd
   # Add TheRock to PATH
   set PATH=C:\dist\therock\bin;%PATH%
   
   # Set HIP platform
   set HIP_PLATFORM=amd
   ```

4. **Verify TheRock installation**:
   ```cmd
   hipconfig -rocmpath -n --hipclangpath
   ```
   
   Should output:
   ```
   C:\dist\therock
   C:\dist\therock\lib\llvm\bin
   ```

### 5. Build hipDNN

1. **Clone rocm-libraries repository** (sparse checkout for faster clone):
   ```bash
   cd C:\Develop\m\source
   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipdnn
   git checkout develop
   ```

2. **Configure hipDNN**:
   ```cmd
   cd C:\Develop\m\source\rocm-libraries\projects\hipdnn
   mkdir build
   cd build
   
   # Replace gfx1103 with your GPU architecture from step 3
   cmake -GNinja -DGPU_TARGETS=gfx1103 -DROCM_CMAKE_PATH=C:/dist/therock -DLLVM_TOOLS_SEARCH_PREFIX=C:/dist/clang -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local/hipdnn ..
   ```

3. **Build hipDNN** (this may take a while):
   ```cmd
   ninja -j4
   ```
   
   Note: Use `-j4` or `-j2` to limit parallel jobs if you have limited RAM.

4. **Install hipDNN**:
   ```cmd
   ninja install
   ```

   This will install hipDNN to `C:/Develop/m/local/hipdnn` with:
   - Headers in `C:/Develop/m/local/hipdnn/include`
   - Libraries in `C:/Develop/m/local/hipdnn/lib`
   - CMake configs in `C:/Develop/m/local/hipdnn/lib/cmake/hipdnn_frontend` and `hipdnn_backend`

### 6. Build onnx-hipdnn-ep with hipDNN

Now that hipDNN is installed, configure and build onnx-hipdnn-ep:

1. **Set environment variables** (in your shell):
   ```cmd
   set THEROCK_DIST=C:\dist\therock
   set HIP_PLATFORM=amd
   ```

2. **Clean previous build**:
   ```powershell
   Remove-Item -Recurse -Force C:\Develop\m\build\onnx-hipdnn-ep
   ```

3. **Configure with hipDNN**:
   ```cmd
   cd C:\Develop\m\source\morphizen-hipdnn
   cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -B C:/Develop/m/build/morphizen-hipdnn -S . -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local -DCMAKE_PREFIX_PATH=C:/Develop/m/local/hipdnn
   ```

4. **Build morphizen-hipdnn**:
   ```cmd
   cmake --build C:/Develop/m/build/morphizen-hipdnn
   ```

## Verification

After successful build, verify the hipDNN graph validation is working:

1. Check that the level-1-pass library includes hipDNN symbols
2. Run a test with `MORPHIZEN_DEBUG_HIPDNN=1` to see validation logs
3. The pass should log "hipDNN graph validation succeeded" for compatible Conv operations

## Troubleshooting

### Issue: "Could not find hipdnn_frontend"
**Solution**: Ensure `CMAKE_PREFIX_PATH` includes the hipDNN install directory and that the cmake config files exist:
- Check: `C:/Develop/m/local/hipdnn/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake`
- Check: `C:/Develop/m/local/hipdnn/lib/cmake/hipdnn_backend/hipdnn_backendConfig.cmake`

### Issue: GPU_TARGETS not set
**Solution**: Always specify `-DGPU_TARGETS=gfxXXXX` when configuring hipDNN on Windows (auto-detection doesn't work).

### Issue: Out of memory during build
**Solution**: Reduce parallel jobs with `ninja -j2` or `ninja -j4`.

### Issue: Clang tool version mismatch
**Solution**: Install the correct Clang versions and use `-DLLVM_TOOLS_SEARCH_PREFIX=C:/dist/clang`.

### Issue: PATH conflicts with Visual Studio
**Solution**: Do NOT use "x64 Native Tools Command Prompt for VS 2022". Use regular PowerShell or CMD with the TheRock bin directory in PATH.

## References

- [hipDNN Building Instructions](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/Building.md)
- [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [TheRock Nightly Tarballs](https://therock-nightly-tarball.s3.amazonaws.com/index.html)
