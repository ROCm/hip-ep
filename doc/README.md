# Documentation Index

This directory contains comprehensive documentation for the onnx-hipdnn-ep project.

## Getting Started

### Build Guides

- **[Linux Build Guide](linux_build_guide.md)** - Complete setup, build, and testing instructions for Linux
  - Prerequisites and system requirements
  - TheRock SDK installation
  - Building hipDNNEP and onnx-hipdnn-ep
  - Running tests and validation
  - Troubleshooting common issues

- **[Windows Build Guide](windows_build_guide.md)** - Complete setup, build, and testing instructions for Windows
  - Windows-specific prerequisites
  - TheRock SDK installation for Windows
  - Building with Clang/LLVM toolchain
  - Platform-specific troubleshooting
  - Patches and workarounds

- **[Windows hipDNN Setup](HIPDNN_WINDOWS_SETUP.md)** - Dedicated guide for hipDNN installation on Windows
  - System configuration (long paths, symlinks)
  - GPU architecture detection
  - TheRock SDK hardcoded path fixes
  - Building hipDNN from source

## Testing

- **[ResNet50 End-to-End Test](resnet50_e2e_test.md)** - Quick reference for running the classification test
  - Quick start commands
  - Expected output
  - Command-line options
  - Test data description

## Implementation Guides

### Architecture and Design

- **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** - Comprehensive implementation architecture
  - Overall system design
  - Protocol buffer changes
  - Level-1 pass implementation
  - Custom operation implementation
  - Graph serialization approach
  - UID extraction and management
  - Complete code examples

- **[hipDNN Graph API Guide](04_Graph_API_Guide.md)** - Working with hipDNN Frontend API
  - Building hipDNN graphs
  - Graph serialization and deserialization
  - Compiling graphs for execution
  - Execution with variant packs
  - Complete examples
  - Limitations and best practices

### Operation-Specific Guides

- **[Conv Implementation](CONV_IMPLEMENTATION.md)** - Convolution operation support details
  - Conv node pattern recognition
  - Attribute extraction
  - Graph building for Conv operations
  - Custom operator execution

## Documentation Quick Reference

| Document | Purpose | Audience |
|----------|---------|----------|
| [linux_build_guide.md](linux_build_guide.md) | Linux build & test | Developers (Linux) |
| [windows_build_guide.md](windows_build_guide.md) | Windows build & test | Developers (Windows) |
| [HIPDNN_WINDOWS_SETUP.md](HIPDNN_WINDOWS_SETUP.md) | hipDNN setup on Windows | Windows developers |
| [resnet50_e2e_test.md](resnet50_e2e_test.md) | ResNet50 testing | All users |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | Architecture & design | Advanced developers |
| [04_Graph_API_Guide.md](04_Graph_API_Guide.md) | hipDNN API usage | Advanced developers |
| [CONV_IMPLEMENTATION.md](CONV_IMPLEMENTATION.md) | Conv operation | Advanced developers |

## Additional Resources

- **Main README**: [../README.md](../README.md) - Project overview and quick start
- **Test Data README**: [../test/data/README_image_to_bin.md](../test/data/README_image_to_bin.md) - Image preprocessing tool

## Contributing

When adding new documentation:
1. Create the document in this `doc/` directory
2. Add an entry to this index with a brief description
3. Update cross-references in other documents as needed
4. Ensure the document follows the existing format and style

---

**Documentation Version**: 2.0  
**Last Updated**: January 26, 2026  
**Maintainer**: onnx-hipdnn-ep Team
