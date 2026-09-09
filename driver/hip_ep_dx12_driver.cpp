/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * hip-ep-dx12-driver -- standalone smoke-test tool for Dx12Runner.
 *
 * Architecture
 * ============
 * hip-ep's normal execution path compiles a HIP-dialect model to LLVM
 * bitcode / a native .dll that, at runtime, calls back into HIP kernels
 * from lib/Runtime/Kernels/hip via the ROCm/HIP runtime (hipLaunchKernelGGL).
 * That artifact does not contain a standalone, relocatable GPU code object
 * for a single kernel -- there is currently no pass that emits one.
 *
 * This tool bridges that gap for a single kernel at a time: it loads a
 * precompiled GPU ELF (HSA or PAL) for one HIP kernel -- e.g. one built from
 * driver/dx12/kernels/hip_ep_add_f32.hip via
 * driver/dx12/kernels/build_add_kernel.ps1 -- and drives it through
 * Dx12Runner (AMD Cross-Compile D3D12 extension API) instead of the normal
 * HIP/ROCm runtime. It is not part of the ONNX Runtime execution provider.
 *
 * Usage:
 *   hip-ep-dx12-driver.exe --kernel <path.elf> [options]
 *
 * See print_usage() below for the full option list.
 */

#include "dx12/dx12_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options
{
    std::string kernel_path;
    std::string entry_point   = "hip_ep_add_f32";
    std::string elf_format    = "hsa_dyn";
    std::string adapter       = "";
    uint64_t    num_elements  = 16; // matches sample_hip_add_compiler.hip.mlir (1x4x4xf32)
    uint32_t    group_size    = 256;
    unsigned    seed          = 42;
    bool        verbose       = false;
    bool        assume_gfx1151 = false;
};

void print_usage(const char* argv0)
{
    std::cout <<
        "Usage: " << argv0 << " --kernel <path.elf> [options]\n"
        "\n"
        "Runs a single precompiled GPU add-kernel ELF (lhs + rhs -> output,\n"
        "float32) through Dx12Runner and checks the result on the CPU.\n"
        "\n"
        "Options:\n"
        "  --kernel <path>     Path to the compiled kernel ELF (required)\n"
        "  --entry <name>      Kernel entry point symbol (default: hip_ep_add_f32)\n"
        "  --format <fmt>      hsa_rel | hsa_dyn | pal_rel | pal_dyn (default: hsa_dyn)\n"
        "  --elements <n>      Number of float32 elements (default: 16)\n"
        "  --group-size <n>    Threads per group (default: 256)\n"
        "  --adapter <sel>     Dx12Runner adapter selector (default: auto)\n"
        "  --assume-gfx1151    Accept the selected AMD adapter as gfx1151\n"
        "  --seed <n>          RNG seed for random inputs (default: 42)\n"
        "  --verbose           Verbose Dx12Runner logging\n";
}

bool parse_args(int argc, char** argv, Options& opts)
{
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if(i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };

        if(arg == "--kernel")
            opts.kernel_path = next("--kernel");
        else if(arg == "--entry")
            opts.entry_point = next("--entry");
        else if(arg == "--format")
            opts.elf_format = next("--format");
        else if(arg == "--elements")
            opts.num_elements = std::stoull(next("--elements"));
        else if(arg == "--group-size")
            opts.group_size = static_cast<uint32_t>(std::stoul(next("--group-size")));
        else if(arg == "--adapter")
            opts.adapter = next("--adapter");
        else if(arg == "--seed")
            opts.seed = static_cast<unsigned>(std::stoul(next("--seed")));
        else if(arg == "--assume-gfx1151")
            opts.assume_gfx1151 = true;
        else if(arg == "--verbose")
            opts.verbose = true;
        else if(arg == "--help" || arg == "-h")
            return false;
        else
        {
            std::cerr << "error: unrecognised argument \x27" << arg << "\x27\n";
            return false;
        }
    }
    return !opts.kernel_path.empty();
}

hip_ep::dx12::ElfFormat parse_elf_format(const std::string& s)
{
    if(s == "hsa_rel")
        return hip_ep::dx12::ElfFormat::HsaRel;
    if(s == "hsa_dyn")
        return hip_ep::dx12::ElfFormat::HsaDyn;
    if(s == "pal_rel")
        return hip_ep::dx12::ElfFormat::PalRel;
    if(s == "pal_dyn")
        return hip_ep::dx12::ElfFormat::PalDyn;
    throw std::runtime_error("unknown --format \x27" + s + "\x27");
}

std::vector<uint8_t> read_file_bytes(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if(!f)
        throw std::runtime_error("cannot open kernel ELF \x27" + path + "\x27");
    auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    return buf;
}

std::vector<char> to_char_buf(const std::vector<float>& v)
{
    std::vector<char> out(v.size() * sizeof(float));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    Options opts;
    try
    {
        if(!parse_args(argc, argv, opts))
        {
            print_usage(argv[0]);
            return 1;
        }

        // Build the add(lhs, rhs) -> output kernel descriptor. Matches the
        // "add-f32" ABI of driver/dx12/kernels/hip_ep_add_f32.hip:
        //   (const float* lhs, const float* rhs, float* output, int64_t n)
        hip_ep::dx12::KernelDescriptor kd;
        kd.entry_point           = opts.entry_point;
        kd.hsaco_data            = read_file_bytes(opts.kernel_path);
        kd.elf_format            = parse_elf_format(opts.elf_format);
        kd.group_size_x          = opts.group_size;
        kd.dispatch_x            = static_cast<uint32_t>(
            (opts.num_elements + opts.group_size - 1) / opts.group_size);
        kd.output_arg_index      = 2; // lhs=0, rhs=1, output=2
        kd.output_element_bytes  = sizeof(float);
        kd.arg_sizes             = {
            opts.num_elements * sizeof(float), // lhs
            opts.num_elements * sizeof(float), // rhs
            opts.num_elements * sizeof(float), // output
        };
        kd.scalar_slots.push_back(hip_ep::dx12::ScalarSlot{
            /*arg_index=*/3,
            /*value_bytes=*/std::vector<char>(sizeof(int64_t))});
        std::memcpy(kd.scalar_slots.back().value_bytes.data(), &opts.num_elements,
                    sizeof(int64_t));

        std::mt19937 rng(opts.seed);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        std::vector<float> lhs(opts.num_elements), rhs(opts.num_elements);
        for(uint64_t i = 0; i < opts.num_elements; ++i)
        {
            lhs[i] = dist(rng);
            rhs[i] = dist(rng);
        }

        std::vector<std::vector<char>> inputs;
        inputs.push_back(to_char_buf(lhs));
        inputs.push_back(to_char_buf(rhs));

        std::cout << "Loading kernel ELF: " << opts.kernel_path
                   << " (entry=" << opts.entry_point << ", format=" << opts.elf_format
                   << ", elements=" << opts.num_elements << ")\n";

        hip_ep::dx12::Dx12Runner runner(opts.adapter, opts.verbose, opts.assume_gfx1151);
        std::vector<float> output = runner.execute(kd, inputs);

        double max_abs_diff = 0.0;
        for(uint64_t i = 0; i < opts.num_elements; ++i)
        {
            float expected = lhs[i] + rhs[i];
            max_abs_diff   = (std::max)(max_abs_diff,
                                      static_cast<double>(std::fabs(output[i] - expected)));
        }

        std::cout << "max |output - (lhs+rhs)| = " << max_abs_diff << "\n";
        if(max_abs_diff > 1e-4)
        {
            std::cerr << "FAIL: output does not match CPU-computed lhs+rhs\n";
            return 1;
        }
        std::cout << "PASS\n";
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}