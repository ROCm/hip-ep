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
#include "dx12/hsa_metadata.hpp"
#include "dx12/elf_single_kernel.hpp"

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
    std::string entry_point   = ""; // empty = use the ELF single kernel
    std::string elf_format    = "hsa_rel"; // ET_DYN removes the device under PAL
    std::string adapter       = "";
    uint64_t    num_elements  = 16; // matches sample_hip_add_compiler.hip.mlir (1x4x4xf32)
    uint32_t    group_size    = 256;
    unsigned    seed          = 42;
    std::string verify        = "none"; // none | add | sqrt
    std::vector<std::int64_t> scalars;  // by_value overrides, in ABI order
    bool        dump_abi      = false;
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
        "  --entry <name>      Kernel symbol (default: the ELF single kernel)\n"
        "  --format <fmt>      hsa_rel | hsa_dyn | pal_rel | pal_dyn (default: hsa_rel)\n"
        "  --elements <n>      Number of float32 elements (default: 16)\n"
        "  --group-size <n>    Threads per group (default: 256)\n"
        "  --adapter <sel>     Dx12Runner adapter selector (default: auto)\n"
        "  --assume-gfx1151    Accept the selected AMD adapter as gfx1151\n"
        "  --verify <mode>     none | add | sqrt CPU reference (default: none)\n"
        "  --scalars a,b       by_value arg values in ABI order (default: --elements)\n"
        "  --dump-abi          Print kernel ABI from ELF metadata and exit\n"
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
        else if(arg == "--dump-abi")
            opts.dump_abi = true;
        else if(arg == "--verify")
            opts.verify = next("--verify");
        else if(arg == "--scalars")
        {
            const std::string s = next("--scalars");
            for(std::size_t p = 0; p <= s.size(); )
            {
                const std::size_t comma = s.find(44, p);
                const std::string tok = s.substr(p, comma == std::string::npos ? comma : comma - p);
                if(!tok.empty()) opts.scalars.push_back(std::stoll(tok));
                if(comma == std::string::npos) break;
                p = comma + 1;
            }
        }
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

        if(opts.dump_abi)
        {
            auto elf  = read_file_bytes(opts.kernel_path);
            auto abis = hip_ep::dx12::hsa_md::parse(elf);
            std::cout << "kernels: " << abis.size() << "\n";
            for(const auto& a : abis)
            {
                std::cout << "\n  " << a.name
                          << "\n    kernarg_segment_size=" << a.kernarg_segment_size
                          << " max_flat_workgroup_size=" << a.max_flat_workgroup_size
                          << " group_segment=" << a.group_segment_fixed_size
                          << "\n    args (" << a.explicit_arg_count() << " explicit / "
                          << a.args.size() << " total):\n";
                for(const auto& g : a.args)
                {
                    using AA = hip_ep::dx12::hsa_md::ArgAccess;
                    const char* acc = g.access == AA::ReadOnly  ? "read_only"
                                    : g.access == AA::WriteOnly ? "write_only"
                                    : g.access == AA::ReadWrite ? "read_write" : "-";
                    std::cout << "      off=" << g.offset << " size=" << g.size
                              << " kind=" << g.value_kind << " access=" << acc
                              << (g.name.empty() ? std::string() : (" name=" + g.name)) << "\n";
                }
            }
            return 0;
        }

        // The code object is self-describing: recover the kernel ABI from its
        // AMDHSA note rather than hard-coding one shape per kernel.
        auto elf  = read_file_bytes(opts.kernel_path);
        auto abis = hip_ep::dx12::hsa_md::parse(elf);
        if(abis.empty())
            throw std::runtime_error("no AMDHSA kernel metadata in " + opts.kernel_path);

        const hip_ep::dx12::hsa_md::KernelAbi* abi = nullptr;
        if(opts.entry_point.empty())
        {
            if(abis.size() != 1)
                throw std::runtime_error("ELF holds " + std::to_string(abis.size()) +
                                         " kernels; select one with --entry");
            abi = &abis.front();
        }
        else
        {
            for(const auto& a : abis)
                if(a.name == opts.entry_point) { abi = &a; break; }
            if(abi == nullptr)
                throw std::runtime_error("entry not found in ELF: " + opts.entry_point);
        }

        // PAL resolves one kernel via its .kd symbol; a multi-kernel object removes the device.
        if(abis.size() > 1)
        {
            auto single = hip_ep::dx12::elf_split::build_single_kernel(elf, abi->name);
            if(single.empty())
                throw std::runtime_error("failed to extract single kernel: " + abi->name);
            std::cout << "extracted single-kernel ELF: " << elf.size() << " -> "
                      << single.size() << " bytes\n";
            elf = std::move(single);
        }

        using AK = hip_ep::dx12::hsa_md::ArgKind;
        using AA = hip_ep::dx12::hsa_md::ArgAccess;

        hip_ep::dx12::KernelDescriptor kd;
        kd.entry_point          = abi->name;
        kd.hsaco_data           = elf;
        kd.elf_format           = parse_elf_format(opts.elf_format);
        kd.group_size_x         = opts.group_size;
        kd.dispatch_x           = static_cast<uint32_t>(
            (opts.num_elements + opts.group_size - 1) / opts.group_size);
        kd.output_element_bytes = sizeof(float);

        std::size_t arg_index = 0; // position among explicit (non-hidden) args
        std::size_t n_scalars = 0;
        bool        have_out  = false;
        for(const auto& a : abi->args)
        {
            if(a.kind == AK::Hidden)
                continue;
            if(a.kind == AK::GlobalBuffer)
            {
                kd.arg_sizes.push_back(opts.num_elements * sizeof(float));
                if(a.access == AA::WriteOnly || a.access == AA::ReadWrite)
                {
                    kd.output_arg_index = kd.arg_sizes.size() - 1;
                    have_out            = true;
                }
            }
            else if(a.kind == AK::ByValue)
            {
                // by_value args default to the element count; --scalars overrides in order.
                std::int64_t v = (n_scalars < opts.scalars.size())
                                     ? opts.scalars[n_scalars]
                                     : static_cast<std::int64_t>(opts.num_elements);
                std::vector<char> bytes(a.size, 0);
                std::memcpy(bytes.data(), &v, (std::min)(a.size, sizeof(std::int64_t)));
                kd.scalar_slots.push_back(
                    hip_ep::dx12::ScalarSlot{arg_index, std::move(bytes)});
                ++n_scalars;
            }
            else
            {
                throw std::runtime_error("unsupported arg kind: " + a.value_kind);
            }
            ++arg_index;
        }
        if(!have_out)
            throw std::runtime_error("kernel ABI has no write_only buffer");
        if(kd.arg_sizes.empty())
            throw std::runtime_error("kernel ABI has no buffer arguments");

        const std::size_t n_inputs = kd.arg_sizes.size() - 1;
        if(opts.verify == "add" && n_inputs < 2)
            throw std::runtime_error("--verify add needs at least two input buffers");

        std::mt19937 rng(opts.seed);
        std::uniform_real_distribution<float> dist(
            opts.verify == "sqrt" ? 0.25f : -4.0f, 4.0f);
        std::vector<std::vector<float>> host_in(
            n_inputs, std::vector<float>(opts.num_elements));
        for(auto& buf : host_in)
            for(auto& x : buf) x = dist(rng);

        std::vector<std::vector<char>> inputs;
        for(const auto& buf : host_in)
            inputs.push_back(to_char_buf(buf));

        std::cout << "Loading kernel ELF: " << opts.kernel_path
                  << " (entry=" << kd.entry_point
                  << ", format=" << opts.elf_format
                  << ", buffers=" << kd.arg_sizes.size()
                  << ", scalars=" << kd.scalar_slots.size()
                  << ", out_index=" << kd.output_arg_index
                  << ", elements=" << opts.num_elements << ")\n";

        hip_ep::dx12::Dx12Runner runner(opts.adapter, opts.verbose, opts.assume_gfx1151);
        std::vector<float> output = runner.execute(kd, inputs);

        if(opts.verify == "none")
        {
            double lo = output.empty() ? 0.0 : output[0];
            double hi = lo;
            for(float v : output)
            {
                lo = (std::min)(lo, static_cast<double>(v));
                hi = (std::max)(hi, static_cast<double>(v));
            }
            std::cout << "dispatched; output range [" << lo << ", " << hi << "]\nPASS\n";
            return 0;
        }

        double max_abs_diff = 0.0;
        for(uint64_t i = 0; i < opts.num_elements; ++i)
        {
            float expected = (opts.verify == "sqrt")
                                 ? std::sqrt(host_in[0][i])
                                 : (host_in[0][i] + host_in[1][i]);
            max_abs_diff   = (std::max)(max_abs_diff,
                                      static_cast<double>(std::fabs(output[i] - expected)));
        }
        std::cout << "max |output - "
                  << (opts.verify == "sqrt" ? "sqrt(input)" : "(lhs+rhs)")
                  << "| = " << max_abs_diff << "\n";
        if(max_abs_diff > 1e-4)
        {
            std::cerr << "FAIL: output does not match CPU reference\n";
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
