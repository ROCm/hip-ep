/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "spool_writer.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace ktrace::collector {
namespace {
const char *OpName(std::uint32_t op) {
  switch (op) {
  case 0:
    return "dispatch";
  case 1:
    return "copy";
  case 2:
    return "barrier";
  default:
    return "other";
  }
}

const char *KindName(std::uint32_t kind) {
  switch (kind) {
  case 0x0000:
    return "InternalMarker";
  case 0x11F0:
    return "KernelExecution";
  case 0x11F1:
    return "Task";
  case 0x11F2:
    return "NativeKernel";
  case 0x11F3:
    return "CopyDeviceToHost";
  case 0x11F4:
    return "CopyHostToDevice";
  case 0x11F5:
    return "CopyDeviceToDevice";
  case 0x11F6:
    return "ReadImage";
  case 0x11F7:
    return "WriteImage";
  case 0x11F8:
    return "CopyImage";
  case 0x11F9:
    return "CopyImageToBuffer";
  case 0x11FA:
    return "CopyBufferToImage";
  case 0x11FB:
    return "MapBuffer";
  case 0x11FC:
    return "MapImage";
  case 0x11FD:
    return "UnmapMemObject";
  case 0x11FE:
    return "Marker";
  case 0x11FF:
    return "AcquireGLObjects";
  case 0x1200:
    return "ReleaseGLObjects";
  case 0x1201:
    return "CopyDeviceToHost2D";
  case 0x1202:
    return "CopyHostToDevice2D";
  case 0x1203:
    return "CopyDeviceToDevice2D";
  case 0x1204:
    return "User";
  case 0x1205:
    return "Barrier";
  case 0x1206:
    return "MigrateMemObjects";
  case 0x1207:
    return "FillBuffer";
  case 0x1208:
    return "FillImage";
  case 0x1209:
    return "SvmFree";
  case 0x120A:
    return "SvmMemcpy";
  case 0x120B:
    return "SvmMemFill";
  case 0x120C:
    return "SvmMap";
  case 0x120D:
    return "SvmUnmap";
  default:
    return "Other";
  }
}

const char *RoctxName(std::uint32_t operation) {
  switch (operation) {
  case 0:
    return "MARK";
  case 1:
    return "RANGE_PUSH";
  case 2:
    return "RANGE_POP";
  case 3:
    return "RANGE_START";
  case 4:
    return "RANGE_STOP";
  default:
    return "UNKNOWN";
  }
}

std::string Quote(const char *value) {
  std::string result = "\"";
  for (const char *cursor = value; cursor && *cursor; ++cursor) {
    if (*cursor == '"')
      result.push_back('"');
    result.push_back(*cursor);
  }
  result.push_back('"');
  return result;
}
} // namespace

SpoolWriter::SpoolWriter(fs::path output_dir, std::string prefix,
                         std::uint64_t resume_timestamp_ns)
    : output_dir_(std::move(output_dir)), prefix_(std::move(prefix)),
      resume_timestamp_ns_(resume_timestamp_ns) {
  fs::create_directories(output_dir_);
}

fs::path SpoolWriter::Base(std::uint32_t pid) const {
  return output_dir_ / ("." + prefix_ + "_" + std::to_string(pid) + "_ops.csv");
}

SpoolWriter::Files &SpoolWriter::Process(std::uint32_t pid) {
  auto [it, inserted] = processes_.try_emplace(pid);
  Files &files = it->second;
  if (inserted)
    files.pid = pid;
  return files;
}

void SpoolWriter::EnsureOps(Files &files) {
  if (files.ops_file.is_open())
    return;
  files.ops_file.open(Base(files.pid), std::ios::binary | std::ios::trunc);
  if (!files.ops_file)
    throw std::runtime_error("cannot create ops spool");
  files.ops_file << "Id,Op,Kind,Kernel_Name,Bytes,Device,Queue,Corr,"
                    "Begin_us,End_us,Duration_us\n";
  files.ops_file << std::fixed << std::setprecision(4);
}

void SpoolWriter::Write(const ipc::Record &record) {
  process_ids_.insert(record.process_id);
  if (record.thread_id)
    thread_ids_.insert(record.thread_id);
  Files &files = Process(record.process_id);
  EnsureOps(files);
  const fs::path base = Base(record.process_id);
  switch (record.type) {
  case ipc::RecordType::op: {
    const auto &op = record.op;
    files.ops_file << files.op_id++ << ',' << OpName(op.op) << ','
                   << KindName(op.kind) << ',' << op.kernel_name << ','
                   << op.bytes << ',' << op.device_id << ',' << op.queue_id
                   << ',' << op.correlation_id << ',' << op.begin_ns / 1000.0
                   << ',' << op.end_ns / 1000.0 << ','
                   << (op.end_ns - op.begin_ns) / 1000.0 << '\n';
    ++files.ops;
    break;
  }
  case ipc::RecordType::api: {
    if (!files.api_file.is_open()) {
      files.api_file.open(base.string() + ".api.csv",
                          std::ios::binary | std::ios::trunc);
      files.api_file << "Corr,Api,Api_Id,Thread,Enter_us,Exit_us,Host_us,Args\n"
                     << std::fixed << std::setprecision(4);
    }
    const auto &api = record.api;
    files.api_file << api.correlation_id << ',' << api.api_name << ','
                   << api.api_id << ',' << record.thread_id << ','
                   << api.enter_ns / 1000.0 << ',' << api.exit_ns / 1000.0
                   << ',' << (api.exit_ns - api.enter_ns) / 1000.0 << ','
                   << api.arguments << '\n';
    ++files.apis;
    break;
  }
  case ipc::RecordType::roctx: {
    if (!files.roctx_file.is_open()) {
      files.roctx_file.open(base.string() + ".roctx.csv",
                            std::ios::binary | std::ios::trunc);
      files.roctx_file
          << "Op,Op_Id,Process,Thread,Range_Id,Timestamp_us,Message\n"
          << std::fixed << std::setprecision(4);
    }
    const auto &roctx = record.roctx;
    files.roctx_file << RoctxName(roctx.operation_id) << ','
                     << roctx.operation_id << ',' << record.process_id << ','
                     << record.thread_id << ',' << roctx.range_id << ','
                     << roctx.timestamp_ns / 1000.0 << ','
                     << Quote(roctx.message) << '\n';
    ++files.roctx;
    break;
  }
  case ipc::RecordType::vram: {
    if (!files.vram_file.is_open()) {
      files.vram_file.open(base.string() + ".vram.csv",
                           std::ios::binary | std::ios::trunc);
      files.vram_file << "Time_us,Free_Bytes,Total_Bytes,Used_Bytes\n"
                      << std::fixed << std::setprecision(4);
    }
    const auto &vram = record.vram;
    files.vram_file << vram.timestamp_ns / 1000.0 << ',' << vram.free_bytes
                    << ',' << vram.total_bytes << ','
                    << (vram.total_bytes - vram.free_bytes) << '\n';
    break;
  }
  case ipc::RecordType::kernel: {
    if (!files.kernel_file.is_open()) {
      files.kernel_file.open(base.string() + ".kernels.csv",
                             std::ios::binary | std::ios::trunc);
      files.kernel_file
          << "Kernel_Name,Func_Addr,VGPR,Scratch_Bytes,Static_LDS,"
             "Max_Threads,Binary_Version,Occ_Block_Size,Occ_Dyn_LDS,"
             "Theoretical_Blocks_Per_CU,Theoretical_Occupancy_Pct\n";
    }
    const auto &kernel = record.kernel;
    files.kernel_file << kernel.name << ",0x" << std::hex << kernel.func_addr
                      << std::dec << ',' << kernel.vgpr << ',' << kernel.scratch
                      << ',' << kernel.static_lds << ',' << kernel.max_threads
                      << ',' << kernel.binary_version << ','
                      << kernel.block_size << ',' << kernel.dynamic_lds << ','
                      << kernel.blocks_per_cu << ',' << std::fixed
                      << std::setprecision(1) << kernel.occupancy_pct << '\n';
    break;
  }
  case ipc::RecordType::device:
    files.device = record.device;
    files.have_device = true;
    if (record.device.runtime_version > 0) {
      const int version = record.device.runtime_version;
      std::cout << "[collector] HIP runtime PID " << record.process_id
                << " version " << version / 10000000 << '.'
                << (version / 100000) % 100 << '.' << version % 100000 << " ("
                << version << ")\n";
    }
    break;
  }
}

void SpoolWriter::WriteSummary(Files &files, std::uint64_t pushed,
                               std::uint64_t dropped,
                               std::uint64_t high_watermark) {
  const fs::path base = Base(files.pid);
  std::ofstream summary(base.string() + ".summary.txt",
                        std::ios::binary | std::ios::trunc);
  summary << "pid              : " << files.pid << "\n\n";
  if (files.have_device) {
    const auto &d = files.device;
    summary << "device           : " << d.name << '\n'
            << "  CUs            : " << d.cu_count << '\n'
            << "  warp size      : " << d.warp_size << '\n'
            << "  threads/CU     : " << d.threads_per_cu << '\n'
            << "  regs/block     : " << d.registers_per_block << '\n'
            << "  LDS/CU         : " << d.lds_per_cu << '\n'
            << "  clock / mem    : " << d.clock_khz << " kHz / "
            << d.memory_clock_khz << " kHz, bus " << d.memory_bus_width
            << " bit\n  driver / hip   : " << d.driver_version << " / "
            << d.runtime_version << "\n\n";
  }
  summary << "records          : " << files.ops << '\n'
          << "API calls       : " << files.apis << '\n'
          << "ROCTx events    : " << files.roctx << '\n'
          << "IPC pushed      : " << pushed << '\n'
          << "IPC dropped     : " << dropped << '\n'
          << "ring peak       : " << high_watermark << " / "
          << ipc::kRingCapacity << '\n';

  std::ofstream launch(base.string() + ".launch.csv",
                       std::ios::binary | std::ios::trunc);
  launch << "Process_Id,Resume_Timestamp_ns\n"
         << files.pid << ',' << resume_timestamp_ns_ << '\n';
}

void SpoolWriter::Finish(std::uint64_t pushed, std::uint64_t dropped,
                         std::uint64_t high_watermark) {
  for (auto &[pid, files] : processes_) {
    (void)pid;
    files.ops_file.flush();
    files.api_file.flush();
    files.roctx_file.flush();
    files.vram_file.flush();
    files.kernel_file.flush();
    WriteSummary(files, pushed, dropped, high_watermark);
  }
}
} // namespace ktrace::collector
