/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./create-session.hpp"
#include "./wide-string.hpp"
#include <fstream>
namespace test_onnx_runner {

static std::vector<uint8_t> ReadBinaryFile(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    return buffer;
  } else {
    throw std::runtime_error("Failed to read model file");
  }
}
std::unique_ptr<Ort::Session>
create_session(const std::filesystem::path& model_path, Ort::Env& env,
               Ort::SessionOptions& session_options, const Config& config) {

  auto use_mempry_model = config.use_memory_model();

  std::unique_ptr<Ort::Session> p_session;
  auto create_session_start_time = std::chrono::steady_clock::now();
  try {
    if (use_mempry_model) {
      auto model_data = ReadBinaryFile(model_path.string());
      p_session = std::make_unique<Ort::Session>(
          env, model_data.data(), model_data.size(), session_options);
    } else {
      p_session = std::make_unique<Ort::Session>(
          env, PathToString<ORTCHAR_T>()(model_path).c_str(), session_options);
    }
  } catch (Ort::Exception& e) {
    std::cout << "Catched Ort Exception: " << e.what() << std::endl;
    exit(1);
  }
  auto create_session_end_time = std::chrono::steady_clock::now();
  std::cout << " ONNXRuntime session create :  "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   create_session_end_time - create_session_start_time)
                   .count()
            << " us" << std::endl;
  return std::move(p_session);
}
} // namespace test_onnx_runner
