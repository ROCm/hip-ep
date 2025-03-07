/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#pragma once
#include "./_sanity_check.hpp"
#include "vaip/graph.hpp"
#include "vaip/my_ort.h"
#include <memory>
#include <string>

namespace vaip_core {
VAIP_DLL_SPEC ModelPtr model_load(const std::string& filename);
VAIP_DLL_SPEC void model_set_meta_data(Model& model, const std::string& key,
                                       const std::string& value);
VAIP_DLL_SPEC ModelPtr model_clone(const Model& model,
                                   int64_t external_data_threshold = 64);

} // namespace vaip_core

namespace vaip_cxx {
class VAIP_DLL_SPEC Model {
public:
  ~Model();
  /** @brief the name of model
   *
   * @return a Model object
   */
  /**
   * Creates a new instance of the Model class.
   *
   * @param model_path The path to the model file.
   * @param opset A vector of pairs representing the operator set version.
   *              Each pair consists of a string representing the operator
   * domain and an int64_t representing the operator version.
   * @return A unique pointer to the created Model instance.
   */
  static std::unique_ptr<Model>
  create(const std::filesystem::path& model_path,
         const std::vector<std::pair<std::string, int64_t>>& opset);
  /** @brief the name of model
   *
   * @return a Model object
   */
  static std::unique_ptr<Model> load(const std::filesystem::path& model_path);

  /** @brief the name of model
   *
   * @return the name of the main graph
   */
  const std::string& name() const;

  /**
   * @brief Sets the metadata for the model.
   *
   * This function sets the metadata for the model with the specified name and
   * value.
   *
   * @param name The name of the metadata.
   * @param value The value of the metadata.
   * @return A reference to the updated Model object.
   */
  Model& set_metadata(const std::string& name, const std::string& value);
  /**
   * Retrieves the metadata value associated with the given name.
   *
   * @param name The name of the metadata to retrieve.
   * @return The metadata value associated with the given name.
   */
  std::string get_metadata(const std::string& name) const;
  /**
   * @brief Checks if the specified metadata exists.
   *
   * This function checks if the metadata with the given name exists in the
   * model.
   *
   * @param name The name of the metadata to check.
   * @return `true` if the metadata exists, `false` otherwise.
   */
  bool has_metadata(const std::string& name) const;
  /**
   * @brief Conversion operator to retrieve a const reference to the underlying
   * onnxruntime::Model object.
   *
   * @return const onnxruntime::Model& A const reference to the underlying
   * onnxruntime::Model object.
   */
  operator const onnxruntime::Model&() const { return *self_.get(); }
  /**
   * @brief Implicit conversion operator to retrieve the underlying
   * onnxruntime::Model object.
   *
   * This operator allows implicit conversion of an object to an
   * onnxruntime::Model reference. It returns a reference to the underlying
   * onnxruntime::Model object.
   *
   * @return Reference to the underlying onnxruntime::Model object.
   */
  operator onnxruntime::Model&() { return *self_.get(); }

  /**
   * @brief Retrieves the main graph of the model.
   *
   * @return A reference to the main graph of the model.
   */
  GraphRef main_graph();
  /**
   * @brief Retrieves the main graph of the model.
   *
   * @return A const reference to the main graph of the model.
   */
  const GraphRef main_graph() const;

  /**
   *  @brief Clones the model.
   *
   *  @return A new Model object that is a clone of the current Model object.
   */
  std::unique_ptr<Model> clone(int64_t external_data_threshold = 64) const;

private:
  Model(vaip_core::ModelPtr&& ptr);

private:
  vaip_core::ModelPtr self_;
};
} // namespace vaip_cxx
