/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/_sanity_check.hpp"
#include "morphizen/graph.hpp"
#include <memory>
#include <morphizen/my_ort.h>
#include <string>

namespace morphizen {
MORPHIZEN_DLL_SPEC ModelPtr model_load(const std::string &filename);
MORPHIZEN_DLL_SPEC void model_set_meta_data(Model &model,
                                            const std::string &key,
                                            const std::string &value);
MORPHIZEN_DLL_SPEC ModelPtr model_clone(const Model &model,
                                        int64_t external_data_threshold = 64);

} // namespace morphizen

namespace morphizen_cxx {
class Model;

class ModelConstRef {
public:
  /**
   * @brief A constant reference wrapper for the morphizen::Model object.
   *
   * This constructor initializes the ModelConstRef with a constant reference
   * to an existing morphizen::Model instance.
   *
   * @param model A constant reference to a morphizen::Model object.
   */
  ModelConstRef(const morphizen::Model &model);
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
  operator const onnxruntime::Model &() { return self_; }

  /** @brief the name of model
   *
   * @return the name of the main graph
   */
  const std::string &name() const;
  /**
   * Retrieves the metadata value associated with the given name.
   *
   * @param name The name of the metadata to retrieve.
   * @return The metadata value associated with the given name.
   */
  std::string get_metadata(const std::string &name) const;

  /**
   * @brief Checks if the specified metadata exists.
   *
   * This function checks if the metadata with the given name exists in the
   * model.
   *
   * @param name The name of the metadata to check.
   * @return `true` if the metadata exists, `false` otherwise.
   */
  bool has_metadata(const std::string &name) const;
  /**
   * @brief Retrieves the main graph of the model.
   *
   * @return A const reference to the main graph of the model.
   */
  GraphConstRef main_graph() const;

  /**
   *  @brief Clones the model.
   *
   *  @return A new Model object that is a clone of the current Model object.
   */
  std::unique_ptr<Model> clone(int64_t external_data_threshold = 64) const;
  /**
   * @brief Retrieves the model path.
   * @return The path to the model file.
   *
   * This function returns the path to the model file associated with the
   * current Model object.
   *
   * @note The returned path is a std::filesystem::path object.
   */
  std::filesystem::path model_path() const;

private:
  const morphizen::Model &self_;
};
class MORPHIZEN_DLL_SPEC Model {
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
  create(const std::filesystem::path &model_path,
         const std::vector<std::pair<std::string, int64_t>> &opset);
  /** @brief the name of model
   *
   * @return a Model object
   */
  static std::unique_ptr<Model> load(const std::filesystem::path &model_path);

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
  Model &set_metadata(const std::string &name, const std::string &value);

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
  operator onnxruntime::Model &() { return *self_.get(); }
  ModelConstRef ref() const { return ModelConstRef(*self_.get()); }

  /**
   * @brief Retrieves the main graph of the model.
   *
   * @return A reference to the main graph of the model.
   */
  GraphRef main_graph();

private:
  Model(morphizen::ModelPtr &&ptr);

private:
  morphizen::ModelPtr self_;
  friend class ModelConstRef;
};
} // namespace morphizen_cxx
