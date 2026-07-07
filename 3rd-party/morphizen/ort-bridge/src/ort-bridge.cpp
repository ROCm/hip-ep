/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./morphizen-ep-factory.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>
#include <iostream>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_ORT_EP_API, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ORT_EP_API) >= n)

extern "C" {

OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories,
                             size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  // initialize the API in this dll.
  Ort::InitApi(ort_api);
  const OrtEpApi* ort_ep_api = ort_api->GetEpApi();
  MY_LOG(1) << "ORT is initalized, ORT_API_VERSION=" << ORT_API_VERSION
            << ", ptr=" << (void*)ort_ep_api << ", registration_name="
            << "\"" << registration_name << "\"";

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory =
      std::make_unique<morphizen::MorphiZenEpFactory>(
          registration_name, morphizen::ApiPtrs{*ort_api, *ort_ep_api},
          *default_logger);

  if (max_factories < 1) {
    return ort_api->CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Not enough space to return EP factory. Need at least one.");
  }
  factories[0] = factory.release();
  MY_LOG(1) << "CreateEpFactories: this=" << (void*)factories[0]
            << ", ep_name=" << factories[0]->GetName(factories[0])
            << ", vendor=" << factories[0]->GetVendor(factories[0]);
  *num_factories = 1;

  return nullptr;
}
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  MY_LOG(1) << " ReleaseEpFactory: this=" << (void*)factory;
  delete factory;
  return nullptr;
}
} // extern "C"
