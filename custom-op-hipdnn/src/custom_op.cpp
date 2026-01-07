/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// clang-format off
#include "morphizen/onnxruntime_api.hpp"

#include <glog/logging.h>
#include <sstream>
#include <fstream>
#include <morphizen/vaip.hpp>
#include <morphizen/env_config.hpp>
#include "./custom_op.hpp"
#include "google/protobuf/util/json_util.h"
#include <cstdlib>
#include <filesystem>
#include <thread>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)

namespace hipdnn {

	HipdnnCustomOp::HipdnnCustomOp(std::shared_ptr<const PassContext> context,
		const std::shared_ptr<MetaDefProto>& meta_def,
		onnxruntime::Model* model)
		: CustomOpImp(context, meta_def, model){
		MY_LOG(1) << "HipdnnCustomOp constructor: ";
		auto hipdnn_json_str = get_meta_def_param();
		auto status = google::protobuf::util::JsonStringToMessage(hipdnn_json_str,&hipdnn_proto_);
		if(status.ok()) {
			LOG(INFO) << "hipdnn_json_str: " << hipdnn_json_str;
		} else {
			LOG(FATAL) << "failed to parse hipdnn_json_str: " << status.ToString();
			return;
		}
		std::string ep_context_data;
		ep_context_data.resize((size_t)hipdnn_proto_.ep_context_file_size());
		auto dat_file = context->open_file_for_read(hipdnn_proto_.ep_context_file_name());
		if(dat_file == nullptr) {
			LOG(FATAL) << "failed to open file: " << hipdnn_proto_.ep_context_file_name();
		}
		auto read_size = dat_file->fread(ep_context_data.data(), hipdnn_proto_.ep_context_file_size());
		if(read_size != hipdnn_proto_.ep_context_file_size()) {
			LOG(FATAL) << "failed to read file: " << hipdnn_proto_.ep_context_file_name();
		}
		LOG(INFO) << "read ep_context_data size: " << ep_context_data;
		LOG(INFO) << "HIP DNN device_id: " << hipdnn_proto_.device_id();
		LOG(INFO) << "HIP DNN kernel_type: " << hipdnn_proto_.kernel_type();
	}

	HipdnnCustomOp::~HipdnnCustomOp() {}

	static std::string shape_to_string(const std::vector<int64_t>& shape) {
		std::ostringstream str;
		str << "[";
		int c = 0;
		for (auto s : shape) {
			if (c != 0) {
				str << ",";
			}
			str << s;
			c = c + 1;
		}
		str << "]";
		return str.str();
	}
	void HipdnnCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
		Ort::KernelContext ctx(context);
		auto num_inputs = ctx.GetInputCount();
		auto num_outputs = ctx.GetOutputCount();
		LOG(INFO) << "num_inputs " << num_inputs << " "   //
			<< "num_outputs " << num_outputs << " " //
			;
		auto tensor_shape = std::vector<int64_t>();
		for (auto idx = 0u; idx < num_inputs; ++idx) {
			auto input_tensor = ctx.GetInput(idx);
			auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
			auto tensor_type = tensor_info.GetElementType();
			tensor_shape = tensor_info.GetShape();
			auto element_num = tensor_info.GetElementCount();
			LOG(INFO)
				<< "input [" << idx << "] " <<
				"element_num " << element_num << " " //
				<< "tensor_type " << tensor_type << " " //
				<< "shape: " << shape_to_string(tensor_shape);
		}

		MY_LOG(1) << "sample_string: " << hipdnn_proto_.sample_string();
		MY_LOG(1) << "sample_int: " << hipdnn_proto_.sample_int();
		auto index = 0u;
		for(auto str : hipdnn_proto_.sample_strings()) {
			MY_LOG(1) << "sample_string[" << index << "]: " << str;
			index = index + 1;
		}
		index = 0u;
		for(auto i : hipdnn_proto_.sample_ints()) {
			MY_LOG(1) << "sample_int[" << index << "]: " << i;
			index = index + 1;
		}
		auto idx = 0u;
		// NOTE: must call this GetOutput function, otherwise the output tensor will not be created
		// it will crash when trying to access the output tensor
		auto output_tensor = ctx.GetOutput(idx, tensor_shape);
		auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
		auto tensor_type = tensor_info.GetElementType();
		auto element_num = tensor_info.GetElementCount();
		MY_LOG(1) << "output [" << idx << "] " << "element_num " << element_num << " " //
			<< "tensor_type " << tensor_type << " " //
			<< "shape: " << shape_to_string(tensor_shape);
		
		// HIP DNN kernel execution placeholder
		auto in_base = ctx.GetInput(idx).GetTensorData<float>();
  		auto out_base = output_tensor.GetTensorMutableData<int8_t>();
		for(auto i = 0; i < element_num; ++i) {
			// TODO: Replace with actual HIP DNN kernel implementation
			out_base[i] = static_cast<int8_t>(in_base[i]);
		}
		return;
	}

} // namespace hipdnn
