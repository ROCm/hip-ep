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
#include <unordered_set>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_HIPDNN) >= n)

namespace hipdnn {

// Helper to load graph from file
static std::vector<uint8_t> LoadGraphFromFile(const std::string& filepath) {
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);
	if (!file) {
		throw std::runtime_error("Failed to open file for reading: " + filepath);
	}
	
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	
	std::vector<uint8_t> buffer(size);
	if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
		throw std::runtime_error("Failed to read graph from file: " + filepath);
	}
	
	return buffer;
}

HipdnnCustomOp::HipdnnCustomOp(std::shared_ptr<const PassContext> context,
	const std::shared_ptr<MetaDefProto>& meta_def,
	onnxruntime::Model* model)
	: CustomOpImp(context, meta_def, model), handle_(nullptr) {
	
	MY_LOG(1) << "HipdnnCustomOp constructor";
	
	// Parse proto
	auto hipdnn_json_str = get_meta_def_param();
	auto status = google::protobuf::util::JsonStringToMessage(hipdnn_json_str, &hipdnn_proto_);
	
	if (!status.ok()) {
		LOG(FATAL) << "failed to parse hipdnn_json_str: " << status.ToString();
		return;
	}
	
	LOG(INFO) << "Graph file: " << hipdnn_proto_.graph_file_name();
	
	// Create hipDNN handle
	hipdnnStatus_t hipdnn_status = hipdnnCreate(&handle_);
	if (hipdnn_status != HIPDNN_STATUS_SUCCESS) {
		LOG(FATAL) << "Failed to create hipDNN handle";
		return;
	}
	
	// Load and compile the graph
	try {
		LoadAndCompileGraph();
		LOG(INFO) << "Graph compiled successfully";
	} catch (const std::exception& ex) {
		LOG(FATAL) << "Failed to compile graph: " << ex.what();
	}
}

HipdnnCustomOp::~HipdnnCustomOp() {
	if (handle_) {
		hipdnnDestroy(handle_);
		handle_ = nullptr;
	}
}

void HipdnnCustomOp::LoadAndCompileGraph() {
	using namespace hipdnn_frontend;
	
	// Load serialized graph from file
	std::string filename = hipdnn_proto_.graph_file_name();
	std::vector<uint8_t> buffer = LoadGraphFromFile(filename);
	
	MY_LOG(1) << "Loaded graph file: " << filename 
	          << " (" << buffer.size() << " bytes)";
	
	// Step 1: Create graph descriptor from buffer
	// (same as Graph::build_operation_graph does internally)
	graphDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
	    buffer.data(), buffer.size());
	
	if (!graphDesc_ || !graphDesc_->valid()) {
		throw std::runtime_error("Failed to create graph descriptor from buffer");
	}
	
	// Set handle on graph descriptor
	hipdnnStatus_t status = hipdnnBackend()->backendSetAttribute(
	    graphDesc_->get(),
	    HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
	    HIPDNN_TYPE_HANDLE,
	    1,
	    &handle_);
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to set handle on graph descriptor");
	}
	
	// Finalize graph descriptor
	status = hipdnnBackend()->backendFinalize(graphDesc_->get());
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to finalize graph descriptor");
	}
	
	// Step 2: Create execution plans (same as Graph::create_execution_plans)
	InitializeHeuristicDescriptor();
	InitializeEngineConfig();
	
	// Step 3: Build execution plan
	executionPlanDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
	    HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
	
	if (!executionPlanDesc_ || !executionPlanDesc_->valid()) {
		throw std::runtime_error("Failed to create execution plan descriptor");
	}
	
	status = hipdnnBackend()->backendSetAttribute(
	    executionPlanDesc_->get(),
	    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
	    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
	    1,
	    &engineConfigDesc_->get());
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to set engine config on execution plan");
	}
	
	status = hipdnnBackend()->backendFinalize(executionPlanDesc_->get());
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to finalize execution plan");
	}
	
	// Step 4: Get workspace size
	int64_t workspace_size = 0;
	status = hipdnnBackend()->backendGetAttribute(
	    executionPlanDesc_->get(),
	    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
	    HIPDNN_TYPE_INT64,
	    1,
	    nullptr,
	    &workspace_size);
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to get workspace size");
	}
	
	if (workspace_size > 0) {
		workspace_.resize(workspace_size);
		MY_LOG(1) << "Allocated workspace: " << workspace_size << " bytes";
	}
	
	// Step 5: Extract UIDs for variant pack mapping
	ExtractUIDsFromSerializedGraph(buffer);
}

void HipdnnCustomOp::InitializeHeuristicDescriptor() {
	using namespace hipdnn_frontend;
	
	engineHeuristicDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
	    HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
	
	hipdnnStatus_t status = hipdnnBackend()->backendSetAttribute(
	    engineHeuristicDesc_->get(),
	    HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
	    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
	    1,
	    &graphDesc_->get());
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to set operation graph on heuristic descriptor");
	}
	
	// Set heuristic mode to FALLBACK
	hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
	status = hipdnnBackend()->backendSetAttribute(
	    engineHeuristicDesc_->get(),
	    HIPDNN_ATTR_ENGINEHEUR_MODE,
	    HIPDNN_TYPE_HEUR_MODE,
	    1,
	    &mode);
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to set mode on heuristic descriptor");
	}
	
	status = hipdnnBackend()->backendFinalize(engineHeuristicDesc_->get());
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to finalize heuristic descriptor");
	}
}

void HipdnnCustomOp::InitializeEngineConfig() {
	using namespace hipdnn_frontend;
	
	// Get number of available engine configurations
	int64_t availableEngineCount = 0;
	hipdnnStatus_t status = hipdnnBackend()->backendGetAttribute(
	    engineHeuristicDesc_->get(),
	    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
	    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
	    0,
	    &availableEngineCount,
	    nullptr);
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to get engine count from heuristic descriptor");
	}
	
	if (availableEngineCount == 0) {
		throw std::runtime_error("No engine configurations available");
	}
	
	// Get the first (best) engine configuration
	auto engineCfgDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
	    HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
	
	if (!engineCfgDesc || !engineCfgDesc->valid()) {
		throw std::runtime_error("Failed to create engine config descriptor");
	}
	
	hipdnnBackendDescriptor_t engineCfgPtr = engineCfgDesc->get();
	int64_t count = 0;
	status = hipdnnBackend()->backendGetAttribute(
	    engineHeuristicDesc_->get(),
	    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
	    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
	    1,
	    &count,
	    &engineCfgPtr);
	
	if (status != HIPDNN_STATUS_SUCCESS || count == 0) {
		throw std::runtime_error("Failed to get engine configuration");
	}
	
	// Finalize engine config
	status = hipdnnBackend()->backendFinalize(engineCfgPtr);
	if (status != HIPDNN_STATUS_SUCCESS) {
		throw std::runtime_error("Failed to finalize engine config");
	}
	
	engineConfigDesc_ = std::move(engineCfgDesc);
	
	MY_LOG(1) << "Selected engine configuration for execution";
}

void HipdnnCustomOp::ExtractUIDsFromSerializedGraph(const std::vector<uint8_t>& buffer) {
	using namespace hipdnn_plugin_sdk;
	
	// Use GraphWrapper to read the serialized graph structure
	GraphWrapper graphWrapper(buffer.data(), buffer.size());
	
	if (!graphWrapper.isValid()) {
		throw std::runtime_error("Invalid serialized graph");
	}
	
	// Get tensor map: UID → TensorAttributes
	auto tensorMap = graphWrapper.getTensorMap();
	
	// Build set of output UIDs by examining node attributes
	std::unordered_set<int64_t> outputUids;
	
	for (uint32_t i = 0; i < graphWrapper.nodeCount(); ++i) {
		auto& node = graphWrapper.getNode(i);
		
		// Check node type and extract output UID
		if (node.attributes_type() == hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes) {
			auto* conv_attrs = node.attributes_as_ConvolutionFwdAttributes();
			if (conv_attrs) {
				outputUids.insert(conv_attrs->y_tensor_uid());
			}
		}
		// Add other node types as needed
	}
	
	// Classify non-virtual tensors as inputs or outputs
	// We need to maintain order, so we'll iterate through UIDs in sorted order
	std::vector<std::pair<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>> sortedTensors;
	for (const auto& [uid, tensor] : tensorMap) {
		if (!tensor->virtual_()) {
			sortedTensors.push_back({uid, tensor});
		}
	}
	
	// Sort by UID to maintain consistent ordering
	std::sort(sortedTensors.begin(), sortedTensors.end(),
	          [](const auto& a, const auto& b) { return a.first < b.first; });
	
	// Now classify as inputs or outputs
	for (const auto& [uid, tensor] : sortedTensors) {
		if (outputUids.count(uid) > 0) {
			// This is a graph output
			output_uids_.push_back(uid);
			
			// Extract shape for output allocation
			auto dims = tensor->dims();
			if (dims) {
				std::vector<int64_t> shape(dims->begin(), dims->end());
				output_shapes_.push_back(shape);
			} else {
				throw std::runtime_error("Output tensor missing dimensions");
			}
		} else {
			// This is a graph input
			input_uids_.push_back(uid);
		}
	}
	
	MY_LOG(1) << "Extracted UIDs: " 
	          << input_uids_.size() << " inputs, "
	          << output_uids_.size() << " outputs";
}

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
	
	MY_LOG(1) << "Executing hipDNN graph: "
	          << "num_inputs " << num_inputs << ", "
	          << "num_outputs " << num_outputs;
	
	// Validate input/output counts
	if (num_inputs != input_uids_.size()) {
		LOG(ERROR) << "Input count mismatch: expected " << input_uids_.size() 
		           << ", got " << num_inputs;
		return;
	}
	
	if (num_outputs != output_uids_.size()) {
		LOG(ERROR) << "Output count mismatch: expected " << output_uids_.size() 
		           << ", got " << num_outputs;
		return;
	}
	
	// Build variant pack: UID → tensor pointer mapping
	std::unordered_map<int64_t, void*> variant_pack;
	
	// Map inputs
	for (size_t i = 0; i < input_uids_.size(); ++i) {
		Ort::ConstValue input = ctx.GetInput(i);
		auto tensor_info = input.GetTensorTypeAndShapeInfo();
		auto shape = tensor_info.GetShape();
		
		MY_LOG(1) << "Input [" << i << "] UID=" << input_uids_[i] 
		          << " shape=" << shape_to_string(shape);
		
		variant_pack[input_uids_[i]] = const_cast<void*>(input.GetTensorRawData());
	}
	
	// Allocate and map outputs
	for (size_t i = 0; i < output_uids_.size(); ++i) {
		Ort::UnownedValue output = ctx.GetOutput(i, output_shapes_[i]);
		
		MY_LOG(1) << "Output [" << i << "] UID=" << output_uids_[i] 
		          << " shape=" << shape_to_string(output_shapes_[i]);
		
		variant_pack[output_uids_[i]] = output.GetTensorMutableRawData();
	}
	
	// Create variant pack descriptor
	using namespace hipdnn_frontend;
	auto variantPackDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
	    HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
	
	if (!variantPackDesc || !variantPackDesc->valid()) {
		LOG(ERROR) << "Failed to create variant pack descriptor";
		return;
	}
	
	// Split variant_pack into keys and values
	std::vector<int64_t> keys;
	std::vector<void*> values;
	keys.reserve(variant_pack.size());
	values.reserve(variant_pack.size());
	
	for (const auto& [key, value] : variant_pack) {
		keys.push_back(key);
		values.push_back(value);
	}
	
	// Set data pointers
	hipdnnStatus_t status = hipdnnBackend()->backendSetAttribute(
	    variantPackDesc->get(),
	    HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
	    HIPDNN_TYPE_VOID_PTR,
	    static_cast<int64_t>(values.size()),
	    values.data());
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		LOG(ERROR) << "Failed to set variant pack data pointers";
		return;
	}
	
	// Set UIDs
	status = hipdnnBackend()->backendSetAttribute(
	    variantPackDesc->get(),
	    HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
	    HIPDNN_TYPE_INT64,
	    static_cast<int64_t>(keys.size()),
	    keys.data());
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		LOG(ERROR) << "Failed to set variant pack UIDs";
		return;
	}
	
	// Set workspace
	void* workspace_ptr = workspace_.empty() ? nullptr : const_cast<void*>(static_cast<const void*>(workspace_.data()));
	status = hipdnnBackend()->backendSetAttribute(
	    variantPackDesc->get(),
	    HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
	    HIPDNN_TYPE_VOID_PTR,
	    1,
	    &workspace_ptr);
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		LOG(ERROR) << "Failed to set variant pack workspace";
		return;
	}
	
	// Finalize variant pack
	status = hipdnnBackend()->backendFinalize(variantPackDesc->get());
	if (status != HIPDNN_STATUS_SUCCESS) {
		LOG(ERROR) << "Failed to finalize variant pack";
		return;
	}
	
	// Execute!
	status = hipdnnBackend()->backendExecute(
	    handle_, executionPlanDesc_->get(), variantPackDesc->get());
	
	if (status != HIPDNN_STATUS_SUCCESS) {
		LOG(ERROR) << "Execution failed";
		return;
	}
	
	MY_LOG(1) << "Execution completed successfully";
}

} // namespace hipdnn
