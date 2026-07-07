/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// clang-format off
#include <glog/logging.h>

#include <limits>
#include <regex>
#include <string>
#include <fstream>
#include <unordered_map>
#include <exception>
#include <boost/program_options.hpp>
#include "morphizen/vaip.hpp"
#include "morphizen/env_config.hpp"

namespace po = boost::program_options;

// Validate file path to mitigate path traversal risks
static bool validate_path(const std::string& path, const std::string& param_name) {
  if (path.empty()) {
    return true;  // Empty paths are handled by subsequent checks
  }

  // Check for null bytes (potential for path truncation attacks)
  if (path.find('\0') != std::string::npos) {
    std::cerr << "Error: " << param_name << " contains null byte: " << path << std::endl;
    return false;
  }

  // Warn about suspicious patterns but don't block them (users may legitimately need them)
  if (path.find("..") != std::string::npos) {
    std::cerr << "Warning: " << param_name << " contains '..': " << path << std::endl;
  }

  return true;
}

DEF_ENV_PARAM(IGNORE_CONSTANT, "1")
DEF_ENV_PARAM_2(NODE_FORMAT, "$cxx_name<br>$shape, id=$pattern_id, ty=$type<br>$node_arg_name",std::string)
DEF_ENV_PARAM_2(INPUT_FORMAT, "$cxx_name<br>$shape, id=$pattern_id, ty=$type<br>$node_arg_name",std::string)
DEF_ENV_PARAM_2(CONSTANT_FORMAT, "$cxx_name<br>$shape, id=$pattern_id, ty=$type<br>$node_arg_name",std::string)
DEF_ENV_PARAM(ENABLE_CONSTANT_SHARING, "0")

// clang-format on

namespace {
using namespace morphizen;
struct NodePattern {
public:
  NodePattern(const NodeInput& node_input, const std::string& hint,
              const std::shared_ptr<Pattern> pattern,
              const std::string& cxx_name,
              const std::vector<std::shared_ptr<NodePattern>>& inputs)
      : node_input_(node_input), hint_(hint), pattern_(pattern),
        cxx_name_(cxx_name), inputs_(inputs) {}
  const std::shared_ptr<Pattern> pattern() const { return pattern_; }
  const std::string& node_arg_name() const {
    return node_arg_get_name(*node_input_.node_arg);
  }
  /**
   * Builds a c++ snipets for genreate a pattern using the given hint and
   * inputs.
   *
   * @return The string,  a c++ snipets representation of the pattern.
   */
  const std::string cxx_build_self() const {
    std::ostringstream str;
    if (hint_ == "constant") {
      str << "builder.constant()";
    } else if (hint_ == "input") {
      str << "builder.wildcard()";
    } else {
      CHECK(node_input_.node != nullptr) << "hint_= " << hint_;
      auto& node = *node_input_.node;
      auto pattern_op_type = std::string();
      auto& op_type = node_op_type(node);
      auto& op_domain = node_op_domain(node);
      if (op_domain.empty()) {
        pattern_op_type = op_type;
      } else {
        pattern_op_type = op_domain + ":" + op_type;
      }
      str << "builder.node2(\"" << pattern_op_type << "\"";
      str << ",{";
      auto sep = std::string("");
      for (auto& i : inputs_) {
        str << sep << i->cxx_name_;
        sep = ",";
      }
      str << "})";
    }
    return str.str();
  }

  /**
   * Builds a string representation of the node in a mermaid diagram
   *
   * @param output the output node name, it is used for styling the end node.
   * @return The string representing a mermaid node.
   */
  const std::string
  mmd_build_self(const std::vector<std::string>& outputs) const {
    std::ostringstream str;
    auto indent = "    ";

    if (hint_ == "constant") {
      if (!ENV_PARAM(IGNORE_CONSTANT)) {
        str << indent << cxx_name_ << "{{\"" //
            << node_label(ENV_PARAM(CONSTANT_FORMAT), cxx_name_,
                          pattern_->get_id(),
                          node_input_) //
            << "\"}}" << std::endl;
      }
    } else if (hint_ == "input") {
      str << indent << cxx_name_ << "[\\\"" //
          << node_label(ENV_PARAM(INPUT_FORMAT), cxx_name_, pattern_->get_id(),
                        node_input_)
          << "\"/]" << std::endl;
    } else {
      auto node_arg_name = node_arg_get_name(*node_input_.node_arg);
      auto is_root_pattern = std::find(outputs.begin(), outputs.end(),
                                       node_arg_name) != outputs.end();
      str << indent << cxx_name_ << "[";
      if (!is_root_pattern) {
        str << "\"";
      } else {
        str << "[\"";
      }
      str << node_label(ENV_PARAM(NODE_FORMAT), cxx_name_, pattern_->get_id(),
                        node_input_); //
      if (!is_root_pattern) {
        str << "\"";
      } else {
        str << "\"]";
      }
      str << "]";
      str << std::endl;
      for (auto& ni : inputs_) {
        auto draw_it = true;
        if (ENV_PARAM(IGNORE_CONSTANT)) {
          if (ni->hint_ == "constant") {
            draw_it = false;
          }
        }
        if (draw_it) {
          str << indent << ni->cxx_name_
              << (is_root_pattern ? " --o " : " -.-> ") << cxx_name_
              << std::endl;
        }
      }
    }
    maybe_class_defs(hint_, cxx_name_, str);
    return str.str();
  }

  /**
   * Generates a node label for a mermaid diagram based on the provided format
   * and input parameters.
   *
   * @param format The format string used to generate the node label.
   * @param cxx_name The C++ name used in the node label.
   * @param pattern_id The pattern ID used in the node label.
   * @param ni The NodeInput object used in the node label.
   * @return The generated node label.
   */
  std::string node_label(const std::string format, const std::string& cxx_name,
                         int pattern_id, const NodeInput& ni) const {
    auto ret = format;
    auto str_pattern_id = std::to_string(pattern_id);
    auto str_node_arg = node_arg_as_string(*ni.node_arg);
    auto shape_ptr = node_arg_get_shape_i64(*ni.node_arg);
    auto shape = shape_ptr == nullptr
                     ? std::string("N/A")
                     : morphizen::container_as_string(*shape_ptr);
    auto type = std::to_string(node_arg_get_element_type(*ni.node_arg));
    auto node_arg_name = node_arg_get_name(*ni.node_arg);
    auto op_type =
        ni.node == nullptr ? std::string("N/A") : node_op_type(*ni.node);
    auto op_domain =
        ni.node == nullptr ? std::string("N/A") : node_op_domain(*ni.node);
    ret = std::regex_replace(ret, std::regex("\\$pattern_id"), str_pattern_id);
    ret = std::regex_replace(ret, std::regex("\\$cxx_name"), cxx_name);
    ret =
        std::regex_replace(ret, std::regex("\\$node_arg_name"), node_arg_name);
    ret = std::regex_replace(ret, std::regex("\\$node_arg"), str_node_arg);
    ret = std::regex_replace(ret, std::regex("\\$shape"), shape);
    ret = std::regex_replace(ret, std::regex("\\$type"), type);
    ret = std::regex_replace(ret, std::regex("\\$op_type"), op_type);
    ret = std::regex_replace(ret, std::regex("\\$domain"), op_domain);
    return ret;
  }

  /**
   * @brief For generating mermaid diagram, checks if a given hint exists in the
   * class_defs map and generates class definitions accordingly.
   *
   * @param hint The hint string to check in the class_defs map.
   * @param cxx_name The name of the C++ class.
   * @param str The output stream to write the generated class definitions.
   */
  void maybe_class_defs(const std::string& hint, const std::string& cxx_name,
                        std::ostream& str) const {
    static const std::unordered_map<std::string, std::string> class_defs = {
        // Light Blue Fill with Dark Blue Stroke
        {"input", "fill:#add8e6,stroke:#00008b,stroke-width:2px;"},
        // Light Pink Fill with Dark Red Stroke
        {"Div", "fill:#ffb6c1,stroke:#8b0000,stroke-width:2px;"},
        // Light Green Fill with Dark Green Stroke (repeated for demonstration)
        {"Mul", "fill:#90ee90,stroke:#006400,stroke-width:2px;"},
        // Light Grey Fill with Charcoal Stroke (repeated for demonstration)
        {"ReLU", "fill:#d3d3d3,stroke:#36454f,stroke-width:2px;"},
        // Light Yellow Fill with Dark Orange Stroke (repeated for
        // demonstration)
        {"Concat", "fill:#ffffe0,stroke:#ff8c00,stroke-width:2px;"},
        // Light Blue Fill with Dark Blue Stroke (repeated for demonstration)
        {"MaxPool", "fill:#add8e6,stroke:#00008b,stroke-width:2px;"},
        // Light Coral Fill with Dark Red Stroke (repeated for demonstration)
        {"Conv", "fill:#f08080,stroke:#8b0000,stroke-width:2px;"},
        // Light Cyan Fill with Teal Stroke (repeated for demonstration)
        {"Gemm", "fill:#e0ffff,stroke:#008080,stroke-width:2px;"},
        // Beige Fill with Brown Stroke (repeated for demonstration)
        {"BatchNormalization", "fill:#f5f5dc,stroke:#8b4513,stroke-width:2px;"},
        // Lavender Fill with Dark Violet Stroke (repeated for demonstration)
        {"Softmax", "fill:#e6e6fa,stroke:#9400d3,stroke-width:2px;"},
        // Peach Fill with Dark Brown Stroke (repeated for demonstration)
        {"Flatten", "fill:#ffdab9,stroke:#8b4513,stroke-width:2px;"},
        // Light Pink Fill with Dark Red Stroke (repeated for demonstration)
        {"Div", "fill:#ffb6c1,stroke:#8b0000,stroke-width:2px;"},

        // Additional ONNX common operations with different style strings
        // Light Salmon Fill with Dark Red Stroke
        {"Add", "fill:#ffa07a,stroke:#8b0000,stroke-width:2px;"},
        // Light Goldenrod Yellow Fill with Dark Goldenrod Stroke
        {"Sub", "fill:#fafad2,stroke:#b8860b,stroke-width:2px;"},
        // Light Steel Blue Fill with Dark Slate Blue Stroke
        {"AveragePool", "fill:#b0c4de,stroke:#483d8b,stroke-width:2px;"},
        // Light Sea Fill with Dark Green Stroke
        {"GlobalAveragePool", "fill:#20b2aa,stroke:#006400,stroke-width:2px;"},
        // Light Sky Blue Fill with Dark Blue Stroke
        {"GlobalMaxPool", "fill:#87cefa,stroke:#00008b,stroke-width:2px;"},
        // Light Slate Gray Fill with Dark Slate Gray Stroke
        {"Dropout", "fill:#778899,stroke:#2f4f4f,stroke-width:2px;"},
        // Light Yellow Fill with Dark Orange Stroke
        {"LRN", "fill:#ffffe0,stroke:#ff8c00,stroke-width:2px;"},
        // Light Coral Fill with Dark Red Stroke
        {"PRelu", "fill:#ffb6c1,stroke:#8b0000,stroke-width:2px;"},
        {"Relu", "fill:#ffb6c1,stroke:#8b0000,stroke-width:2px;"},
        // Light Cyan Fill with Teal Stroke
        {"LeakyRelu", "fill:#e0ffff,stroke:#008080,stroke-width:2px;"},
        // Light Pink Fill with Dark Red Stroke
        {"Sigmoid", "fill:#ffb6c1,stroke:#8b0000,stroke-width:2px;"},
        // Light Green Fill with Dark Green Stroke
        {"Tanh", "fill:#90ee90,stroke:#006400,stroke-width:2px;"},
        // Light Grey Fill with Charcoal Stroke
        {"HardSigmoid", "fill:#d3d3d3,stroke:#36454f,stroke-width:2px;"},
        // Light Yellow Fill with Dark Orange Stroke
        {"HardSwish", "fill:#ffffe0,stroke:#ff8c00,stroke-width:2px;"},
        // Light Blue Fill with Dark Blue Stroke
        {"Elu", "fill:#add8e6,stroke:#00008b,stroke-width:2px;"},
        // Light Coral Fill with Dark Red Stroke
        {"Selu", "fill:#f08080,stroke:#8b0000,stroke-width:2px;"},
        // Light Cyan Fill with Teal Stroke
        {"ThresholdedRelu", "fill:#e0ffff,stroke:#008080,stroke-width:2px;"},
        // Beige Fill with Brown Stroke
        {"Softplus", "fill:#f5f5dc,stroke:#8b4513,stroke-width:2px;"},
        // Lavender Fill with Dark Violet Stroke
        {"Softsign", "fill:#e6e6fa,stroke:#9400d3,stroke-width:2px;"},
        // Peach Fill with Dark Brown Stroke
        {"LogSoftmax", "fill:#ffdab9,stroke:#8b4513,stroke-width:2px;"},
        // Light Salmon Fill with Dark Red Stroke
        {"QuantizeLinear", "fill:#ffa07a,stroke:#8b0000,stroke-width:2px;"},
        {"com_microsoft_QuantizeLinear",
         "fill:#ffa07a,stroke:#8b0000,stroke-width:2px;"},
        // Light Goldenrod Yellow Fill with Dark Goldenrod Stroke
        {"DequantizeLinear", "fill:#fafad2,stroke:#b8860b,stroke-width:2px;"},
        {"com_microsoft_DequantizeLinear",
         "fill:#fafad2,stroke:#b8860b,stroke-width:2px;"}};

    static std::unordered_map<std::string, int> class_defs_used;
    auto it = class_defs.find(hint);
    if (it == class_defs.end()) {
      return;
    }
    if (class_defs_used[hint] == 0) {
      class_defs_used[hint] = class_defs_used[hint] + 1;
      str << "    "
          << "classDef " << hint << " " << it->second << ";" << std::endl;
    }
    str << "    "
        << "class " << cxx_name << " " << hint << ";" << std::endl;
    return;
  }

public:
  const NodeInput node_input_;
  const std::string hint_;
  const std::shared_ptr<Pattern> pattern_;
  const std::string cxx_name_;
  const std::vector<std::shared_ptr<NodePattern>> inputs_;
};

struct NodePatternBuiler {
public:
  /**
   * Builds a collection of node patterns from a vector of nodes.
   *
   * @param nodes A vector of pointers to nodes.
   * @return A shared pointer to the last node pattern in the collection.
   */
  std::shared_ptr<NodePattern>
  build_nodes(std::vector<morphizen_cxx::NodeConstRef> nodes) {
    for (auto node : nodes) {
      build_node(node);
    }
    return vector_node_patterns_.back();
  }
  /**
   * Builds a node pattern based on the given node.
   *
   * @param node The node to build the pattern for.
   * @return A shared pointer to the built node pattern.
   */
  std::shared_ptr<NodePattern> build_node(const Node& node) {
    auto hint = get_hint(node);
    auto cxx_name = new_unique_name(hint);
    auto node_inputs = node_get_inputs(node);
    auto& node_arg = node_get_first_output_node_arg(node);
    auto ni_name = node_arg_get_name(node_arg);
    auto pattern_args = std::vector<std::shared_ptr<Pattern>>{};
    auto node_pattern_inputs = std::vector<std::shared_ptr<NodePattern>>{};
    for (auto ni : node_inputs) {
      auto ni_name_1 = node_arg_get_name(*ni.node_arg);
      auto it = map_node_patterns_.find(ni_name_1);
      auto arg_node_pattern = std::shared_ptr<NodePattern>();
      if (it != map_node_patterns_.end()) {
        if (ni.node == nullptr) {
          if (ENV_PARAM(ENABLE_CONSTANT_SHARING)) {
            arg_node_pattern = it->second;
          } else {
            arg_node_pattern = build_constant_pattern(*ni.node_arg);
          }
        } else {
          arg_node_pattern = it->second;
        }
      } else {
        if (ni.node == nullptr) {
          arg_node_pattern = build_constant_pattern(*ni.node_arg);
        } else {
          arg_node_pattern = build_input_pattern(*ni.node_arg);
        }
      }
      pattern_args.push_back(arg_node_pattern->pattern());
      node_pattern_inputs.push_back(arg_node_pattern);
    }
    auto op_type = node_op_type(node);
    auto op_domain = node_op_domain(node);
    auto pattern_op_type = std::string();
    if (node_op_domain(node).empty()) {
      pattern_op_type = op_type;
    } else {
      pattern_op_type = op_domain + ":" + op_type;
    }
    auto pattern = builder_.node2(pattern_op_type, pattern_args);
    auto& node_arg_name = node_arg_get_name(node_arg);
    builder_.bind(node_arg_name, pattern);
    return push_back_build_node_pattern({&node, &node_arg}, hint, pattern,
                                        cxx_name, node_pattern_inputs);
  }
  /**
   * Builds a constant pattern for the given NodeArg.
   *
   * @param node_arg The NodeArg for which the constant pattern is built.
   * @return A shared pointer to the built NodePattern.
   */
  std::shared_ptr<NodePattern> build_constant_pattern(const NodeArg& node_arg) {
    auto& node_arg_name = node_arg_get_name(node_arg);
    LOG(INFO) << "create constant for " << node_arg_name;
    if (ENV_PARAM(ENABLE_CONSTANT_SHARING)) {
      auto it = map_node_patterns_.find(node_arg_name);
      if (it != map_node_patterns_.end()) {
        LOG(INFO) << "re-use constant " << it->second->pattern()->get_id();
        return it->second;
      }
    }
    auto pattern = builder_.constant();
    builder_.bind(node_arg_name, pattern);
    auto hint = std::string("constant");
    auto cxx_name = new_unique_name(hint);
    LOG(INFO) << "create a new constant " << pattern->get_id();
    return push_back_build_node_pattern(
        {nullptr, &node_arg}, hint, pattern, cxx_name,
        std::vector<std::shared_ptr<NodePattern>>{});
  }

  /**
   * Builds a node pattern for the given input node argument.
   *
   * @param node_arg The input node argument.
   * @return A shared pointer to the built node pattern.
   */
  std::shared_ptr<NodePattern> build_input_pattern(const NodeArg& node_arg) {
    auto& node_arg_name = node_arg_get_name(node_arg);
    auto it = map_node_patterns_.find(node_arg_name);
    if (it != map_node_patterns_.end()) {
      return it->second;
    }
    auto pattern = builder_.wildcard();
    builder_.bind(node_arg_name, pattern);
    auto hint = std::string("input");
    auto cxx_name = new_unique_name(hint);
    return push_back_build_node_pattern(
        {nullptr, &node_arg}, hint, pattern, cxx_name,
        std::vector<std::shared_ptr<NodePattern>>{});
  }

  /**
   * @brief Adds a new node pattern to the vector_node_patterns_ and
   * map_node_patterns_.
   *
   * This function takes in the node_input, hint, pattern, cxx_name, and inputs
   * parameters and creates a new NodePattern object using these parameters. The
   * newly created NodePattern object is then added to the vector_node_patterns_
   * and map_node_patterns_ containers.
   *
   * @param node_input The NodeInput object.
   * @param hint The hint string.
   * @param pattern The shared pointer to the Pattern object.
   * @param cxx_name The cxx_name string.
   * @param inputs The vector of shared pointers to NodePattern objects.
   */
  std::shared_ptr<NodePattern> push_back_build_node_pattern(
      const NodeInput& node_input, const std::string& hint,
      const std::shared_ptr<Pattern> pattern, const std::string& cxx_name,
      const std::vector<std::shared_ptr<NodePattern>>& inputs) {
    auto& node_arg_name = node_arg_get_name(*node_input.node_arg);
    auto ret = std::make_shared<NodePattern>(node_input, hint, pattern,
                                             cxx_name, inputs);
    vector_node_patterns_.push_back(ret);
    map_node_patterns_[node_arg_name] = ret;
    return ret;
  }
  /**
   * Returns a hint for the given node.
   *
   * The hint is generated based on the node's operation type and domain.
   * If the node's domain is empty, the hint is simply the operation type.
   * Otherwise, the hint is a combination of the domain and operation type,
   * separated by an underscore. hint is used to generate c++ variable name and
   * mmd node id
   *
   * @param node The node for which to generate the hint.
   * @return The hint for the given node.
   */
  std::string get_hint(const Node& node) {
    auto to_cxx_id = [](const std::string& v_name) -> std::string {
      auto name = v_name;
      std::transform(
          name.begin(), name.end(), name.begin(),
          [](std::string::value_type c) { return std::isalnum(c) ? c : '_'; });
      return name;
    };
    auto op_type = node_op_type(node);
    if (node_op_domain(node).empty()) {
      return op_type;
    } else {
      return to_cxx_id(node_op_domain(node)) + "_" + op_type;
    }
  }

  /**
   * Generates a new unique name based on the given hint.
   *
   * @param hint The hint for the new unique name.
   * @return The new unique name.
   */
  std::string new_unique_name(const std::string& hint) {
    auto c = name_counter_[hint];
    auto ret = hint + "_" + std::to_string(c);
    name_counter_[hint] = c + 1;
    return ret;
  };
  /**
   * @brief Tests the pattern matching algorithm on a given graph.
   *
   * This function takes a graph and an output node as input and performs
   * pattern matching on the graph using the root pattern obtained from the
   * output node. It prints the matching pattern and returns the result of the
   * matching process.
   *
   * @param graph The graph to perform pattern matching on.
   * @param output The output node of the graph to start pattern matching from.
   */
  std::shared_ptr<morphizen::Pattern>
  test(morphizen_cxx::GraphConstRef graph,
       const std::vector<std::string>& outputs) {
    auto patterns = std::vector<std::shared_ptr<morphizen::Pattern>>{};
    patterns.reserve(outputs.size());
    auto first_node = std::optional<morphizen_cxx::NodeConstRef>();
    for (auto output : outputs) {
      auto node = graph.find_node(output);
      CHECK(node != std::nullopt)
          << "- Error: Cannot find the root pattern "
             "-------------------------------------------"
          << std::endl;
      if (first_node == std::nullopt) {
        first_node = node;
      }
      // Get root pattern
      auto root_pattern = builder_.get_pattern(output);
      CHECK(root_pattern != nullptr)
          << "- Error: Cannot find the root pattern "
             "-------------------------------------------"
          << std::endl;

      std::cout << "- Matching pattern "
                   "---------------------------------------------------"
                << std::endl;
      morphizen::Pattern::enable_trace(1);
      auto match = root_pattern->match(graph, *node);
      patterns.push_back(root_pattern);
      CHECK(match != nullptr)
          << "- Error: Cannot match the root pattern " << std::endl;
      std::cout << "---------------------------------------------------"
                << std::endl;
      std::cout << "- Pattern matched ?: " << (match ? "true" : "false") << "\n"
                << "- Pattern: " << root_pattern->debug_string() << "\n"
                << "- Node: " << *node << std::endl;
      std::cout << "---------------------------------------------------"
                << std::endl;
    }
    auto ret = std::shared_ptr<morphizen::Pattern>();
    if (patterns.size() > 1) {
      std::cout << "- Test Multiple output patterns: " << std::endl;
      auto seq = builder_.sequence(patterns);
      auto match = seq->match(graph, *first_node);
      CHECK(match != nullptr)
          << "- Error: Cannot match the sequence pattern " << std::endl;
      ret = seq;
    } else {
      ret = patterns[0];
    }
    return ret;
  }
  /**
   * Generates C++ code based on the provided inputs and writes it to a file.
   *
   * @param opt_inputs A vector of strings representing the optional inputs.
   * @param opt_output A string representing the optional output.
   * @param opt_onnx_file A string representing the optional ONNX file.
   * @param opt_cxx_file A string representing the optional C++ file to write
   * the generated code to.
   */
  void generate_cxx_code(const std::vector<std::string>& opt_inputs,
                         const std::vector<std::string>& opt_outputs,
                         const std::string opt_onnx_file,
                         const std::string& opt_cxx_file) const {
    std::ostringstream cxx_src_stream;
    cxx_src_stream                                                           //
        << "/** generated by the following command:\n"
        << "env \\\n"                                                        //
        << " IGNORE_CONSTANT=" << ENV_PARAM(IGNORE_CONSTANT) << " \\\n"      //
        << " ENABLE_CONSTNAT_SHARING=" << ENV_PARAM(ENABLE_CONSTANT_SHARING) //
        << " \\\n"                                                           //
        << " $BUILD/morphizen/onnxruntime_morphizen_ep/onnx_pattern_gen \\\n";
    for (auto& input : opt_inputs) {
      cxx_src_stream << " -i " << input << " \\\n";
    }
    for (auto& output : opt_outputs) {
      cxx_src_stream << " -o " << output << " \\\n";
    }
    cxx_src_stream << " -f " << opt_onnx_file << "\\\n";
    cxx_src_stream << " -c " << opt_cxx_file << "\n";
    cxx_src_stream << "*/\n";
    for (auto np : vector_node_patterns_) {
      cxx_src_stream << "auto " << np->cxx_name_ << " = "
                     << np->cxx_build_self() << "; // "
                     << " id = " << np->pattern_->get_id() << " "
                     << " node_arg_name = " << np->node_arg_name() << "\n"
                     << "builder.bind(\"" << np->node_arg_name() << "\","
                     << np->cxx_name_ << ");\n";
    }
    if (opt_outputs.size() == 1) {
      cxx_src_stream << "ret = " << vector_node_patterns_.back()->cxx_name_
                     << ";" << std::endl;
    } else {
      cxx_src_stream
          << "ret = "
             "builder.sequence(std::vector<std::shared_ptr<morphizen_"
             "core::Pattern>>{";
      for (auto output : opt_outputs) {
        cxx_src_stream << map_node_patterns_.at(output)->cxx_name_ << ",";
      }
      cxx_src_stream << "});";
    }

    auto code = cxx_src_stream.str();
    // dump the file
    auto inc = opt_cxx_file;
    CHECK(std::ofstream(inc).write(code.data(), code.size()).good())
        << " failed to write to " << inc;
    LOG(INFO) << "write generated c++ code to " << inc;
  }

  /**
   * Generates a Mermaid code for a flowchart based on the given input
   * parameters and writes it to a file.
   *
   * @param opt_inputs A vector of strings representing the optional inputs.
   * @param opt_output A string representing the optional output.
   * @param opt_onnx_file A string representing the optional ONNX file.
   * @param opt_mmd_file A string representing the path to the output Mermaid
   * file.
   */
  void generate_mmd_code(const std::vector<std::string>& /*opt_inputs*/,
                         const std::vector<std::string>& opt_outputs,
                         const std::string /*opt_onnx_file*/,
                         const std::string& opt_mmd_file) const {
    std::ostringstream mmd_stream;
    mmd_stream << "flowchart TB" << std::endl;
    for (auto np : vector_node_patterns_) {
      mmd_stream << np->mmd_build_self(opt_outputs);
    }
    auto code = mmd_stream.str();
    // dump the file
    auto inc = opt_mmd_file;
    CHECK(std::ofstream(inc).write(code.data(), code.size()).good())
        << " failed to write to " << inc;
    LOG(INFO) << "write generated mermaid diagram to " << inc;
  }

  void dump_subgraph_to_onnx(morphizen_cxx::GraphRef& graph_ref,
                             const std::vector<std::string>& opt_inputs,
                             const std::vector<std::string>& opt_outputs,
                             const std::string& out_onnx) {
    graph_ref.resolve();
    auto [meta_def, error] = graph_ref.try_fuse("onnx_pattern_gen", opt_inputs,
                                                opt_outputs, {}, "no_device");
    if (meta_def == nullptr) {
      LOG(FATAL) << "try_fuse error: " << error.comments;
    }
    auto node = graph_ref.fuse(*meta_def);
    auto subgraph = node.get_function_body();
    std::filesystem::path onnx_path = out_onnx;

    subgraph.save(onnx_path);

    LOG(INFO) << "Subgraph dumped to: " << out_onnx;
  }

private:
  PatternBuilder builder_;
  std::unordered_map<std::string, int> name_counter_;
  std::unordered_map<std::string, std::shared_ptr<NodePattern>>
      map_node_patterns_;
  std::vector<std::shared_ptr<NodePattern>> vector_node_patterns_;
};
} // namespace

/**
 * Retrieves a vector of nodes from the given graph based on the specified
 * criteria.
 *
 * @param graph The graph to search for nodes.
 * @param to_ops A vector of strings representing the names of the target nodes
 * to search for. If empty, all output nodes in the graph will be considered as
 * target nodes.
 * @param from_op_names A vector of strings representing the names of the source
 * nodes to search from.
 * @return A vector of const Node pointers representing the nodes that satisfy
 * the search criteria.
 */
template <typename T>
std::vector<morphizen_cxx::NodeConstRef>
get_nodes(morphizen_cxx::GraphConstRef graph, const T& node_output_names) {
  std::vector<morphizen_cxx::NodeConstRef> nodes;
  nodes.reserve(node_output_names.size());
  for (auto it = node_output_names.rbegin(); it != node_output_names.rend();
       ++it) {
    const auto& name = *it;
    auto node = graph.find_node(name);
    if (node == std::nullopt) {
      LOG(FATAL) << "cannot find node:" << name;
    }
    nodes.push_back(*node);
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const auto& a, const auto& b) { return a.index() < b.index(); });
  return nodes;
}

// example usage:

// clang-format off
// $BUILD/morphizen/onnxruntime_morphizen_ep/onnx_pattern_gen -i 38 -o 62 -f $BUILD/../morphizen_regression/5/Resnet18_int.onnx
// clang-format on
int main(int argc, char* argv[]) {
  try {
    // Define command line options
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "input,i", po::value<std::vector<std::string>>()->multitoken(),
        "input of an onnx model (can be specified multiple times)")(
        "output,o", po::value<std::vector<std::string>>()->multitoken(),
        "output of an onnx model (can be specified multiple times)")(
        "file,f", po::value<std::string>(), "file of an onnx model")(
        "cxx,c", po::value<std::string>(), "C++ output file")(
        "mermaid,m", po::value<std::string>(), "mermaid output file")(
        "json,j", po::value<std::string>(), "json output file")(
        "onnx,x", po::value<std::string>(), "onnx output file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Handle help option
    if (vm.count("help")) {
      std::cout << "Usage: " << argv[0] << " [options]\n" << desc << std::endl;
      return 0;
    }

    // Extract option values
    auto opt_onnx_file =
        vm.count("file") ? vm["file"].as<std::string>() : std::string();
    auto opt_cxx_output_file =
        vm.count("cxx") ? vm["cxx"].as<std::string>() : std::string("");
    auto opt_mmd_output_file =
        vm.count("mermaid") ? vm["mermaid"].as<std::string>() : std::string("");
    auto opt_json_output_file =
        vm.count("json") ? vm["json"].as<std::string>() : std::string("");
    auto opt_onnx_output_file =
        vm.count("onnx") ? vm["onnx"].as<std::string>() : std::string("");
    auto opt_inputs = vm.count("input")
                          ? vm["input"].as<std::vector<std::string>>()
                          : std::vector<std::string>{};
    auto opt_outputs = vm.count("output")
                           ? vm["output"].as<std::vector<std::string>>()
                           : std::vector<std::string>{};

    // Validate all paths at entry point
    if (!validate_path(opt_onnx_file, "file")) {
      return 1;
    }
    if (!validate_path(opt_cxx_output_file, "cxx")) {
      return 1;
    }
    if (!validate_path(opt_mmd_output_file, "mermaid")) {
      return 1;
    }
    if (!validate_path(opt_json_output_file, "json")) {
      return 1;
    }
    if (!validate_path(opt_onnx_output_file, "onnx")) {
      return 1;
    }

    // Check command line args
    CHECK_NE(opt_onnx_file, "")
        << " -f <model.onnx> is required. " << opt_onnx_file;
    CHECK_NE(opt_inputs.size(), 0) << " -i <input> is required";
    CHECK_GE(opt_outputs.size(), 1) << " -o <output> is required";

    // Load model
    auto model = morphizen_cxx::Model::load(opt_onnx_file);
    auto graph = model->main_graph();
    graph.resolve();
    auto [meta_def, error] = graph.try_fuse("onnx_pattern_gen", opt_inputs,
                                            opt_outputs, {}, "no_device");

    if (meta_def == nullptr) {
      LOG(FATAL) << "try_fuse error: " << error.comments;
    }

    /// get all nodes in the subgraph.
    auto nodes = get_nodes(graph, meta_def->nodes());
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "- Found subgraph as below: "
                 "-----------------------------------------"
              << std::endl;
    std::cout << meta_def->DebugString() << std::endl;
    std::cout << std::endl;
    std::cout << "- Generate pattern "
                 "-----------------------------------------"
              << std::endl;
    auto builder = NodePatternBuiler();
    auto node_pattern = builder.build_nodes(nodes);
    auto root_pattern = builder.test(graph, opt_outputs);
    if (opt_cxx_output_file != "") {
      builder.generate_cxx_code(opt_inputs, opt_outputs, opt_onnx_file,
                                opt_cxx_output_file);
      std::cout << " generate c++ code: " << opt_cxx_output_file << std::endl;
    }
    if (opt_json_output_file != "") {
      // node_patern->(opt_json_output_file);
      std::ofstream(opt_json_output_file) << root_pattern->to_json();
      std::cout << " generate json code: " << opt_json_output_file << std::endl;
    }
    if (opt_mmd_output_file != "") {
      builder.generate_mmd_code(opt_inputs, opt_outputs, opt_onnx_file,
                                opt_mmd_output_file);
      std::cout << " generate mermaid diagram: " << opt_mmd_output_file
                << std::endl;
    }
    if (opt_onnx_output_file != "") {
      builder.dump_subgraph_to_onnx(graph, opt_inputs, opt_outputs,
                                    opt_onnx_output_file);
      std::cout << " generate spilt onnx model: " << opt_onnx_output_file
                << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }

  return 0;
}
