#include "vaiml/vaiml_logging.hpp"

namespace vaip_vaiml {
/**
 * @brief Convert string vector to string for print
 */
std::string stringVectorToString(std::vector<std::string>& string_vec) {
  std::string tmpStr;

  for (auto i = 0; i < string_vec.size(); i++) {
    if (i == string_vec.size() - 1) {
      tmpStr += " " + string_vec[i];
    } else {
      tmpStr += " " + string_vec[i] + ",";
    }
  }
  return tmpStr;
}

/**
 * @brief Format a number in 000,000 format for easier reading
 * The code is to make sure that it works even the locale is not set correctly
 * on the running machine.
 */
std::string formatNumberWithCommas(int64_t number) {
  std::string num_str = std::to_string(number);
  int insert_position = (int)num_str.length() - 3;
  while (insert_position > 0) {
    num_str.insert(insert_position, ",");
    insert_position -= 3;
  }
  return num_str;
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

std::string shapeToString(std::vector<int64_t> shape) {
  std::string tmpStr;
  for (size_t i = 0; i < shape.size(); i++) {
    tmpStr += std::to_string(shape[i]) + " ";
  }
  return tmpStr;
}
std::vector<int64_t> shapeFromString(const std::string& shape_str,
                                     char delimiter) {
  std::vector<int64_t> shape;
  auto shape_str_vec = splitString(shape_str, delimiter);
  for (auto elem : shape_str_vec) {
    shape.push_back(std::stoi(elem));
  }
  return shape;
}

} // namespace vaip_vaiml