/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- Error.h - How the compile-time tools refuse ---------------------===//
//
// Every refusal in these tools is a statement about the input: this graph uses
// an operation with no typed semantics, this value names a storage its
// operands do not reach, this file is not there. So there is one way to say
// it, and the message is built where the facts are rather than assembled by
// the caller.
//
// The runtime refuses differently -- it returns a `Status` so that a failure
// crossing the C ABI is a value rather than an exception -- and does not use
// this.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_FRONTEND_GGML_ERROR_H
#define HIP_FRONTEND_GGML_ERROR_H

#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace hip::ggml {

/// An input the tools cannot accept. Catching this rather than
/// `std::runtime_error` distinguishes a refusal from a defect.
class Error : public std::runtime_error {
public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};

namespace detail {

inline void appendTo(std::string &message, std::string_view part) {
  message.append(part);
}
inline void appendTo(std::string &message, const char *part) {
  message.append(part);
}
inline void appendTo(std::string &message, const std::string &part) {
  message.append(part);
}
inline void appendTo(std::string &message, bool part) {
  message.append(part ? "true" : "false");
}
inline void appendTo(std::string &message, char part) {
  message.push_back(part);
}
template <typename Number, std::enable_if_t<std::is_arithmetic_v<Number> &&
                                                !std::is_same_v<Number, bool> &&
                                                !std::is_same_v<Number, char>,
                                            int> = 0>
void appendTo(std::string &message, Number part) {
  message.append(std::to_string(part));
}

} // namespace detail

/// Builds the message from `parts` and throws it.
///
/// Numbers spell themselves, so a message reads as the sentence it is instead
/// of as a chain of conversions:
///
///     fail("node ", id, " fills GGML slot ", slot, " from operand ", index);
template <typename... Parts> [[noreturn]] void fail(const Parts &...parts) {
  std::string message;
  (detail::appendTo(message, parts), ...);
  throw Error(message);
}

} // namespace hip::ggml

#endif // HIP_FRONTEND_GGML_ERROR_H
