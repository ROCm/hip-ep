/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <cassert>
#include <iostream>
#include <morphizen-utils/morphizen-utils.hpp>

// Test environment parameters
DEF_ENV_PARAM(TEST_INT_VALUE, "42");
DEF_ENV_PARAM_2(TEST_STRING_VALUE, "hello", std::string);
DEF_ENV_PARAM_2(TEST_BOOL_VALUE, "true", bool);

// Test class for weak patterns
class TestService {
public:
  TestService(int value = 0) : value_(value) {
    std::cout << "TestService created with value: " << value_ << std::endl;
  }

  ~TestService() { std::cout << "TestService destroyed" << std::endl; }

  void initialize() {
    initialized_ = true;
    std::cout << "TestService initialized" << std::endl;
  }

  int get_value() const { return value_; }
  bool is_initialized() const { return initialized_; }

private:
  int value_;
  bool initialized_ = false;
};

void test_env_config() {
  std::cout << "\n=== Testing Environment Configuration ===" << std::endl;

  // Test basic types
  int int_val = ENV_PARAM(TEST_INT_VALUE);
  std::string str_val = ENV_PARAM(TEST_STRING_VALUE);
  bool bool_val = ENV_PARAM(TEST_BOOL_VALUE);

  std::cout << "INT: " << int_val << " (expected: 42)" << std::endl;
  std::cout << "STRING: " << str_val << " (expected: hello)" << std::endl;
  std::cout << "BOOL: " << bool_val << " (expected: 1)" << std::endl;

  assert(int_val == 42);
  assert(str_val == "hello");
  assert(bool_val == true);
}

void test_parse_value() {
  std::cout << "\n=== Testing Parse Value ===" << std::endl;

  // Test hex parsing
  int hex_val;
  morphizen::utils::parse_value("0xFF", hex_val);
  assert(hex_val == 255);
  std::cout << "Hex parsing: 0xFF -> " << hex_val << std::endl;

  // Test boolean parsing
  bool bool_val;
  morphizen::utils::parse_value("yes", bool_val);
  assert(bool_val == true);
  std::cout << "Bool parsing: 'yes' -> " << bool_val << std::endl;

  morphizen::utils::parse_value("no", bool_val);
  assert(bool_val == false);
  std::cout << "Bool parsing: 'no' -> " << bool_val << std::endl;
}

void test_weak_singleton() {
  std::cout << "\n=== Testing Weak Singleton ===" << std::endl;

  // Test singleton creation
  auto service1 = morphizen::utils::WeakSingleton<TestService>::create(100);
  auto service2 = morphizen::utils::WeakSingleton<TestService>::create(200);

  // Should be the same instance
  assert(service1.get() == service2.get());
  assert(service1->get_value() == 100); // First creation wins

  std::cout << "Singleton instances are same: "
            << (service1.get() == service2.get()) << std::endl;
  std::cout << "Value: " << service1->get_value() << std::endl;
}

void test_weak_store() {
  std::cout << "\n=== Testing Weak Store ===" << std::endl;

  // Test store creation and retrieval
  auto conn1 =
      morphizen::utils::WeakStore<std::string, TestService>::create("db1", 300);
  auto conn2 =
      morphizen::utils::WeakStore<std::string, TestService>::create("db2", 400);
  auto conn1_again =
      morphizen::utils::WeakStore<std::string, TestService>::create("db1", 500);

  // conn1 and conn1_again should be the same
  assert(conn1.get() == conn1_again.get());
  assert(conn1->get_value() == 300); // Original value preserved
  assert(conn2->get_value() == 400);
  assert(conn1->is_initialized()); // Should be initialized

  std::cout << "Store test passed - same key returns same instance"
            << std::endl;
  std::cout << "DB1 value: " << conn1->get_value()
            << ", initialized: " << conn1->is_initialized() << std::endl;
  std::cout << "DB2 value: " << conn2->get_value()
            << ", initialized: " << conn2->is_initialized() << std::endl;
}

int main() {
  std::cout << "MorphiZen Utils Test Suite" << std::endl;
  std::cout << "=========================" << std::endl;

  try {
    test_env_config();
    test_parse_value();
    test_weak_singleton();
    test_weak_store();

    std::cout << "\n=== All Tests Passed! ===" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
