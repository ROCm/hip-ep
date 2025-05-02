/** Copyright(C) 2023 – 2024 Advanced Micro Devices,
    Inc.All rights reserved.*Licensed under the MIT License.*/
#include "morphizen/vaip.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
namespace {

class UnitTestOps : public OpRegister {
public:
  UnitTestOps(void* state, add_op_t add_op) : OpRegister(state, add_op) {}
  int register_ops() override {
    // Register your ops here
    auto op_name = std::string("add");
    AddOp("com.test.unit",
          std::make_unique<vaip_core::XilinxCustomOp>(op_name));
    return 0;
  }
};
static void register_ops(void* state, add_op_t add_op) {
  UnitTestOps(state, add_op).register_ops();
}

static ::vaip_core::StaticPluginRegister
    my_register("UnitTestOpDef", "morphizen_register_ops", (void*)register_ops);
} // namespace

// TEST(OpDefTest, TestAddAndRemove) {
//   {
//     auto unit_test_op_add =
//         vaip_core::Plugin::invoke<OrtCustomOp*, const char*, const char*>(
//             "onnxruntime_vitisai_ep", "morphizen_get_registered_custom_op",
//             "com.test.unit", "add");
//     ASSERT_TRUE(unit_test_op_add != nullptr);
//   }
// }
// todo long filename test
