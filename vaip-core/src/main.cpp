#include "morphizen/onnxruntime_vitisai_ep.hpp"
#include "morphizen/vaip.hpp"
extern "C" {
VAIP_DLL_SPEC
void initialize_onnxruntime_vitisai_ep(
    vaip_core::OrtApiForVaip* api, std::vector<OrtCustomOpDomain*>& ret_domain);
VAIP_DLL_SPEC
void deinitialize_onnxruntime_vitisai_ep();
}

typedef void* voidp;
static struct {
  const char* name;
  void* symbol;
} table[] = {
    {"initialize_onnxruntime_vitisai_ep",
     vaip_core::initialize_onnxruntime_vitisai_ep},
    {"deinitialize_onnxruntime_vitisai_ep",
     deinitialize_onnxruntime_vitisai_ep},
};
extern "C" VAIP_DLL_SPEC int morphizen_main(int argc, char* argv[]) {
  if (argc >= 1) {
    auto cmd = std::string(argv[1]);
    for (int i = 0; i < sizeof(table) / sizeof(table[0]); ++i) {
      if (cmd == table[i].name) {
        std::cout << "table[" << i << "]: " << table[i].symbol << std::endl;
      }
    }
  }
  return 0;
}
