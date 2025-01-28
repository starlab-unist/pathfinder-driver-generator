#include "torch_generator.h"

const std::string tensor_method_self_var = "self";

TorchFDG::TorchFDG() : FuzzDriverGenerator<TorchAPI>() {}
std::vector<std::string> TorchFDG::header() const {
  return {
      "#include <stdint.h>",
      "#include <stddef.h>",
      "#include <c10/util/irange.h>",
      "#include <cassert>",
      "#include <cstdlib>",
      "#include <torch/torch.h>",
      "#include \"pathfinder.h\"",
      "#include \"fuzzer_util.h\"\n",

      "using namespace fuzzer_util;\n",

      "extern \"C\" {\n",
  };
}
std::vector<std::string> TorchFDG::setup() const {
  std::vector<std::string> setup_code;

  setup_code.push_back("void PathFinderSetup() {");

  concat(setup_code, "  ", arg_setup_code());
  concat(setup_code, "  ", hard_constraint_code());
  concat(setup_code, "  ", soft_constraint_code());

  setup_code.push_back("}\n");

  return setup_code;
}
std::vector<std::string> TorchFDG::callback() const {
  std::vector<std::string> callback_code;

  callback_code.push_back(
      "int PathFinderTestOneInput(const pathfinder::Input& " +
      callback_input_var + ") {");
  callback_code.push_back("  torch::set_num_threads(1);\n");

  concat(callback_code, "  ", input_pass_condition_code());

  callback_code.push_back("\n  try {");
  concat(callback_code, "    ", arg_initialization_code());
  concat(callback_code, "    ", api_call_code());
  callback_code.push_back("  } catch (c10::Error& e) {");
  callback_code.push_back(
      "    return abort_if_pytorch_internal_assertion_failed(e.what());");
  callback_code.push_back("  } catch (std::exception& e) {");
  callback_code.push_back(
      "    return abort_if_not_expected_exception(e.what());");
  callback_code.push_back("  }\n");
  callback_code.push_back("  return 0;");
  callback_code.push_back("}\n");

  return callback_code;
}
std::vector<std::string> TorchFDG::footer() const {
  return {
      "}  // extern \"C\"\n",
      "int main(int argc, char **argv) {",
      "  pathfinder::parse_arg(argc, argv);",
      "  return pathfinder::driver(PathFinderTestOneInput);",
      "}\n",
  };
}

// std::vector<std::string> TorchAPI::pov_header() const {
//   return {
//       "#include <stdint.h>",
//       "#include <stddef.h>",
//       "#include <c10/util/irange.h>",
//       "#include <cassert>",
//       "#include <cstdlib>",
//       "#include <torch/torch.h>",
//       "#include \"fuzzer_util.h\"\n",

//       "using namespace fuzzer_util;\n",

//   };
// }
// std::vector<std::string> TorchAPI::pov_callback() const {
//   std::vector<std::string> callback_code;

//   callback_code.push_back("int main() {");
//   callback_code.push_back("  torch::set_num_threads(1);\n");

//   callback_code.push_back("\n  try {");
//   concat(callback_code, "    ", arg_initialization_code());
//   concat(callback_code, "    ", pov_api_call_code());
//   callback_code.push_back("  } catch (c10::Error& e) {");
//   callback_code.push_back(
//       "    return abort_if_pytorch_internal_assertion_failed(e.what());");
//   callback_code.push_back("  } catch (std::exception& e) {");
//   callback_code.push_back(
//       "    return abort_if_not_expected_exception(e.what());");
//   callback_code.push_back("  }\n");
//   callback_code.push_back("  return 0;");
//   callback_code.push_back("}\n");

//   return callback_code;
// }

TorchFuncFDG::TorchFuncFDG() : TorchFDG() {}
std::vector<std::string> TorchFuncFDG::api_call_code() const {
  std::string api_call = api->get_name() + "(";
  for (size_t i = 0; i < api->get_symargs().size(); i++) {
    api_call += api->get_symargs()[i]->expr(MODE_DRIVER);
    if (i != api->get_symargs().size() - 1) api_call += comma;
  }
  api_call += ")";

  return {
      "PathFinderExecuteTarget(",
      "  " + api_call + ");",
  };
}

TorchTensorMethodFDG::TorchTensorMethodFDG() : TorchFDG() {}
std::vector<std::string> TorchTensorMethodFDG::api_call_code() const {
  std::string api_call = tensor_method_self_var + "." + api->get_name() + "(";
  for (size_t i = 1; i < api->get_symargs().size(); i++) {
    api_call += api->get_symargs()[i]->expr(MODE_DRIVER);
    if (i != api->get_symargs().size() - 1) api_call += comma;
  }
  api_call += ")";

  return {
      "PathFinderExecuteTarget(",
      "  " + api_call + ");",
  };
}

TorchModuleFDG::TorchModuleFDG() : TorchFDG() {}
std::vector<std::string> TorchModuleFDG::api_call_code() const {
  const TorchModule* module_api = dynamic_cast<const TorchModule*>(api);
  PDG_ASSERT(module_api != nullptr);

  const std::string module_var = "module";

  std::string module_init =
      "auto " + module_var + assign + api->get_name() + "(";
  for (size_t i = 0; i < module_api->get_ctor_symargs().size(); i++) {
    module_init += module_api->get_ctor_symargs()[i]->expr(MODE_DRIVER);
    if (i != module_api->get_ctor_symargs().size() - 1) module_init += comma;
  }
  module_init += ");\n";

  std::string forward_call = module_var + "->forward(";
  for (size_t i = 0; i < module_api->get_forward_symargs().size(); i++) {
    forward_call += module_api->get_forward_symargs()[i]->expr(MODE_DRIVER);
    if (i != module_api->get_forward_symargs().size() - 1)
      forward_call += comma;
  }
  forward_call += ")";

  return {
      "PathFinderExecuteTarget(",
      "  " + module_init,
      "  " + module_var + "->to" +
          bracket(module_api->get_module_dtype()->expr(MODE_DRIVER)) +
          semicolon + newline,
      "  " + forward_call + ");",
  };
}

TorchPOVG::TorchPOVG() : POVGenerator<TorchAPI>() {}
std::vector<std::string> TorchPOVG::header() const {
  return {
      "#include <stdint.h>",
      "#include <stddef.h>",
      "#include <c10/util/irange.h>",
      "#include <cassert>",
      "#include <cstdlib>",
      "#include <torch/torch.h>",
      "#include \"fuzzer_util.h\"\n",

      "using namespace fuzzer_util;\n",
  };
}
std::vector<std::string> TorchPOVG::callback() const {
  std::vector<std::string> callback_code;

  callback_code.push_back("int main() {");
  callback_code.push_back("  torch::set_num_threads(1);\n");

  callback_code.push_back("  try {");
  concat(callback_code, "    ", arg_initialization_code());
  concat(callback_code, "    ", api_call_code());
  callback_code.push_back("  } catch (c10::Error& e) {");
  callback_code.push_back(
      "    return abort_if_pytorch_internal_assertion_failed(e.what());");
  callback_code.push_back("  } catch (std::exception& e) {");
  callback_code.push_back(
      "    return abort_if_not_expected_exception(e.what());");
  callback_code.push_back("  }\n");
  callback_code.push_back("  return 0;");
  callback_code.push_back("}\n");

  return callback_code;
}

TorchFuncPOVG::TorchFuncPOVG() : TorchPOVG() {}
std::vector<std::string> TorchFuncPOVG::api_call_code() const {
  std::string api_call = api->get_name() + "(";
  for (size_t i = 0; i < api->get_symargs().size(); i++) {
    api_call += api->get_symargs()[i]->expr(MODE_POV);
    if (i != api->get_symargs().size() - 1) api_call += comma;
  }
  api_call += ");";

  return {api_call};
}

TorchTensorMethodPOVG::TorchTensorMethodPOVG() : TorchPOVG() {}
std::vector<std::string> TorchTensorMethodPOVG::api_call_code() const {
  std::string api_call = tensor_method_self_var + "." + api->get_name() + "(";
  for (size_t i = 1; i < api->get_symargs().size(); i++) {
    api_call += api->get_symargs()[i]->expr(MODE_POV);
    if (i != api->get_symargs().size() - 1) api_call += comma;
  }
  api_call += ");";

  return {api_call};
}

TorchModulePOVG::TorchModulePOVG() : TorchPOVG() {}
std::vector<std::string> TorchModulePOVG::api_call_code() const {
  const TorchModule* module_api = dynamic_cast<const TorchModule*>(api);
  PDG_ASSERT(module_api != nullptr);

  const std::string module_var = "module";

  std::string module_init =
      "auto " + module_var + assign + api->get_name() + "(";
  for (size_t i = 0; i < module_api->get_ctor_symargs().size(); i++) {
    module_init += module_api->get_ctor_symargs()[i]->expr(MODE_POV);
    if (i != module_api->get_ctor_symargs().size() - 1) module_init += comma;
  }
  module_init += ");\n";

  std::string forward_call = module_var + "->forward(";
  for (size_t i = 0; i < module_api->get_forward_symargs().size(); i++) {
    forward_call += module_api->get_forward_symargs()[i]->expr(MODE_POV);
    if (i != module_api->get_forward_symargs().size() - 1)
      forward_call += comma;
  }
  forward_call += ");";

  return {
      module_init,
      module_var + "->to" +
          bracket(module_api->get_module_dtype()->expr(MODE_POV)) + semicolon +
          newline,
      forward_call,
  };
}
