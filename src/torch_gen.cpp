#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>

#include "options.h"
#include "torch_api.h"
#include "torch_generator.h"
#include "utils.h"
#include "writer.h"
namespace fs = std::filesystem;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

bool is_unknown_type(const json& param_type) {
  return param_type["type_kind"] == "Unknown";
}

std::unique_ptr<TorchSymArg> gen_torch_symarg(const std::string& param_name,
                                              const json& param_type) {
  std::string param_type_kind = param_type["type_kind"];

  std::unique_ptr<TorchSymArg> symarg;
  if (param_type_kind == "Enum") {
    std::string enum_name = param_type["enum_name"];
    symarg = std::make_unique<TorchEnumSymArg>(param_name, enum_name);
  } else if (param_type_kind == "Int") {
    std::string specifier = param_type["specifier"];
    std::unique_ptr<TorchIntSymArg> int_symarg =
        std::make_unique<TorchIntSymArg>(param_name, specifier);
    if (param_type.contains("default_value"))
      int_symarg->set_default(param_type["default_value"]);
    symarg = std::move(int_symarg);
  } else if (param_type_kind == "SymInt") {
    std::unique_ptr<TorchSymIntSymArg> symint_symarg =
        std::make_unique<TorchSymIntSymArg>(param_name);
    if (param_type.contains("default_value"))
      symint_symarg->set_default(param_type["default_value"]);
    symarg = std::move(symint_symarg);
  } else if (param_type_kind == "UnsignedInt") {
    std::unique_ptr<TorchUnsignedIntSymArg> uint_symarg =
        std::make_unique<TorchUnsignedIntSymArg>(param_name);
    if (param_type.contains("default_value"))
      uint_symarg->set_default(param_type["default_value"]);
    symarg = std::move(uint_symarg);
  } else if (param_type_kind == "Dtype") {
    symarg = std::make_unique<TorchDtypeSymArg>(param_name);
  } else if (param_type_kind == "Bool") {
    symarg = std::make_unique<TorchBoolSymArg>(param_name);
  } else if (param_type_kind == "String") {
    bool is_view = param_type["is_view"];
    symarg = std::make_unique<TorchStringSymArg>(param_name, is_view);
  } else if (param_type_kind == "Float") {
    symarg = std::make_unique<TorchFloatSymArg>(param_name);
  } else if (param_type_kind == "Double") {
    symarg = std::make_unique<TorchDoubleSymArg>(param_name);
  } else if (param_type_kind == "MemoryFormat") {
    symarg = std::make_unique<TorchMemoryFormatSymArg>(param_name);
  } else if (param_type_kind == "Layout") {
    symarg = std::make_unique<TorchLayoutSymArg>(param_name);
  } else if (param_type_kind == "Device") {
    symarg = std::make_unique<TorchDeviceSymArg>(param_name);
  } else if (param_type_kind == "Variant") {
    assert(param_type["types"].is_array());
    std::vector<std::unique_ptr<TorchSymArg>> symargs;
    size_t i = 0;
    for (auto& type : param_type["types"]) {
      if (is_unknown_type(type)) return nullptr;
      symargs.push_back(
          gen_torch_symarg(param_name + "_" + std::to_string(i), type));
      i++;
    }
    symarg =
        std::make_unique<TorchVariantSymArg>(param_name, std::move(symargs));
  } else if (param_type_kind == "Vector") {
    assert(param_type["value_type"].is_object());
    if (is_unknown_type(param_type["value_type"])) {
      symarg = std::make_unique<TorchVectorSymArg>(
          param_name, param_type["value_type"]["type_str"]);
    } else {
      std::vector<std::unique_ptr<TorchSymArg>> symargs;
      for (size_t i = 0; i < TORCH_MAX_VECTOR_SIZE; i++)
        symargs.push_back(gen_torch_symarg(param_name + "_" + std::to_string(i),
                                           param_type["value_type"]));
      symarg =
          std::make_unique<TorchVectorSymArg>(param_name, std::move(symargs));
    }
  } else if (param_type_kind == "ArrayRef") {
    assert(param_type["value_type"].is_object());
    if (is_unknown_type(param_type["value_type"])) {
      symarg = std::make_unique<TorchArrayRefSymArg>(
          param_name, param_type["value_type"]["type_str"]);
    } else {
      std::vector<std::unique_ptr<TorchSymArg>> symargs;
      for (size_t i = 0; i < TORCH_MAX_ARRAYREF_SIZE; i++)
        symargs.push_back(gen_torch_symarg(param_name + "_" + std::to_string(i),
                                           param_type["value_type"]));
      symarg =
          std::make_unique<TorchArrayRefSymArg>(param_name, std::move(symargs));
    }
  } else if (param_type_kind == "OptionalArrayRef") {
    assert(param_type["value_type"].is_object());
    if (is_unknown_type(param_type["value_type"])) {
      std::unique_ptr<TorchArrayRefSymArg> arrayref_symarg =
          std::make_unique<TorchArrayRefSymArg>(
              param_name + "_arrayref", param_type["value_type"]["type_str"]);
      symarg = std::make_unique<TorchOptionalSymArg>(
          param_name, std::move(arrayref_symarg));
    } else {
      std::vector<std::unique_ptr<TorchSymArg>> symargs;
      for (size_t i = 0; i < TORCH_MAX_ARRAYREF_SIZE; i++)
        symargs.push_back(gen_torch_symarg(param_name + "_" + std::to_string(i),
                                           param_type["value_type"]));
      std::unique_ptr<TorchArrayRefSymArg> arrayref_symarg =
          std::make_unique<TorchArrayRefSymArg>(param_name + "_arrayref",
                                                std::move(symargs));
      symarg = std::make_unique<TorchOptionalSymArg>(
          param_name, std::move(arrayref_symarg));
    }
  } else if (param_type_kind == "ExpandingArray") {
    size_t size = param_type["size"];
    assert(param_type["value_type"].is_object() &&
           !is_unknown_type(param_type["value_type"]));
    std::vector<std::unique_ptr<TorchSymArg>> symargs;
    for (size_t i = 0; i < size; i++)
      symargs.push_back(gen_torch_symarg(param_name + "_" + std::to_string(i),
                                         param_type["value_type"]));
    symarg = std::make_unique<TorchExpandingArraySymArg>(param_name, size,
                                                         std::move(symargs));
  } else if (param_type_kind == "ExpandingArrayWithOptionalElem") {
    size_t size = param_type["size"];
    assert(param_type["value_type"].is_object() &&
           !is_unknown_type(param_type["value_type"]));
    std::vector<std::unique_ptr<TorchSymArg>> symargs;
    for (size_t i = 0; i < size; i++) {
      auto symarg_base =
          gen_torch_symarg(param_name + "_" + std::to_string(i) + "_base",
                           param_type["value_type"]);
      auto symarg_opt = std::make_unique<TorchOptionalSymArg>(
          param_name + "_" + std::to_string(i), std::move(symarg_base));
      symargs.push_back(std::move(symarg_opt));
    }
    symarg = std::make_unique<TorchExpandingArrayWithOptionalElemSymArg>(
        param_name, size, std::move(symargs));
  } else if (param_type_kind == "Tuple") {
    assert(param_type["types"].is_array());
    size_t size = param_type["types"].size();
    std::vector<std::unique_ptr<TorchSymArg>> symargs;
    size_t i = 0;
    for (auto& type : param_type["types"]) {
      assert(!is_unknown_type(type));
      symargs.push_back(
          gen_torch_symarg(param_name + "_" + std::to_string(i), type));
      i++;
    }
    symarg = std::make_unique<TorchTupleSymArg>(param_name, size,
                                                std::move(symargs));
  } else if (param_type_kind == "Pair") {
    assert(param_type["types"].is_array());
    size_t size = param_type["types"].size();
    assert(size == 2);
    std::vector<std::unique_ptr<TorchSymArg>> symargs;
    size_t i = 0;
    for (auto& type : param_type["types"]) {
      assert(!is_unknown_type(type));
      symargs.push_back(
          gen_torch_symarg(param_name + "_" + std::to_string(i), type));
      i++;
    }
    symarg = std::make_unique<TorchPairSymArg>(param_name, std::move(symargs));
  } else if (param_type_kind == "Tensor") {
    symarg = std::make_unique<TorchTensorSymArg>(param_name);
  } else if (param_type_kind == "Scalar") {
    symarg = std::make_unique<TorchScalarSymArg>(param_name);
  } else if (param_type_kind == "Optional") {
    assert(param_type["value_type"].is_object());
    if (is_unknown_type(param_type["value_type"])) {
      symarg = std::make_unique<TorchOptionalSymArg>(
          param_name, param_type["value_type"]["type_str"]);
    } else {
      std::unique_ptr<TorchSymArg> base =
          gen_torch_symarg(param_name + "_base", param_type["value_type"]);
      symarg =
          std::make_unique<TorchOptionalSymArg>(param_name, std::move(base));
    }
  } else if (param_type_kind == "APIOptions") {
    std::string class_name = param_type["class_name"];
    std::vector<std::unique_ptr<TorchSymArg>> ctor_symargs;
    for (auto& ctor_param : param_type["ctor_params"]) {
      std::string ctor_param_name = ctor_param["param_name"];
      const json& ctor_param_type = ctor_param["param_type"];
      std::unique_ptr<TorchSymArg> ctor_symarg =
          gen_torch_symarg(ctor_param_name, ctor_param_type);
      if (ctor_symarg == nullptr) return nullptr;
      ctor_symargs.push_back(std::move(ctor_symarg));
    }
    std::vector<std::unique_ptr<TorchSymArg>> member_symargs;
    for (auto& member_param : param_type["member_params"]) {
      std::string member_param_name = member_param["param_name"];
      const json& member_param_type = member_param["param_type"];
      std::unique_ptr<TorchSymArg> member_symarg =
          gen_torch_symarg(member_param_name, member_param_type);
      if (member_symarg == nullptr) continue;
      member_symargs.push_back(std::move(member_symarg));
    }
    symarg = std::make_unique<TorchAPIOptionsSymArg>(param_name, class_name,
                                                     std::move(ctor_symargs),
                                                     std::move(member_symargs));
  }

  return symarg;
}

std::unique_ptr<TorchFunction> read_torch_func_sig(const json& torch_api_sig) {
  assert(torch_api_sig["API_kind"] == "Function");
  const json& params = torch_api_sig["params"];
  assert(params.is_array());
  std::vector<std::unique_ptr<TorchSymArg>> symargs;
  for (auto& param : params) {
    std::string param_name = param["param_name"];
    const json& param_type = param["param_type"];
    std::unique_ptr<TorchSymArg> symarg =
        gen_torch_symarg(param_name, param_type);
    if (symarg == nullptr) return nullptr;
    symargs.push_back(std::move(symarg));
  }
  return std::make_unique<TorchFunction>(torch_api_sig["API_name"],
                                         std::move(symargs));
}

std::unique_ptr<TorchTensorMethod> read_torch_tensor_method_sig(
    const json& torch_api_sig) {
  assert(torch_api_sig["API_kind"] == "TensorMethod");
  std::unique_ptr<TorchTensorSymArg> self =
      std::make_unique<TorchTensorSymArg>(tensor_method_self_var);
  const json& params = torch_api_sig["params"];
  assert(params.is_array());
  std::vector<std::unique_ptr<TorchSymArg>> symargs;
  for (auto& param : params) {
    std::string param_name = param["param_name"];
    const json& param_type = param["param_type"];
    std::unique_ptr<TorchSymArg> symarg =
        gen_torch_symarg(param_name, param_type);
    if (symarg == nullptr) return nullptr;
    symargs.push_back(std::move(symarg));
  }
  return std::make_unique<TorchTensorMethod>(
      torch_api_sig["API_name"], std::move(self), std::move(symargs));
}

std::unique_ptr<TorchModule> read_torch_module_sig(const json& torch_api_sig) {
  assert(torch_api_sig["API_kind"] == "Module");

  std::unique_ptr<TorchDtypeSymArg> module_dtype =
      std::make_unique<TorchDtypeSymArg>("module_dtype");

  const json& ctor_params = torch_api_sig["ctor_params"];
  assert(ctor_params.is_array());
  std::vector<std::unique_ptr<TorchSymArg>> ctor_symargs;
  for (auto& ctor_param : ctor_params) {
    std::string param_name = ctor_param["param_name"];
    const json& param_type = ctor_param["param_type"];
    std::unique_ptr<TorchSymArg> ctor_symarg =
        gen_torch_symarg(param_name, param_type);
    if (ctor_symarg == nullptr) return nullptr;
    ctor_symargs.push_back(std::move(ctor_symarg));
  }

  const json& forward_params = torch_api_sig["forward_params"];
  assert(forward_params.is_array());
  std::vector<std::unique_ptr<TorchSymArg>> forward_symargs;
  for (auto& forward_param : forward_params) {
    std::string param_name = forward_param["param_name"];
    const json& param_type = forward_param["param_type"];
    std::unique_ptr<TorchSymArg> forward_symarg =
        gen_torch_symarg(param_name, param_type);
    if (forward_symarg == nullptr) return nullptr;
    forward_symargs.push_back(std::move(forward_symarg));
  }

  return std::make_unique<TorchModule>(
      torch_api_sig["API_name"], std::move(module_dtype),
      std::move(ctor_symargs), std::move(forward_symargs));
}

std::map<std::string, std::unique_ptr<TorchAPI>> read_torch_api_sigs() {
  std::string file_name = "torch" + DLL_VERSION + ".json";
  fs::path torch_api_sig_filepath = api_signatures_dir_path() / file_name;
  std::ifstream torch_api_sig_file(torch_api_sig_filepath);
  json torch_api_sigs = json::parse(torch_api_sig_file);
  assert(torch_api_sigs.is_array());

  std::map<std::string, std::unique_ptr<TorchAPI>> apis;
  for (auto& torch_api_sig : torch_api_sigs) {
    std::string api_name = torch_api_sig["API_name"];
    std::string api_kind = torch_api_sig["API_kind"];

    // std::cout << "Generating API `" << api_name << "`..." << std::endl;

    std::unique_ptr<TorchAPI> api;
    if (api_kind == "Function") {
      api = std::move(read_torch_func_sig(torch_api_sig));
    } else if (api_kind == "TensorMethod") {
      api = std::move(read_torch_tensor_method_sig(torch_api_sig));
    } else if (api_kind == "Module") {
      api = std::move(read_torch_module_sig(torch_api_sig));
    } else {
      UNREACHABLE;
    }

    if (api == nullptr) continue;

    if (apis.find(api_name) == apis.end())
      apis.insert({api_name, std::move(api)});
  }

  return apis;
}

void gen_torch_drivers(const fs::path& output_dir) {
  std::map<std::string, std::unique_ptr<TorchAPI>> apis = read_torch_api_sigs();

  TorchFuncFDG generator_func;
  TorchTensorMethodFDG generator_tensor_method;
  TorchModuleFDG generator_module;

  Writer writer("torch", "driver", output_dir);
  for (auto& api : apis) {
    std::string api_name = api.first;
    if (auto torch_tensor_method =
            dynamic_cast<TorchTensorMethod*>(api.second.get()))
      api_name = "torch::Tensor::" + api_name;
    api_name = std::regex_replace(api_name, std::regex("::"), "_");
    std::string code;
    if (auto func_api = dynamic_cast<TorchFunction*>(api.second.get())) {
      code = generator_func.gen(api.second.get());
    } else if (auto tensor_method_api =
                   dynamic_cast<TorchTensorMethod*>(api.second.get())) {
      code = generator_tensor_method.gen(api.second.get());
    } else if (auto module_api = dynamic_cast<TorchModule*>(api.second.get())) {
      code = generator_module.gen(api.second.get());
    }
    writer.add(api_name, code);
  }
  writer.write();
}

std::string gen_torch_pov_code(const std::string& target_api,
                               std::vector<long> concrete_args) {
  std::map<std::string, std::unique_ptr<TorchAPI>> apis = read_torch_api_sigs();

  TorchFuncPOVG generator_func;
  TorchTensorMethodPOVG generator_tensor_method;
  TorchModulePOVG generator_module;

  for (auto& api : apis) {
    std::string api_name = api.first;
    if (auto torch_tensor_method =
            dynamic_cast<TorchTensorMethod*>(api.second.get()))
      api_name = "torch::Tensor::" + api_name;
    api_name = std::regex_replace(api_name, std::regex("::"), "_");
    if (api_name == target_api) {
      api.second->assign(concrete_args);
      std::string code;
      if (auto func_api = dynamic_cast<TorchFunction*>(api.second.get())) {
        return generator_func.gen(api.second.get());
      } else if (auto tensor_method_api =
                     dynamic_cast<TorchTensorMethod*>(api.second.get())) {
        return generator_tensor_method.gen(api.second.get());
      } else if (auto module_api =
                     dynamic_cast<TorchModule*>(api.second.get())) {
        return generator_module.gen(api.second.get());
      }
    }
  }
  PDG_CHECK(false, "Unknown API `" + target_api + "`");
}

void gen_torch_pov(const fs::path& buggy_input_dir,
                   const fs::path& output_dir) {
  Writer writer("torch", "pov", output_dir);
  for (const auto& entry : fs::directory_iterator(buggy_input_dir)) {
    auto path = entry.path();
    if (fs::is_directory(path)) {
      std::string api_name = path.filename();
      size_t idx = 1;
      for (const auto& entry : fs::directory_iterator(path)) {
        auto seed_path = entry.path();
        if (!fs::is_regular_file(seed_path)) continue;
        std::vector<long> concrete_args =
            uint8_vec_to_long_vec(file_to_vector(seed_path));
        std::string file_name;
        if (idx == 1) {
          file_name = api_name;
        } else {
          file_name = api_name + "_" + std::to_string(idx);
        }
        std::string code = gen_torch_pov_code(api_name, concrete_args);
        writer.add(file_name, code);
        idx++;
      }
    }
  }
  writer.write();
}
