#include "tf_api.h"

#include <cassert>

const std::string target_var = "target";

TFAPI::TFAPI(std::string api_name_,
             std::vector<std::unique_ptr<TFSymArg>> dependent_symargs_,
             std::vector<std::unique_ptr<TFSymArg>> required_symargs_,
             std::vector<std::unique_ptr<TFSymArg>> optional_symargs_)
    : API(api_name_, {callback_input_var, tf_scope_var, target_var}) {
  std::unique_ptr<TFScopeSymArg> scope_ = std::make_unique<TFScopeSymArg>();
  scope = scope_.get();
  symargs.push_back(std::move(scope_));

  for (auto& symarg : dependent_symargs_) {
    dependent_symargs.push_back(symarg.get());
    symargs.push_back(std::move(symarg));
  }

  for (auto& symarg : required_symargs_) {
    ordinary_symargs.push_back(symarg.get());
    symargs.push_back(std::move(symarg));
  }

  if (!optional_symargs_.empty()) {
    std::vector<std::pair<std::string, std::unique_ptr<TFSymArg>>> setters;
    for (auto& symarg : optional_symargs_) {
      std::string setter_name = snake_to_camel(symarg->get_name());
      setters.push_back({setter_name, std::move(symarg)});
    }
    auto attrs = std::make_unique<TFAPIAttrsSymArg>(
        "attrs", get_qualified_name() + "::Attrs", std::move(setters));
    ordinary_symargs.push_back(attrs.get());
    symargs.push_back(std::move(attrs));
  }

  resolve_name_conflict();
}
std::string TFAPI::get_qualified_name() const { return "ops::" + get_name(); }
std::string TFAPI::get_unqualified_name() const { return get_name(); }
std::string TFAPI::pathfinder_setup_sig() const {
  return "void PathFinderSetup_" + get_unqualified_name() + "()";
}
std::string TFAPI::pathfinder_test_one_input_sig() const {
  return "int PathFinderTestOneInput_" + get_unqualified_name() +
         "(const pathfinder::Input& " + callback_input_var + ")";
}
TFScopeSymArg* TFAPI::get_scope() const { return scope; }
const std::vector<TFSymArg*>& TFAPI::get_ordinary_symargs() const {
  return ordinary_symargs;
}
