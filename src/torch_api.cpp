#include "torch_api.h"

TorchAPI::TorchAPI(std::string api_name_)
    : API(api_name_, {callback_input_var}) {}

TorchFunction::TorchFunction(std::string func_name,
                             std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchAPI(func_name) {
  symargs = std::move(symargs_);

  resolve_name_conflict();
}

TorchTensorMethod::TorchTensorMethod(
    std::string method_name, std::unique_ptr<TorchTensorSymArg> self_,
    std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchAPI(method_name) {
  self = self_.get();
  symargs.push_back(std::move(self_));
  for (auto& symarg : symargs_) symargs.push_back(std::move(symarg));

  resolve_name_conflict();
}
TorchTensorSymArg* TorchTensorMethod::get_self() { return self; }

TorchModule::TorchModule(
    std::string module_name, std::unique_ptr<TorchDtypeSymArg> module_dtype_,
    std::vector<std::unique_ptr<TorchSymArg>> ctor_symargs_,
    std::vector<std::unique_ptr<TorchSymArg>> forward_symargs_)
    : TorchAPI(module_name) {
  module_dtype = module_dtype_.get();
  symargs.push_back(std::move(module_dtype_));
  for (auto& symarg : forward_symargs_) {
    forward_symargs.push_back(symarg.get());
    symargs.push_back(std::move(symarg));
  }
  for (auto& symarg : ctor_symargs_) {
    ctor_symargs.push_back(symarg.get());
    symargs.push_back(std::move(symarg));
  }

  resolve_name_conflict();
}
const TorchDtypeSymArg* TorchModule::get_module_dtype() const {
  return module_dtype;
}
const std::vector<TorchSymArg*>& TorchModule::get_ctor_symargs() const {
  return ctor_symargs;
}
const std::vector<TorchSymArg*>& TorchModule::get_forward_symargs() const {
  return forward_symargs;
}
