#ifndef PATHFINDER_DRIVER_GENERATOR_TORCH_API
#define PATHFINDER_DRIVER_GENERATOR_TORCH_API

#include "api.h"
#include "torch_symarg.h"

class TorchAPI : public API {
 public:
  TorchAPI(std::string api_name_);
  virtual ~TorchAPI() = default;
};

class TorchFunction : public TorchAPI {
 public:
  TorchFunction(std::string func_name,
                std::vector<std::unique_ptr<TorchSymArg>> symargs_);
};

class TorchTensorMethod : public TorchAPI {
 public:
  TorchTensorMethod(std::string method_name,
                    std::unique_ptr<TorchTensorSymArg> self_,
                    std::vector<std::unique_ptr<TorchSymArg>> symargs_);
  TorchTensorSymArg* get_self();

 private:
  TorchTensorSymArg* self;
};

class TorchModule : public TorchAPI {
 public:
  TorchModule(std::string module_name,
              std::unique_ptr<TorchDtypeSymArg> module_dtype_,
              std::vector<std::unique_ptr<TorchSymArg>> ctor_symargs_,
              std::vector<std::unique_ptr<TorchSymArg>> forward_symargs_);
  const TorchDtypeSymArg* get_module_dtype() const;
  const std::vector<TorchSymArg*>& get_ctor_symargs() const;
  const std::vector<TorchSymArg*>& get_forward_symargs() const;

 private:
  TorchDtypeSymArg* module_dtype;
  std::vector<TorchSymArg*> ctor_symargs;
  std::vector<TorchSymArg*> forward_symargs;
};

#endif
