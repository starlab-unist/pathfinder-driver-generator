#ifndef PATHFINDER_DRIVER_GENERATOR_TORCH_GENERATOR
#define PATHFINDER_DRIVER_GENERATOR_TORCH_GENERATOR

#include "generator.h"
#include "torch_api.h"
#include "torch_symarg.h"

class TorchFDG : public FuzzDriverGenerator<TorchAPI> {
 public:
  TorchFDG();

 private:
  virtual std::vector<std::string> header() const override;
  virtual std::vector<std::string> setup() const override;
  virtual std::vector<std::string> callback() const override;
  virtual std::vector<std::string> footer() const override;
  virtual std::vector<std::string> api_call_code() const = 0;
};

class TorchFuncFDG : public TorchFDG {
 public:
  TorchFuncFDG();

 private:
  virtual std::vector<std::string> api_call_code() const override;
};

extern const std::string tensor_method_self_var;

class TorchTensorMethodFDG : public TorchFDG {
 public:
  TorchTensorMethodFDG();

 private:
  virtual std::vector<std::string> api_call_code() const override;
};

class TorchModuleFDG : public TorchFDG {
 public:
  TorchModuleFDG();

 private:
  virtual std::vector<std::string> api_call_code() const override;
};

class TorchPOVG : public POVGenerator<TorchAPI> {
 public:
  TorchPOVG();

 private:
  virtual std::vector<std::string> header() const override;
  virtual std::vector<std::string> callback() const override;
  virtual std::vector<std::string> api_call_code() const = 0;
};

class TorchFuncPOVG : public TorchPOVG {
 public:
  TorchFuncPOVG();

 private:
  virtual std::vector<std::string> api_call_code() const override;
};

extern const std::string tensor_method_self_var;

class TorchTensorMethodPOVG : public TorchPOVG {
 public:
  TorchTensorMethodPOVG();

 private:
  virtual std::vector<std::string> api_call_code() const override;
};

class TorchModulePOVG : public TorchPOVG {
 public:
  TorchModulePOVG();

 private:
  virtual std::vector<std::string> api_call_code() const override;
};
#endif
